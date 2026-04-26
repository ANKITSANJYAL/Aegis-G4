"""
models.py — Aegis-G4
Async model client wrappers for the Orchestrator and Auditor models
via vLLM's OpenAI-compatible API.

Model selection (override via environment variables):
  AEGIS_ORCHESTRATOR_MODEL  — default: google/gemma-4-e4b-it
  AEGIS_AUDITOR_MODEL       — default: google/gemma-4-e4b-it

Memory guide for vLLM:
  gemma-4-e4b-it  ~8 GB float16  — fits on a single T4 (14.5 GB)
  gemma-4-12b-it  ~24 GB float16 — requires 2xT4 (tensor_parallel_size=2)
  gemma-4-31b-it  ~62 GB float16 — requires 4xA100 or larger; does NOT
                                    fit on 2xT4 even with tensor parallelism

Recommended Kaggle 2xT4 setup (one model per GPU):
  CUDA_VISIBLE_DEVICES=0 vllm serve $AEGIS_ORCHESTRATOR_MODEL --port 8000
  CUDA_VISIBLE_DEVICES=1 vllm serve $AEGIS_AUDITOR_MODEL --port 8001

Locally (dev/test):
  Both models fall back to a deterministic MockModelClient so the
  graph runs end-to-end without GPU hardware.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
import uuid
from abc import ABC, abstractmethod
from typing import Any, Optional

import structlog

logger = structlog.get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Base Client Interface
# ─────────────────────────────────────────────────────────────────────────────

class BaseModelClient(ABC):
    """Abstract base for all model clients."""

    @abstractmethod
    async def chat_complete(
        self,
        messages: list[dict[str, str]],
        tools: Optional[list[dict]] = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        thinking: bool = False,
    ) -> dict[str, Any]:
        """
        Send a chat completion request.

        Returns a dict with:
            - content (str): The model's text response
            - tool_calls (list[dict] | None): Any function calls
            - thinking (str | None): Chain-of-thought if enabled
            - usage (dict): Token usage stats
            - latency_ms (float): Wall-clock call time
        """
        ...


# ─────────────────────────────────────────────────────────────────────────────
# vLLM OpenAI-Compatible Client
# ─────────────────────────────────────────────────────────────────────────────

class VLLMModelClient(BaseModelClient):
    """
    Async client for Gemma 4 served via vLLM's OpenAI-compatible endpoint.
    Handles Gemma 4's native function calling and thinking mode.
    """

    def __init__(
        self,
        model_id: str,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        request_timeout: float = 120.0,
    ) -> None:
        self.model_id = model_id
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.request_timeout = request_timeout
        self._session: Optional[Any] = None

    async def _get_session(self) -> Any:
        """Lazy-initialize aiohttp session."""
        if self._session is None:
            try:
                import aiohttp
                self._session = aiohttp.ClientSession(
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    timeout=aiohttp.ClientTimeout(total=self.request_timeout),
                )
            except ImportError as exc:
                raise RuntimeError(
                    "aiohttp is required for VLLMModelClient. "
                    "Install with: pip install aiohttp"
                ) from exc
        return self._session

    async def chat_complete(
        self,
        messages: list[dict[str, str]],
        tools: Optional[list[dict]] = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        thinking: bool = False,
    ) -> dict[str, Any]:
        log = logger.bind(model=self.model_id)
        start = time.monotonic()

        payload: dict[str, Any] = {
            "model": self.model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        # Gemma 4 thinking mode — enable extended reasoning
        if thinking:
            payload["thinking"] = {
                "type": "enabled",
                "budget_tokens": 1024,
            }

        session = await self._get_session()
        try:
            async with session.post(
                f"{self.base_url}/chat/completions",
                json=payload,
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
        except Exception as exc:
            log.error("vllm_request_failed", error=str(exc))
            raise

        latency_ms = (time.monotonic() - start) * 1000
        choice = data["choices"][0]
        message = choice["message"]

        # Extract thinking trace if present
        thinking_trace = None
        if thinking and "thinking" in message:
            thinking_trace = message["thinking"]

        # Parse tool calls
        tool_calls = None
        if message.get("tool_calls"):
            tool_calls = []
            for tc in message["tool_calls"]:
                fn = tc.get("function", {})
                try:
                    args = json.loads(fn.get("arguments", "{}"))
                except json.JSONDecodeError:
                    args = {}
                tool_calls.append({
                    "id": tc.get("id", str(uuid.uuid4())),
                    "name": fn.get("name", ""),
                    "arguments": args,
                })

        log.info(
            "vllm_call_complete",
            latency_ms=round(latency_ms, 1),
            finish_reason=choice.get("finish_reason"),
            prompt_tokens=data.get("usage", {}).get("prompt_tokens", 0),
            completion_tokens=data.get("usage", {}).get("completion_tokens", 0),
        )

        return {
            "content": message.get("content") or "",
            "tool_calls": tool_calls,
            "thinking": thinking_trace,
            "usage": data.get("usage", {}),
            "latency_ms": latency_ms,
        }

    async def close(self) -> None:
        if self._session:
            await self._session.close()
            self._session = None


# ─────────────────────────────────────────────────────────────────────────────
# LiteRT (on-device) Client Wrapper
# ─────────────────────────────────────────────────────────────────────────────

class LiteRTModelClient(BaseModelClient):
    """
    Client for Gemma 4 E4B running via Google AI Edge LiteRT (TFLite successor).
    Intended for the Auditor on-device deployment path.
    """

    def __init__(
        self,
        model_path: str,
        num_threads: int = 4,
    ) -> None:
        self.model_path = model_path
        self.num_threads = num_threads
        self._model: Optional[Any] = None

    def _load_model(self) -> None:
        try:
            from ai_edge_litert.interpreter import Interpreter  # type: ignore
            self._model = Interpreter(
                model_path=self.model_path,
                num_threads=self.num_threads,
            )
            self._model.allocate_tensors()
        except ImportError as exc:
            raise RuntimeError(
                "ai-edge-litert package required. Install with: pip install ai-edge-litert"
            ) from exc

    async def chat_complete(
        self,
        messages: list[dict[str, str]],
        tools: Optional[list[dict]] = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        thinking: bool = False,
    ) -> dict[str, Any]:
        # LiteRT is synchronous — run in executor
        return await asyncio.get_running_loop().run_in_executor(
            None,
            self._sync_infer,
            messages,
            temperature,
            max_tokens,
        )

    def _sync_infer(
        self,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
    ) -> dict[str, Any]:
        if self._model is None:
            self._load_model()
        # Simplified inference stub — real implementation would use
        # the LiteRT token streaming API
        raise NotImplementedError(
            "LiteRT inference pipeline requires model-specific tokenizer integration. "
            "See Kaggle-entrypoint.ipynb for the full setup."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Mock Client (for local dev / CI without GPU)
# ─────────────────────────────────────────────────────────────────────────────

class MockOrchestratorClient(BaseModelClient):
    """
    Deterministic mock of gemma-4-31b-it.
    Emits realistic function-calling JSON for testing the graph flow.
    """

    async def chat_complete(
        self,
        messages: list[dict[str, str]],
        tools: Optional[list[dict]] = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        thinking: bool = False,
    ) -> dict[str, Any]:
        await asyncio.sleep(0.1)  # Simulate latency

        # Extract patient context from the user message
        user_msg = next(
            (m["content"] for m in messages if m["role"] == "user"), ""
        )

        # Determine action type from the request
        if "lab" in user_msg.lower() or "blood test" in user_msg.lower():
            return {
                "content": None,
                "tool_calls": [{
                    "id": str(uuid.uuid4()),
                    "name": "order_lab_test",
                    "arguments": {
                        "patient_id": "PT-TOKEN-7842",
                        "test_code": "HbA1c",
                        "priority": "ROUTINE",
                        "indication_icd10": "E11.9",
                        "ordering_provider_id": "NPI-1234567890",
                    },
                }],
                "thinking": None,
                "usage": {"prompt_tokens": 512, "completion_tokens": 128},
                "latency_ms": 100.0,
            }
        else:
            return {
                "content": None,
                "tool_calls": [{
                    "id": str(uuid.uuid4()),
                    "name": "prescribe_medication",
                    "arguments": {
                        "patient_id": "PT-TOKEN-7842",
                        "drug_name": "metformin",
                        "dose": "1000mg",
                        "frequency": "twice daily",
                        "duration_days": 90,
                        "indication_icd10": "E11.9",
                        "prescriber_id": "NPI-1234567890",
                    },
                }],
                "thinking": None,
                "usage": {"prompt_tokens": 480, "completion_tokens": 112},
                "latency_ms": 95.0,
            }


class MockAuditorClient(BaseModelClient):
    """
    Deterministic mock of gemma-4-e4b-it.
    Returns a structured compliance reasoning trace with policy citations.
    """

    def __init__(self, always_approve: bool = True) -> None:
        self.always_approve = always_approve

    async def chat_complete(
        self,
        messages: list[dict[str, str]],
        tools: Optional[list[dict]] = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        thinking: bool = False,
    ) -> dict[str, Any]:
        await asyncio.sleep(0.08)

        thinking_trace = (
            "Step 1: Identify the proposed action type — PRESCRIPTION of metformin 1000mg.\n"
            "Step 2: Retrieve relevant NIST policies. NIST-H-03 (Drug-Allergy Verification) "
            "is most critical. NIST-H-01 (Transparency) applies universally. "
            "NIST-H-02 (High-Risk Medications) — metformin is not a controlled substance, "
            "so this does NOT apply. NIST-H-05 (PHI Minimization) — patient_id uses 'PT-TOKEN' "
            "prefix indicating proper tokenization, compliant.\n"
            "Step 3: Cross-reference patient allergies. No penicillin/sulfa interaction with "
            "metformin. Drug class is Biguanide — no cross-reactivity documented.\n"
            "Step 4: Assess overall risk. Metformin is a first-line T2DM medication. "
            "Risk score: 1.5/10. Verdict: APPROVED.\n"
        )

        verdict_text = "APPROVED" if self.always_approve else "REJECTED"
        content = json.dumps({
            "verdict": verdict_text,
            "overall_risk_score": 1.5 if self.always_approve else 8.5,
            "auditor_confidence": 0.95 if self.always_approve else 0.92,
            "summary": (
                "Proposal is compliant with all applicable NIST AI RMF 2026 Healthcare policies. "
                "No allergy conflicts detected. PHI is properly tokenized."
                if self.always_approve else
                "Proposal rejected. One or more mandatory NIST AI RMF 2026 Healthcare policies "
                "are violated. Manual clinician review required before any action is taken."
            ),
            "compliance_citations": [
                {
                    "policy_id": "NIST-H-01",
                    "policy_title": "AI System Transparency in Clinical Decision Support",
                    "relevance_score": 0.92,
                    "violated": False,
                    "violation_detail": None,
                    "remediation_suggestion": None,
                },
                {
                    "policy_id": "NIST-H-03",
                    "policy_title": "Drug-Allergy Interaction Verification",
                    "relevance_score": 0.98,
                    "violated": False,
                    "violation_detail": None,
                    "remediation_suggestion": None,
                },
                {
                    "policy_id": "NIST-H-05",
                    "policy_title": "PHI Minimization in AI Inference Pipelines",
                    "relevance_score": 0.85,
                    "violated": False,
                    "violation_detail": None,
                    "remediation_suggestion": None,
                },
            ],
        })

        return {
            "content": content,
            "tool_calls": None,
            "thinking": thinking_trace,
            "usage": {"prompt_tokens": 890, "completion_tokens": 420},
            "latency_ms": 80.0,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Client Factory
# ─────────────────────────────────────────────────────────────────────────────

_ORCHESTRATOR_MODEL = os.getenv("AEGIS_ORCHESTRATOR_MODEL", "google/gemma-4-e4b-it")
_AUDITOR_MODEL = os.getenv("AEGIS_AUDITOR_MODEL", "google/gemma-4-e4b-it")


def create_orchestrator_client(
    mode: str = "mock",
    vllm_base_url: str = "http://localhost:8000/v1",
) -> BaseModelClient:
    """
    Factory for the Orchestrator model client.

    Args:
        mode: "vllm" | "mock"
        vllm_base_url: Base URL of the vLLM server (for mode="vllm")
    """
    if mode == "vllm":
        return VLLMModelClient(
            model_id=_ORCHESTRATOR_MODEL,
            base_url=vllm_base_url,
        )
    return MockOrchestratorClient()


def create_auditor_client(
    mode: str = "mock",
    vllm_base_url: str = "http://localhost:8001/v1",
    litert_model_path: Optional[str] = None,
) -> BaseModelClient:
    """
    Factory for the Auditor model client.

    Args:
        mode: "vllm" | "litert" | "mock"
        vllm_base_url: Base URL of the vLLM server (for mode="vllm")
        litert_model_path: Path to the LiteRT model file (for mode="litert")
    """
    if mode == "vllm":
        return VLLMModelClient(
            model_id=_AUDITOR_MODEL,
            base_url=vllm_base_url,
        )
    if mode == "litert":
        if not litert_model_path:
            raise ValueError("litert_model_path must be provided for mode='litert'")
        return LiteRTModelClient(model_path=litert_model_path)
    return MockAuditorClient()
