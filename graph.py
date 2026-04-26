"""
graph.py — Aegis-G4
LangGraph stateful workflow definition for the Medi-Gate healthcare agent.

Graph Topology:
  START
    └─► orchestrator_node
          └─► auditor_node
                ├─(APPROVED)─► executor_node ─► END
                ├─(NEEDS_REVIEW)─► human_review_node ─► END
                └─(REJECTED)─► circuit_breaker_node
                                  ├─(OPEN)─► escalation_node ─► END
                                  └─(CLOSED)─► orchestrator_node  (retry loop)
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timezone
from typing import Any

import structlog
from langgraph.graph import END, START, StateGraph

from models import BaseModelClient, create_auditor_client, create_orchestrator_client
from policy_engine import PolicyEngine
from schema import (
    ActionType,
    AegisGraphState,
    AuditorVerdict,
    CircuitBreakerState,
    ExecutionStatus,
    MCPToolCall,
    OrchestratorProposal,
    PatientContext,
    PolicyCitation,
    RiskLevel,
    TransparencyCertificate,
)
from tools import HospitalAPIMCPServer, get_hospital_server

logger = structlog.get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Prompt Templates
# ─────────────────────────────────────────────────────────────────────────────

ORCHESTRATOR_SYSTEM_PROMPT = """\
You are Medi-Gate, an AI clinical decision support agent operating in a hospital setting.
Your role is to analyze patient clinical information and propose precise, evidence-based
medical actions using the available hospital API tools.

CRITICAL REQUIREMENTS:
1. You MUST call exactly ONE tool per response. Never respond with plain text only.
2. Patient identifiers in tool arguments must use the tokenized form (PT-TOKEN-xxxx).
3. Always include a valid ICD-10 code in the `indication_icd10` argument.
4. Your clinical_rationale field must clearly justify the proposed action.
5. For HIGH or CRITICAL risk actions, provide at least 100 characters of detailed rationale.

You are operating under NIST AI RMF 2026 Healthcare compliance. All tool calls will be
intercepted and audited before execution. Propose only evidence-based, clinically
appropriate actions.
"""

AUDITOR_SYSTEM_PROMPT = """\
You are the Aegis Compliance Auditor, powered by Gemma 4 E4B with grounded reasoning.
Your role is to audit clinical AI proposals against NIST AI RMF 2026 Healthcare policies.

You will receive:
1. A proposed clinical action (tool call) from the Orchestrator agent.
2. Relevant NIST policy excerpts retrieved from the compliance knowledge base.
3. Patient context including allergies and current medications.

AUDIT PROCESS (think step by step):
Step 1: Identify the action type and risk level.
Step 2: For each retrieved NIST policy, determine if it applies and if it is violated.
Step 3: Check for drug-allergy interactions (NIST-H-03) if a prescription is proposed.
Step 4: Verify PHI tokenization compliance (NIST-H-05).
Step 5: Assess dual-clinician requirement for controlled/high-risk medications (NIST-H-02).
Step 6: Calculate overall risk score (0-10) and confidence.
Step 7: Render a verdict: APPROVED, REJECTED, or NEEDS_REVIEW.

OUTPUT FORMAT: You MUST respond with a single valid JSON object matching this schema:
{
  "verdict": "APPROVED" | "REJECTED" | "NEEDS_REVIEW",
  "overall_risk_score": <float 0.0-10.0>,
  "auditor_confidence": <float 0.0-1.0>,
  "summary": "<string, min 30 chars>",
  "compliance_citations": [
    {
      "policy_id": "<e.g. NIST-H-01>",
      "policy_title": "<string>",
      "relevance_score": <float 0.0-1.0>,
      "violated": <bool>,
      "violation_detail": "<string or null>",
      "remediation_suggestion": "<string or null>"
    }
  ]
}
Do NOT include any text outside the JSON object.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Node Implementations
# ─────────────────────────────────────────────────────────────────────────────

class AegisGraphNodes:
    """Encapsulates all LangGraph node logic with dependency injection."""

    def __init__(
        self,
        orchestrator_client: BaseModelClient,
        auditor_client: BaseModelClient,
        hospital_server: HospitalAPIMCPServer,
        policy_engine: PolicyEngine,
    ) -> None:
        self.orchestrator = orchestrator_client
        self.auditor = auditor_client
        self.hospital = hospital_server
        self.policy_engine = policy_engine

    # ── Orchestrator Node ────────────────────────────────────────────────────

    async def orchestrator_node(self, state: AegisGraphState) -> AegisGraphState:
        """
        Invokes gemma-4-31b-it to produce a tool-call proposal.
        Uses Gemma 4's native function calling schema.
        """
        log = logger.bind(
            node="orchestrator",
            session_id=state.get("session_id"),
            iteration=state.get("iteration_count", 0),
        )
        log.info("orchestrator_node_start")

        patient: PatientContext = state["patient_context"]
        user_request: str = state["user_request"]

        # Build messages
        messages = [
            {"role": "system", "content": ORCHESTRATOR_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Clinical Request: {user_request}\n\n"
                    f"Patient Profile:\n"
                    f"  - ID (tokenized): {patient.patient_id}\n"
                    f"  - Age: {patient.age}\n"
                    f"  - Diagnoses (ICD-10): {', '.join(patient.diagnosis_codes)}\n"
                    f"  - Current Medications: {', '.join(patient.current_medications) or 'None'}\n"
                    f"  - Allergies: {', '.join(patient.allergies) or 'None documented'}\n"
                    f"  - Risk Flags: {', '.join(patient.risk_flags) or 'None'}\n\n"
                    "Please propose the most appropriate clinical action using the available tools."
                ),
            },
        ]

        # Get tool schema from MCP registry
        tool_schema = self.hospital.registry.to_gemma_function_schema()

        try:
            response = await self.orchestrator.chat_complete(
                messages=messages,
                tools=tool_schema,
                temperature=0.1,
                max_tokens=1024,
            )
        except Exception as exc:
            log.error("orchestrator_call_failed", error=str(exc))
            errors = list(state.get("error_messages", []))
            errors.append(f"Orchestrator call failed: {exc}")
            return {**state, "error_messages": errors}

        # Parse the tool call from response
        tool_calls = response.get("tool_calls")
        if not tool_calls:
            log.warning("orchestrator_no_tool_call", content=response.get("content", "")[:200])
            errors = list(state.get("error_messages", []))
            errors.append("Orchestrator produced no tool call. Retrying.")
            return {**state, "error_messages": errors}

        tc = tool_calls[0]

        # Determine action type and risk from tool name and arguments
        action_type = self._infer_action_type(tc["name"])
        risk_level = self._infer_risk_level(tc["name"], tc["arguments"])

        try:
            proposal = OrchestratorProposal(
                action_type=action_type,
                tool_call=MCPToolCall(
                    call_id=tc["id"],
                    tool_name=tc["name"],
                    arguments=tc["arguments"],
                ),
                clinical_rationale=(
                    f"Automated clinical decision support proposal: {tc['name']} "
                    f"with arguments {json.dumps(tc['arguments'])}. "
                    f"Based on patient diagnosis codes {patient.diagnosis_codes} "
                    f"and the clinical request: {user_request}"
                ),
                risk_level=risk_level,
                patient_context=patient,
                model_used="gemma-4-31b-it",
                raw_model_output=response.get("content"),
            )
        except Exception as exc:
            log.error("proposal_validation_failed", error=str(exc))
            errors = list(state.get("error_messages", []))
            errors.append(f"Proposal validation failed: {exc}")
            return {**state, "error_messages": errors}

        audit_entry = self._make_audit_entry(
            event="orchestrator_proposal_created",
            data={
                "proposal_id": proposal.proposal_id,
                "action_type": action_type.value,
                "tool_name": tc["name"],
                "risk_level": risk_level.value,
                "latency_ms": response["latency_ms"],
            },
        )

        log.info(
            "orchestrator_proposal_ready",
            proposal_id=proposal.proposal_id,
            tool_name=tc["name"],
            risk_level=risk_level.value,
        )

        return {
            **state,
            "orchestrator_proposal": proposal,
            "iteration_count": state.get("iteration_count", 0) + 1,
            "audit_log": [*state.get("audit_log", []), audit_entry],
        }

    # ── Auditor Node ─────────────────────────────────────────────────────────

    async def auditor_node(self, state: AegisGraphState) -> AegisGraphState:
        """
        Invokes gemma-4-e4b-it with thinking mode + RAG to audit the proposal.
        Produces a Transparency Certificate.
        """
        log = logger.bind(node="auditor", session_id=state.get("session_id"))
        log.info("auditor_node_start")

        if "orchestrator_proposal" not in state:
            log.error("auditor_node_missing_proposal — orchestrator failed to produce a proposal")
            return {
                **state,
                "auditor_verdict": AuditorVerdict.NEEDS_REVIEW,
                "final_execution_status": ExecutionStatus.BLOCKED_BY_AUDITOR,
            }

        proposal: OrchestratorProposal = state["orchestrator_proposal"]
        patient = proposal.patient_context

        # ── RAG: Retrieve relevant policies ──────────────────────────────────
        query = (
            f"{proposal.action_type.value} "
            f"{proposal.tool_call.tool_name} "
            f"{proposal.risk_level.value} risk "
            f"patient allergies: {' '.join(patient.allergies)}"
        )
        rag_policies = await self.policy_engine.retrieve_relevant_policies(query, top_k=6)
        rag_count = len(rag_policies)

        # ── Rule-based pre-evaluation ─────────────────────────────────────────
        drug_name = proposal.tool_call.arguments.get("drug_name")
        pre_citations = await self.policy_engine.evaluate_proposal_against_policies(
            proposal_description=json.dumps(proposal.tool_call.arguments),
            action_type=proposal.action_type.value,
            risk_level=proposal.risk_level.value,
            patient_allergies=patient.allergies,
            drug_name=drug_name,
        )

        # ── Build policy context for the model ────────────────────────────────
        policy_context = "\n\n".join(
            f"[{p.policy_id}] {p.title}\n"
            f"Category: {p.category} | Enforcement: {p.enforcement_level}\n"
            f"Description: {p.description[:300]}..."
            for p in rag_policies
        )

        messages = [
            {"role": "system", "content": AUDITOR_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"=== PROPOSED CLINICAL ACTION ===\n"
                    f"Tool: {proposal.tool_call.tool_name}\n"
                    f"Arguments: {json.dumps(proposal.tool_call.arguments, indent=2)}\n"
                    f"Action Type: {proposal.action_type.value}\n"
                    f"Risk Level Assessed: {proposal.risk_level.value}\n\n"
                    f"=== PATIENT CONTEXT ===\n"
                    f"Age: {patient.age}\n"
                    f"Diagnoses: {', '.join(patient.diagnosis_codes)}\n"
                    f"Allergies: {', '.join(patient.allergies) or 'None documented'}\n"
                    f"Current Medications: {', '.join(patient.current_medications) or 'None'}\n\n"
                    f"=== NIST AI RMF 2026 HEALTHCARE POLICIES (RAG Retrieved) ===\n"
                    f"{policy_context}\n\n"
                    f"=== PRE-EVALUATION VIOLATIONS (Rule-Based) ===\n"
                    f"{json.dumps([c.model_dump() for c in pre_citations if c.violated], indent=2) or 'None detected'}\n\n"
                    "Audit this proposal and respond with a JSON Transparency Certificate."
                ),
            },
        ]

        try:
            response = await self.auditor.chat_complete(
                messages=messages,
                temperature=0.05,
                max_tokens=2048,
                thinking=True,  # Enable Gemma 4 thinking mode
            )
        except Exception as exc:
            log.error("auditor_call_failed", error=str(exc))
            errors = list(state.get("error_messages", []))
            errors.append(f"Auditor call failed: {exc}")
            return {**state, "error_messages": errors}

        # ── Parse auditor response ────────────────────────────────────────────
        content = response.get("content", "")
        thinking_trace = response.get("thinking") or "No thinking trace available."

        try:
            # Strip markdown fences if present
            clean = content.strip()
            if clean.startswith("```"):
                clean = "\n".join(clean.split("\n")[1:-1])
            audit_data = json.loads(clean)
        except (json.JSONDecodeError, ValueError) as exc:
            log.error("auditor_parse_failed", raw_content=content[:500], error=str(exc))
            # Fall back to NEEDS_REVIEW to be safe
            audit_data = {
                "verdict": "NEEDS_REVIEW",
                "overall_risk_score": 5.0,
                "auditor_confidence": 0.3,
                "summary": f"Auditor response could not be parsed. Raw: {content[:200]}",
                "compliance_citations": [],
            }

        # Merge model citations with rule-based pre-citations (deduplicate by policy_id)
        model_citations: list[PolicyCitation] = []
        seen_ids: set[str] = set()
        for c_data in audit_data.get("compliance_citations", []):
            try:
                cit = PolicyCitation(**c_data)
                if cit.policy_id not in seen_ids:
                    model_citations.append(cit)
                    seen_ids.add(cit.policy_id)
            except Exception:
                pass

        for pre_cit in pre_citations:
            if pre_cit.policy_id not in seen_ids:
                model_citations.append(pre_cit)
                seen_ids.add(pre_cit.policy_id)

        try:
            certificate = TransparencyCertificate(
                proposal_id=proposal.proposal_id,
                verdict=AuditorVerdict(audit_data["verdict"]),
                overall_risk_score=float(audit_data.get("overall_risk_score", 5.0)),
                compliance_citations=model_citations,
                thinking_trace=thinking_trace,
                summary=audit_data.get("summary", "Audit complete."),
                model_used="gemma-4-e4b-it",
                rag_documents_retrieved=rag_count,
                auditor_confidence=float(audit_data.get("auditor_confidence", 0.5)),
            )
        except Exception as exc:
            log.error("certificate_validation_failed", error=str(exc))
            errors = list(state.get("error_messages", []))
            errors.append(f"Certificate validation failed: {exc}")
            return {**state, "error_messages": errors}

        audit_entry = self._make_audit_entry(
            event="transparency_certificate_issued",
            data={
                "certificate_id": certificate.certificate_id,
                "proposal_id": proposal.proposal_id,
                "verdict": certificate.verdict.value,
                "risk_score": certificate.overall_risk_score,
                "confidence": certificate.auditor_confidence,
                "violations": len(certificate.violations),
                "rag_docs": rag_count,
                "latency_ms": response["latency_ms"],
            },
        )

        log.info(
            "auditor_verdict_issued",
            verdict=certificate.verdict.value,
            risk_score=certificate.overall_risk_score,
            violations=len(certificate.violations),
            confidence=certificate.auditor_confidence,
        )

        return {
            **state,
            "auditor_verdict": certificate.verdict,
            "transparency_certificate": certificate,
            "compliance_citations": model_citations,
            "audit_log": [*state.get("audit_log", []), audit_entry],
        }

    # ── Circuit Breaker Node ─────────────────────────────────────────────────

    async def circuit_breaker_node(self, state: AegisGraphState) -> AegisGraphState:
        """
        Tracks consecutive rejections. Opens the circuit after MAX_REJECTIONS,
        preventing further retries and flagging for human intervention.
        """
        log = logger.bind(node="circuit_breaker", session_id=state.get("session_id"))

        cb: CircuitBreakerState = state.get(
            "circuit_breaker", CircuitBreakerState()
        )

        cert: TransparencyCertificate | None = state.get("transparency_certificate")
        reason = "Unknown rejection reason"
        if cert:
            violations = cert.violations
            reason = (
                "; ".join(f"{v.policy_id}: {v.violation_detail}" for v in violations)
                if violations
                else cert.summary
            )

        cb.record_rejection(reason)

        audit_entry = self._make_audit_entry(
            event="circuit_breaker_updated",
            data={
                "rejection_count": cb.rejection_count,
                "is_open": cb.is_open,
                "max_rejections": cb.max_rejections,
                "latest_reason": reason,
            },
        )

        log.warning(
            "circuit_breaker_updated",
            rejection_count=cb.rejection_count,
            is_open=cb.is_open,
        )

        if cb.is_open:
            log.error(
                "circuit_breaker_OPEN",
                total_rejections=cb.rejection_count,
                reasons=cb.rejection_reasons,
            )

        return {
            **state,
            "circuit_breaker": cb,
            "audit_log": [*state.get("audit_log", []), audit_entry],
        }

    # ── Executor Node ────────────────────────────────────────────────────────

    async def executor_node(self, state: AegisGraphState) -> AegisGraphState:
        """
        Executes the approved tool call via the Hospital MCP server.
        Only reached when Auditor verdict is APPROVED.
        """
        log = logger.bind(node="executor", session_id=state.get("session_id"))
        log.info("executor_node_start")

        proposal: OrchestratorProposal = state["orchestrator_proposal"]

        tool_result = await self.hospital.execute_tool(proposal.tool_call)

        status = ExecutionStatus.EXECUTED if tool_result.success else ExecutionStatus.BLOCKED_BY_AUDITOR

        audit_entry = self._make_audit_entry(
            event="tool_executed",
            data={
                "call_id": tool_result.call_id,
                "tool_name": tool_result.tool_name,
                "success": tool_result.success,
                "duration_ms": tool_result.duration_ms,
                "error": tool_result.error,
            },
        )

        log.info(
            "executor_complete",
            tool_name=tool_result.tool_name,
            success=tool_result.success,
            duration_ms=round(tool_result.duration_ms, 2),
        )

        return {
            **state,
            "tool_result": tool_result,
            "final_execution_status": status,
            "audit_log": [*state.get("audit_log", []), audit_entry],
        }

    # ── Escalation Node ──────────────────────────────────────────────────────

    async def escalation_node(self, state: AegisGraphState) -> AegisGraphState:
        """
        Triggered when the circuit breaker opens.
        Calls the flag_for_human_review tool and terminates execution.
        """
        log = logger.bind(node="escalation", session_id=state.get("session_id"))
        log.error("escalation_node_triggered — circuit breaker open")

        proposal: OrchestratorProposal = state["orchestrator_proposal"]
        cb: CircuitBreakerState = state.get("circuit_breaker", CircuitBreakerState())

        escalation_call = MCPToolCall(
            tool_name="flag_for_human_review",
            arguments={
                "patient_id": proposal.patient_context.patient_id,
                "proposal_id": proposal.proposal_id,
                "reason": (
                    f"Circuit breaker opened after {cb.rejection_count} consecutive "
                    f"auditor rejections. Reasons: {'; '.join(cb.rejection_reasons[-3:])}"
                ),
                "urgency": "WITHIN_1_HOUR",
            },
        )

        tool_result = await self.hospital.execute_tool(escalation_call)

        audit_entry = self._make_audit_entry(
            event="escalated_to_human",
            data={
                "rejection_count": cb.rejection_count,
                "escalation_ticket": tool_result.result.get("escalation_ticket_id") if tool_result.result else None,
                "success": tool_result.success,
            },
        )

        log.error(
            "escalation_complete",
            ticket_id=tool_result.result.get("escalation_ticket_id") if tool_result.result else None,
        )

        return {
            **state,
            "tool_result": tool_result,
            "final_execution_status": ExecutionStatus.CIRCUIT_BREAKER_OPEN,
            "audit_log": [*state.get("audit_log", []), audit_entry],
        }

    # ── Human Review Node ────────────────────────────────────────────────────

    async def human_review_node(self, state: AegisGraphState) -> AegisGraphState:
        """
        Handles NEEDS_REVIEW verdict: escalates but with ROUTINE urgency.
        """
        log = logger.bind(node="human_review", session_id=state.get("session_id"))
        log.info("human_review_node_triggered")

        proposal: OrchestratorProposal = state["orchestrator_proposal"]
        cert: TransparencyCertificate | None = state.get("transparency_certificate")

        review_call = MCPToolCall(
            tool_name="flag_for_human_review",
            arguments={
                "patient_id": proposal.patient_context.patient_id,
                "proposal_id": proposal.proposal_id,
                "reason": cert.summary if cert else "Auditor requested human review.",
                "urgency": "WITHIN_4_HOURS",
            },
        )

        tool_result = await self.hospital.execute_tool(review_call)

        audit_entry = self._make_audit_entry(
            event="human_review_requested",
            data={
                "ticket": tool_result.result.get("escalation_ticket_id") if tool_result.result else None,
                "success": tool_result.success,
            },
        )

        return {
            **state,
            "tool_result": tool_result,
            "final_execution_status": ExecutionStatus.ESCALATED_TO_HUMAN,
            "audit_log": [*state.get("audit_log", []), audit_entry],
        }

    # ── Routing Functions ────────────────────────────────────────────────────

    @staticmethod
    def route_after_auditor(state: AegisGraphState) -> str:
        """Conditional edge: route based on Auditor verdict."""
        verdict = state.get("auditor_verdict")
        if verdict == AuditorVerdict.APPROVED:
            return "executor_node"
        if verdict == AuditorVerdict.NEEDS_REVIEW:
            return "human_review_node"
        # REJECTED or PENDING → circuit breaker
        return "circuit_breaker_node"

    @staticmethod
    def route_after_circuit_breaker(state: AegisGraphState) -> str:
        """Conditional edge: if circuit breaker is open, escalate; else retry."""
        cb: CircuitBreakerState | None = state.get("circuit_breaker")
        if cb and cb.is_open:
            return "escalation_node"
        return "orchestrator_node"

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _infer_action_type(tool_name: str) -> ActionType:
        mapping = {
            "prescribe_medication": ActionType.PRESCRIPTION,
            "order_lab_test": ActionType.LAB_ORDER,
            "flag_for_human_review": ActionType.PROCEDURE,
        }
        return mapping.get(tool_name, ActionType.PROCEDURE)

    @staticmethod
    def _infer_risk_level(tool_name: str, arguments: dict) -> RiskLevel:
        high_risk_drugs = {
            "oxycodone", "fentanyl", "morphine", "warfarin", "methotrexate",
            "cyclosporine", "tacrolimus", "hydrocodone",
        }
        if tool_name == "prescribe_medication":
            drug = arguments.get("drug_name", "").lower()
            if drug in high_risk_drugs:
                return RiskLevel.HIGH
            return RiskLevel.MEDIUM
        if tool_name == "order_lab_test":
            return RiskLevel.LOW
        return RiskLevel.MEDIUM

    @staticmethod
    def _make_audit_entry(event: str, data: dict[str, Any]) -> dict[str, Any]:
        return {
            "event": event,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **data,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Graph Builder
# ─────────────────────────────────────────────────────────────────────────────

def build_aegis_graph(
    orchestrator_client: BaseModelClient | None = None,
    auditor_client: BaseModelClient | None = None,
    hospital_server: HospitalAPIMCPServer | None = None,
    policy_engine: PolicyEngine | None = None,
    model_mode: str = "mock",
) -> StateGraph:
    """
    Assemble and compile the full Aegis-G4 LangGraph workflow.

    Args:
        orchestrator_client: Gemma 4 31B client (defaults to factory based on model_mode)
        auditor_client: Gemma 4 E4B client (defaults to factory based on model_mode)
        hospital_server: MCP hospital API server
        policy_engine: ChromaDB policy retrieval engine
        model_mode: "mock" | "vllm" | "litert"

    Returns:
        Compiled LangGraph StateGraph ready for invocation.
    """
    # Dependency injection with defaults
    orch_client = orchestrator_client or create_orchestrator_client(mode=model_mode)
    aud_client = auditor_client or create_auditor_client(mode=model_mode)
    hosp_server = hospital_server or get_hospital_server()
    pol_engine = policy_engine or PolicyEngine()

    nodes = AegisGraphNodes(
        orchestrator_client=orch_client,
        auditor_client=aud_client,
        hospital_server=hosp_server,
        policy_engine=pol_engine,
    )

    # ── Graph construction ────────────────────────────────────────────────────
    graph = StateGraph(AegisGraphState)

    # Register nodes
    graph.add_node("orchestrator_node", nodes.orchestrator_node)
    graph.add_node("auditor_node", nodes.auditor_node)
    graph.add_node("circuit_breaker_node", nodes.circuit_breaker_node)
    graph.add_node("executor_node", nodes.executor_node)
    graph.add_node("escalation_node", nodes.escalation_node)
    graph.add_node("human_review_node", nodes.human_review_node)

    # Entry point
    graph.add_edge(START, "orchestrator_node")

    # Orchestrator always flows to Auditor
    graph.add_edge("orchestrator_node", "auditor_node")

    # Auditor has conditional routing
    graph.add_conditional_edges(
        "auditor_node",
        nodes.route_after_auditor,
        {
            "executor_node": "executor_node",
            "human_review_node": "human_review_node",
            "circuit_breaker_node": "circuit_breaker_node",
        },
    )

    # Circuit breaker: retry or escalate
    graph.add_conditional_edges(
        "circuit_breaker_node",
        nodes.route_after_circuit_breaker,
        {
            "orchestrator_node": "orchestrator_node",
            "escalation_node": "escalation_node",
        },
    )

    # Terminal nodes
    graph.add_edge("executor_node", END)
    graph.add_edge("human_review_node", END)
    graph.add_edge("escalation_node", END)

    return graph.compile()


# ─────────────────────────────────────────────────────────────────────────────
# High-Level Runner
# ─────────────────────────────────────────────────────────────────────────────

async def run_aegis(
    user_request: str,
    patient_context: PatientContext,
    model_mode: str = "mock",
    session_id: str | None = None,
) -> AegisGraphState:
    """
    Convenience async runner for the Aegis-G4 graph.

    Args:
        user_request: Natural language clinical request
        patient_context: Patient profile
        model_mode: "mock" | "vllm" | "litert"
        session_id: Optional session identifier for logging

    Returns:
        Final AegisGraphState after graph completion
    """
    sid = session_id or str(uuid.uuid4())
    log = logger.bind(session_id=sid, model_mode=model_mode)
    log.info("aegis_run_start", request_preview=user_request[:100])

    # Initialize policy engine
    policy_engine = PolicyEngine()
    await policy_engine.initialize()

    # Build graph
    compiled_graph = build_aegis_graph(
        policy_engine=policy_engine,
        model_mode=model_mode,
    )

    # Initial state
    initial_state: AegisGraphState = {
        "session_id": sid,
        "patient_context": patient_context,
        "user_request": user_request,
        "circuit_breaker": CircuitBreakerState(),
        "iteration_count": 0,
        "error_messages": [],
        "audit_log": [],
    }

    # Execute
    final_state = await compiled_graph.ainvoke(initial_state)

    log.info(
        "aegis_run_complete",
        final_status=final_state.get("final_execution_status"),
        verdict=final_state.get("auditor_verdict"),
        iterations=final_state.get("iteration_count"),
        audit_entries=len(final_state.get("audit_log", [])),
    )

    return final_state
