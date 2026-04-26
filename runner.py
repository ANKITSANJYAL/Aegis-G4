"""
runner.py - Aegis-G4 CLI demo runner.

Runs three clinical scenarios through the Aegis safety agent and prints
structured results. Defaults to mock mode — no GPU required.

Usage:
    python runner.py                    # all scenarios, mock models
    python runner.py --mode vllm        # all scenarios, real Gemma models
    python runner.py --scenario 2       # single scenario
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
from datetime import datetime, timezone

import structlog
import structlog.dev
import structlog.stdlib

from graph import run_aegis
from schema import AuditorVerdict, ExecutionStatus, PatientContext

_log = structlog.get_logger(__name__)


async def wait_for_vllm(
    server_urls: list[str],
    timeout_seconds: int = 600,
    poll_interval: int = 10,
) -> None:
    """
    Poll each vLLM /health endpoint until all servers respond 200 or timeout.

    Args:
        server_urls: Base server URLs WITHOUT a path suffix, e.g.
                     ["http://localhost:8000", "http://localhost:8001"].
                     The /v1 path is only for inference calls; the health
                     endpoint lives at the root: GET /health.
        timeout_seconds: Give up after this many seconds (default 600 = 10 min).
        poll_interval: Seconds between polls.
    """
    try:
        import aiohttp
    except ImportError:
        _log.warning("aiohttp not installed — skipping vLLM health check")
        return

    deadline = asyncio.get_running_loop().time() + timeout_seconds
    pending = set(server_urls)

    async with aiohttp.ClientSession() as session:
        while pending:
            if asyncio.get_running_loop().time() > deadline:
                raise TimeoutError(
                    f"vLLM servers not ready after {timeout_seconds}s: {pending}"
                )
            for url in list(pending):
                with contextlib.suppress(Exception):
                    async with session.get(
                        f"{url}/health",
                        timeout=aiohttp.ClientTimeout(total=5),
                    ) as r:
                        if r.status == 200:
                            _log.info("vllm_server_ready", url=url)
                            pending.discard(url)
            if pending:
                _log.info(
                    "waiting_for_vllm_servers",
                    remaining=list(pending),
                    poll_interval=poll_interval,
                )
                await asyncio.sleep(poll_interval)

structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.dev.ConsoleRenderer(),
    ]
)

# ---------------------------------------------------------------------------
# Clinical scenarios
# ---------------------------------------------------------------------------

SCENARIOS: list[dict] = [
    {
        "id": 1,
        "title": "Routine T2DM Management — Metformin Prescription",
        "description": "First-line medication for type 2 diabetes. Expected: APPROVED, EXECUTED.",
        "request": "Order metformin 1000mg twice daily for 90 days for type 2 diabetes management.",
        "patient": PatientContext(
            patient_id="PT-TOKEN-7842",
            age=54,
            diagnosis_codes=["E11.9", "E11.65"],
            current_medications=["lisinopril 10mg"],
            allergies=["Penicillin", "Sulfa"],
            risk_flags=[],
        ),
    },
    {
        "id": 2,
        "title": "High-Risk Controlled Substance — Oxycodone Prescription",
        "description": (
            "Schedule II opioid requiring dual-clinician review. "
            "Expected: REJECTED, circuit breaker opens, ESCALATED."
        ),
        "request": "Prescribe oxycodone 10mg every 6 hours for post-operative pain management.",
        "patient": PatientContext(
            patient_id="PT-TOKEN-3391",
            age=42,
            diagnosis_codes=["M54.5", "Z96.641"],
            current_medications=["ibuprofen 400mg"],
            allergies=["Latex"],
            risk_flags=["Post-operative Day 1", "Opioid-naive"],
        ),
    },
    {
        "id": 3,
        "title": "Routine Lab Order — HbA1c Monitoring",
        "description": "Quarterly glycated hemoglobin check. Expected: APPROVED, EXECUTED.",
        "request": "Order HbA1c lab test for quarterly diabetes monitoring.",
        "patient": PatientContext(
            patient_id="PT-TOKEN-9901",
            age=61,
            diagnosis_codes=["E11.9"],
            current_medications=["metformin 1000mg", "glipizide 5mg"],
            allergies=[],
            risk_flags=[],
        ),
    },
]


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

_WIDTH = 72


def _divider(label: str = "") -> None:
    if label:
        pad = (_WIDTH - len(label) - 2) // 2
        print(f"\n{'=' * pad} {label} {'=' * pad}")
    else:
        print("=" * _WIDTH)


def _print_scenario_header(scenario: dict) -> None:
    _divider(f"SCENARIO {scenario['id']}")
    print(f"  Title    : {scenario['title']}")
    print(f"  Expected : {scenario['description']}")
    print(f"  Request  : {scenario['request']}")
    patient: PatientContext = scenario["patient"]
    print(f"  Patient  : {patient.patient_id} | Age {patient.age}")
    print(f"  Diagnoses: {', '.join(patient.diagnosis_codes)}")
    print(f"  Allergies: {', '.join(patient.allergies) or 'None'}")
    if patient.risk_flags:
        print(f"  Flags    : {', '.join(patient.risk_flags)}")


def _print_result(state: dict) -> None:
    verdict: AuditorVerdict | None = state.get("auditor_verdict")
    status: ExecutionStatus | None = state.get("final_execution_status")
    cert = state.get("transparency_certificate")
    cb = state.get("circuit_breaker")
    errors: list[str] = state.get("error_messages", [])

    print(f"\n  --- Result ---")
    print(f"  Verdict          : {verdict.value if verdict else 'N/A'}")
    print(f"  Execution Status : {status.value if status else 'N/A'}")
    print(f"  Iterations       : {state.get('iteration_count', 0)}")

    if cert:
        print(f"  Risk Score       : {cert.overall_risk_score:.1f} / 10.0")
        print(f"  Confidence       : {cert.auditor_confidence:.0%}")
        print(f"  RAG Policies     : {cert.rag_documents_retrieved}")
        print(f"  Violations       : {len(cert.violations)}")
        summary = f"{cert.summary[:117]}..." if len(cert.summary) > 120 else cert.summary
        print(f"  Summary          : {summary}")

        if cert.violations:
            print("\n  Policy Violations:")
            for v in cert.violations:
                print(f"    [{v.policy_id}] {v.policy_title}")
                print(f"      Detail : {v.violation_detail}")
                if v.remediation_suggestion:
                    print(f"      Action : {v.remediation_suggestion}")

    if tool_result := state.get("tool_result"):
        print("\n  Tool Execution:")
        print(f"    Tool    : {tool_result.tool_name}")
        print(f"    Success : {tool_result.success}")
        if tool_result.result:
            if order_id := (
                tool_result.result.get("order_id")
                or tool_result.result.get("escalation_ticket_id")
            ):
                print(f"    Order   : {order_id}")
        if tool_result.error:
            print(f"    Error   : {tool_result.error}")

    if cb and cb.rejection_count > 0:
        print(
            f"\n  Circuit Breaker  : {cb.rejection_count}/{cb.max_rejections} rejections"
            f" | open={cb.is_open}"
        )

    audit_log: list[dict] = state.get("audit_log", [])
    print(f"\n  Audit Trail ({len(audit_log)} entries):")
    for entry in audit_log:
        ts = entry.get("timestamp", "")[:19].replace("T", " ")
        event = entry.get("event", "")
        print(f"    {ts}  {event}")

    if errors:
        print(f"\n  Errors:")
        for err in errors:
            print(f"    {err}")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


async def run_scenario(scenario: dict, mode: str) -> None:
    _print_scenario_header(scenario)
    print(f"\n  Running in {mode.upper()} mode ...")

    state = await run_aegis(
        user_request=scenario["request"],
        patient_context=scenario["patient"],
        model_mode=mode,
    )
    _print_result(state)


async def main(mode: str, scenario_ids: list[int]) -> None:
    _divider("AEGIS-G4  HEALTHCARE AI SAFETY AGENT")
    print(f"  Mode     : {mode.upper()}")
    print(f"  Scenarios: {scenario_ids}")
    print(f"  Started  : {datetime.now(timezone.utc).isoformat()}")

    if mode == "vllm":
        print("\n  Waiting for vLLM servers to be ready ...")
        await wait_for_vllm(
            server_urls=["http://localhost:8000", "http://localhost:8001"],
            timeout_seconds=600,
            poll_interval=10,
        )

    for scenario in SCENARIOS:
        if scenario["id"] in scenario_ids:
            await run_scenario(scenario, mode)

    _divider("ALL SCENARIOS COMPLETE")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aegis-G4 healthcare AI safety agent demo runner.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python runner.py                   # all scenarios, mock mode\n"
            "  python runner.py --mode vllm       # real Gemma 4 models (needs GPU)\n"
            "  python runner.py --scenario 2      # oxycodone rejection scenario only\n"
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["mock", "vllm"],
        default="mock",
        help="Model backend: 'mock' for local dev, 'vllm' for GPU (Kaggle/production).",
    )
    parser.add_argument(
        "--scenario",
        type=int,
        choices=[1, 2, 3],
        default=None,
        help="Run a single scenario (1-3). Omit to run all three.",
    )
    args = parser.parse_args()
    ids = [args.scenario] if args.scenario else [1, 2, 3]

    asyncio.run(main(mode=args.mode, scenario_ids=ids))
