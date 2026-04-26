"""
tools.py — Aegis-G4
MCP-compliant HospitalAPI tool server and tool registry.
All tools are async and use Pydantic for input/output validation.
"""

from __future__ import annotations

import asyncio
import random
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import structlog

from schema import (
    ActionType,
    MCPToolCall,
    MCPToolDefinition,
    MCPToolParameter,
    MCPToolResult,
)

logger = structlog.get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# MCP Tool Registry
# ─────────────────────────────────────────────────────────────────────────────

class MCPToolRegistry:
    """Central registry for all MCP tool definitions available to the Orchestrator."""

    def __init__(self) -> None:
        self._tools: dict[str, MCPToolDefinition] = {}

    def register(self, tool: MCPToolDefinition) -> None:
        self._tools[tool.name] = tool
        logger.debug("mcp_tool_registered", tool_name=tool.name, server=tool.server)

    def get(self, name: str) -> Optional[MCPToolDefinition]:
        return self._tools.get(name)

    def list_tools(self) -> list[MCPToolDefinition]:
        return list(self._tools.values())

    def to_gemma_function_schema(self) -> list[dict[str, Any]]:
        """
        Serialize tools to Gemma 4 native function calling schema format.
        Compatible with the `tools` parameter in the Gemma chat template.
        """
        schema = []
        for tool in self._tools.values():
            properties = {}
            required = []
            for param in tool.parameters:
                prop: dict[str, Any] = {
                    "type": param.type,
                    "description": param.description,
                }
                if param.enum_values:
                    prop["enum"] = param.enum_values
                properties[param.name] = prop
                if param.required:
                    required.append(param.name)

            schema.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    },
                },
            })
        return schema


# ─────────────────────────────────────────────────────────────────────────────
# Mock Hospital API Server (MCP Server)
# ─────────────────────────────────────────────────────────────────────────────

class HospitalAPIMCPServer:
    """
    Mock Hospital EHR/LIS API implemented as an MCP server.
    Simulates real hospital system integrations with realistic latency,
    occasional failures, and structured audit logging.
    """

    SERVER_NAME = "hospital-api-mcp"
    SERVER_VERSION = "2.1.0"

    # Simulated drug database
    _DRUG_DATABASE: dict[str, dict] = {
        "amoxicillin": {
            "class": "Penicillin Antibiotic",
            "schedule": None,
            "requires_dual_sign": False,
            "typical_doses": ["500mg", "875mg"],
        },
        "metformin": {
            "class": "Biguanide Antidiabetic",
            "schedule": None,
            "requires_dual_sign": False,
            "typical_doses": ["500mg", "1000mg"],
        },
        "oxycodone": {
            "class": "Opioid Analgesic",
            "schedule": "Schedule II",
            "requires_dual_sign": True,
            "typical_doses": ["5mg", "10mg"],
        },
        "warfarin": {
            "class": "Anticoagulant",
            "schedule": None,
            "requires_dual_sign": True,
            "typical_doses": ["2mg", "5mg", "10mg"],
        },
        "lisinopril": {
            "class": "ACE Inhibitor",
            "schedule": None,
            "requires_dual_sign": False,
            "typical_doses": ["5mg", "10mg", "20mg"],
        },
        "methotrexate": {
            "class": "Antimetabolite / DMARDs",
            "schedule": None,
            "requires_dual_sign": True,
            "typical_doses": ["2.5mg", "7.5mg", "15mg"],
        },
    }

    # Simulated lab test catalog
    _LAB_CATALOG: dict[str, dict] = {
        "CBC": {
            "full_name": "Complete Blood Count",
            "turnaround_hours": 2,
            "requires_fasting": False,
        },
        "BMP": {
            "full_name": "Basic Metabolic Panel",
            "turnaround_hours": 3,
            "requires_fasting": True,
        },
        "HbA1c": {
            "full_name": "Glycated Hemoglobin",
            "turnaround_hours": 4,
            "requires_fasting": False,
        },
        "PT_INR": {
            "full_name": "Prothrombin Time / INR",
            "turnaround_hours": 2,
            "requires_fasting": False,
        },
        "CMP": {
            "full_name": "Comprehensive Metabolic Panel",
            "turnaround_hours": 4,
            "requires_fasting": True,
        },
        "LIPID_PANEL": {
            "full_name": "Lipid Panel",
            "turnaround_hours": 6,
            "requires_fasting": True,
        },
        "URINALYSIS": {
            "full_name": "Urinalysis with Microscopy",
            "turnaround_hours": 1,
            "requires_fasting": False,
        },
        "CULTURE_BLOOD": {
            "full_name": "Blood Culture x2",
            "turnaround_hours": 48,
            "requires_fasting": False,
        },
    }

    def __init__(self, failure_rate: float = 0.05) -> None:
        """
        Args:
            failure_rate: Probability of simulating a transient API failure (0–1).
        """
        self.failure_rate = failure_rate
        self._order_store: dict[str, dict] = {}
        self.registry = MCPToolRegistry()
        self._register_tools()

    def _register_tools(self) -> None:
        """Register all Hospital API tools in the MCP registry."""
        tools = [
            MCPToolDefinition(
                name="prescribe_medication",
                description=(
                    "Submit a medication prescription order to the hospital EHR system. "
                    "Creates a verified prescription record and queues it for pharmacy review."
                ),
                server=self.SERVER_NAME,
                parameters=[
                    MCPToolParameter(
                        name="patient_id",
                        type="string",
                        description="Tokenized patient identifier (not raw PHI)",
                    ),
                    MCPToolParameter(
                        name="drug_name",
                        type="string",
                        description="Generic drug name (lowercase)",
                    ),
                    MCPToolParameter(
                        name="dose",
                        type="string",
                        description="Dose with unit (e.g., '500mg')",
                    ),
                    MCPToolParameter(
                        name="frequency",
                        type="string",
                        description="Dosing frequency (e.g., 'twice daily', 'every 8 hours')",
                    ),
                    MCPToolParameter(
                        name="duration_days",
                        type="integer",
                        description="Duration of prescription in days",
                    ),
                    MCPToolParameter(
                        name="indication_icd10",
                        type="string",
                        description="ICD-10 code for the clinical indication",
                    ),
                    MCPToolParameter(
                        name="prescriber_id",
                        type="string",
                        description="Prescribing clinician's NPI or internal ID",
                    ),
                ],
            ),
            MCPToolDefinition(
                name="order_lab_test",
                description=(
                    "Place a laboratory test order in the hospital LIS (Lab Information System). "
                    "Returns an order ID and expected turnaround time."
                ),
                server=self.SERVER_NAME,
                parameters=[
                    MCPToolParameter(
                        name="patient_id",
                        type="string",
                        description="Tokenized patient identifier",
                    ),
                    MCPToolParameter(
                        name="test_code",
                        type="string",
                        description="Lab test code (e.g., 'CBC', 'BMP', 'HbA1c')",
                        enum_values=list(self._LAB_CATALOG.keys()),
                    ),
                    MCPToolParameter(
                        name="priority",
                        type="string",
                        description="Order priority level",
                        enum_values=["STAT", "ROUTINE", "TIMED"],
                    ),
                    MCPToolParameter(
                        name="indication_icd10",
                        type="string",
                        description="ICD-10 code justifying the test",
                    ),
                    MCPToolParameter(
                        name="ordering_provider_id",
                        type="string",
                        description="Ordering provider's internal ID or NPI",
                    ),
                ],
            ),
            MCPToolDefinition(
                name="get_patient_allergies",
                description=(
                    "Retrieve the documented allergy and adverse drug reaction list "
                    "for a patient from the EHR."
                ),
                server=self.SERVER_NAME,
                parameters=[
                    MCPToolParameter(
                        name="patient_id",
                        type="string",
                        description="Tokenized patient identifier",
                    ),
                ],
            ),
            MCPToolDefinition(
                name="get_active_medications",
                description=(
                    "Retrieve the current active medication list for a patient, "
                    "including dosage and prescribing provider."
                ),
                server=self.SERVER_NAME,
                parameters=[
                    MCPToolParameter(
                        name="patient_id",
                        type="string",
                        description="Tokenized patient identifier",
                    ),
                ],
            ),
            MCPToolDefinition(
                name="flag_for_human_review",
                description=(
                    "Escalate a clinical action proposal for mandatory human clinician review. "
                    "Creates a high-priority alert in the clinical workflow system."
                ),
                server=self.SERVER_NAME,
                parameters=[
                    MCPToolParameter(
                        name="patient_id",
                        type="string",
                        description="Tokenized patient identifier",
                    ),
                    MCPToolParameter(
                        name="proposal_id",
                        type="string",
                        description="The orchestrator proposal ID being escalated",
                    ),
                    MCPToolParameter(
                        name="reason",
                        type="string",
                        description="Clear text reason for escalation",
                    ),
                    MCPToolParameter(
                        name="urgency",
                        type="string",
                        description="Urgency level for the human review",
                        enum_values=["IMMEDIATE", "WITHIN_1_HOUR", "WITHIN_4_HOURS", "ROUTINE"],
                    ),
                ],
            ),
        ]
        for tool in tools:
            self.registry.register(tool)

    # ── Tool Execution Dispatcher ────────────────────────────────────────────

    async def execute_tool(self, tool_call: MCPToolCall) -> MCPToolResult:
        """Route a tool call to the appropriate handler with latency simulation."""
        start = time.monotonic()
        log = logger.bind(
            tool_name=tool_call.tool_name,
            call_id=tool_call.call_id,
        )
        log.info("mcp_tool_executing", arguments=tool_call.arguments)

        # Simulate transient failure
        if random.random() < self.failure_rate:
            duration = (time.monotonic() - start) * 1000
            log.warning("mcp_tool_simulated_failure")
            return MCPToolResult(
                call_id=tool_call.call_id,
                tool_name=tool_call.tool_name,
                success=False,
                error="Simulated transient API failure (503). Retry recommended.",
                duration_ms=duration,
            )

        # Simulate network latency
        await asyncio.sleep(random.uniform(0.05, 0.2))

        handlers = {
            "prescribe_medication": self._handle_prescribe_medication,
            "order_lab_test": self._handle_order_lab_test,
            "get_patient_allergies": self._handle_get_patient_allergies,
            "get_active_medications": self._handle_get_active_medications,
            "flag_for_human_review": self._handle_flag_for_human_review,
        }

        handler = handlers.get(tool_call.tool_name)
        if handler is None:
            duration = (time.monotonic() - start) * 1000
            return MCPToolResult(
                call_id=tool_call.call_id,
                tool_name=tool_call.tool_name,
                success=False,
                error=f"Unknown tool: '{tool_call.tool_name}'",
                duration_ms=duration,
            )

        try:
            result_data = await handler(tool_call.arguments)
            duration = (time.monotonic() - start) * 1000
            log.info("mcp_tool_success", duration_ms=round(duration, 2))
            return MCPToolResult(
                call_id=tool_call.call_id,
                tool_name=tool_call.tool_name,
                success=True,
                result=result_data,
                duration_ms=duration,
            )
        except Exception as exc:
            duration = (time.monotonic() - start) * 1000
            log.error("mcp_tool_error", error=str(exc))
            return MCPToolResult(
                call_id=tool_call.call_id,
                tool_name=tool_call.tool_name,
                success=False,
                error=str(exc),
                duration_ms=duration,
            )

    # ── Individual Tool Handlers ─────────────────────────────────────────────

    async def _handle_prescribe_medication(self, args: dict) -> dict:
        drug_name = args.get("drug_name", "").lower()
        drug_info = self._DRUG_DATABASE.get(drug_name, {
            "class": "Unknown",
            "schedule": None,
            "requires_dual_sign": False,
        })
        order_id = f"RX-{uuid.uuid4().hex[:8].upper()}"
        self._order_store[order_id] = {
            "type": "PRESCRIPTION",
            "status": "PENDING_PHARMACY_REVIEW" if drug_info.get("requires_dual_sign") else "QUEUED",
            "args": args,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        return {
            "order_id": order_id,
            "status": self._order_store[order_id]["status"],
            "drug_schedule": drug_info.get("schedule"),
            "requires_dual_clinician_sign": drug_info.get("requires_dual_sign"),
            "drug_class": drug_info.get("class"),
            "estimated_pharmacy_review_minutes": 15 if drug_info.get("requires_dual_sign") else 5,
            "message": (
                "Order created. Routed to dual-clinician review queue."
                if drug_info.get("requires_dual_sign")
                else "Order created and queued for pharmacy dispensing."
            ),
        }

    async def _handle_order_lab_test(self, args: dict) -> dict:
        test_code = args.get("test_code", "").upper()
        lab_info = self._LAB_CATALOG.get(test_code, {
            "full_name": test_code,
            "turnaround_hours": 24,
            "requires_fasting": False,
        })
        priority = args.get("priority", "ROUTINE")
        multiplier = {"STAT": 0.25, "ROUTINE": 1.0, "TIMED": 0.75}.get(priority, 1.0)
        turnaround = max(1, int(lab_info["turnaround_hours"] * multiplier))

        order_id = f"LAB-{uuid.uuid4().hex[:8].upper()}"
        expected_at = datetime.now(timezone.utc) + timedelta(hours=turnaround)
        self._order_store[order_id] = {
            "type": "LAB_ORDER",
            "status": "SUBMITTED_TO_LIS",
            "args": args,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        return {
            "order_id": order_id,
            "status": "SUBMITTED_TO_LIS",
            "test_full_name": lab_info["full_name"],
            "priority": priority,
            "requires_fasting": lab_info["requires_fasting"],
            "expected_results_by": expected_at.isoformat(),
            "turnaround_hours": turnaround,
        }

    async def _handle_get_patient_allergies(self, args: dict) -> dict:
        # Simulate EHR allergy retrieval
        return {
            "patient_id": args["patient_id"],
            "allergies": [
                {"allergen": "Penicillin", "reaction": "Anaphylaxis", "severity": "SEVERE"},
                {"allergen": "Sulfa", "reaction": "Rash", "severity": "MODERATE"},
                {"allergen": "Latex", "reaction": "Contact Dermatitis", "severity": "MILD"},
            ],
            "last_updated": "2026-01-15T10:00:00Z",
            "verified_by": "Dr. Sarah Chen, PharmD",
        }

    async def _handle_get_active_medications(self, args: dict) -> dict:
        return {
            "patient_id": args["patient_id"],
            "medications": [
                {
                    "drug": "metformin",
                    "dose": "1000mg",
                    "frequency": "twice daily",
                    "prescriber": "Dr. James Okafor",
                    "start_date": "2025-06-01",
                },
                {
                    "drug": "lisinopril",
                    "dose": "10mg",
                    "frequency": "once daily",
                    "prescriber": "Dr. James Okafor",
                    "start_date": "2025-06-01",
                },
            ],
            "last_reconciled": "2026-04-20T08:30:00Z",
        }

    async def _handle_flag_for_human_review(self, args: dict) -> dict:
        ticket_id = f"ESC-{uuid.uuid4().hex[:6].upper()}"
        urgency = args.get("urgency", "ROUTINE")
        eta = {
            "IMMEDIATE": "Now — paging on-call physician",
            "WITHIN_1_HOUR": "1 hour",
            "WITHIN_4_HOURS": "4 hours",
            "ROUTINE": "Next available clinician",
        }.get(urgency, "Next available clinician")

        return {
            "escalation_ticket_id": ticket_id,
            "status": "ESCALATED",
            "urgency": urgency,
            "expected_review_eta": eta,
            "proposal_id": args.get("proposal_id"),
            "reason": args.get("reason"),
            "assigned_to": "On-Call Clinical Decision Team",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Singleton Server Instance
# ─────────────────────────────────────────────────────────────────────────────

_hospital_server: Optional[HospitalAPIMCPServer] = None


def get_hospital_server(failure_rate: float = 0.05) -> HospitalAPIMCPServer:
    """Get or create the singleton HospitalAPIMCPServer instance."""
    global _hospital_server
    if _hospital_server is None:
        _hospital_server = HospitalAPIMCPServer(failure_rate=failure_rate)
    return _hospital_server
