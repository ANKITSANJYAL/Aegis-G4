"""
schema.py — Aegis-G4
All Pydantic v2+ data models and LangGraph TypedDict state definitions.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator
from typing_extensions import TypedDict


# ─────────────────────────────────────────────────────────────────────────────
# Enumerations
# ─────────────────────────────────────────────────────────────────────────────


class AuditorVerdict(str, Enum):
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    PENDING = "PENDING"


class ExecutionStatus(str, Enum):
    NOT_STARTED = "NOT_STARTED"
    EXECUTED = "EXECUTED"
    BLOCKED_BY_AUDITOR = "BLOCKED_BY_AUDITOR"
    ESCALATED_TO_HUMAN = "ESCALATED_TO_HUMAN"
    CIRCUIT_BREAKER_OPEN = "CIRCUIT_BREAKER_OPEN"


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class ActionType(str, Enum):
    PRESCRIPTION = "PRESCRIPTION"
    LAB_ORDER = "LAB_ORDER"
    REFERRAL = "REFERRAL"
    PROCEDURE = "PROCEDURE"
    DISCHARGE = "DISCHARGE"


# ─────────────────────────────────────────────────────────────────────────────
# MCP Tool Call Schema (Gemma 4 native function calling)
# ─────────────────────────────────────────────────────────────────────────────


class MCPToolParameter(BaseModel):
    """Single parameter in an MCP tool definition."""

    name: str = Field(..., description="Parameter name")
    type: str = Field(..., description="JSON schema type")
    description: str = Field(..., description="Human-readable description")
    required: bool = Field(default=True)
    enum_values: Optional[list[str]] = Field(default=None, alias="enum")

    model_config = {"populate_by_name": True}


class MCPToolDefinition(BaseModel):
    """Full MCP-compliant tool definition used in function calling."""

    name: str = Field(..., description="Tool name (snake_case)")
    description: str = Field(..., description="What this tool does")
    parameters: list[MCPToolParameter] = Field(default_factory=list)
    version: str = Field(default="1.0.0")
    server: str = Field(default="hospital-api-mcp")


class MCPToolCall(BaseModel):
    """A proposed tool invocation emitted by the Orchestrator."""

    call_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tool_name: str = Field(..., description="Name of the MCP tool to invoke")
    arguments: dict[str, Any] = Field(..., description="Tool arguments as key-value pairs")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("tool_name")
    @classmethod
    def tool_name_must_be_snake_case(cls, v: str) -> str:
        if not v.replace("_", "").isalnum():
            raise ValueError(f"tool_name must be snake_case alphanumeric, got: {v!r}")
        return v


class MCPToolResult(BaseModel):
    """Result returned by a tool execution."""

    call_id: str
    tool_name: str
    success: bool
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    executed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    duration_ms: float = Field(default=0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator Proposal
# ─────────────────────────────────────────────────────────────────────────────


class PatientContext(BaseModel):
    """Minimal patient context passed to the orchestrator."""

    patient_id: str = Field(..., description="Unique patient identifier")
    age: int = Field(..., ge=0, le=150)
    diagnosis_codes: list[str] = Field(..., description="ICD-10 codes")
    current_medications: list[str] = Field(default_factory=list)
    allergies: list[str] = Field(default_factory=list)
    risk_flags: list[str] = Field(default_factory=list)

    @field_validator("diagnosis_codes")
    @classmethod
    def at_least_one_diagnosis(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("At least one diagnosis code is required")
        return v


class OrchestratorProposal(BaseModel):
    """Structured output from the Orchestrator (gemma-4-31b-it)."""

    proposal_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    action_type: ActionType
    tool_call: MCPToolCall
    clinical_rationale: str = Field(..., min_length=20, description="Clinical reasoning text")
    risk_level: RiskLevel
    patient_context: PatientContext
    model_used: str = Field(default="gemma-4-31b-it")
    raw_model_output: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def high_risk_needs_rationale(self) -> "OrchestratorProposal":
        if self.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL):
            if len(self.clinical_rationale) < 100:
                raise ValueError(
                    "HIGH/CRITICAL risk proposals require at least 100 chars of rationale"
                )
        return self


# ─────────────────────────────────────────────────────────────────────────────
# Policy & Compliance
# ─────────────────────────────────────────────────────────────────────────────


class NISTPolicy(BaseModel):
    """A single NIST AI RMF 2026 Healthcare policy record stored in ChromaDB."""

    policy_id: str = Field(..., description="e.g. NIST-H-01")
    category: str = Field(..., description="e.g. GOVERN, MAP, MEASURE, MANAGE")
    title: str
    description: str
    healthcare_applicability: str
    risk_threshold: RiskLevel = Field(default=RiskLevel.MEDIUM)
    enforcement_level: Literal["MANDATORY", "RECOMMENDED", "INFORMATIONAL"] = "MANDATORY"


class PolicyCitation(BaseModel):
    """A single policy citation produced by the Auditor."""

    policy_id: str
    policy_title: str
    relevance_score: float = Field(..., ge=0.0, le=1.0)
    violated: bool
    violation_detail: Optional[str] = None
    remediation_suggestion: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# Transparency Certificate (Auditor Output)
# ─────────────────────────────────────────────────────────────────────────────


class TransparencyCertificate(BaseModel):
    """
    JSON reasoning trace produced by the Auditor (gemma-4-e4b-it).
    This is the core 'Glass Box' artifact for regulatory compliance.
    """

    certificate_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    proposal_id: str
    verdict: AuditorVerdict
    overall_risk_score: float = Field(..., ge=0.0, le=10.0, description="0=safe, 10=critical")
    compliance_citations: list[PolicyCitation]
    thinking_trace: str = Field(..., description="Raw Gemma 4 thinking chain-of-thought")
    summary: str = Field(..., min_length=30)
    model_used: str = Field(default="gemma-4-e4b-it")
    rag_documents_retrieved: int = Field(default=0)
    issued_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    auditor_confidence: float = Field(..., ge=0.0, le=1.0)

    @property
    def violations(self) -> list[PolicyCitation]:
        return [c for c in self.compliance_citations if c.violated]

    @property
    def has_critical_violations(self) -> bool:
        return any(
            c.violated
            for c in self.compliance_citations
            if c.policy_id.startswith("NIST-H-0")  # Core mandatory policies
        )


# ─────────────────────────────────────────────────────────────────────────────
# Circuit Breaker
# ─────────────────────────────────────────────────────────────────────────────


class CircuitBreakerState(BaseModel):
    """Tracks consecutive auditor rejections for the circuit breaker node."""

    rejection_count: int = Field(default=0)
    max_rejections: int = Field(default=3)
    is_open: bool = Field(default=False)
    opened_at: Optional[datetime] = None
    rejection_reasons: list[str] = Field(default_factory=list)

    def record_rejection(self, reason: str) -> None:
        self.rejection_count += 1
        self.rejection_reasons.append(reason)
        if self.rejection_count >= self.max_rejections:
            self.is_open = True
            self.opened_at = datetime.now(timezone.utc)

    def reset(self) -> None:
        self.rejection_count = 0
        self.is_open = False
        self.opened_at = None
        self.rejection_reasons.clear()


# ─────────────────────────────────────────────────────────────────────────────
# LangGraph State (TypedDict)
# ─────────────────────────────────────────────────────────────────────────────


class AegisGraphState(TypedDict, total=False):
    """
    The canonical state object threaded through the entire LangGraph workflow.
    All nodes read from and write to this shared state.
    """

    # ── Input ────────────────────────────────────────────────────────────────
    session_id: str
    patient_context: PatientContext
    user_request: str                    # Natural language clinical request

    # ── Orchestrator output ──────────────────────────────────────────────────
    orchestrator_proposal: OrchestratorProposal

    # ── Auditor output ───────────────────────────────────────────────────────
    auditor_verdict: AuditorVerdict
    transparency_certificate: TransparencyCertificate
    compliance_citations: list[PolicyCitation]

    # ── Circuit breaker ──────────────────────────────────────────────────────
    circuit_breaker: CircuitBreakerState

    # ── Execution ────────────────────────────────────────────────────────────
    tool_result: MCPToolResult
    final_execution_status: ExecutionStatus

    # ── Routing & metadata ───────────────────────────────────────────────────
    iteration_count: int
    error_messages: list[str]
    audit_log: list[dict[str, Any]]      # Append-only structured log entries
