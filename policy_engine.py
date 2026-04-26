"""
policy_engine.py — Aegis-G4
ChromaDB-backed RAG engine for NIST AI RMF 2026 Healthcare policies.
Provides semantic retrieval for the Auditor node's grounded reasoning.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Optional

import structlog

try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

from schema import NISTPolicy, PolicyCitation, RiskLevel

logger = structlog.get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# NIST AI RMF 2026 Healthcare Policy Corpus
# In production this would be loaded from a secured policy store / database.
# ─────────────────────────────────────────────────────────────────────────────

NIST_HEALTHCARE_POLICIES: list[dict] = [
    {
        "policy_id": "NIST-H-01",
        "category": "GOVERN",
        "title": "AI System Transparency in Clinical Decision Support",
        "description": (
            "All AI-generated clinical recommendations must include a machine-readable "
            "reasoning trace that documents the data sources, model version, confidence "
            "scores, and relevant policy citations used to generate the recommendation. "
            "The trace must be retained for a minimum of 7 years per HIPAA §164.530."
        ),
        "healthcare_applicability": (
            "Applies to all AI systems that generate, suggest, or assist in clinical "
            "decisions including prescriptions, lab orders, surgical planning, and discharge."
        ),
        "risk_threshold": "HIGH",
        "enforcement_level": "MANDATORY",
    },
    {
        "policy_id": "NIST-H-02",
        "category": "MAP",
        "title": "High-Risk Medication Prescribing Controls",
        "description": (
            "AI systems proposing prescriptions for controlled substances (Schedule II–V), "
            "chemotherapy agents, anticoagulants, or immunosuppressants MUST trigger a "
            "mandatory dual-clinician review workflow before any order is placed. "
            "Automated execution of such orders without human sign-off is strictly prohibited."
        ),
        "healthcare_applicability": (
            "Applies to any AI system with write access to EHR prescribing modules, "
            "including autonomous agent pipelines."
        ),
        "risk_threshold": "HIGH",
        "enforcement_level": "MANDATORY",
    },
    {
        "policy_id": "NIST-H-03",
        "category": "MEASURE",
        "title": "Drug-Allergy Interaction Verification",
        "description": (
            "Prior to proposing any prescription or medication change, the AI system must "
            "perform a cross-reference check against the patient's documented allergy list "
            "and active medication list. Any potential interaction with severity MODERATE or "
            "above must block automated execution and generate an adverse event alert."
        ),
        "healthcare_applicability": (
            "Applicable to prescription tools, medication reconciliation agents, "
            "and discharge planning AI systems."
        ),
        "risk_threshold": "CRITICAL",
        "enforcement_level": "MANDATORY",
    },
    {
        "policy_id": "NIST-H-04",
        "category": "MANAGE",
        "title": "Lab Order Clinical Necessity Documentation",
        "description": (
            "All AI-ordered laboratory tests must include a documented clinical indication "
            "linked to a validated ICD-10 diagnosis code. Orders without a linked indication "
            "are considered medically unnecessary and must be flagged for physician review "
            "before submission to the lab information system (LIS)."
        ),
        "healthcare_applicability": (
            "Applies to AI agents interfacing with LIS, CPOE systems, or any order-entry "
            "modules in hospital information systems."
        ),
        "risk_threshold": "MEDIUM",
        "enforcement_level": "MANDATORY",
    },
    {
        "policy_id": "NIST-H-05",
        "category": "GOVERN",
        "title": "PHI Minimization in AI Inference Pipelines",
        "description": (
            "AI inference requests must not transmit Protected Health Information (PHI) "
            "beyond the minimum necessary standard (HIPAA §164.502(b)). Patient identifiers "
            "must be tokenized or pseudonymized before being included in model prompts sent "
            "to any external API endpoint. De-identification must comply with HIPAA Safe Harbor "
            "or Expert Determination standards."
        ),
        "healthcare_applicability": (
            "Mandatory for all AI pipelines processing EHR data, regardless of whether "
            "the model is hosted on-premise or via a cloud API."
        ),
        "risk_threshold": "CRITICAL",
        "enforcement_level": "MANDATORY",
    },
    {
        "policy_id": "NIST-H-06",
        "category": "MAP",
        "title": "AI Model Version and Drift Monitoring",
        "description": (
            "Healthcare organizations must maintain a model registry that tracks the exact "
            "version, training cutoff date, and validation metrics (AUROC, sensitivity, "
            "specificity) for each AI model used in clinical workflows. Model drift exceeding "
            "5% degradation in primary metric over a 30-day rolling window must trigger "
            "automatic suspension pending re-validation."
        ),
        "healthcare_applicability": (
            "Applies to all AI/ML models in production clinical environments including "
            "diagnostic, predictive, and prescriptive systems."
        ),
        "risk_threshold": "HIGH",
        "enforcement_level": "MANDATORY",
    },
    {
        "policy_id": "NIST-H-07",
        "category": "MEASURE",
        "title": "Bias Auditing for Demographic Equity",
        "description": (
            "AI systems in clinical use must undergo quarterly bias audits across protected "
            "demographic attributes (age, sex, race/ethnicity, socioeconomic status, disability "
            "status). Disparate impact exceeding 10% across any protected group in treatment "
            "recommendations must be reported to the hospital's AI Ethics Committee and "
            "remediated within 90 days."
        ),
        "healthcare_applicability": (
            "Applies to all AI-assisted triage, treatment recommendation, and resource "
            "allocation systems."
        ),
        "risk_threshold": "HIGH",
        "enforcement_level": "MANDATORY",
    },
    {
        "policy_id": "NIST-H-08",
        "category": "MANAGE",
        "title": "Human Override and Intervention Capability",
        "description": (
            "Every AI agent system in a clinical setting must implement a hard-stop mechanism "
            "that allows any authorized clinician to immediately halt AI-generated actions. "
            "The system must provide a clear escalation path to a human decision-maker when "
            "the AI confidence falls below 0.75 or when consecutive rejections exceed the "
            "configured threshold. Audit trails of human overrides must be maintained."
        ),
        "healthcare_applicability": (
            "Universal applicability to all agentic AI systems operating in clinical workflows, "
            "including autonomous prescription, scheduling, and diagnostic agents."
        ),
        "risk_threshold": "CRITICAL",
        "enforcement_level": "MANDATORY",
    },
    {
        "policy_id": "NIST-H-09",
        "category": "GOVERN",
        "title": "Informed Consent for AI-Assisted Care",
        "description": (
            "Patients must be informed when AI systems play a material role in their care "
            "decisions. Consent documentation must specify: (a) that AI tools were used, "
            "(b) the general nature of the AI's role, and (c) the patient's right to request "
            "AI-free clinical assessment. This consent must be recorded in the EHR."
        ),
        "healthcare_applicability": (
            "Applies to all patient-facing AI interactions and any AI system whose output "
            "directly influences a clinical care decision."
        ),
        "risk_threshold": "MEDIUM",
        "enforcement_level": "MANDATORY",
    },
    {
        "policy_id": "NIST-H-10",
        "category": "MEASURE",
        "title": "Adversarial Robustness Testing for Clinical AI",
        "description": (
            "AI models used in clinical decision support must undergo adversarial robustness "
            "testing prior to deployment and after each major version update. Testing must "
            "include prompt injection attacks, out-of-distribution patient profiles, and "
            "data poisoning scenarios. Results must be documented in the AI System Card."
        ),
        "healthcare_applicability": (
            "Applicable to all generative AI and LLM-based systems used in clinical "
            "environments, particularly those with tool-calling or agentic capabilities."
        ),
        "risk_threshold": "HIGH",
        "enforcement_level": "RECOMMENDED",
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Simple Keyword-Based Fallback Embedder (no ML deps required)
# ─────────────────────────────────────────────────────────────────────────────

class KeywordEmbedder:
    """
    Deterministic TF-style keyword embedder used when sentence-transformers
    is unavailable (e.g., CPU-only Kaggle environment without the package).
    Projects text into a fixed 128-dim sparse vector via term frequency hashing.
    """

    DIM = 128

    def __init__(self) -> None:
        import hashlib

        self._hash = hashlib.md5

    def encode(self, texts: list[str]) -> list[list[float]]:
        results = []
        for text in texts:
            vec = [0.0] * self.DIM
            words = text.lower().split()
            for word in words:
                idx = int(self._hash(word.encode()).hexdigest(), 16) % self.DIM
                vec[idx] += 1.0
            # L2 normalize
            norm = sum(x**2 for x in vec) ** 0.5 or 1.0
            results.append([x / norm for x in vec])
        return results


# ─────────────────────────────────────────────────────────────────────────────
# Policy Engine
# ─────────────────────────────────────────────────────────────────────────────

class PolicyEngine:
    """
    Manages the NIST AI RMF 2026 Healthcare policy corpus in ChromaDB
    and provides async semantic retrieval for the Auditor node.
    """

    COLLECTION_NAME = "nist_healthcare_policies"

    def __init__(
        self,
        persist_dir: str = "/tmp/aegis_chroma",
        embedding_model: str = "all-MiniLM-L6-v2",
        top_k: int = 5,
    ) -> None:
        self.persist_dir = persist_dir
        self.embedding_model_name = embedding_model
        self.top_k = top_k
        self._client: Optional[Any] = None
        self._collection: Optional[Any] = None
        self._embedder: Optional[Any] = None
        self._initialized = False

    async def initialize(self) -> None:
        """Bootstrap ChromaDB and load the policy corpus."""
        if self._initialized:
            return

        log = logger.bind(component="PolicyEngine")
        log.info("initializing_policy_engine", persist_dir=self.persist_dir)

        await asyncio.get_running_loop().run_in_executor(None, self._sync_initialize)
        self._initialized = True
        log.info("policy_engine_ready", policy_count=len(NIST_HEALTHCARE_POLICIES))

    def _sync_initialize(self) -> None:
        """Synchronous initialization (runs in executor thread)."""
        if CHROMA_AVAILABLE:
            self._client = chromadb.PersistentClient(
                path=self.persist_dir,
                settings=Settings(anonymized_telemetry=False),
            )
        else:
            # In-memory fallback using a simple dict store
            logging.warning(
                "chromadb not installed — using in-memory policy store fallback"
            )
            self._client = None

        # Load embedder
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            self._embedder = SentenceTransformer(self.embedding_model_name)
        else:
            self._embedder = KeywordEmbedder()

        # Populate collection
        if self._client is not None:
            self._setup_chroma_collection()
        # else: chroma unavailable — _sync_retrieve falls back to NIST_HEALTHCARE_POLICIES directly

    def _setup_chroma_collection(self) -> None:
        """Create or get the ChromaDB collection and upsert all policies."""
        existing = [c.name for c in self._client.list_collections()]
        if self.COLLECTION_NAME in existing:
            self._collection = self._client.get_collection(self.COLLECTION_NAME)
            return

        self._collection = self._client.create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        # Build embeddings
        documents = [
            f"{p['title']}. {p['description']} {p['healthcare_applicability']}"
            for p in NIST_HEALTHCARE_POLICIES
        ]
        embeddings = self._embedder.encode(documents)
        ids = [p["policy_id"] for p in NIST_HEALTHCARE_POLICIES]
        metadatas = [
            {
                "policy_id": p["policy_id"],
                "category": p["category"],
                "title": p["title"],
                "risk_threshold": p["risk_threshold"],
                "enforcement_level": p["enforcement_level"],
            }
            for p in NIST_HEALTHCARE_POLICIES
        ]

        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    async def retrieve_relevant_policies(
        self, query: str, top_k: Optional[int] = None
    ) -> list[NISTPolicy]:
        """
        Perform semantic retrieval against the policy corpus.
        Returns the top-k most relevant NISTPolicy objects.
        """
        if not self._initialized:
            await self.initialize()

        k = top_k or self.top_k
        return await asyncio.get_running_loop().run_in_executor(
            None, self._sync_retrieve, query, k
        )

    def _sync_retrieve(self, query: str, k: int) -> list[NISTPolicy]:
        if self._collection is not None:
            query_embedding = self._embedder.encode([query])[0]
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=min(k, len(NIST_HEALTHCARE_POLICIES)),
                include=["documents", "metadatas", "distances"],
            )
            policies = []
            for meta in results["metadatas"][0]:
                pid = meta["policy_id"]
                raw = next(p for p in NIST_HEALTHCARE_POLICIES if p["policy_id"] == pid)
                policies.append(NISTPolicy(**raw))
            return policies
        else:
            # Fallback: return all policies (small corpus)
            return [NISTPolicy(**p) for p in NIST_HEALTHCARE_POLICIES[:k]]

    async def evaluate_proposal_against_policies(
        self,
        proposal_description: str,
        action_type: str,
        risk_level: str,
        patient_allergies: list[str],
        drug_name: Optional[str] = None,
    ) -> list[PolicyCitation]:
        """
        Rule-based + semantic compliance evaluation.
        Returns a list of PolicyCitations with violation flags.
        """
        # Semantic retrieval
        query = f"{action_type} {proposal_description} {risk_level} risk"
        relevant_policies = await self.retrieve_relevant_policies(query)

        citations: list[PolicyCitation] = []

        for policy in relevant_policies:
            violated = False
            violation_detail = None
            remediation = None

            # ── Rule-based violation checks ──────────────────────────────────

            if policy.policy_id == "NIST-H-02" and action_type == "PRESCRIPTION":
                high_risk_drugs = [
                    "oxycodone", "fentanyl", "morphine", "warfarin", "methotrexate",
                    "cyclosporine", "tacrolimus", "hydrocodone", "adderall", "methylphenidate",
                ]
                if drug_name and any(d in drug_name.lower() for d in high_risk_drugs):
                    violated = True
                    violation_detail = (
                        f"Prescription of '{drug_name}' is a controlled/high-risk medication "
                        "requiring mandatory dual-clinician review before execution."
                    )
                    remediation = (
                        "Route to dual-clinician review workflow. Do not auto-execute. "
                        "Flag for attending physician and clinical pharmacist co-sign."
                    )

            elif policy.policy_id == "NIST-H-03" and action_type == "PRESCRIPTION":
                if drug_name and patient_allergies:
                    for allergy in patient_allergies:
                        if allergy.lower() in (drug_name or "").lower():
                            violated = True
                            violation_detail = (
                                f"Patient has documented allergy to '{allergy}'. "
                                f"Proposed drug '{drug_name}' may trigger adverse reaction."
                            )
                            remediation = (
                                "IMMEDIATE BLOCK: Do not proceed. Select alternative medication "
                                "from a different drug class. Consult clinical pharmacist."
                            )

            elif policy.policy_id == "NIST-H-01":
                # Always require transparency — not a violation per se, just ensure it's tracked
                violated = False

            elif policy.policy_id == "NIST-H-05":
                if re.search(r"\b\d{7,}\b", proposal_description):
                    violated = True
                    violation_detail = (
                        "Proposal description appears to contain an unmasked numeric identifier. "
                        "PHI minimization policy may be violated."
                    )
                    remediation = "Tokenize patient identifiers before including in model prompts."

            elif policy.policy_id == "NIST-H-04" and action_type == "LAB_ORDER":
                if risk_level in ("LOW",) and "icd" not in proposal_description.lower():
                    violated = True
                    violation_detail = (
                        "Lab order lacks a documented ICD-10 clinical indication."
                    )
                    remediation = (
                        "Add a valid ICD-10 diagnosis code to the lab order request "
                        "before submitting to the LIS."
                    )

            relevance_score = 0.85 if violated else 0.60

            citations.append(
                PolicyCitation(
                    policy_id=policy.policy_id,
                    policy_title=policy.title,
                    relevance_score=relevance_score,
                    violated=violated,
                    violation_detail=violation_detail,
                    remediation_suggestion=remediation,
                )
            )

        return citations

    def get_all_policies(self) -> list[NISTPolicy]:
        """Return the full policy corpus as NISTPolicy objects."""
        return [NISTPolicy(**p) for p in NIST_HEALTHCARE_POLICIES]
