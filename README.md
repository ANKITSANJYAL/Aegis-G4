# Aegis-G4 — Safety-Audited Healthcare AI Agent

Dual-model clinical AI with NIST AI RMF 2026 compliance auditing.
Orchestrator: gemma-4-31b-it | Auditor: gemma-4-e4b-it (Thinking Mode + RAG)

## Architecture

```
START
  └─► orchestrator_node  (gemma-4-31b-it — proposes tool call)
        └─► auditor_node  (gemma-4-e4b-it + ChromaDB RAG — issues Transparency Certificate)
              ├─ APPROVED      ─► executor_node      ─► END
              ├─ NEEDS_REVIEW  ─► human_review_node  ─► END
              └─ REJECTED      ─► circuit_breaker_node
                                    ├─ open  ─► escalation_node  ─► END
                                    └─ closed ─► orchestrator_node  (retry loop, max 3)
```

## File Structure

| File | Purpose |
|------|---------|
| schema.py | Pydantic v2 models and LangGraph TypedDict state |
| policy_engine.py | ChromaDB RAG engine — 10 NIST AI RMF 2026 Healthcare policies |
| tools.py | MCP-compliant HospitalAPIMCPServer with 5 async tools |
| models.py | vLLM / LiteRT / Mock clients for Gemma 4 31B and E4B |
| graph.py | LangGraph stateful workflow — 6 nodes, conditional routing |
| runner.py | CLI demo runner — 3 clinical scenarios, mock or vllm mode |
| requirements.txt | Pinned Python dependencies |
| pyproject.toml | Build config, ruff / mypy / pytest settings |

## Quick Start (local, no GPU)

```bash
pip install -r requirements.txt
python runner.py                    # all three scenarios, mock models
python runner.py --scenario 2       # oxycodone rejection scenario only
```

## Kaggle / GPU

See the **Kaggle** section below for full setup with real Gemma 4 models.

## NIST AI RMF 2026 Healthcare Policies

| Policy ID | Category | Title |
|-----------|----------|-------|
| NIST-H-01 | GOVERN | AI System Transparency in Clinical Decision Support |
| NIST-H-02 | MAP | High-Risk Medication Prescribing Controls |
| NIST-H-03 | MEASURE | Drug-Allergy Interaction Verification |
| NIST-H-04 | MANAGE | Lab Order Clinical Necessity Documentation |
| NIST-H-05 | GOVERN | PHI Minimization in AI Inference Pipelines |
| NIST-H-06 | MAP | AI Model Version and Drift Monitoring |
| NIST-H-07 | MEASURE | Bias Auditing for Demographic Equity |
| NIST-H-08 | MANAGE | Human Override and Intervention Capability |
| NIST-H-09 | GOVERN | Informed Consent for AI-Assisted Care |
| NIST-H-10 | MEASURE | Adversarial Robustness Testing for Clinical AI |

## Demo Scenarios

| # | Scenario | Expected Outcome |
|---|----------|-----------------|
| 1 | Metformin 1000mg — T2DM first-line | APPROVED, EXECUTED |
| 2 | Oxycodone 10mg — Schedule II opioid | REJECTED x3, circuit breaker opens, ESCALATED |
| 3 | HbA1c lab order — quarterly monitoring | APPROVED, EXECUTED |
