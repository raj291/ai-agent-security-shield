# AI Agent Security Shield

A multi-agent cybersecurity system that protects AI agents from attacks.

## What It Does

Sits in front of any AI agent and intercepts:
- **Prompt Injection** — direct and indirect instruction override attempts
- **Jailbreaks** — DAN, persona manipulation, developer mode exploits
- **System Prompt Extraction** — attempts to reveal agent instructions
- **Data Exfiltration** — URL injection, encoded payloads, indirect injection
- **PII Leakage** (Day 5) — SSNs, credit cards, emails in outputs
- **Memory Poisoning** (Day 3) — malicious documents in knowledge base
- **Agent Compromise** (Day 7) — behavioral anomaly detection

## Architecture

```
User Input
    │
    ▼
Guardian Commander (LangGraph Orchestrator)
    │
    ├─► Input Guard (3-Layer Firewall)
    │      Layer 1: Pattern Matching (65+ patterns, <1ms)
    │      Layer 2: LLM Classifier + ChromaDB cache (Day 2)
    │      Layer 3: Scope Validator (Day 2)
    │
    ├─► Memory Guard (Day 3)
    │      Hidden text scanner (PyMuPDF)
    │      Memory write validator
    │
    ├─► Tool Guard (Day 4)
    │      Permission matrix
    │      Parameter sanitizer
    │      Rate limiter
    │
    ├─► Output Guard (Day 5)
    │      PII detection (Presidio)
    │      System prompt leak detection
    │
    └─► Trust Agent (Day 7)
           Behavioral fingerprinting
           Auto-isolation (score < 60)
```

## Day 1 Results (Pattern Matching Only)

| Metric | Score |
|--------|-------|
| Detection Rate | 100% (70/70 known attacks) |
| False Positive Rate | 0% (0/50 clean inputs) |
| Latency | < 1ms |
| API Cost | $0 |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Copy and configure environment
cp .env.example .env
# Add your ANTHROPIC_API_KEY to .env

# Run demo
python main.py

# Run tests
pytest tests/ -v
```

## Project Structure

```
ai-agent-security-shield/
├── agents/
│   ├── guardian_commander.py   # LangGraph orchestrator
│   ├── protected_agent.py      # Simulated protected agent
│   └── state.py                # Shared state TypedDict
├── guards/
│   ├── input_guard/            # Day 1-2: Prompt injection firewall
│   ├── memory_guard/           # Day 3: RAG poisoning prevention
│   ├── tool_guard/             # Day 4: Secure tool proxy
│   ├── output_guard/           # Day 5: Data leakage prevention
│   └── trust_agent/            # Day 7: Agent trust scoring
├── attack_samples/
│   ├── malicious/              # 70+ attack samples
│   └── clean/                  # 50 legitimate inputs
├── tests/
│   └── test_input_guard.py
├── TECHNICAL_NOTES.md          # Architecture deep dives + interview prep
├── requirements.txt
└── main.py
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Agent Framework | LangGraph |
| LLM | Claude API (Anthropic) |
| Vector Store | ChromaDB |
| PII Detection | Microsoft Presidio |
| Document Scanning | PyMuPDF |
| Policy Engine | OPA |
| API Server | FastAPI |
| Dashboard | React + Vite |
| Testing | pytest |

## Build Plan

- **Day 1** ✅ Project skeleton + Input Guard Layer 1 (pattern matching)
- **Day 2** — Input Guard Layers 2 & 3 (LLM classifier + scope validator)
- **Day 3** — Memory Guard (RAG poisoning prevention)
- **Day 4** — Tool Guard (secure tool proxy)
- **Day 5** — Output Guard (PII + leakage detection)
- **Day 6** — Commander integration + end-to-end tests
- **Day 7** — Trust Agent (behavioral fingerprinting)
- **Day 8-9** — React dashboard
- **Day 10-14** — Testing, hardening, polish

---

> See `TECHNICAL_NOTES.md` for in-depth architecture explanations and interview prep.
