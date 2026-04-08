---
title: RAG-RL
emoji: 🔍
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# RAG-RL

RAG-RL is an OpenEnv-compatible reinforcement learning environment for **evidence selection in retrieval-augmented generation**.

The agent does not retrieve more text. Instead, it must learn to build the **smallest evidence set that still supports a correct answer** — while avoiding stale, contradictory, redundant, and adversarial chunks under a strict token budget.

---

## Why This Problem Matters

Every production RAG system retrieves more chunks than it needs. Sending all of them to the LLM wastes tokens, inflates cost, and degrades answer quality when contradictory or outdated evidence is included. No existing benchmark teaches agents to make this selection decision.

RAG-RL frames chunk selection as a sequential decision problem where the agent must:
- Identify genuinely supportive chunks without access to internal labels
- Avoid stale contradictions and adversarial summaries that look relevant but aren't
- Compose multi-hop evidence from complementary chunks for complex queries
- Stay within a token budget across multiple refinement steps

---

## Design Decisions

**No metadata leakage.** The agent never sees internal labels like `support_type`, `fact_group`, or `pattern`. It must learn to distinguish gold from noise using only observable signals — similarity scores, document type, staleness flag, and the cross-similarity matrix. This is what a real deployed agent would face.

**Multi-step feedback.** The observation includes `last_answer_quality` from the previous step. This gives the agent an intra-episode learning signal: if quality was low, change the selection; if quality was already high, go minimal for the bonus.

**Cross-domain generalization.** The hard task mixes corporate HR/policy queries (TechNova Corp) with medical domain queries (drug interactions, dosing, HIPAA, clinical trials, informed consent). This tests whether an agent can generalize across domains without overfitting to a single topic vocabulary.

**Per-query required fact groups.** Each query has specific named fact groups it requires (e.g. `vacation_entitlement`, `notification_timeline`, `dosage_regimen`). Evidence recall is non-trivial — selecting any gold-looking chunk is not sufficient.

---

## API

| Endpoint | Method | Description |
|---|---|---|
| `/reset` | POST | Start a new episode. Body: `{"task": "easy"\|"medium"\|"hard"}` |
| `/step` | POST | Submit a chunk selection. Body: `{"selected_chunk_indices": [0, 2]}` |
| `/state` | GET | Inspect current episode state without advancing |
| `/health` | GET | Server liveness check |
| `/docs` | GET | Interactive Swagger UI |

---

## Observation Space

| Field | Type | Description |
|---|---|---|
| `query` | string | The question the agent must answer |
| `retrieved_chunks` | list[ChunkInfo] | Top-k candidates from semantic search |
| `cross_similarity_matrix` | list[list[float]] | Pairwise cosine similarity between candidates |
| `token_budget_remaining` | int | Tokens left in this episode |
| `episode_step` | int | Current step index |
| `steps_remaining` | int | Steps left before forced termination |
| `current_selection` | list[int] | Indices selected in last step |
| `candidate_token_costs` | list[int] | Per-chunk token cost |
| `last_answer_quality` | float\|null | Answer quality score from previous step |

Each `ChunkInfo` exposes: `chunk_id`, `content`, `similarity_score`, `token_count`, `source`, `doc_type`, `is_current`.

---

## Reward System

Nine components combine into a single total score clamped to `[0.0, 1.0]`:

| Component | Weight / Range | Signal |
|---|---|---|
| `answer_quality` | 0.55 × [0,1] | LLM judge score vs. reference answer |
| `evidence_precision` | 0.15 × [0,1] | Fraction of selected chunks that are supportive |
| `evidence_recall` | 0.10 × [0,1] | Fraction of required fact groups covered |
| `diversity_bonus` | up to +0.10 | Low cross-similarity between selected chunks (server-side embedding) |
| `minimality_bonus` | up to +0.10 | Full recall with fewest possible chunks |
| `contradiction_penalty` | −0.12 per bad chunk | Stale / adversarial / contradictory chunks selected |
| `redundancy_penalty` | −0.05 (flat) | Any pair of selected chunks with cosine sim > 0.92 |
| `budget_penalty` | −0.08 (flat) | Selection exhausts the entire remaining token budget |
| `token_cost` | up to −0.25 | Proportional to tokens used / total budget (consistent denominator) |

**Total formula:**
```
total = clamp(
    0.55 × answer_quality
  + 0.15 × evidence_precision
  + 0.10 × evidence_recall
  + diversity_bonus
  + minimality_bonus
  + contradiction_penalty   # ≤ 0
  + redundancy_penalty      # 0 or −0.05
  + budget_penalty          # 0 or −0.08
  + token_cost,             # ≤ 0
  0.0, 1.0
)
```

The `Reward` response also includes a `shaping_bonus` field (informational only, not added to `total`) that records whether the agent selected at least one high-similarity chunk (score > 0.80).

`token_cost` uses the **total episode budget** as denominator (not the remaining budget), so the cost of selecting N tokens is identical on step 1 and step 3. This prevents a perverse incentive where burning tokens early is cheaper than burning them late.

LLM judging uses `Qwen/Qwen2.5-72B-Instruct` via HuggingFace router. If the LLM is unavailable, answer quality falls back to semantic similarity (sentence-transformer cosine score) to prevent spurious zero rewards.

**Early termination:** If `evidence_recall == 1.0` and `answer_quality ≥ 0.85` at any step, the episode ends immediately. An agent that finds the right evidence on the first step is not forced to take unnecessary refinement steps, preserving its token budget and step efficiency.

**Score calibration — worked examples:**

| Scenario | quality | precision | recall | diversity | minimality | contradiction | redundancy | budget | token_cost | **total** |
|---|---|---|---|---|---|---|---|---|---|---|
| Smart: 1 gold chunk, minimal | 0.80 | 1.00 | 1.00 | 0.00 | +0.10 | 0 | 0 | 0 | −0.10 | **0.82** |
| Smart: 2 diverse gold chunks, multi-hop | 0.90 | 1.00 | 1.00 | +0.08 | +0.10 | 0 | 0 | 0 | −0.18 | **0.90** |
| Baseline: 1 best-similarity chunk | 0.70 | 1.00 | 1.00 | 0.00 | +0.10 | 0 | 0 | 0 | −0.10 | **0.675** |
| Bad: selects stale contradiction | 0.20 | 0.00 | 0.00 | 0.00 | 0 | −0.12 | 0 | 0 | −0.05 | **0.0** (clamped) |
| Bad: selects 2 near-duplicate chunks | 0.65 | 1.00 | 0.50 | 0.00 | 0 | 0 | −0.05 | 0 | −0.18 | **0.528** |

---

## Tasks

| Task | Docs | Queries | top_k | Budget | Max Steps | Failure Modes |
|---|---:|---:|---:|---:|---:|---|
| `easy` | 50 | 15 | 4 | 200 tok | 2 | Redundancy, noise |
| `medium` | 200 | 15 | 6 | 250 tok | 3 | Redundancy, stale docs, adversarial, contradictions |
| `hard` | 500 | 20 | 8 | 200 tok | 4 | All above + multi-hop + medical domain |

Difficulty increases by **evidence ambiguity and cost pressure**, not corpus size alone.

---

## Corpus Design

**TechNova Corp (20 topics):** HR and engineering policy — vacation, sick leave, parental leave, stock options, travel policy, software stack, SLA metrics, and more. Each topic includes a golden document, redundant paraphrases, a stale contradiction, a per-topic adversarial summary, and noise.

**Medical domain (5 topics, hard task only):** Drug interactions (warfarin + NSAIDs), amoxicillin dosing for community-acquired pneumonia, informed consent elements, HIPAA breach notification timelines, and clinical trial phase distinctions.

**PyTorch / Meta / Scalar domain (5 topics, hard task only):** `torch.no_grad()` vs `torch.inference_mode()`, FSDP vs DDP distributed training (multi-hop), `torch.compile()` and TorchInductor, Scalar kernel fusion and HBM bandwidth reduction, and LLaMA 3 70B memory requirements and context length.

The hard task mixes all three domains (10 TechNova + 5 Medical + 5 PyTorch/Scalar/Meta = 20 queries). Cross-domain mixing tests whether an agent can generalize without overfitting to a single topic vocabulary — the same selection strategy must work across HR policy, clinical guidelines, and ML systems documentation.

---

## Example Episode

```
[START] task=easy env=rag-rl model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=select([0]) reward=0.67 done=false error=null
[STEP] step=2 action=select([0]) reward=0.67 done=true  error=null
[END]  success=true steps=2 score=0.67 rewards=0.67,0.67
```

A smarter agent that avoids redundant chunks, checks `is_current`, and uses `last_answer_quality` to refine across steps is expected to score **0.80+** on easy and medium tasks.

---

## Baseline

```bash
python inference.py --task easy --episodes 5 --host localhost --port 7860
```

The baseline policy selects the single highest-similarity chunk each step. It is intentionally simple — a trained agent that uses cross-similarity, staleness signals, and multi-step refinement should significantly outperform it.

---

## Running Locally

```bash
docker build -t rag-rl .
docker run -p 7860:7860 \
  -e HF_TOKEN=your_token \
  -e API_BASE_URL=https://router.huggingface.co/v1 \
  -e MODEL_NAME=Qwen/Qwen2.5-72B-Instruct \
  rag-rl
```

Then open `http://localhost:7860/docs` for the interactive API.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `HF_TOKEN` | — | HuggingFace / LLM API key |
| `API_BASE_URL` | `https://router.huggingface.co/v1` | OpenAI-compatible LLM endpoint |
| `MODEL_NAME` | `Qwen/Qwen2.5-72B-Instruct` | Model for answer generation and judging |
