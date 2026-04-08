"""
RAG-RL FastAPI application.
OpenEnv-compatible RL environment for chunk selection in RAG pipelines.
"""

from contextlib import asynccontextmanager
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from env import Observation, RagRLEnvironment, Reward
from tasks import TASKS

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class ResetRequest(BaseModel):
    task: Optional[str] = "easy"


class StepRequest(BaseModel):
    selected_chunk_indices: List[int]


class StepResponse(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: dict


# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------

_envs: Dict[str, RagRLEnvironment] = {}
_active_task: str = "easy"


def _get_env(task_name: str) -> RagRLEnvironment:
    if task_name not in TASKS:
        raise HTTPException(status_code=400, detail=f"Unknown task '{task_name}'. Choose from: {list(TASKS)}")
    if task_name not in _envs:
        _envs[task_name] = RagRLEnvironment(TASKS[task_name])
    return _envs[task_name]


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        _get_env("easy")
    except Exception:
        pass
    yield


# ---------------------------------------------------------------------------
# Shared back-bar CSS (injected into /docs and /redoc)
# ---------------------------------------------------------------------------

_BACK_BAR = """
<style>
  body { margin-top: 44px !important; }
  #rag-bar {
    position: fixed; top: 0; left: 0; right: 0; height: 44px;
    background: rgba(13,17,23,0.97);
    border-bottom: 1px solid #30363d;
    display: flex; align-items: center;
    padding: 0 24px; z-index: 99999;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    backdrop-filter: blur(10px);
    gap: 0;
  }
  #rag-bar a.home {
    color: #ee6c2e; text-decoration: none; font-size: 13px;
    font-weight: 700; display: flex; align-items: center; gap: 6px;
    padding: 6px 12px; border-radius: 7px;
    border: 1px solid rgba(238,108,46,.3);
    transition: all .15s;
  }
  #rag-bar a.home:hover {
    background: rgba(238,108,46,.1); border-color: rgba(238,108,46,.6);
  }
  #rag-bar .bar-title {
    color: #8b949e; font-size: 12px; font-weight: 500;
    flex: 1; text-align: center;
  }
  #rag-bar a.alt {
    color: #58a6ff; text-decoration: none; font-size: 12px;
    font-weight: 500; padding: 6px 12px; border-radius: 7px;
    border: 1px solid rgba(88,166,255,.25); transition: all .15s;
  }
  #rag-bar a.alt:hover {
    background: rgba(88,166,255,.08); border-color: rgba(88,166,255,.5);
  }
</style>
"""

_SWAGGER_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>RAG-RL · API Explorer</title>
<link rel="stylesheet"
  href="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css"/>
""" + _BACK_BAR + """
</head>
<body>
<div id="rag-bar">
  <a href="/" class="home">&#8592; RAG-RL Home</a>
  <span class="bar-title">&#9889; API Explorer &mdash; Interactive endpoint testing</span>
  <a href="/redoc" class="alt">Full Docs &#8594;</a>
</div>
<div id="swagger-ui"></div>
<script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
<script>
  SwaggerUIBundle({
    url: "/openapi.json",
    dom_id: "#swagger-ui",
    presets: [SwaggerUIBundle.presets.apis, SwaggerUIBundle.SwaggerUIStandalonePreset],
    layout: "BaseLayout",
    deepLinking: true,
  });
</script>
</body>
</html>"""

_REDOC_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>RAG-RL · Documentation</title>
""" + _BACK_BAR + """
</head>
<body>
<div id="rag-bar">
  <a href="/" class="home">&#8592; RAG-RL Home</a>
  <span class="bar-title">&#128218; Full Documentation &mdash; Schema &amp; reward reference</span>
  <a href="/docs" class="alt">API Explorer &#8594;</a>
</div>
<div id="redoc-container"></div>
<script src="https://cdn.jsdelivr.net/npm/redoc@2.1.3/bundles/redoc.standalone.js"></script>
<script>
  Redoc.init(
    "/openapi.json",
    {
      theme: {
        colors: { primary: { main: "#ee6c2e" } },
        typography: {
          fontSize: "14px",
          fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"
        },
        sidebar: { backgroundColor: "#0d1117", textColor: "#e6edf3" }
      },
      hideDownloadButton: false,
      scrollYOffset: 44
    },
    document.getElementById("redoc-container")
  );
</script>
</body>
</html>"""

# ---------------------------------------------------------------------------
# Landing page
# ---------------------------------------------------------------------------

_LANDING_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>RAG-RL · OpenEnv</title>
<style>
  :root {
    --bg:      #0d1117;
    --surface: #161b22;
    --border:  #30363d;
    --accent:  #ee6c2e;
    --accent2: #f0a05a;
    --text:    #e6edf3;
    --muted:   #8b949e;
    --green:   #3fb950;
    --blue:    #58a6ff;
    --purple:  #bc8cff;
    --red:     #f85149;
    --radius:  10px;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg); color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    line-height: 1.6; min-height: 100vh;
  }

  /* ── Header ── */
  header {
    border-bottom: 1px solid var(--border);
    padding: 16px 40px;
    display: flex; align-items: center; justify-content: space-between;
    position: sticky; top: 0;
    background: rgba(13,17,23,0.92);
    backdrop-filter: blur(8px); z-index: 100;
  }
  .logo { display: flex; align-items: center; gap: 12px; }
  .logo-icon {
    width: 34px; height: 34px; border-radius: 8px;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    display: flex; align-items: center; justify-content: center;
    font-size: 17px; font-weight: 800; color: #fff;
  }
  .logo-text { font-size: 19px; font-weight: 700; letter-spacing: -.3px; }
  .logo-badge {
    background: var(--surface); border: 1px solid var(--border);
    color: var(--muted); font-size: 11px; padding: 2px 8px;
    border-radius: 20px; font-weight: 500;
  }

  /* ── Hero ── */
  .hero {
    text-align: center; padding: 68px 40px 52px;
    max-width: 820px; margin: 0 auto;
  }
  .hero-tag {
    display: inline-block;
    background: rgba(238,108,46,.12); color: var(--accent2);
    border: 1px solid rgba(238,108,46,.28);
    font-size: 11px; font-weight: 700; letter-spacing: .7px;
    padding: 4px 12px; border-radius: 20px; margin-bottom: 20px;
    text-transform: uppercase;
  }
  h1 {
    font-size: 50px; font-weight: 800; letter-spacing: -1.5px;
    line-height: 1.1; margin-bottom: 18px;
    background: linear-gradient(135deg, #fff 40%, var(--accent2));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  }
  .hero p {
    font-size: 17px; color: var(--muted); max-width: 580px;
    margin: 0 auto 28px;
  }
  .stats {
    display: flex; gap: 10px; justify-content: center;
    flex-wrap: wrap; margin-bottom: 36px;
  }
  .stat {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 20px; padding: 5px 14px;
    font-size: 12.5px; font-weight: 500; color: var(--text);
  }
  .stat span { color: var(--accent2); font-weight: 700; }

  /* ── CTA cards ── */
  .cta-row {
    display: flex; gap: 16px; justify-content: center;
    flex-wrap: wrap; margin-bottom: 16px;
  }
  .cta-card {
    display: flex; flex-direction: column; align-items: flex-start;
    width: 270px; padding: 22px 24px; border-radius: 14px;
    text-decoration: none; cursor: pointer;
    transition: transform .15s, box-shadow .15s, border-color .15s;
    backdrop-filter: blur(6px);
  }
  .cta-card:hover { transform: translateY(-3px); }
  .cta-orange {
    background: rgba(238,108,46,.08);
    border: 1px solid rgba(238,108,46,.35);
  }
  .cta-orange:hover {
    border-color: rgba(238,108,46,.75);
    box-shadow: 0 8px 32px rgba(238,108,46,.18);
  }
  .cta-blue {
    background: rgba(88,166,255,.08);
    border: 1px solid rgba(88,166,255,.35);
  }
  .cta-blue:hover {
    border-color: rgba(88,166,255,.75);
    box-shadow: 0 8px 32px rgba(88,166,255,.18);
  }
  .cta-tag {
    font-size: 10px; font-weight: 700; text-transform: uppercase;
    letter-spacing: .8px; margin-bottom: 10px; display: block;
  }
  .cta-orange .cta-tag { color: var(--accent); }
  .cta-blue   .cta-tag { color: var(--blue); }
  .cta-name {
    font-size: 18px; font-weight: 700; color: var(--text); margin-bottom: 8px;
  }
  .cta-blurb { font-size: 12.5px; color: var(--muted); line-height: 1.55; }

  /* ── Main content grid ── */
  main {
    max-width: 1100px; margin: 0 auto;
    padding: 0 40px 80px;
    display: grid; grid-template-columns: 1fr 1fr; gap: 24px;
  }
  .full { grid-column: 1 / -1; }

  /* ── Cards ── */
  .card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 24px; overflow: hidden;
  }
  .card-title {
    font-size: 12px; font-weight: 600; text-transform: uppercase;
    letter-spacing: .6px; color: var(--muted); margin-bottom: 16px;
    display: flex; align-items: center; gap: 8px;
  }
  .card-title .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--accent); }

  /* ── Reward table ── */
  table { width: 100%; border-collapse: collapse; font-size: 13.5px; }
  th {
    text-align: left; color: var(--muted); font-weight: 500;
    font-size: 12px; letter-spacing: .4px; text-transform: uppercase;
    padding: 0 10px 10px 0; border-bottom: 1px solid var(--border);
  }
  td { padding: 9px 10px 9px 0; border-bottom: 1px solid rgba(48,54,61,.5); }
  tr:last-child td { border-bottom: none; }
  .tag {
    font-family: "SF Mono","Fira Code",monospace;
    font-size: 12px; padding: 2px 8px; border-radius: 5px;
    font-weight: 600; white-space: nowrap;
  }
  .pos     { background: rgba(63,185,80,.12);  color: var(--green); }
  .neg     { background: rgba(248,81,73,.12);  color: var(--red); }
  .neutral { background: rgba(88,166,255,.12); color: var(--blue); }
  .weight  { color: var(--accent2); font-weight: 700; font-size: 13px; }

  /* ── Task cards ── */
  .tasks { display: grid; grid-template-columns: repeat(3,1fr); gap: 14px; }
  .task-card {
    background: var(--bg); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 18px;
  }
  .task-name {
    font-size: 15px; font-weight: 700; margin-bottom: 10px;
    display: flex; align-items: center; gap: 8px;
  }
  .task-pill {
    font-size: 10px; font-weight: 600; text-transform: uppercase;
    letter-spacing: .5px; padding: 2px 8px; border-radius: 10px;
  }
  .easy-pill { background: rgba(63,185,80,.15);  color: var(--green); }
  .med-pill  { background: rgba(240,160,90,.15); color: var(--accent2); }
  .hard-pill { background: rgba(248,81,73,.15);  color: var(--red); }
  .task-row {
    display: flex; justify-content: space-between;
    font-size: 12.5px; color: var(--muted); padding: 3px 0;
  }
  .task-row strong { color: var(--text); }
  .task-desc {
    font-size: 12px; color: var(--muted); margin-top: 10px;
    padding-top: 10px; border-top: 1px solid var(--border);
  }

  /* ── Code block ── */
  .code-wrap {
    background: #010409; border: 1px solid var(--border);
    border-radius: var(--radius); overflow: hidden;
  }
  .code-header {
    background: var(--surface); border-bottom: 1px solid var(--border);
    padding: 8px 16px; font-size: 12px; color: var(--muted);
    display: flex; align-items: center; justify-content: space-between;
  }
  .code-dots { display: flex; gap: 6px; }
  .code-dot  { width: 10px; height: 10px; border-radius: 50%; }
  pre {
    padding: 20px;
    font-family: "SF Mono","Fira Code","Cascadia Code",monospace;
    font-size: 13px; line-height: 1.7;
    overflow-x: auto; color: #c9d1d9;
  }
  .kw { color: #ff7b72; } .fn { color: #d2a8ff; }
  .st { color: #a5d6ff; } .cm { color: #8b949e; }
  .nm { color: #79c0ff; } .op { color: #ff7b72; }

  /* ── Domain panels ── */
  .domains { display: flex; flex-wrap: wrap; gap: 10px; }
  .domain {
    background: var(--bg); border: 1px solid var(--border);
    border-radius: 8px; padding: 12px 16px; flex: 1; min-width: 160px;
  }
  .domain-name  { font-size: 13px; font-weight: 700; margin-bottom: 4px; }
  .domain-count { font-size: 11px; color: var(--muted); margin-bottom: 6px; }
  .domain-topics{ font-size: 12px; color: var(--muted); line-height: 1.5; }

  /* ── Footer ── */
  footer {
    border-top: 1px solid var(--border); text-align: center;
    padding: 24px; font-size: 13px; color: var(--muted);
  }
  footer a { color: var(--blue); text-decoration: none; }
</style>
</head>
<body>

<header>
  <div class="logo">
    <div class="logo-icon">R</div>
    <div class="logo-text">RAG-RL</div>
    <div class="logo-badge">OpenEnv v1.0</div>
  </div>
</header>

<!-- ── Hero ── -->
<section class="hero">
  <div class="hero-tag">Reinforcement Learning Environment</div>
  <h1>Optimal Chunk<br/>Selection for RAG</h1>
  <p>A multi-step RL environment where agents learn to select the right evidence
     chunks from a noisy corpus — balancing quality, precision, diversity,
     and token cost.</p>

  <div class="stats">
    <div class="stat"><span>9</span> Reward Components</div>
    <div class="stat"><span>3</span> Difficulty Tasks</div>
    <div class="stat"><span>3</span> Cross-Domain Corpora</div>
    <div class="stat"><span>500</span> Docs (hard task)</div>
    <div class="stat"><span>4</span> Failure Modes</div>
  </div>

  <!-- ── Primary CTAs ── -->
  <div class="cta-row">
    <a href="/docs" class="cta-card cta-orange">
      <span class="cta-tag">&#9889; Interactive</span>
      <div class="cta-name">API Explorer &nbsp;&#8599;</div>
      <div class="cta-blurb">Try every endpoint live — reset episodes, submit
        chunk selections, and inspect the full reward breakdown in real time.</div>
    </a>
    <a href="/redoc" class="cta-card cta-blue">
      <span class="cta-tag">&#128218; Reference</span>
      <div class="cta-name">Full Docs &nbsp;&#8599;</div>
      <div class="cta-blurb">Complete schema for Observation, Action &amp; Reward
        models — all 9 reward components documented with types and ranges.</div>
    </a>
  </div>
</section>

<!-- ── Content ── -->
<main>

  <!-- Reward table -->
  <div class="card full">
    <div class="card-title"><div class="dot"></div>Reward System &mdash; 9 Components &rarr; total &isin; [0.0, 1.0]</div>
    <table>
      <thead>
        <tr>
          <th>Component</th><th>Weight / Range</th>
          <th>What It Measures</th><th>Type</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td><code class="tag neutral">answer_quality</code></td>
          <td class="weight">0.55 &times; [0,1]</td>
          <td>LLM judge score vs. reference answer (Qwen2.5-72B)</td>
          <td><span class="tag pos">+bonus</span></td>
        </tr>
        <tr>
          <td><code class="tag neutral">evidence_precision</code></td>
          <td class="weight">0.15 &times; [0,1]</td>
          <td>Fraction of selected chunks that are genuinely supportive</td>
          <td><span class="tag pos">+bonus</span></td>
        </tr>
        <tr>
          <td><code class="tag neutral">evidence_recall</code></td>
          <td class="weight">0.10 &times; [0,1]</td>
          <td>Fraction of required fact-groups covered by selection</td>
          <td><span class="tag pos">+bonus</span></td>
        </tr>
        <tr>
          <td><code class="tag neutral">diversity_bonus</code></td>
          <td class="weight">up to +0.10</td>
          <td>Low cross-similarity between selected chunks (server-side embeddings)</td>
          <td><span class="tag pos">+bonus</span></td>
        </tr>
        <tr>
          <td><code class="tag neutral">minimality_bonus</code></td>
          <td class="weight">up to +0.10</td>
          <td>Full recall achieved with the fewest possible chunks</td>
          <td><span class="tag pos">+bonus</span></td>
        </tr>
        <tr>
          <td><code class="tag neutral">contradiction_penalty</code></td>
          <td class="weight">&minus;0.12 per chunk</td>
          <td>Stale, adversarial, or contradictory chunks selected</td>
          <td><span class="tag neg">&minus;penalty</span></td>
        </tr>
        <tr>
          <td><code class="tag neutral">redundancy_penalty</code></td>
          <td class="weight">&minus;0.05 flat</td>
          <td>Any selected pair with cosine similarity &gt; 0.92</td>
          <td><span class="tag neg">&minus;penalty</span></td>
        </tr>
        <tr>
          <td><code class="tag neutral">budget_penalty</code></td>
          <td class="weight">&minus;0.08 flat</td>
          <td>Selection exhausts the entire remaining token budget</td>
          <td><span class="tag neg">&minus;penalty</span></td>
        </tr>
        <tr>
          <td><code class="tag neutral">token_cost</code></td>
          <td class="weight">up to &minus;0.25</td>
          <td>Tokens used &divide; total budget (fixed denominator &mdash; consistent across steps)</td>
          <td><span class="tag neg">&minus;penalty</span></td>
        </tr>
      </tbody>
    </table>
  </div>

  <!-- Tasks -->
  <div class="card full">
    <div class="card-title"><div class="dot"></div>Tasks</div>
    <div class="tasks">
      <div class="task-card">
        <div class="task-name">Easy <span class="task-pill easy-pill">Starter</span></div>
        <div class="task-row"><span>Documents</span><strong>50</strong></div>
        <div class="task-row"><span>Queries</span><strong>15</strong></div>
        <div class="task-row"><span>top_k retrieved</span><strong>4</strong></div>
        <div class="task-row"><span>Token budget</span><strong>200</strong></div>
        <div class="task-row"><span>Max steps</span><strong>2</strong></div>
        <div class="task-row"><span>Target score</span><strong>&ge; 0.75</strong></div>
        <div class="task-desc">FAQ-style corpus. Golden + redundant paraphrases. Noise distractors. Tests basic precision.</div>
      </div>
      <div class="task-card">
        <div class="task-name">Medium <span class="task-pill med-pill">Intermediate</span></div>
        <div class="task-row"><span>Documents</span><strong>200</strong></div>
        <div class="task-row"><span>Queries</span><strong>15</strong></div>
        <div class="task-row"><span>top_k retrieved</span><strong>6</strong></div>
        <div class="task-row"><span>Token budget</span><strong>250</strong></div>
        <div class="task-row"><span>Max steps</span><strong>3</strong></div>
        <div class="task-row"><span>Target score</span><strong>&ge; 0.65</strong></div>
        <div class="task-desc">Adds stale contradictions and adversarial summaries. Agent must distinguish current vs. outdated docs.</div>
      </div>
      <div class="task-card">
        <div class="task-name">Hard <span class="task-pill hard-pill">Expert</span></div>
        <div class="task-row"><span>Documents</span><strong>500</strong></div>
        <div class="task-row"><span>Queries</span><strong>20</strong></div>
        <div class="task-row"><span>top_k retrieved</span><strong>8</strong></div>
        <div class="task-row"><span>Token budget</span><strong>200</strong></div>
        <div class="task-row"><span>Max steps</span><strong>4</strong></div>
        <div class="task-row"><span>Target score</span><strong>&ge; 0.55</strong></div>
        <div class="task-desc">Multi-hop queries, 3-domain corpus, tighter budget. All failure modes active.</div>
      </div>
    </div>
  </div>

  <!-- Corpus domains -->
  <div class="card">
    <div class="card-title"><div class="dot"></div>Corpus Domains (Hard Task)</div>
    <div class="domains">
      <div class="domain">
        <div class="domain-name" style="color:var(--blue)">TechNova Corp</div>
        <div class="domain-count">20 topics &middot; easy + medium + hard</div>
        <div class="domain-topics">Vacation, sick leave, parental leave, stock options, travel policy, 401k, software stack, SLA metrics, data infrastructure&hellip;</div>
      </div>
      <div class="domain">
        <div class="domain-name" style="color:var(--green)">Medical</div>
        <div class="domain-count">5 topics &middot; hard only</div>
        <div class="domain-topics">Warfarin&ndash;NSAID interaction, amoxicillin dosing, informed consent, HIPAA breach timelines, clinical trial phases</div>
      </div>
      <div class="domain">
        <div class="domain-name" style="color:var(--accent2)">PyTorch &middot; Meta &middot; Scalar</div>
        <div class="domain-count">5 topics &middot; hard only</div>
        <div class="domain-topics">no_grad vs inference_mode, FSDP vs DDP (multi-hop), torch.compile + TorchInductor, Scalar kernel fusion, LLaMA 3 70B memory</div>
      </div>
    </div>
  </div>

  <!-- Quick start -->
  <div class="card">
    <div class="card-title"><div class="dot"></div>Quick Start</div>
    <div class="code-wrap">
      <div class="code-header">
        <div class="code-dots">
          <div class="code-dot" style="background:#ff5f57"></div>
          <div class="code-dot" style="background:#febc2e"></div>
          <div class="code-dot" style="background:#28c840"></div>
        </div>
        <span>Python &middot; 3 lines to run an episode</span>
      </div>
<pre><span class="kw">import</span> requests

BASE <span class="op">=</span> <span class="st">"https://shrutianubolu-rag-rl.hf.space"</span>

<span class="cm"># 1. Start a new episode</span>
obs <span class="op">=</span> requests.<span class="fn">post</span>(<span class="st">f"{BASE}/reset"</span>, json<span class="op">=</span>{<span class="st">"task"</span>: <span class="st">"hard"</span>}).<span class="fn">json</span>()

<span class="cm"># 2. Select top-2 chunks by similarity</span>
chunks <span class="op">=</span> obs[<span class="st">"retrieved_chunks"</span>]
top2 <span class="op">=</span> [c[<span class="st">"chunk_id"</span>] <span class="kw">for</span> c <span class="kw">in</span>
        <span class="fn">sorted</span>(chunks, key<span class="op">=</span><span class="kw">lambda</span> x: x[<span class="st">"similarity_score"</span>], reverse<span class="op">=</span><span class="nm">True</span>)[:<span class="nm">2</span>]]

<span class="cm"># 3. Step and read the reward</span>
result <span class="op">=</span> requests.<span class="fn">post</span>(<span class="st">f"{BASE}/step"</span>,
           json<span class="op">=</span>{<span class="st">"selected_chunk_indices"</span>: top2}).<span class="fn">json</span>()
<span class="fn">print</span>(result[<span class="st">"reward"</span>][<span class="st">"total"</span>])  <span class="cm"># e.g. 0.6712</span></pre>
    </div>
  </div>

  <!-- Design decisions -->
  <div class="card full">
    <div class="card-title"><div class="dot"></div>Design Decisions</div>
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-top:4px">
      <div style="padding:16px;background:var(--bg);border:1px solid var(--border);border-radius:8px">
        <div style="font-size:13px;font-weight:700;margin-bottom:6px;color:var(--blue)">No metadata leakage</div>
        <div style="font-size:12px;color:var(--muted)">Agents never see <code>support_type</code>, <code>fact_group</code>, or <code>pattern</code> labels. Only public signals: similarity score, <code>is_current</code>, and the cross-similarity matrix.</div>
      </div>
      <div style="padding:16px;background:var(--bg);border:1px solid var(--border);border-radius:8px">
        <div style="font-size:13px;font-weight:700;margin-bottom:6px;color:var(--green)">Multi-step feedback</div>
        <div style="font-size:12px;color:var(--muted)"><code>last_answer_quality</code> is fed back in the next observation so the agent can learn from prior-step mistakes without rerunning the query.</div>
      </div>
      <div style="padding:16px;background:var(--bg);border:1px solid var(--border);border-radius:8px">
        <div style="font-size:13px;font-weight:700;margin-bottom:6px;color:var(--accent2)">Early exit reward</div>
        <div style="font-size:12px;color:var(--muted)">Episode ends early if <code>recall&nbsp;=&nbsp;1.0</code> and <code>quality&nbsp;&ge;&nbsp;0.85</code>. Smart agents that nail it on step 1 are not penalized with forced extra steps.</div>
      </div>
      <div style="padding:16px;background:var(--bg);border:1px solid var(--border);border-radius:8px">
        <div style="font-size:13px;font-weight:700;margin-bottom:6px;color:var(--purple)">Fixed cost denominator</div>
        <div style="font-size:12px;color:var(--muted)">Token cost uses the <em>total</em> episode budget as denominator so spending N tokens costs the same on every step &mdash; no perverse late-step discount.</div>
      </div>
    </div>
  </div>

</main>

<footer>
  Built for the <strong>OpenEnv Hackathon</strong> &middot; RAG-RL v1.0 &middot;
  <a href="/docs">API Explorer</a> &middot;
  <a href="/redoc">Full Docs</a> &middot;
  <a href="/openenv.yaml">openenv.yaml</a>
</footer>

</body>
</html>"""


# ---------------------------------------------------------------------------
# App  (docs_url/redoc_url=None so we serve our own styled versions)
# ---------------------------------------------------------------------------

app = FastAPI(
    title="RAG-RL",
    description="RL environment for optimal chunk selection in RAG pipelines",
    version="1.0",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def root():
    return HTMLResponse(content=_LANDING_HTML)


@app.get("/docs", response_class=HTMLResponse, include_in_schema=False)
def custom_swagger():
    return HTMLResponse(content=_SWAGGER_HTML)


@app.get("/redoc", response_class=HTMLResponse, include_in_schema=False)
def custom_redoc():
    return HTMLResponse(content=_REDOC_HTML)


@app.post("/reset", response_model=Observation)
def reset(request: ResetRequest = ResetRequest()):
    """
    Start a new episode. Optionally specify a task (easy / medium / hard).
    Returns the initial Observation with the sampled query and retrieved chunks.
    """
    global _active_task
    task_name = (request.task or "easy").lower()
    env = _get_env(task_name)
    _active_task = task_name
    obs = env.reset()
    return obs


@app.post("/step", response_model=StepResponse)
def step(request: StepRequest):
    """
    Take an action: pass selected chunk indices to the LLM, receive a reward.
    """
    env = _envs.get(_active_task)
    if env is None or env.current_query is None:
        raise HTTPException(status_code=400, detail="Call /reset before /step.")
    obs, reward, done, info = env.step(request.selected_chunk_indices)
    return StepResponse(observation=obs, reward=reward, done=done, info=info)


@app.get("/state")
def state():
    """Return current episode state without advancing the environment."""
    env = _envs.get(_active_task)
    if env is None:
        return {"message": "No active environment. Call /reset first."}
    return env.get_state()


@app.get("/health")
def health():
    return {"status": "ok", "active_task": _active_task, "loaded_tasks": list(_envs.keys())}
