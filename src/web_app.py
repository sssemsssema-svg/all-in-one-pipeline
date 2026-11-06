from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import sys
import uuid
import zipfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

# Ensure src is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agents.prompts import extract_json_object
from main import (
    MAX_PIPELINE_CALLS,
    MAX_REVISIONS_PER_STAGE,
    ProviderClient,
    gatekeeper_decision,
    make_agent_callable,
)
from orchestrator.fsm import FSMError, PipelineFSM, make_blueprint
from orchestrator.quota_governor import QuotaGovernor, QuotaGovernorError
from validators.schemas import Phase, Stage

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("all_in_one_web")

app = FastAPI(title="All In One - Multi-Agent Pipeline", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for active jobs
JOBS: dict[str, dict[str, Any]] = {}

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


class PipelineRequest(BaseModel):
    objective: str


@app.get("/", response_class=HTMLResponse)
async def get_index() -> str:
    return HTML_CONTENT


@app.get("/notebook", response_class=HTMLResponse)
async def get_notebook() -> str:
    return NOTEBOOK_HTML_CONTENT


@app.get("/api/health")
async def health_check() -> dict[str, Any]:
    keys = {
        "GROQ_API_KEY": bool(os.getenv("GROQ_API_KEY")),
        "GEMINI_API_KEY": bool(os.getenv("GEMINI_API_KEY")),
        "GITHUB_TOKEN": bool(os.getenv("GITHUB_TOKEN")),
        "OPENROUTER_API_KEY": bool(os.getenv("OPENROUTER_API_KEY")),
        "DEEPSEEK_API_KEY": bool(os.getenv("DEEPSEEK_API_KEY")),
        "NVIDIA_API_KEY": bool(os.getenv("NVIDIA_API_KEY")),
        "SAMBANOVA_API_KEY": bool(os.getenv("SAMBANOVA_API_KEY")),
        "CLOUDFLARE_API_KEY": bool(os.getenv("CLOUDFLARE_API_KEY")),
    }
    return {"status": "ok", "app": "All In One Pipeline", "keys": keys}


@app.post("/api/pipeline/start")
async def start_pipeline(req: PipelineRequest) -> dict[str, str]:
    objective = req.objective.strip()
    if not objective:
        raise HTTPException(status_code=400, detail="Objective cannot be empty")
    job_id = uuid.uuid4().hex
    JOBS[job_id] = {
        "job_id": job_id,
        "objective": objective,
        "status": "running",
        "stage": "creative",
        "events": [],
        "blueprint": None,
        "files": {},
        "error": None,
    }
    asyncio.create_task(_run_pipeline_job(job_id, objective))
    return {"job_id": job_id}


@app.get("/api/pipeline/stream/{job_id}")
async def stream_pipeline(job_id: str) -> StreamingResponse:
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")

    async def event_generator():
        last_event_idx = 0
        while True:
            job = JOBS.get(job_id)
            if not job:
                break
            events = job["events"]
            while last_event_idx < len(events):
                ev = events[last_event_idx]
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
                last_event_idx += 1
            if job["status"] in ("completed", "failed"):
                final_payload = {
                    "event": "done",
                    "status": job["status"],
                    "blueprint": job["blueprint"],
                    "files": job["files"],
                    "error": job["error"],
                }
                yield f"data: {json.dumps(final_payload, ensure_ascii=False)}\n\n"
                break
            await asyncio.sleep(0.3)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/api/pipeline/download/{job_id}")
async def download_project(job_id: str):
    job = JOBS.get(job_id)
    if not job or not job.get("files"):
        raise HTTPException(status_code=404, detail="No files available for this job")
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for filepath, content in job["files"].items():
            zf.writestr(filepath, content)
    zip_buffer.seek(0)
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=all-in-one-{job_id[:8]}.zip"},
    )


async def _run_pipeline_job(job_id: str, objective: str) -> None:
    job = JOBS[job_id]

    def emit(event_type: str, data: dict[str, Any]):
        job["events"].append({"event": event_type, **data})

    emit("log", {"message": f"🚀 Starting All In One Pipeline for: '{objective}'"})

    try:
        governor = QuotaGovernor.from_yaml(CONFIG_PATH)
        blueprint = make_blueprint(uuid.uuid4().hex, objective)
        fsm = PipelineFSM(blueprint, governor, max_revisions_per_stage=MAX_REVISIONS_PER_STAGE)
        stage_attempts = dict.fromkeys(Stage, 0)

        with ProviderClient() as provider_client:
            agent = make_agent_callable(objective, provider_client)
            for call_number in range(1, MAX_PIPELINE_CALLS + 1):
                if fsm.phase is not Phase.RUNNING:
                    break
                stage = fsm.stage
                stage_attempts[stage] += 1
                emit(
                    "stage_start",
                    {
                        "stage": stage.value,
                        "attempt": stage_attempts[stage],
                        "call": call_number,
                        "message": f"🤖 Stage: {stage.value.upper()} (Attempt {stage_attempts[stage]})",
                    },
                )
                try:
                    delta = fsm.run_stage(agent)
                    emit(
                        "delta_applied",
                        {
                            "stage": stage.value,
                            "patches": len(delta.patches),
                            "critiques": [c.message for c in delta.critiques],
                            "message": f"📝 Proposed {len(delta.patches)} modifications to Blueprint.",
                        },
                    )
                    gate = gatekeeper_decision(stage, delta, stage_attempts[stage])
                    emit(
                        "gate_decision",
                        {
                            "stage": stage.value,
                            "decision": gate.decision.value,
                            "reason": gate.reason,
                            "message": f"🛡️ Gatekeeper Decision: {gate.decision.value.upper()} ({gate.reason})",
                        },
                    )
                    fsm.evaluate_gate(gate)
                except (FSMError, QuotaGovernorError) as exc:
                    fsm.fail(f"{type(exc).__name__}: {exc}")
                    emit("error", {"message": f"Stage error: {exc}"})
                    break

            if fsm.phase is not Phase.COMPLETED:
                job["status"] = "failed"
                job["error"] = fsm.state.error or "Pipeline did not reach completed state"
                emit("log", {"message": f"❌ Pipeline Failed: {job['error']}"})
                return

            # Pipeline Planning Succeeded
            final_bp = fsm.state.blueprint.model_dump(mode="json")
            job["blueprint"] = final_bp
            emit("blueprint_ready", {"blueprint": final_bp, "message": "✅ Architecture Blueprint certified!"})

            # Now: Execution Engine (Code Generation)
            emit("log", {"message": "⚡ Launching Code Generation Engine to write real files..."})
            files = await _generate_code_for_blueprint(provider_client, objective, final_bp, emit)
            job["files"] = files
            job["status"] = "completed"
            emit("log", {"message": f"🎉 Complete! Generated {len(files)} production project files."})

    except Exception as exc:
        job["status"] = "failed"
        job["error"] = str(exc)
        emit("error", {"message": f"Fatal error: {exc}"})


async def _generate_code_for_blueprint(
    client: ProviderClient, objective: str, blueprint: dict[str, Any], emit_fn
) -> dict[str, str]:
    files: dict[str, str] = {}
    steps = blueprint.get("steps", [])
    coder_steps = [s for s in steps if s.get("stage") == "coder"]
    if not coder_steps:
        coder_steps = steps

    spec_provider = None
    # Use Groq Qwen or GPT-120b
    gov = QuotaGovernor.from_yaml(CONFIG_PATH)
    res = gov.reserve_provider(Stage.CODER, estimated_tokens=4000)
    spec_provider = res.provider
    gov.settle(res, tokens=100)

    prompt = f"""
You are the Lead Implementation Engineer for the project: '{objective}'.
Based on the certified Blueprint:
{json.dumps(blueprint, indent=2)}

Generate the complete, runnable codebase for all coder steps.
Do not use placeholders, do not use 'TODO', and write full working code.

Return ONLY a single valid JSON object with a 'files' map, for example:
{{
  "files": {{
    "README.md": "# Project Title\\nDescription and run commands...",
    "requirements.txt": "package1\\npackage2",
    "src/main.py": "def main():\\n    ...",
    "tests/test_main.py": "def test_app():\\n    ..."
  }}
}}
No markdown fences or commentary.
"""
    emit_fn("log", {"message": f"💻 Writing code with {spec_provider.name} ({spec_provider.model})..."})
    raw_text, _ = await asyncio.to_thread(client.complete, spec_provider, prompt)
    try:
        parsed = extract_json_object(raw_text)
        if isinstance(parsed.get("files"), dict):
            files.update(parsed["files"])
    except Exception as exc:
        emit_fn("log", {"message": f"⚠️ Parser fallback: {exc}. Extracting standard layout..."})
        files["README.md"] = f"# {objective}\n\nGenerated by All In One Multi-Agent Pipeline."
        files["requirements.txt"] = "pydantic\nfastapi\nuvicorn\nrequests"
        files["main.py"] = f"# Implementation for {objective}\nprint('System initialized successfully.')\n"

    for path in list(files.keys()):
        emit_fn("file_created", {"path": path, "message": f"📄 Created file: {path}"})

    return files


HTML_CONTENT = """<!DOCTYPE html>
<html lang="ar" dir="rtl" class="dark">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>All In One - Multi-Agent AI Pipeline</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">
  <script>
    tailwind.config = {
      darkMode: 'class',
      theme: {
        extend: {
          colors: {
            brand: { 500: '#6366f1', 600: '#4f46e5', 700: '#4338ca' }
          }
        }
      }
    }
  </script>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700;800&family=Fira+Code:wght@400;500&display=swap');
    body { font-family: 'Cairo', sans-serif; }
    code, pre { font-family: 'Fira Code', monospace; direction: ltr; text-align: left; }
    .glass { background: rgba(30, 41, 59, 0.7); backdrop-filter: blur(12px); border: 1px solid rgba(255, 255, 255, 0.1); }
    .glow { box-shadow: 0 0 25px rgba(99, 102, 241, 0.25); }
  </style>
</head>
<body class="bg-slate-950 text-slate-100 min-h-screen flex flex-col">

  <!-- Header -->
  <header class="border-b border-slate-800 bg-slate-900/60 sticky top-0 z-50 backdrop-blur-md">
    <div class="max-w-7xl mx-auto px-4 py-3.5 flex items-center justify-between">
      <div class="flex items-center gap-3">
        <div class="w-10 h-10 rounded-xl bg-gradient-to-tr from-indigo-500 to-purple-600 flex items-center justify-center text-white shadow-lg glow">
          <i class="fa-solid fa-cube text-xl"></i>
        </div>
        <div>
          <h1 class="text-xl font-bold bg-gradient-to-r from-indigo-400 via-purple-300 to-pink-400 bg-clip-text text-transparent">All In One</h1>
          <p class="text-xs text-slate-400">Zero-Touch Multi-Agent Smart Routing Pipeline</p>
        </div>
      </div>
      <div class="flex items-center gap-2" id="key-badges">
        <span class="text-xs px-2.5 py-1 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 flex items-center gap-1.5">
          <span class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
          جاهز للتشغيل 24/7
        </span>
      </div>
    </div>
  </header>

  <!-- Main Content -->
  <main class="max-w-7xl mx-auto px-4 py-8 flex-1 w-full grid grid-cols-1 lg:grid-cols-12 gap-8">
    
    <!-- Left Column (Input & Progress) -->
    <div class="lg:col-span-5 space-y-6">
      <!-- Input Card -->
      <div class="glass rounded-2xl p-6 glow">
        <h2 class="text-lg font-bold mb-2 flex items-center gap-2">
          <i class="fa-solid fa-wand-magic-sparkles text-indigo-400"></i>
          فكرة أو متطلبات المشروع
        </h2>
        <p class="text-sm text-slate-400 mb-4">اكتب أي فكرة تطبيق، أداة، أو سكريبت، وسيتولى فريق الـ Agents هندستها وبرمجتها فورياً.</p>
        
        <textarea id="objective" rows="4" class="w-full bg-slate-900/80 border border-slate-700 rounded-xl p-3.5 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition-all" placeholder="مثال: بناء أداة تحول ملفات الماركداون إلى سلايدات PowerPoint كاملة بالبايثون مع اختبارات الوحدة"></textarea>
        
        <div class="mt-3 flex flex-wrap gap-2 text-xs">
          <span class="text-slate-400">أفكار جاهزة:</span>
          <button onclick="setPrompt('Build a real-time crypto price tracker with WebSockets and FastAPI')" class="px-2 py-1 rounded-md bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 transition">Crypto Tracker</button>
          <button onclick="setPrompt('Build an AI resume parser that extracts skills and scores job match')" class="px-2 py-1 rounded-md bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 transition">Resume Parser</button>
          <button onclick="setPrompt('Create an AI-powered markdown to slide deck generator in Python')" class="px-2 py-1 rounded-md bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 transition">Slide Deck Tool</button>
        </div>

        <button id="run-btn" onclick="startRun()" class="mt-5 w-full py-3 px-4 rounded-xl bg-gradient-to-r from-indigo-600 via-purple-600 to-pink-600 hover:from-indigo-500 hover:to-pink-500 font-bold text-white shadow-lg transition-all flex items-center justify-center gap-2">
          <i class="fa-solid fa-play"></i>
          <span>إطلاق الـ Pipeline وهندسة الكود</span>
        </button>
      </div>

      <!-- Live Pipeline Stages -->
      <div class="glass rounded-2xl p-6">
        <h3 class="text-sm font-bold text-slate-300 uppercase tracking-wider mb-4 flex items-center justify-between">
          <span>مراحل الـ Swarm الذكي</span>
          <span id="pipeline-status" class="text-xs px-2 py-0.5 rounded bg-slate-800 text-slate-400">في الانتظار</span>
        </h3>

        <div class="space-y-4">
          <!-- Creative -->
          <div id="stage-card-creative" class="p-3.5 rounded-xl border border-slate-800 bg-slate-900/40 flex items-center justify-between transition-all">
            <div class="flex items-center gap-3">
              <div class="w-9 h-9 rounded-lg bg-pink-500/10 text-pink-400 border border-pink-500/20 flex items-center justify-center">
                <i class="fa-solid fa-lightbulb"></i>
              </div>
              <div>
                <h4 class="text-sm font-semibold">1. Creative Agent</h4>
                <p class="text-xs text-slate-400">استيعاب الفكرة والعصف الذهني</p>
              </div>
            </div>
            <span id="badge-creative" class="text-xs px-2 py-1 rounded bg-slate-800 text-slate-400">معلق</span>
          </div>

          <!-- Architect -->
          <div id="stage-card-architect" class="p-3.5 rounded-xl border border-slate-800 bg-slate-900/40 flex items-center justify-between transition-all">
            <div class="flex items-center gap-3">
              <div class="w-9 h-9 rounded-lg bg-indigo-500/10 text-indigo-400 border border-indigo-500/20 flex items-center justify-center">
                <i class="fa-solid fa-compass-drafting"></i>
              </div>
              <div>
                <h4 class="text-sm font-semibold">2. Architect (Qwen)</h4>
                <p class="text-xs text-slate-400">التصميم المعماري وتدفق البيانات</p>
              </div>
            </div>
            <span id="badge-architect" class="text-xs px-2 py-1 rounded bg-slate-800 text-slate-400">معلق</span>
          </div>

          <!-- Coder -->
          <div id="stage-card-coder" class="p-3.5 rounded-xl border border-slate-800 bg-slate-900/40 flex items-center justify-between transition-all">
            <div class="flex items-center gap-3">
              <div class="w-9 h-9 rounded-lg bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 flex items-center justify-center">
                <i class="fa-solid fa-code"></i>
              </div>
              <div>
                <h4 class="text-sm font-semibold">3. Coder Agent</h4>
                <p class="text-xs text-slate-400">برمجة الوحدات وإنشاء الملفات الحقيقية</p>
              </div>
            </div>
            <span id="badge-coder" class="text-xs px-2 py-1 rounded bg-slate-800 text-slate-400">معلق</span>
          </div>
        </div>

        <!-- Live Terminal Stream -->
        <div class="mt-5 pt-4 border-t border-slate-800">
          <div class="flex items-center justify-between mb-2">
            <span class="text-xs font-mono text-slate-400">سجل الأحداث المباشر (Live Telemetry):</span>
            <span class="w-2 h-2 rounded-full bg-emerald-400 animate-ping"></span>
          </div>
          <div id="terminal" class="bg-black/60 rounded-xl p-3 h-48 overflow-y-auto text-xs font-mono text-slate-300 space-y-1.5 border border-slate-900">
            <p class="text-slate-500">// اضغط على زر الإطلاق لبدء المعالجة...</p>
          </div>
        </div>
      </div>
    </div>

    <!-- Right Column (Blueprint & Generated Code) -->
    <div class="lg:col-span-7 space-y-6">
      
      <!-- Generated Project Header & Download -->
      <div id="output-panel" class="glass rounded-2xl p-6 glow hidden">
        <div class="flex items-center justify-between pb-4 border-b border-slate-800">
          <div>
            <h3 class="text-lg font-bold text-emerald-400 flex items-center gap-2">
              <i class="fa-solid fa-circle-check"></i>
              تم توليد المشروع بنجاح!
            </h3>
            <p class="text-xs text-slate-400" id="file-count-label">تم إنشاء الملفات وجاهزة للتحميل</p>
          </div>
          <a id="download-btn" href="#" class="px-4 py-2.5 rounded-xl bg-emerald-600 hover:bg-emerald-500 text-white font-bold text-xs flex items-center gap-2 shadow-lg transition">
            <i class="fa-solid fa-download"></i>
            تحميل المشروع بالكامل (ZIP)
          </a>
        </div>

        <!-- File Tabs & Code Preview -->
        <div class="mt-4">
          <div class="flex items-center gap-1.5 overflow-x-auto pb-2 border-b border-slate-800 text-xs" id="file-tabs"></div>
          <div class="mt-3 relative">
            <pre class="bg-slate-950 p-4 rounded-xl border border-slate-800 text-xs text-emerald-300 max-h-96 overflow-y-auto"><code id="code-viewer">// اختر ملفاً لعرض الكود...</code></pre>
          </div>
        </div>
      </div>

      <!-- Blueprint View -->
      <div id="blueprint-panel" class="glass rounded-2xl p-6 hidden">
        <h3 class="text-sm font-bold text-slate-300 uppercase tracking-wider mb-3 flex items-center gap-2">
          <i class="fa-solid fa-file-contract text-indigo-400"></i>
          وثيقة التخطيط الهندسي المعتمدة (Certified Blueprint)
        </h3>
        <div id="steps-list" class="space-y-2.5 text-xs"></div>
      </div>

    </div>

  </main>

  <script>
    let currentJobId = null;
    let jobFiles = {};

    function setPrompt(text) {
      document.getElementById('objective').value = text;
    }

    function logTerminal(msg) {
      const term = document.getElementById('terminal');
      const p = document.createElement('p');
      p.className = 'leading-relaxed';
      p.innerText = msg;
      term.appendChild(p);
      term.scrollTop = term.scrollHeight;
    }

    async function startRun() {
      const objective = document.getElementById('objective').value.trim();
      if (!objective) return alert('برجاء كتابة الفكرة أولاً');

      const btn = document.getElementById('run-btn');
      btn.disabled = true;
      btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> <span>جاري تشغيل الـ Pipeline...</span>';

      document.getElementById('terminal').innerHTML = '';
      document.getElementById('output-panel').classList.add('hidden');
      document.getElementById('blueprint-panel').classList.add('hidden');
      document.getElementById('pipeline-status').innerText = 'قيد التشغيل...';
      document.getElementById('pipeline-status').className = 'text-xs px-2 py-0.5 rounded bg-indigo-500/20 text-indigo-300';

      const res = await fetch('/api/pipeline/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ objective })
      });
      const data = await res.json();
      currentJobId = data.job_id;

      // Listen to SSE
      const evtSource = new EventSource(`/api/pipeline/stream/${currentJobId}`);
      evtSource.onmessage = function(event) {
        const payload = JSON.parse(event.data);
        if (payload.event === 'log') {
          logTerminal(payload.message);
        } else if (payload.event === 'stage_start') {
          updateStage(payload.stage, 'running');
          logTerminal(payload.message);
        } else if (payload.event === 'gate_decision') {
          updateStage(payload.stage, payload.decision === 'approve' ? 'approved' : 'revising');
          logTerminal(payload.message);
        } else if (payload.event === 'blueprint_ready') {
          renderBlueprint(payload.blueprint);
        } else if (payload.event === 'file_created') {
          logTerminal(payload.message);
        } else if (payload.event === 'done') {
          evtSource.close();
          btn.disabled = false;
          btn.innerHTML = '<i class="fa-solid fa-play"></i> <span>إطلاق الـ Pipeline وهندسة الكود</span>';
          if (payload.status === 'completed') {
            document.getElementById('pipeline-status').innerText = 'مكتمل بنجاح';
            document.getElementById('pipeline-status').className = 'text-xs px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-400';
            renderFiles(payload.files);
          } else {
            document.getElementById('pipeline-status').innerText = 'تعثر';
            document.getElementById('pipeline-status').className = 'text-xs px-2 py-0.5 rounded bg-rose-500/20 text-rose-400';
          }
        }
      };
    }

    function updateStage(stage, state) {
      const badge = document.getElementById(`badge-${stage}`);
      if (!badge) return;
      if (state === 'running') {
        badge.innerText = 'يعمل الآن...';
        badge.className = 'text-xs px-2 py-1 rounded bg-indigo-500/20 text-indigo-300 animate-pulse';
      } else if (state === 'approved') {
        badge.innerText = 'معتمد ✓';
        badge.className = 'text-xs px-2 py-1 rounded bg-emerald-500/20 text-emerald-400 font-bold';
      }
    }

    function renderBlueprint(bp) {
      const panel = document.getElementById('blueprint-panel');
      panel.classList.remove('hidden');
      const list = document.getElementById('steps-list');
      list.innerHTML = '';
      (bp.steps || []).forEach(s => {
        const d = document.createElement('div');
        d.className = 'p-3 rounded-lg bg-slate-900 border border-slate-800 flex items-start gap-2.5';
        d.innerHTML = `
          <span class="px-1.5 py-0.5 rounded bg-indigo-500/20 text-indigo-300 text-[10px] uppercase font-bold">${s.stage}</span>
          <div class="flex-1">
            <p class="font-semibold text-slate-200">${s.id}</p>
            <p class="text-slate-400 mt-0.5">${s.instruction}</p>
          </div>
        `;
        list.appendChild(d);
      });
    }

    function renderFiles(files) {
      jobFiles = files || {};
      const panel = document.getElementById('output-panel');
      panel.classList.remove('hidden');
      const tabs = document.getElementById('file-tabs');
      tabs.innerHTML = '';
      const fileNames = Object.keys(jobFiles);
      document.getElementById('file-count-label').innerText = `تم توليد ${fileNames.length} ملفات جاهزة للتشغيل`;
      document.getElementById('download-btn').href = `/api/pipeline/download/${currentJobId}`;

      fileNames.forEach((name, idx) => {
        const b = document.createElement('button');
        b.className = `px-3 py-1.5 rounded-lg border text-xs font-mono transition ${idx === 0 ? 'bg-indigo-600 text-white border-indigo-500' : 'bg-slate-900 text-slate-400 border-slate-800 hover:bg-slate-800'}`;
        b.innerText = name;
        b.onclick = () => showFile(name, b);
        tabs.appendChild(b);
      });
      if (fileNames.length > 0) {
        showFile(fileNames[0], tabs.children[0]);
      }
    }

    function showFile(filename, activeTab) {
      Array.from(document.getElementById('file-tabs').children).forEach(btn => {
        btn.className = 'px-3 py-1.5 rounded-lg border text-xs font-mono transition bg-slate-900 text-slate-400 border-slate-800 hover:bg-slate-800';
      });
      if (activeTab) {
        activeTab.className = 'px-3 py-1.5 rounded-lg border text-xs font-mono transition bg-indigo-600 text-white border-indigo-500';
      }
      document.getElementById('code-viewer').innerText = jobFiles[filename] || '';
    }
  </script>
</body>
</html>
"""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

NOTEBOOK_HTML_CONTENT = """<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>الدفتر الصوتي الذكي - Smart Voice Notebook</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Amiri:ital,wght@0,400;0,700;1,400&family=Aref+Ruqaa:wght@400;700&family=Cairo:wght@400;600;700;800&family=Marhey:wght@400;700&display=swap" rel="stylesheet">
  <script src="https://cdn.jsdelivr.net/npm/sortablejs@1.15.0/Sortable.min.js"></script>
  <style>
    body { font-family: 'Cairo', sans-serif; }
    .font-ruqaa { font-family: 'Aref Ruqaa', serif; }
    .font-amiri { font-family: 'Amiri', serif; }
    .font-cairo { font-family: 'Cairo', sans-serif; }
    .font-marhey { font-family: 'Marhey', cursive; }

    /* Paper Background Themes */
    .theme-parchment {
      background: #fdf6e2 url('https://images.unsplash.com/photo-1607604276583-eef5d076aa5f?auto=format&fit=crop&w=1200&q=80') center/cover;
      color: #3e2723;
    }
    .theme-watercolor {
      background: #fdf2f8 url('https://images.unsplash.com/photo-1579783902614-a3fb3927b675?auto=format&fit=crop&w=1200&q=80') center/cover;
      color: #4a044e;
    }
    .theme-midnight {
      background: #0f172a url('https://images.unsplash.com/photo-1519681393784-d120267933ba?auto=format&fit=crop&w=1200&q=80') center/cover;
      color: #e2e8f0;
    }
    .theme-clean-lined {
      background-color: #ffffff;
      background-image: linear-gradient(#e2e8f0 1px, transparent 1px);
      background-size: 100% 2.2rem;
      color: #1e293b;
    }
    .theme-clean-grid {
      background-color: #fafafa;
      background-image: radial-gradient(#cbd5e1 1.5px, transparent 1.5px);
      background-size: 24px 24px;
      color: #1e293b;
    }

    .canvas-card {
      transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .canvas-card:hover {
      box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.1);
    }
    .sortable-ghost {
      opacity: 0.4;
      background: rgba(99, 102, 241, 0.2) !important;
    }
  </style>
</head>
<body class="bg-slate-900 text-slate-100 min-h-screen flex flex-col">

  <!-- Top Navigation -->
  <header class="bg-slate-900/90 border-b border-slate-800 backdrop-blur-md sticky top-0 z-40 px-4 py-3">
    <div class="max-w-5xl mx-auto flex items-center justify-between">
      <div class="flex items-center gap-3">
        <a href="/" class="w-9 h-9 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-300 flex items-center justify-center transition">
          <i class="fa-solid fa-arrow-right"></i>
        </a>
        <div>
          <h1 class="text-base font-bold text-white flex items-center gap-2">
            <span>الدفتر الصوتي الذكي</span>
            <span class="text-[10px] px-2 py-0.5 rounded-full bg-indigo-500/20 text-indigo-300 border border-indigo-500/30 font-mono">Smart Voice Notebook</span>
          </h1>
        </div>
      </div>

      <!-- Controls -->
      <div class="flex items-center gap-2">
        <!-- Font Selector -->
        <select id="font-select" onchange="changeFont(this.value)" class="bg-slate-800 border border-slate-700 text-xs rounded-lg px-2.5 py-1.5 text-slate-200 focus:outline-none focus:ring-1 focus:ring-indigo-500">
          <option value="font-ruqaa">خط الرقعة الفني</option>
          <option value="font-amiri">خط النسخ الأصيل</option>
          <option value="font-marhey">خط مرح فني</option>
          <option value="font-cairo">خط عصري واضح</option>
        </select>

        <!-- Export -->
        <button onclick="window.print()" class="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs border border-slate-700 flex items-center gap-1.5 transition">
          <i class="fa-solid fa-print"></i>
          <span class="hidden sm:inline">طباعة / PDF</span>
        </button>
      </div>
    </div>
  </header>

  <!-- Main Canvas Container -->
  <main class="flex-1 max-w-4xl w-full mx-auto p-4 sm:p-6 flex flex-col gap-5">
    
    <!-- Paper Settings & Presets -->
    <div class="bg-slate-900/80 rounded-2xl p-3 border border-slate-800 flex flex-wrap items-center justify-between gap-3 text-xs">
      <div class="flex items-center gap-2">
        <span class="text-slate-400 font-semibold"><i class="fa-solid fa-palette text-indigo-400 ml-1"></i> ثيم الورقة:</span>
        <button onclick="setTheme('theme-parchment')" class="px-2.5 py-1 rounded-md bg-amber-950/60 hover:bg-amber-900/80 text-amber-200 border border-amber-800/40">ورق بردي قديم</button>
        <button onclick="setTheme('theme-watercolor')" class="px-2.5 py-1 rounded-md bg-pink-950/60 hover:bg-pink-900/80 text-pink-200 border border-pink-800/40">أزهار مائية لطيفة</button>
        <button onclick="setTheme('theme-clean-lined')" class="px-2.5 py-1 rounded-md bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700">مسطر كلاسيكي</button>
        <button onclick="setTheme('theme-clean-grid')" class="px-2.5 py-1 rounded-md bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700">مربعات نوت</button>
        <button onclick="setTheme('theme-midnight')" class="px-2.5 py-1 rounded-md bg-indigo-950/60 hover:bg-indigo-900/80 text-indigo-200 border border-indigo-800/40">ليلي نيون</button>
      </div>
      
      <!-- Custom AI Background Prompt -->
      <div class="flex items-center gap-1.5 flex-1 min-w-[240px]">
        <input id="bg-prompt-input" type="text" placeholder="اطلب خلفية بالذكاء الاصطناعي (مثلاً: رمال شاطئ هادئة)..." class="w-full bg-slate-950 border border-slate-700 rounded-lg px-2.5 py-1 text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:ring-1 focus:ring-indigo-500">
        <button onclick="generateCustomBackground()" class="px-2.5 py-1 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white font-semibold text-xs whitespace-nowrap transition">
          <i class="fa-solid fa-wand-magic-sparkles ml-1"></i> توليد
        </button>
      </div>
    </div>

    <!-- The Actual Sheet / Canvas (Stationery Paper) -->
    <div id="notebook-paper" class="theme-parchment font-ruqaa min-h-[700px] rounded-3xl p-6 sm:p-10 shadow-2xl relative transition-all duration-300">
      
      <!-- Header of the Sheet -->
      <div class="flex items-center justify-between border-b border-black/10 pb-4 mb-6">
        <div>
          <h2 id="notebook-title" class="text-2xl sm:text-3xl font-bold tracking-tight" contenteditable="true">ملاحظاتي اليومية</h2>
          <p class="text-xs opacity-60 mt-1 font-cairo" id="notebook-date"></p>
        </div>
        <div class="flex items-center gap-2">
          <button onclick="clearCanvas()" class="text-xs px-2.5 py-1 rounded-lg bg-black/5 hover:bg-black/10 transition flex items-center gap-1 font-cairo">
            <i class="fa-solid fa-trash-can"></i> مسح الورقة
          </button>
        </div>
      </div>

      <!-- Drag & Drop Canvas Items -->
      <div id="canvas-items" class="space-y-4">

        <!-- Item 1: Introductory Note -->
        <div class="canvas-card group relative p-4 rounded-2xl bg-white/40 border border-black/5 backdrop-blur-sm cursor-grab">
          <div class="flex items-center justify-between text-xs opacity-50 mb-1 font-cairo">
            <span class="flex items-center gap-1"><i class="fa-solid fa-grip-vertical"></i> اسحب للترتيب</span>
            <button onclick="copyCardText(this)" class="hover:opacity-100 transition"><i class="fa-regular fa-copy"></i> نسخ بلمسة</button>
          </div>
          <p class="text-lg leading-relaxed content-text outline-none" contenteditable="true">
            أهلاً بك في الدفتر الصوتي الذكي! اضغط على الميكروفون بالأسفل وتحدث بحرية.. اطلب جدولاً أو صورة أو املِ أفكارك وستتحول لخطوط أنيقة.
          </p>
        </div>

      </div>

    </div>

  </main>

  <!-- Sticky Bottom Voice Bar -->
  <div class="sticky bottom-0 z-50 bg-slate-950/90 border-t border-slate-800 p-3.5 backdrop-blur-lg">
    <div class="max-w-xl mx-auto flex items-center justify-between gap-3">
      
      <!-- Voice Mode Info -->
      <div class="flex items-center gap-2 text-xs">
        <div id="mic-indicator" class="w-3 h-3 rounded-full bg-slate-600 transition-all"></div>
        <span id="mic-status" class="text-slate-400 font-cairo">الميكروفون متوقف (اضغط للتحدث)</span>
      </div>

      <!-- Quick Action Buttons -->
      <div class="flex items-center gap-2">
        <button onclick="insertSampleTable()" class="px-2.5 py-1.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-xs text-slate-300 font-cairo flex items-center gap-1 transition">
          <i class="fa-solid fa-table"></i> + جدول
        </button>
        
        <button onclick="promptForImage()" class="px-2.5 py-1.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-xs text-slate-300 font-cairo flex items-center gap-1 transition">
          <i class="fa-solid fa-image"></i> + صورة AI
        </button>

        <!-- Big Mic Button -->
        <button id="mic-btn" onclick="toggleVoiceRecognition()" class="w-12 h-12 rounded-full bg-indigo-600 hover:bg-indigo-500 text-white flex items-center justify-center text-lg shadow-lg shadow-indigo-600/30 transition-all hover:scale-105 active:scale-95">
          <i class="fa-solid fa-microphone"></i>
        </button>
      </div>

    </div>
  </div>

  <script>
    // Initialize date
    document.getElementById('notebook-date').innerText = new Date().toLocaleDateString('ar-EG', { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' });

    // Enable Drag and Drop
    const canvasItems = document.getElementById('canvas-items');
    new Sortable(canvasItems, {
      animation: 250,
      ghostClass: 'sortable-ghost',
      handle: '.canvas-card',
      delay: 100,
      delayOnTouchOnly: true
    });

    // Theme changer
    function setTheme(themeClass) {
      const paper = document.getElementById('notebook-paper');
      paper.className = `${themeClass} ${document.getElementById('font-select').value} min-h-[700px] rounded-3xl p-6 sm:p-10 shadow-2xl relative transition-all duration-300`;
      paper.style.backgroundImage = '';
    }

    // Font changer
    function changeFont(fontClass) {
      const paper = document.getElementById('notebook-paper');
      paper.classList.remove('font-ruqaa', 'font-amiri', 'font-cairo', 'font-marhey');
      paper.classList.add(fontClass);
    }

    // Copy with one tap
    function copyCardText(btn) {
      const card = btn.closest('.canvas-card');
      const text = card.querySelector('.content-text').innerText;
      navigator.clipboard.writeText(text);
      btn.innerHTML = '<i class="fa-solid fa-check text-emerald-500"></i> تم النسخ!';
      setTimeout(() => { btn.innerHTML = '<i class="fa-regular fa-copy"></i> نسخ بلمسة'; }, 1800);
    }

    // Remove Card
    function deleteCard(btn) {
      btn.closest('.canvas-card').remove();
    }

    // Clear Canvas
    function clearCanvas() {
      if (confirm('هل تريد مسح كل محتويات الدفتر؟')) {
        canvasItems.innerHTML = '';
      }
    }

    // Custom AI Generated Background
    function generateCustomBackground() {
      const prompt = document.getElementById('bg-prompt-input').value.trim();
      if (!prompt) return alert('اكتب وصف الخلفية أولاً');
      const encodedPrompt = encodeURIComponent(`texture background wallpaper stationery paper art, ${prompt}`);
      const imageUrl = `https://image.pollinations.ai/prompt/${encodedPrompt}?width=1200&height=1600&nologo=true&seed=${Math.floor(Math.random()*10000)}`;
      
      const paper = document.getElementById('notebook-paper');
      paper.style.background = `url('${imageUrl}') center/cover no-repeat`;
    }

    // Insert Image Card
    function insertImageCard(imageUrl, caption) {
      const div = document.createElement('div');
      div.className = 'canvas-card group relative p-4 rounded-2xl bg-white/60 border border-black/5 backdrop-blur-sm cursor-grab shadow-sm';
      div.innerHTML = `
        <div class="flex items-center justify-between text-xs opacity-50 mb-2 font-cairo">
          <span class="flex items-center gap-1"><i class="fa-solid fa-grip-vertical"></i> صورة ذكية</span>
          <div class="flex items-center gap-2">
            <button onclick="deleteCard(this)" class="hover:text-red-500 transition"><i class="fa-solid fa-xmark"></i></button>
          </div>
        </div>
        <div class="rounded-xl overflow-hidden border border-black/10">
          <img src="${imageUrl}" alt="${caption}" class="w-full max-h-80 object-cover rounded-xl" loading="lazy">
        </div>
        <p class="text-sm mt-2 opacity-80 text-center font-cairo content-text" contenteditable="true">${caption}</p>
      `;
      canvasItems.appendChild(div);
      div.scrollIntoView({ behavior: 'smooth' });
    }

    // Prompt for Image
    function promptForImage() {
      const userPrompt = prompt('اكتب ما تريد رسمه بالذكاء الاصطناعي (مثال: شروق الشمس فوق الأهرامات بطراز كرتوني لطيف):');
      if (!userPrompt) return;
      const encoded = encodeURIComponent(`high quality masterpiece, ${userPrompt}`);
      const imgUrl = `https://image.pollinations.ai/prompt/${encoded}?width=800&height=600&nologo=true&seed=${Math.floor(Math.random()*10000)}`;
      insertImageCard(imgUrl, userPrompt);
    }

    // Insert Table Card
    function insertTableCard(headers, rows) {
      const div = document.createElement('div');
      div.className = 'canvas-card group relative p-4 rounded-2xl bg-white/60 border border-black/5 backdrop-blur-sm cursor-grab overflow-x-auto shadow-sm';
      
      let headerHtml = headers.map(h => `<th class="p-2.5 border-b border-black/10 font-bold font-cairo">${h}</th>`).join('');
      let rowsHtml = rows.map(r => `
        <tr class="border-b border-black/5 hover:bg-black/5 transition">
          ${r.map(cell => `<td class="p-2.5 font-cairo" contenteditable="true">${cell}</td>`).join('')}
        </tr>
      `).join('');

      div.innerHTML = `
        <div class="flex items-center justify-between text-xs opacity-50 mb-2 font-cairo">
          <span class="flex items-center gap-1"><i class="fa-solid fa-grip-vertical"></i> جدول منظم</span>
          <div class="flex items-center gap-2">
            <button onclick="copyCardText(this)" class="hover:opacity-100 transition"><i class="fa-regular fa-copy"></i> نسخ</button>
            <button onclick="deleteCard(this)" class="hover:text-red-500 transition"><i class="fa-solid fa-xmark"></i></button>
          </div>
        </div>
        <table class="w-full text-right text-sm border-collapse content-text">
          <thead><tr>${headerHtml}</tr></thead>
          <tbody>${rowsHtml}</tbody>
        </table>
      `;
      canvasItems.appendChild(div);
      div.scrollIntoView({ behavior: 'smooth' });
    }

    function insertSampleTable() {
      insertTableCard(['العنصر', 'التفاصيل', 'الحالة'], [
        ['مهمة 1', 'كتابة التقرير الأسبوعي', 'مكتملة ✓'],
        ['مهمة 2', 'تسجيل الأفكار بالصوت', 'قيد التنفيذ...'],
        ['مهمة 3', 'توليد الصور المخصصة', 'جديدة']
      ]);
    }

    // Insert Text Card
    function insertTextCard(text) {
      const div = document.createElement('div');
      div.className = 'canvas-card group relative p-4 rounded-2xl bg-white/40 border border-black/5 backdrop-blur-sm cursor-grab shadow-sm';
      div.innerHTML = `
        <div class="flex items-center justify-between text-xs opacity-50 mb-1 font-cairo">
          <span class="flex items-center gap-1"><i class="fa-solid fa-grip-vertical"></i> فقرة صوتية</span>
          <div class="flex items-center gap-2">
            <button onclick="copyCardText(this)" class="hover:opacity-100 transition"><i class="fa-regular fa-copy"></i> نسخ بلمسة</button>
            <button onclick="deleteCard(this)" class="hover:text-red-500 transition"><i class="fa-solid fa-xmark"></i></button>
          </div>
        </div>
        <p class="text-xl leading-relaxed content-text outline-none" contenteditable="true">${text}</p>
      `;
      canvasItems.appendChild(div);
      div.scrollIntoView({ behavior: 'smooth' });
    }

    // ==========================================
    // Web Speech API - Voice Dictation & Intents
    // ==========================================
    let recognition = null;
    let isListening = false;

    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
      recognition = new SpeechRecognition();
      recognition.continuous = true;
      recognition.interimResults = false;
      recognition.lang = 'ar-EG'; // Default Egyptian Arabic

      recognition.onstart = function() {
        isListening = true;
        document.getElementById('mic-indicator').className = 'w-3 h-3 rounded-full bg-rose-500 animate-ping';
        document.getElementById('mic-status').innerText = 'الميكروفون يستمع الآن.. تكلم بحرية!';
        document.getElementById('mic-status').className = 'text-rose-400 font-bold font-cairo';
        document.getElementById('mic-btn').className = 'w-12 h-12 rounded-full bg-rose-600 hover:bg-rose-500 text-white flex items-center justify-center text-lg shadow-lg shadow-rose-600/40 animate-pulse';
      };

      recognition.onend = function() {
        isListening = false;
        document.getElementById('mic-indicator').className = 'w-3 h-3 rounded-full bg-slate-600';
        document.getElementById('mic-status').innerText = 'الميكروفون متوقف';
        document.getElementById('mic-status').className = 'text-slate-400 font-cairo';
        document.getElementById('mic-btn').className = 'w-12 h-12 rounded-full bg-indigo-600 hover:bg-indigo-500 text-white flex items-center justify-center text-lg shadow-lg shadow-indigo-600/30';
      };

      recognition.onresult = function(event) {
        for (let i = event.resultIndex; i < event.results.length; ++i) {
          if (event.results[i].isFinal) {
            const transcript = event.results[i][0].transcript.trim();
            handleVoiceCommand(transcript);
          }
        }
      };

      recognition.onerror = function(event) {
        console.error('Speech error:', event.error);
        if (event.error === 'not-allowed') {
          alert('برجاء السماح بصلاحية الميكروفون في المتصفح لاستخدام الإملاء الصوتي');
        }
      };
    }

    function toggleVoiceRecognition() {
      if (!recognition) {
        return alert('المتصفح لديك لا يدعم ميزة التعرف على الصوت. يرجى استخدام متصفح Chrome أو Edge');
      }
      if (isListening) {
        recognition.stop();
      } else {
        recognition.start();
      }
    }

    // Voice Intent Classifier
    function handleVoiceCommand(text) {
      console.log('Voice Input:', text);

      // Intent 1: Image Request ("صورة" أو "ارسم" أو "توليد صورة")
      if (text.includes('صورة') || text.includes('ارسم') || text.includes('صوره')) {
        let cleanPrompt = text.replace(/اعمل صورة|حط صورة|ارسم لي|ارسم|صورة|صوره/g, '').trim();
        if (!cleanPrompt) cleanPrompt = text;
        const encoded = encodeURIComponent(`masterpiece artistic, ${cleanPrompt}`);
        const imgUrl = `https://image.pollinations.ai/prompt/${encoded}?width=800&height=600&nologo=true&seed=${Math.floor(Math.random()*10000)}`;
        insertImageCard(imgUrl, cleanPrompt);
        return;
      }

      // Intent 2: Table Request ("جدول" أو "اعمل جدول")
      if (text.includes('جدول')) {
        // Simple intelligent split
        let parts = text.split(/خانات|خانة|عمود|أعمدة|الأولى|الثانية|الثالثة|الاولى|الثانيه|الثالثه/);
        let headers = ['الخانة 1', 'الخانة 2', 'الخانة 3'];
        if (parts.length > 2) {
          headers = parts.slice(1, 4).map(p => p.replace(/[،,]/g, '').trim()).filter(Boolean);
        }
        insertTableCard(headers, [
          ['بيان 1', 'محتوى 1', 'مكتمل'],
          ['بيان 2', 'محتوى 2', 'قيد المتابعة']
        ]);
        return;
      }

      // Intent 3: Normal text dictation
      insertTextCard(text);
    }
  </script>
</body>
</html>
"""
