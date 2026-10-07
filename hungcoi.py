# VHƯNG PRO MAX — LOCAL Z-IMAGE-TURBO / GEMINI STABLE BUILD V40
# V33: Gemini Directive UI source-of-truth fix — workflow_config.json wins over stale runtime settings.
# V12: SQLite SOT, durable sessions, bounded async executor; no live JSON mirror.
from __future__ import annotations
import warnings
import logging

# Ẩn cảnh báo nội bộ của google-genai về automatic function calling khỏi CMD.
warnings.filterwarnings("ignore", message=r".*Direct use of automatic function calling.*")
# google-genai emits the AFC message through its logger, not Python warnings.
# Keep SDK chatter out of CMD; real workflow errors are still captured by workflow_cmd_log().
for _logger_name in ("google_genai", "google.genai", "google.genai.models"):
    logging.getLogger(_logger_name).setLevel(logging.ERROR)

import base64
import atexit
import gc
import json
import os
import re
import shutil
import subprocess
import sys
import time
import threading
import asyncio
import inspect
import contextvars
import webbrowser
import tempfile
import html
import hashlib
import secrets
import uuid
import sqlite3
from collections import deque
from datetime import datetime, timezone, timedelta
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import gradio as gr
import requests

try:
    import psutil
except Exception:
    psutil = None

# Managed AI child-process cleanup. We deliberately do NOT put Chrome/CapCut/user apps
# in this job: only AI server processes launched by ServerManager belong here.
_MANAGED_CHILD_PIDS = set()
_SETTINGS_WRITE_LOCK = threading.RLock()
_GEMINI_KEY_CURSOR = threading.local()
_RUNTIME_PROJECT_LOCK = threading.RLock()
_RUNTIME_PROJECT_STATE = {}
# Heavy local GPU work is serialized explicitly. A second browser/tab receives a
# clear BUSY response instead of consuming another Gradio worker while waiting
# for the same 12 GB GPU/ComfyUI queue.
_COMFY_GENERATION_LOCK = threading.Lock()
_COMFY_ACTIVE_PROMPT_ID = None
_COMFY_ACTIVE_PROMPT_LOCK = threading.RLock()
_GRADIO_TEMP_LOCK = threading.RLock()
_GRADIO_TEMP_LAST_CLEAN = 0.0
_GRADIO_TEMP_CLEAN_INTERVAL = 600.0
_GRADIO_TEMP_MAX_AGE = 24 * 3600
_GRADIO_TEMP_MAX_BYTES = 2 * 1024 * 1024 * 1024
GEMINI_KEY_UX_BUILD = "GEMINI-KEY-UX-V4"

# Z-IMAGE-TURBO MAX-SPEED production policy:
# - one ComfyUI process / one browser tab on :8188
# - HighVRAM
# - preview OFF
# - model/CLIP/VAE cache HOT across the entire batch
# - NO gc/CUDA cache flush after each image
# - 1024x576 / 8 steps / CFG 1.0 by default

# Z-IMAGE SESSION HOT CACHE: bootstrap ComfyUI/model indexing exactly once per
# Master image batch. The generation loop owns this session; individual Scenes
# must never re-run the bootstrap path unless the session is explicitly reset.
_ZIMAGE_SESSION_LOCK = threading.RLock()
_ZIMAGE_SESSION_READY = False
_ZIMAGE_SESSION_PROJECT = ""

def _zimage_session_reset(reason=""):
    global _ZIMAGE_SESSION_READY, _ZIMAGE_SESSION_PROJECT
    with _ZIMAGE_SESSION_LOCK:
        _ZIMAGE_SESSION_READY = False
        _ZIMAGE_SESSION_PROJECT = ""
    if reason:
        workflow_cmd_log("Z-IMAGE-TURBO", "SESSION_RESET", reason=str(reason)[:500], level="DEBUG")

def _zimage_session_bootstrap(project, status_cb=None):
    """Bootstrap Z-Image/ComfyUI once for the whole Master image batch."""
    global _ZIMAGE_SESSION_READY, _ZIMAGE_SESSION_PROJECT
    project = safe_name(project or SETTINGS.get("selected_project", "Project_001"))
    with _ZIMAGE_SESSION_LOCK:
        if _ZIMAGE_SESSION_READY and _ZIMAGE_SESSION_PROJECT == project and _comfy_is_alive():
            workflow_cmd_log("Z-IMAGE-TURBO", "SESSION_HOT_REUSE", project=project, level="DEBUG")
            return True, "SESSION_HOT_REUSE"
        started = time.monotonic()
        workflow_cmd_log("Z-IMAGE-TURBO", "SESSION_BOOTSTRAP_START", project=project)
        ok, msg, root = _diffusion_bootstrap(status_cb=status_cb, project=project, load_pipeline=True)
        if not ok:
            _ZIMAGE_SESSION_READY = False
            _ZIMAGE_SESSION_PROJECT = ""
            return False, msg
        _ZIMAGE_SESSION_READY = True
        _ZIMAGE_SESSION_PROJECT = project
        workflow_cmd_log("Z-IMAGE-TURBO", "SESSION_HOT_READY", project=project, bootstrap_seconds=round(time.monotonic()-started, 3), root=str(root or ""))
        return True, msg


# ============================================================
# LocalAIStudio FULL V4
# Full workflow + isolated voice servers.
# - Main UI environment is separate from every voice engine.
# - Heavy engines run in their own processes/venvs.
# - App startup never installs or updates anything automatically.
# - Install/update/repair/model refresh happen only on explicit click.
# ============================================================

BASE = Path(__file__).resolve().parent
APP_DISPLAY_NAME = "HƯNG PRO WORKFLOW"
DATA = BASE / "data"
# Project Archive storage: keep archive metadata outside individual project folders.
PROJECT_ARCHIVE_DIR = BASE / "tool" / "ProjectArchive"
PROJECT_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

def project_archive_file(project_name: str) -> Path:
    return PROJECT_ARCHIVE_DIR / (safe_name(str(project_name or "project")) + ".json")
CONFIG = DATA / "config"
PROJECTS = DATA / "projects"
VOICE_LIBRARY = BASE / "voice"
SERVERS = DATA / "servers"
ENV_ROOT = BASE / "envs"
# Shared tool-owned model storage. Z-Image-Turbo NEVER lives under a Project/prj folder.
# Tool-owned model storage is explicit and only created when a model is actually
# downloaded/installed. Startup must NEVER create a stray `models\<engine>` tree.
MODEL_ROOT = BASE / "tool" / "models"
SETTINGS_FILE = CONFIG / "settings.json"
# Gemini credential persistence: same config directory, explicit and deterministic.
GEMINI_KEY_FILE = CONFIG / "gemini_api_key.txt"
APP_STATE_DIR = BASE / "LocalAIStudio_Data"
# Workflow configuration is deliberately separate from settings.json.
# It owns user workflow choices: Voice, Server/Engine, Voice ID, Gemini visual
# directive, Project binding and workflow-specific references.
WORKFLOW_CONFIG_FILE = APP_STATE_DIR / "workflow_config.json"
WORKFLOW_CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
# Workflow profiles are persisted in LocalAIStudio_Data/workflow_config.json.
WORKFLOW_PROFILES_FILE = WORKFLOW_CONFIG_FILE
ENGINE_STATE_FILE = SERVERS / "engine_state.json"



# ============================================================
# CREATIVE STUDIO CORE V1
# ============================================================
# Durable control-plane contracts for Job Queue / Scheduler /
# Worker Registry / Account Pool / Heartbeat / Recovery / Retry /
# CAPTCHA / QC / Engine Routing.
#
# IMPORTANT:
# - This layer never fakes Google Flow execution.
# - A real Playwright/Flow provider worker attaches to these contracts.
# - GOOGLE_FLOW is PRIMARY; Z_IMAGE is LOCAL ALTERNATIVE; FLUX is not registered.

CREATIVE_JOB_STAGES = ("IMAGE", "DEPTH", "MOTION", "POSTPROCESS", "QC", "TIMELINE")
CREATIVE_MAX_RETRY = 3
CREATIVE_HEARTBEAT_TIMEOUT = 60
CREATIVE_ENGINE_AUTO = "AUTO"
CREATIVE_ENGINE_FLOW = "GOOGLE_FLOW"
CREATIVE_ENGINE_ZIMAGE = "Z_IMAGE"

CREATIVE_ACCOUNT_STATES = (
    "ACTIVE", "BUSY", "RATE_LIMITED", "QUOTA", "CAPTCHA",
    "AUTH_ERROR", "DISABLED", "COOLDOWN",
)

def _creative_now():
    return _batch_now()

def _creative_json(value, default="[]"):
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return default

def _creative_load_json(value, default):
    try:
        return json.loads(value or "")
    except Exception:
        return default

def _creative_stage_dependencies(stage, media_type="image"):
    mt = str(media_type or "image").lower()
    if stage == "IMAGE":
        return []
    if stage == "DEPTH":
        return ["IMAGE"]
    if stage == "MOTION":
        return ["DEPTH"] if mt == "video" else ["IMAGE"]
    if stage == "POSTPROCESS":
        return ["MOTION"] if mt == "video" else ["IMAGE"]
    if stage == "QC":
        return ["POSTPROCESS"]
    if stage == "TIMELINE":
        return ["QC"]
    return []

def _creative_priority(kind="NORMAL"):
    return {
        "FAILED_RETRY": 100,
        "CAPTCHA_RESUME": 90,
        "NORMAL": 50,
        "BACKGROUND": 10,
    }.get(str(kind).upper(), 50)

def _creative_job_id(project, scene_id, stage):
    return f"{_batch_project_name(project)}::{scene_id}::{stage}"

def _creative_db_init():
    with _batch_db() as db:
        db.executescript("""
        CREATE TABLE IF NOT EXISTS creative_accounts (
            account_id TEXT PRIMARY KEY,
            provider TEXT NOT NULL DEFAULT 'GOOGLE_FLOW',
            status TEXT NOT NULL DEFAULT 'ACTIVE',
            session_profile TEXT,
            last_success TEXT,
            last_error TEXT,
            cooldown_until TEXT,
            usage INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS creative_workers (
            worker_id TEXT PRIMARY KEY,
            worker_type TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'IDLE',
            account_id TEXT,
            job_id TEXT,
            last_heartbeat TEXT,
            started_at TEXT,
            stopped_at TEXT,
            error TEXT,
            metadata_json TEXT DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS creative_sessions (
            session_id TEXT PRIMARY KEY,
            account_id TEXT NOT NULL,
            provider TEXT NOT NULL DEFAULT 'GOOGLE_FLOW',
            status TEXT NOT NULL DEFAULT 'NEW',
            profile_path TEXT,
            worker_id TEXT,
            browser_pid INTEGER,
            last_used TEXT,
            last_error TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS creative_jobs (
            job_id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL,
            scene_id TEXT NOT NULL,
            stage TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'PENDING',
            worker_id TEXT,
            account_id TEXT,
            priority INTEGER NOT NULL DEFAULT 50,
            retry_count INTEGER NOT NULL DEFAULT 0,
            max_retry INTEGER NOT NULL DEFAULT 3,
            ready_at TEXT,
            started_at TEXT,
            completed_at TEXT,
            last_heartbeat TEXT,
            error_class TEXT,
            error TEXT,
            output TEXT,
            requested_engine TEXT NOT NULL DEFAULT 'AUTO',
            actual_engine TEXT,
            dependency_json TEXT NOT NULL DEFAULT '[]',
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(project_id, scene_id, stage)
        );

        CREATE TABLE IF NOT EXISTS creative_captcha (
            captcha_id INTEGER PRIMARY KEY AUTOINCREMENT,
            account_id TEXT NOT NULL,
            job_id TEXT,
            scene_id TEXT,
            status TEXT NOT NULL DEFAULT 'OPEN',
            detected_at TEXT NOT NULL,
            resolved_at TEXT,
            note TEXT
        );

        CREATE TABLE IF NOT EXISTS creative_retry (
            retry_id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id TEXT NOT NULL,
            retry_no INTEGER NOT NULL,
            reason TEXT,
            error_class TEXT,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS creative_qc (
            qc_id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id TEXT NOT NULL,
            scene_id TEXT NOT NULL,
            stage TEXT NOT NULL,
            result TEXT NOT NULL,
            checks_json TEXT NOT NULL DEFAULT '{}',
            error TEXT,
            created_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_creative_jobs_ready
            ON creative_jobs(project_id,status,priority DESC,ready_at,created_at);
        CREATE INDEX IF NOT EXISTS idx_creative_jobs_heartbeat
            ON creative_jobs(status,last_heartbeat);
        CREATE INDEX IF NOT EXISTS idx_creative_workers_heartbeat
            ON creative_workers(status,last_heartbeat);
        CREATE INDEX IF NOT EXISTS idx_creative_accounts_pool
            ON creative_accounts(provider,status,cooldown_until);
        """)

def _creative_activate_verified_flow_account(account_id=None, profile_path=None, worker_id=None):
    """Mark a real, successfully verified Flow profile ACTIVE."""
    try:
        _creative_db_init()
        now = _creative_now()
        with _batch_db() as db:
            aid = str(account_id or "").strip()
            profile = str(profile_path or "").strip()
            if not aid and profile:
                row = db.execute(
                    "SELECT account_id FROM creative_accounts "
                    "WHERE session_profile=? LIMIT 1",
                    (profile,),
                ).fetchone()
                aid = str(row[0]) if row else ""
            if not aid and profile:
                aid = Path(profile).name
            if not aid:
                return False
            row = db.execute(
                "SELECT account_id FROM creative_accounts "
                "WHERE account_id=? OR session_profile=? LIMIT 1",
                (aid, profile),
            ).fetchone()
            if not row:
                return False
            aid = str(row[0])
            db.execute(
                "UPDATE creative_accounts SET "
                "status='ACTIVE', last_error=NULL, cooldown_until=NULL, "
                "last_success=?, updated_at=? WHERE account_id=?",
                (now, now, aid),
            )
            if worker_id:
                db.execute(
                    "UPDATE creative_workers SET account_id=?, "
                    "updated_at=? WHERE worker_id=?",
                    (aid, now, str(worker_id)),
                )
        workflow_cmd_log(
            "FLOW",
            "PROFILE_VERIFIED_ACTIVE",
            account_id=aid,
            profile=profile,
            worker_id=str(worker_id or ""),
        )
        return True
    except Exception as exc:
        try:
            workflow_cmd_log(
                "FLOW",
                "PROFILE_ACTIVE_UPDATE_FAILED",
                error=str(exc)[:500],
                account_id=str(account_id or ""),
            )
        except Exception:
            pass
        return False

def _creative_create_jobs(project, scenes, requested_engine=CREATIVE_ENGINE_AUTO):
    _creative_db_init()
    project = _batch_project_name(project)
    now = _creative_now()
    with _batch_db() as db:
        for scene in scenes or []:
            sid = str(scene.get("scene_id") or "").strip()
            if not sid:
                continue
            media_type = scene_media_type(scene, "image")
            for stage in CREATIVE_JOB_STAGES:
                if stage == "IMAGE" and media_type == "video":
                    status = "SKIPPED"
                elif stage == "TIMELINE" and media_type != "video":
                    status = "SKIPPED"
                else:
                    status = "PENDING"
                jid = _creative_job_id(project, sid, stage)
                deps = _creative_stage_dependencies(stage, media_type)
                db.execute(
                    "INSERT INTO creative_jobs("
                    "job_id,project_id,scene_id,stage,status,priority,max_retry,ready_at,"
                    "requested_engine,dependency_json,created_at,updated_at"
                    ") VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?) "
                    "ON CONFLICT(project_id,scene_id,stage) DO UPDATE SET "
                    "dependency_json=excluded.dependency_json,"
                    "requested_engine=excluded.requested_engine,"
                    "updated_at=excluded.updated_at",
                    (
                        jid, project, sid, stage, status,
                        _creative_priority("NORMAL"), CREATIVE_MAX_RETRY,
                        now, requested_engine, _creative_json(deps), now, now
                    ),
                )

def _creative_dependencies_ready(db, project, scene_id, stage):
    row = db.execute(
        "SELECT dependency_json FROM creative_jobs "
        "WHERE project_id=? AND scene_id=? AND stage=?",
        (project, scene_id, stage),
    ).fetchone()
    deps = _creative_load_json(row[0], []) if row else []
    if not deps:
        return True
    placeholders = ",".join("?" for _ in deps)
    rows = db.execute(
        f"SELECT stage,status FROM creative_jobs "
        f"WHERE project_id=? AND scene_id=? AND stage IN ({placeholders})",
        [project, scene_id, *deps],
    ).fetchall()
    state = {r[0]: r[1] for r in rows}
    return all(state.get(dep) in ("DONE", "SKIPPED") for dep in deps)

def _creative_scheduler_pick(project):
    _creative_db_init()
    project = _batch_project_name(project)
    now = _creative_now()
    with _batch_db() as db:
        rows = db.execute(
            "SELECT job_id,scene_id,stage,status,ready_at "
            "FROM creative_jobs WHERE project_id=? "
            "AND status IN ('PENDING','WAITING_DEPENDENCY','RETRY') "
            "ORDER BY priority DESC,COALESCE(ready_at,created_at),created_at LIMIT 500",
            (project,),
        ).fetchall()
        for row in rows:
            jid, sid, stage, status, ready_at = row
            if status == "RETRY" and ready_at and str(ready_at) > now:
                continue
            if _creative_dependencies_ready(db, project, sid, stage):
                db.execute(
                    "UPDATE creative_jobs SET status='READY',updated_at=? "
                    "WHERE job_id=? AND status IN ('PENDING','WAITING_DEPENDENCY','RETRY')",
                    (now, jid),
                )
            else:
                db.execute(
                    "UPDATE creative_jobs SET status='WAITING_DEPENDENCY',updated_at=? "
                    "WHERE job_id=?",
                    (now, jid),
                )
        row = db.execute(
            "SELECT job_id,scene_id,stage,requested_engine,priority,retry_count,max_retry "
            "FROM creative_jobs WHERE project_id=? AND status='READY' "
            "ORDER BY priority DESC,COALESCE(ready_at,created_at),created_at LIMIT 1",
            (project,),
        ).fetchone()
        return dict(row) if row else None

def _creative_claim_job(project, worker_id):
    return _creative_atomic_claim_job(project, worker_id)

def _creative_claim_account(worker_id, provider=CREATIVE_ENGINE_FLOW):
    _creative_db_init()
    now = _creative_now()
    with _batch_db() as db:
        row = db.execute(
            "SELECT account_id FROM creative_accounts "
            "WHERE provider=? AND status='ACTIVE' "
            "AND (cooldown_until IS NULL OR cooldown_until<=?) "
            "ORDER BY COALESCE(last_success,''),updated_at LIMIT 1",
            (provider, now),
        ).fetchone()
        if not row:
            return None
        account_id = row[0]
        cur = db.execute(
            "UPDATE creative_accounts SET status='BUSY',usage=usage+1,updated_at=? "
            "WHERE account_id=? AND status='ACTIVE'",
            (now, account_id),
        )
        if cur.rowcount != 1:
            return None
        db.execute(
            "UPDATE creative_workers SET account_id=?,updated_at=? WHERE worker_id=?",
            (account_id, now, worker_id),
        )
        return account_id

def _creative_release_account(account_id, success=False, error=None):
    if not account_id:
        return
    _creative_db_init()
    now = _creative_now()
    with _batch_db() as db:
        db.execute(
            "UPDATE creative_accounts SET status='ACTIVE',"
            "last_success=CASE WHEN ? THEN ? ELSE last_success END,"
            "last_error=CASE WHEN ? IS NOT NULL THEN ? ELSE last_error END,"
            "updated_at=? WHERE account_id=?",
            (1 if success else 0, now, error, error, now, account_id),
        )

def _creative_set_account_state(account_id, status, error=None, cooldown_until=None):
    _creative_db_init()
    with _batch_db() as db:
        db.execute(
            "UPDATE creative_accounts SET status=?,last_error=?,cooldown_until=?,updated_at=? "
            "WHERE account_id=?",
            (status, error, cooldown_until, _creative_now(), account_id),
        )

def _creative_heartbeat(worker_id, job_id=None):
    _creative_db_init()
    now = _creative_now()
    with _batch_db() as db:
        db.execute(
            "UPDATE creative_workers SET last_heartbeat=?,job_id=?,"
            "status=CASE WHEN ? IS NULL THEN status ELSE 'RUNNING' END,updated_at=? "
            "WHERE worker_id=?",
            (now, job_id, job_id, now, worker_id),
        )
        if job_id:
            db.execute(
                "UPDATE creative_jobs SET last_heartbeat=?,updated_at=? "
                "WHERE job_id=? AND status='RUNNING'",
                (now, now, job_id),
            )

def _creative_mark_done(job_id, output=None, actual_engine=None):
    _creative_db_init()
    now = _creative_now()
    with _batch_db() as db:
        db.execute(
            "UPDATE creative_jobs SET status='DONE',completed_at=?,last_heartbeat=?,"
            "output=?,actual_engine=?,error=NULL,error_class=NULL,updated_at=? WHERE job_id=?",
            (now, now, output, actual_engine, now, job_id),
        )

def _creative_retry(job_id, error_class, error_text, account_id=None, cooldown_until=None):
    _creative_db_init()
    now = _creative_now()
    with _batch_db() as db:
        row = db.execute(
            "SELECT retry_count,max_retry FROM creative_jobs WHERE job_id=?",
            (job_id,),
        ).fetchone()
        if not row:
            return "MISSING"
        retry_count, max_retry = row
        if error_class == "CAPTCHA":
            db.execute(
                "UPDATE creative_jobs SET status='CAPTCHA',error_class='CAPTCHA',"
                "error=?,account_id=?,worker_id=NULL,updated_at=? WHERE job_id=?",
                (error_text, account_id, now, job_id),
            )
            if account_id:
                _creative_set_account_state(account_id, "CAPTCHA", error_text)
            return "CAPTCHA"
        if error_class == "RATE_LIMITED":
            if account_id:
                _creative_set_account_state(account_id, "COOLDOWN", error_text, cooldown_until)
            db.execute(
                "UPDATE creative_jobs SET status='RETRY',error_class=?,error=?,"
                "ready_at=?,account_id=NULL,worker_id=NULL,updated_at=? WHERE job_id=?",
                (error_class, error_text, cooldown_until or now, now, job_id),
            )
            return "RETRY"
        if error_class == "AUTH_ERROR":
            if account_id:
                _creative_set_account_state(account_id, "AUTH_ERROR", error_text)
            db.execute(
                "UPDATE creative_jobs SET status='RETRY',error_class=?,error=?,"
                "ready_at=?,account_id=NULL,worker_id=NULL,updated_at=? WHERE job_id=?",
                (error_class, error_text, now, now, job_id),
            )
            return "RETRY"
        if int(retry_count or 0) < int(max_retry or CREATIVE_MAX_RETRY):
            n = int(retry_count or 0) + 1
            db.execute(
                "UPDATE creative_jobs SET status='RETRY',retry_count=?,error_class=?,"
                "error=?,ready_at=?,account_id=NULL,updated_at=? WHERE job_id=?",
                (n, error_class, error_text, now, now, job_id),
            )
            db.execute(
                "INSERT INTO creative_retry(job_id,retry_no,reason,error_class,created_at) "
                "VALUES(?,?,?,?,?)",
                (job_id, n, error_text, error_class, now),
            )
            return "RETRY"
        db.execute(
            "UPDATE creative_jobs SET status='FAILED',error_class=?,error=?,"
            "account_id=NULL,worker_id=NULL,updated_at=? WHERE job_id=?",
            (error_class, error_text, now, job_id),
        )
        return "FAILED"

def _creative_recover_stale(timeout_sec=CREATIVE_HEARTBEAT_TIMEOUT):
    _creative_db_init()
    from datetime import datetime, timezone
    now_dt = datetime.now(timezone.utc)
    recovered = 0
    with _batch_db() as db:
        rows = db.execute(
            "SELECT worker_id,job_id,account_id,last_heartbeat "
            "FROM creative_workers WHERE status='RUNNING' AND job_id IS NOT NULL"
        ).fetchall()
        for worker_id, job_id, account_id, hb in rows:
            try:
                dt = datetime.fromisoformat(str(hb).replace("Z","+00:00")) if hb else None
                stale = not dt or (now_dt - dt).total_seconds() > timeout_sec
            except Exception:
                stale = True
            if not stale:
                continue
            now = _creative_now()
            db.execute(
                "UPDATE creative_jobs SET status='RETRY',worker_id=NULL,account_id=NULL,"
                "error_class='WORKER_TIMEOUT',error='Worker heartbeat quá hạn; đưa Job về hàng đợi.',"
                "ready_at=?,updated_at=? WHERE job_id=? AND status='RUNNING'",
                (now, now, job_id),
            )
            db.execute(
                "UPDATE creative_workers SET status='IDLE',job_id=NULL,account_id=NULL,"
                "error='Heartbeat timeout',updated_at=? WHERE worker_id=?",
                (now, worker_id),
            )
            if account_id:
                db.execute(
                    "UPDATE creative_accounts SET status='ACTIVE',updated_at=? "
                    "WHERE account_id=? AND status='BUSY'",
                    (now, account_id),
                )
            recovered += 1
    return recovered

def _creative_record_captcha(account_id, job_id, scene_id, note=""):
    _creative_db_init()
    now = _creative_now()
    with _batch_db() as db:
        db.execute(
            "INSERT INTO creative_captcha(account_id,job_id,scene_id,status,detected_at,note) "
            "VALUES(?,?,?,?,?,?)",
            (account_id, job_id, scene_id, "OPEN", now, note),
        )
        db.execute(
            "UPDATE creative_accounts SET status='CAPTCHA',last_error=?,updated_at=? "
            "WHERE account_id=?",
            (note, now, account_id),
        )
        db.execute(
            "UPDATE creative_jobs SET status='CAPTCHA',error_class='CAPTCHA',error=?,"
            "account_id=?,updated_at=? WHERE job_id=?",
            (note, account_id, now, job_id),
        )

def _creative_resolve_captcha(account_id, note="Đã xử lý CAPTCHA"):
    _creative_db_init()
    now = _creative_now()
    with _batch_db() as db:
        db.execute(
            "UPDATE creative_captcha SET status='RESOLVED',resolved_at=?,note=? "
            "WHERE account_id=? AND status='OPEN'",
            (now, note, account_id),
        )
        db.execute(
            "UPDATE creative_accounts SET status='ACTIVE',last_error=NULL,"
            "cooldown_until=NULL,updated_at=? WHERE account_id=?",
            (now, account_id),
        )
        db.execute(
            "UPDATE creative_jobs SET status='READY',error=NULL,error_class=NULL,"
            "account_id=NULL,ready_at=?,updated_at=? "
            "WHERE account_id=? AND status='CAPTCHA'",
            (now, now, account_id),
        )

def _creative_engine_route(requested_engine, flow_available=True):
    """Legacy compatibility helper: scheduler must not persist actual_engine here."""
    requested = str(requested_engine or CREATIVE_ENGINE_AUTO).upper()
    if requested == CREATIVE_ENGINE_ZIMAGE:
        return CREATIVE_ENGINE_ZIMAGE
    if requested == CREATIVE_ENGINE_FLOW:
        return CREATIVE_ENGINE_FLOW if flow_available else None
    # This is informational only; actual_engine is decided by the worker.
    return CREATIVE_ENGINE_FLOW if flow_available else CREATIVE_ENGINE_ZIMAGE

def _creative_qc_record(project, scene_id, stage, result, checks=None, error=None):
    _creative_db_init()
    with _batch_db() as db:
        db.execute(
            "INSERT INTO creative_qc(project_id,scene_id,stage,result,checks_json,error,created_at) "
            "VALUES(?,?,?,?,?,?,?)",
            (
                project, scene_id, stage, result,
                _creative_json(checks or {}, "{}"), error, _creative_now()
            ),
        )

def _creative_worker_register(worker_id, worker_type):
    _creative_db_init()
    now = _creative_now()
    with _batch_db() as db:
        db.execute(
            "INSERT INTO creative_workers(worker_id,worker_type,status,last_heartbeat,created_at,updated_at) "
            "VALUES(?,?,?,?,?,?) "
            "ON CONFLICT(worker_id) DO UPDATE SET worker_type=excluded.worker_type,"
            "status='IDLE',last_heartbeat=excluded.last_heartbeat,updated_at=excluded.updated_at",
            (worker_id, worker_type, "IDLE", now, now, now),
        )


# ============================================================
# CREATIVE STUDIO CORE V2 — ATOMIC ENGINE + RECOVERY CONTRACTS
# ============================================================
# V2 hardens the V1 control plane before real Flow execution:
# - SQLite BEGIN IMMEDIATE claims: one Job/Account/Worker at a time.
# - Real scheduler loop with worker/account dispatch.
# - actual_engine is written ONLY after a worker selects an engine.
# - CAPTCHA is a paused state, never a retry counter.
# - Recovery verifies output before requeueing a stale RUNNING job.
# - Output Contract + project manifest are durable sources of truth.
# - Z Image delegates to the existing run_zimage_txt2img_queue() only;
#   no second renderer/queue is created.

CREATIVE_OUTPUT_VERSION = 1
CREATIVE_MANIFEST_NAME = "creative_manifest.json"
CREATIVE_SCHEDULER_INTERVAL = 0.5
_CREATIVE_RUNTIME_LOCK = threading.RLock()
_CREATIVE_SCHEDULER_THREADS = {}
_CREATIVE_SCHEDULER_STOPS = {}
_CREATIVE_WORKER_THREADS = {}
_CREATIVE_PROJECT_Z_LOCKS = {}


def _creative_atomic_claim_job(project, worker_id):
    """Claim exactly one READY job with an SQLite write transaction."""
    _creative_db_init()
    project = _batch_project_name(project)
    now = _creative_now()
    with _batch_db() as db:
        db.execute("BEGIN IMMEDIATE")
        row = db.execute(
            "SELECT * FROM creative_jobs WHERE project_id=? AND status='READY' "
            "AND (worker_id IS NULL OR worker_id='') "
            "ORDER BY priority DESC,COALESCE(ready_at,created_at),created_at LIMIT 1",
            (project,),
        ).fetchone()
        if not row:
            db.execute("COMMIT")
            return None
        jid = row["job_id"]
        cur = db.execute(
            "UPDATE creative_jobs SET status='RUNNING',worker_id=?,started_at=?,"
            "last_heartbeat=?,updated_at=? WHERE job_id=? AND status='READY' "
            "AND (worker_id IS NULL OR worker_id='')",
            (worker_id, now, now, now, jid),
        )
        if cur.rowcount != 1:
            db.execute("ROLLBACK")
            return None
        db.execute(
            "UPDATE creative_workers SET status='RUNNING',job_id=?,last_heartbeat=?,updated_at=? "
            "WHERE worker_id=? AND status IN ('IDLE','RUNNING')",
            (jid, now, now, worker_id),
        )
        db.execute("COMMIT")
        return dict(row)


def _creative_atomic_claim_account(worker_id, job_id, provider=CREATIVE_ENGINE_FLOW):
    """Atomically bind one ACTIVE account to one worker/job."""
    _creative_db_init()
    now = _creative_now()
    with _batch_db() as db:
        db.execute("BEGIN IMMEDIATE")
        wr = db.execute(
            "SELECT account_id,job_id,status FROM creative_workers WHERE worker_id=?",
            (worker_id,),
        ).fetchone()
        if not wr or wr["job_id"] != job_id or wr["status"] != "RUNNING":
            db.execute("ROLLBACK")
            return None
        row = db.execute(
            "SELECT account_id FROM creative_accounts WHERE provider=? AND status='ACTIVE' "
            "AND (cooldown_until IS NULL OR cooldown_until<=?) "
            "ORDER BY COALESCE(last_success,''),updated_at LIMIT 1",
            (provider, now),
        ).fetchone()
        if not row:
            db.execute("ROLLBACK")
            return None
        account_id = row["account_id"]
        cur = db.execute(
            "UPDATE creative_accounts SET status='BUSY',usage=usage+1,updated_at=? "
            "WHERE account_id=? AND provider=? AND status='ACTIVE' "
            "AND (cooldown_until IS NULL OR cooldown_until<=?)",
            (now, account_id, provider, now),
        )
        if cur.rowcount != 1:
            db.execute("ROLLBACK")
            return None
        db.execute(
            "UPDATE creative_workers SET account_id=?,updated_at=? WHERE worker_id=? "
            "AND status='RUNNING' AND job_id=?",
            (account_id, now, worker_id, job_id),
        )
        db.execute(
            "UPDATE creative_jobs SET account_id=?,updated_at=? WHERE job_id=? "
            "AND status='RUNNING' AND worker_id=? AND account_id IS NULL",
            (account_id, now, job_id, worker_id),
        )
        db.execute("COMMIT")
        return account_id


def _creative_release_worker(worker_id, status="IDLE", error=None):
    _creative_db_init()
    now = _creative_now()
    with _batch_db() as db:
        db.execute(
            "UPDATE creative_workers SET status=?,job_id=NULL,account_id=NULL,error=?,"
            "updated_at=? WHERE worker_id=?",
            (status, error, now, worker_id),
        )


def _creative_output_contract(job, path, engine, account_id=None, worker_id=None):
    """Build the only output shape accepted by Job Engine."""
    fp = Path(path).expanduser().resolve()
    if not fp.is_file():
        raise FileNotFoundError(str(fp))
    import hashlib
    h = hashlib.sha256()
    with fp.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    width = height = None
    try:
        from PIL import Image
        with Image.open(fp) as im:
            width, height = im.size
            im.verify()
    except Exception as exc:
        raise ValueError(f"Output ảnh lỗi/không đọc được: {exc}")
    return {
        "contract_version": CREATIVE_OUTPUT_VERSION,
        "job_id": job.get("job_id"),
        "project_id": job.get("project_id"),
        "scene_id": job.get("scene_id"),
        "stage": job.get("stage"),
        "engine": engine,
        "status": "DONE",
        "output": {
            "path": str(fp),
            "width": width,
            "height": height,
            "size": fp.stat().st_size,
            "sha256": h.hexdigest(),
        },
        "account_id": account_id,
        "worker_id": worker_id,
        "job_metadata": job.get("metadata_json") or {},
        "created_at": _creative_now(),
    }


def _creative_verify_output_contract(contract, project=None, scene_id=None):
    """Re-verify a persisted contract before declaring a job DONE."""
    if not isinstance(contract, dict) or contract.get("status") != "DONE":
        return False, "Output contract không hợp lệ."
    out = contract.get("output") or {}
    fp = Path(str(out.get("path") or "")).expanduser()
    if not fp.is_file() or fp.stat().st_size <= 0:
        return False, "Không tìm thấy output hoặc file rỗng."
    if project and str(contract.get("project_id")) != str(_batch_project_name(project)):
        return False, "Sai project_id."
    if scene_id and str(contract.get("scene_id")) != str(scene_id):
        return False, "Sai scene_id."
    try:
        actual = _creative_output_contract(
            {"job_id": contract.get("job_id"), "project_id": contract.get("project_id"),
             "scene_id": contract.get("scene_id"), "stage": contract.get("stage")},
            fp, contract.get("engine"), contract.get("account_id"), contract.get("worker_id"))
        a = actual["output"]
        try:
            raw_meta = contract.get("job_metadata") or {}
            meta = _creative_load_json(raw_meta, {}) if isinstance(raw_meta, str) else dict(raw_meta)
        except Exception:
            meta = {}
        expected_w = meta.get("expected_width")
        expected_h = meta.get("expected_height")
        if expected_w and int(a["width"] or 0) != int(expected_w):
            return False, f"Sai chiều rộng output: {a['width']} != {expected_w}."
        if expected_h and int(a["height"] or 0) != int(expected_h):
            return False, f"Sai chiều cao output: {a['height']} != {expected_h}."
        if int(a["width"] or 0) != int(out.get("width") or 0):
            return False, "Sai chiều rộng output."
        if int(a["height"] or 0) != int(out.get("height") or 0):
            return False, "Sai chiều cao output."
        if str(a["sha256"]) != str(out.get("sha256")):
            return False, "Hash output không khớp."
        return True, "OK"
    except Exception as exc:
        return False, str(exc)


def _creative_job_output_from_db(row):
    try:
        return _creative_load_json(row["output"], None)
    except Exception:
        return None


def _creative_recover_stale_v2(timeout_sec=CREATIVE_HEARTBEAT_TIMEOUT):
    """Recover only after checking worker liveness AND persisted output."""
    _creative_db_init()
    from datetime import datetime, timezone
    now_dt = datetime.now(timezone.utc)
    recovered = 0
    with _batch_db() as db:
        rows = db.execute(
            "SELECT j.*,w.status AS worker_status,w.last_heartbeat AS worker_heartbeat "
            "FROM creative_jobs j LEFT JOIN creative_workers w ON w.worker_id=j.worker_id "
            "WHERE j.status='RUNNING'"
        ).fetchall()
        for row in rows:
            hb = row["last_heartbeat"] or row["worker_heartbeat"]
            try:
                dt = datetime.fromisoformat(str(hb).replace("Z", "+00:00")) if hb else None
                stale = not dt or (now_dt - dt).total_seconds() > timeout_sec
            except Exception:
                stale = True
            if not stale:
                continue
            contract = _creative_job_output_from_db(row)
            ok, _ = _creative_verify_output_contract(contract, row["project_id"], row["scene_id"]) if contract else (False, "")
            now = _creative_now()
            if ok:
                db.execute(
                    "UPDATE creative_jobs SET status='DONE',completed_at=?,updated_at=?,"
                    "error=NULL,error_class=NULL WHERE job_id=? AND status='RUNNING'",
                    (now, now, row["job_id"]),
                )
            else:
                db.execute(
                    "UPDATE creative_jobs SET status='RETRY',worker_id=NULL,account_id=NULL,"
                    "error_class='WORKER_TIMEOUT',error='Worker mất heartbeat; output chưa xác thực.',"
                    "ready_at=?,updated_at=? WHERE job_id=? AND status='RUNNING'",
                    (now, now, row["job_id"]),
                )
            if row["worker_id"]:
                db.execute(
                    "UPDATE creative_workers SET status='IDLE',job_id=NULL,account_id=NULL,"
                    "error='Recovery heartbeat timeout',updated_at=? WHERE worker_id=?",
                    (now, row["worker_id"]),
                )
            if row["account_id"] and not ok:
                db.execute(
                    "UPDATE creative_accounts SET status='ACTIVE',updated_at=? "
                    "WHERE account_id=? AND status='BUSY'",
                    (now, row["account_id"]),
                )
            recovered += 1
    return recovered


def _creative_engine_available(engine):
    """Availability check only; does not assign actual_engine."""
    engine = str(engine or "").upper()
    if engine == CREATIVE_ENGINE_FLOW:
        # A usable ACTIVE account is the durable signal available to the core.
        _creative_db_init()
        with _batch_db() as db:
            row = db.execute(
                "SELECT 1 FROM creative_accounts WHERE provider='GOOGLE_FLOW' "
                "AND status='ACTIVE' AND (cooldown_until IS NULL OR cooldown_until<=?) LIMIT 1",
                (_creative_now(),),
            ).fetchone()
        return bool(row)
    if engine == CREATIVE_ENGINE_ZIMAGE:
        return callable(globals().get("run_zimage_txt2img_queue"))
    return False


def _creative_choose_engine_for_worker(requested_engine):
    """Strict user-selected image engine. Never silently fallback."""
    selected = _selected_image_engine()
    if selected == "Z_IMAGE":
        return CREATIVE_ENGINE_ZIMAGE, None
    if _creative_engine_available(CREATIVE_ENGINE_FLOW):
        return CREATIVE_ENGINE_FLOW, None
    return None, "Google Flow đã được chọn nhưng chưa CONNECTED. Không tự chuyển sang Z Image."


def _creative_set_actual_engine(job_id, engine, reason=None):
    _creative_db_init()
    now = _creative_now()
    with _batch_db() as db:
        db.execute(
            "UPDATE creative_jobs SET actual_engine=?,error=CASE WHEN ? IS NULL THEN error ELSE ? END,"
            "updated_at=? WHERE job_id=? AND status='RUNNING'",
            (engine, reason, reason, now, job_id),
        )


def _creative_fallback_allowed(error_class):
    return str(error_class or "").upper() in {"RATE_LIMITED", "UNAVAILABLE", "NETWORK", "SERVER_BUSY", "FLOW_UNAVAILABLE"}


def _creative_mark_output_done(job, contract):
    ok, reason = _creative_verify_output_contract(contract, job.get("project_id"), job.get("scene_id"))
    if not ok:
        return False, reason
    _creative_mark_done(job["job_id"], _creative_json(contract, "{}"), contract.get("engine"))
    _creative_release_worker(job.get("worker_id") or "", "IDLE", None)
    if job.get("account_id"):
        _creative_release_account(job["account_id"], success=True)
    return True, "OK"


def _creative_project_manifest_path(project):
    return project_dir(_batch_project_name(project)) / CREATIVE_MANIFEST_NAME


def _creative_build_manifest(project, scenes):
    project = _batch_project_name(project)
    _creative_db_init()
    with _batch_db() as db:
        rows = db.execute(
            "SELECT * FROM creative_jobs WHERE project_id=? ORDER BY scene_id,stage",
            (project,),
        ).fetchall()
    by_scene = {}
    for row in rows:
        sid = row["scene_id"]
        by_scene.setdefault(sid, {})[row["stage"]] = dict(row)
    items = []
    for scene in scenes or []:
        sid = str(scene.get("scene_id") or "").strip()
        if not sid:
            continue
        jobs = by_scene.get(sid, {})
        image_job = jobs.get("IMAGE", {})
        image_out = _creative_job_output_from_db(image_job) if image_job else None
        items.append({
            "scene_id": sid,
            "scene_type": scene_media_type(scene, "image"),
            "prompt": scene.get("prompt") or scene.get("image_prompt") or "",
            "image_required": scene_media_type(scene, "image") != "video",
            "video_required": scene_media_type(scene, "image") == "video",
            "requested_engine": image_job.get("requested_engine", CREATIVE_ENGINE_AUTO),
            "actual_engine": image_job.get("actual_engine"),
            "current_stage": next((st for st in CREATIVE_JOB_STAGES if jobs.get(st, {}).get("status") not in ("DONE", "SKIPPED")), "DONE"),
            "output": image_out,
            "QC": [dict(r) for r in rows if r["scene_id"] == sid and r["stage"] == "QC"],
        })
    manifest = {
        "manifest_version": 1,
        "project_id": project,
        "updated_at": _creative_now(),
        "scenes": items,
    }
    path = _creative_project_manifest_path(project)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, manifest)
    return manifest


def _creative_load_manifest(project):
    path = _creative_project_manifest_path(project)
    try:
        return read_json(path, {}) or {}
    except Exception:
        return {}


def _creative_prepare_project(project, scenes, requested_engine=CREATIVE_ENGINE_AUTO):
    """Idempotent project preparation: no duplicate Job rows are created."""
    _creative_create_jobs(project, scenes, requested_engine=requested_engine)
    return _creative_build_manifest(project, scenes)


def _creative_scheduler_dispatch(project):
    """One scheduler tick: dependencies -> idle worker -> atomic claim -> dispatch."""
    project = _batch_project_name(project)
    _creative_db_init()
    _creative_recover_stale_v2()
    with _batch_db() as db:
        workers = db.execute(
            "SELECT worker_id,worker_type,status,job_id FROM creative_workers "
            "WHERE status='IDLE' ORDER BY worker_id"
        ).fetchall()
    for worker in workers:
        picked = _creative_scheduler_pick(project)
        if not picked:
            break
        job = _creative_atomic_claim_job(project, worker["worker_id"])
        if job:
            _creative_dispatch_worker(worker["worker_id"], job)
    return True


def _creative_scheduler_loop(project):
    stop = _CREATIVE_SCHEDULER_STOPS.setdefault(_batch_project_name(project), threading.Event())
    while not stop.is_set():
        try:
            _creative_scheduler_dispatch(project)
        except Exception as exc:
            print(f"[CREATIVE] Scheduler lỗi | {str(exc)[:500]}", flush=True)
        stop.wait(CREATIVE_SCHEDULER_INTERVAL)


def _creative_start_scheduler(project):
    project = _batch_project_name(project)
    with _CREATIVE_RUNTIME_LOCK:
        th = _CREATIVE_SCHEDULER_THREADS.get(project)
        if th and th.is_alive():
            return th
        stop = _CREATIVE_SCHEDULER_STOPS.setdefault(project, threading.Event())
        stop.clear()
        th = threading.Thread(target=_creative_scheduler_loop, args=(project,), name=f"CreativeScheduler-{project}", daemon=True)
        _CREATIVE_SCHEDULER_THREADS[project] = th
        th.start()
        return th


def _creative_stop_scheduler(project):
    stop = _CREATIVE_SCHEDULER_STOPS.get(_batch_project_name(project))
    if stop:
        stop.set()


class CreativeEngineAdapter:
    def run(self, job, worker_id, account_id=None):
        raise NotImplementedError


class ZImageAdapter(CreativeEngineAdapter):
    """Adapter only. The existing Z Image queue remains the renderer."""
    def run(self, job, worker_id, account_id=None):
        project = job["project_id"]
        sid = job["scene_id"]
        lock = _CREATIVE_PROJECT_Z_LOCKS.setdefault(project, threading.RLock())
        with lock:
            # Never create a second queue/renderer. Start the existing project queue once.
            existing = _creative_find_existing_scene_output(project, sid)
            if existing:
                return _creative_output_contract(job, existing, CREATIVE_ENGINE_ZIMAGE, account_id, worker_id)
            runner = globals().get("run_zimage_txt2img_queue")
            if not callable(runner):
                raise RuntimeError("Không tìm thấy pipeline Z Image hiện tại.")
            for _pct, _status, _hud in runner(project):
                _creative_heartbeat(worker_id, job["job_id"])
            existing = _creative_find_existing_scene_output(project, sid)
            if not existing:
                raise RuntimeError(f"Z Image queue chạy xong nhưng chưa có output cho {sid}.")
            return _creative_output_contract(job, existing, CREATIVE_ENGINE_ZIMAGE, account_id, worker_id)


def _creative_find_existing_scene_output(project, scene_id):
    root = project_dir(project)
    candidates = [
        root / "Hình ảnh" / f"{scene_id}.png",
        root / "Hình ảnh" / f"{scene_id}.jpg",
        root / "Hình ảnh" / f"{scene_id}.jpeg",
    ]
    for fp in candidates:
        if fp.is_file() and fp.stat().st_size > 0:
            return fp
    # Reuse the application's existing stage candidate resolver when available.
    fn = globals().get("_batch_stage_output_candidates")
    if callable(fn):
        try:
            for fp in fn(project, scene_id, "IMAGE") or []:
                fp = Path(fp)
                if fp.is_file() and fp.stat().st_size > 0:
                    return fp
        except Exception:
            pass
    return None


# V11.6: Flow execution uses installed Chrome Stable over CDP; Playwright is only the controller.
class GoogleFlowAdapter(CreativeEngineAdapter):
    """Playwright-backed Flow adapter. CAPTCHA is detected, never bypassed."""
    FLOW_URL = os.getenv("VHUNG_GOOGLE_FLOW_URL", "https://labs.google/fx/tools/flow")

    def __init__(self):
        self._contexts = {}

    def _context(self, account_id):
        """Connect Playwright to the tool-owned Google Chrome Stable profile.

        IMPORTANT: this never launches Playwright/Chromium. Chrome is launched
        separately with a deterministic CDP port and the verified Flow profile.
        The Playwright controller belongs exclusively to the calling worker
        thread and is disconnected by _flow_run_v7 when that scene finishes.
        """
        try:
            _playwright_prepare_runtime()
            from playwright.sync_api import sync_playwright
        except Exception as exc:
            raise RuntimeError(f"Thiếu Playwright runtime: {exc}")

        account_dir = APP_STATE_DIR / "flow_profiles" / safe_name(account_id)
        row = _creative_get_account(account_id) if callable(globals().get("_creative_get_account")) else None
        profile = str((row or {}).get("session_profile") or account_dir).strip()
        port = int((row or {}).get("debug_port") or _flow_debug_port(account_id))

        if not _flow_cdp_ready(port, timeout=2):
            try:
                _flow_launch_stable_chrome(account_id, profile, self.FLOW_URL)
            except Exception as exc:
                raise RuntimeError(f"Không mở được Chrome Stable cho {account_id}: {exc}")
            if not _flow_cdp_ready(port, timeout=15):
                raise RuntimeError(
                    f"Chrome Stable đã mở nhưng CDP chưa sẵn sàng · Account={account_id} · Port={port}"
                )

        pw = sync_playwright().start()
        try:
            browser = pw.chromium.connect_over_cdp(
                f"http://127.0.0.1:{port}", timeout=15000
            )
            contexts = browser.contexts
            if not contexts:
                raise RuntimeError("Chrome CDP connected nhưng không có BrowserContext.")
            context = contexts[0]
            pages = context.pages
            page = None
            for candidate in pages:
                try:
                    if "flow" in str(candidate.url or "").lower():
                        page = candidate
                        break
                except Exception:
                    pass
            if page is None:
                page = pages[0] if pages else context.new_page()
            if "flow" not in str(page.url or "").lower():
                page.goto(self.FLOW_URL, wait_until="domcontentloaded", timeout=60000)
            return pw, browser, page
        except Exception:
            try:
                pw.stop()
            except Exception:
                pass
            raise

    @staticmethod
    def _captcha_visible(page):
        text = (page.locator("body").inner_text(timeout=5000) or "").lower()
        keys = ("captcha", "verify you are human", "unusual traffic", "security check")
        return any(k in text for k in keys)

    def run(self, job, worker_id, account_id=None):
        if not account_id:
            raise RuntimeError("Flow Job chưa được cấp Account.")
        _creative_set_actual_engine(job["job_id"], CREATIVE_ENGINE_FLOW)
        _pw, _browser, page = self._context(account_id)
        _creative_heartbeat(worker_id, job["job_id"])
        if self._captcha_visible(page):
            raise CreativeCaptchaError("Google Flow yêu cầu CAPTCHA.")
        prompt = str(job.get("metadata_json") or "")
        metadata = _creative_load_json(prompt, {}) if prompt else {}
        prompt_text = str(metadata.get("prompt") or metadata.get("image_prompt") or "").strip()
        if not prompt_text:
            raise CreativeInvalidRequest("Job Flow không có Prompt.")
        # Prefer accessible textbox/contenteditable controls. No fixed CSS tied to one UI build.
        box = page.get_by_role("textbox").last
        if not box.count():
            box = page.locator("textarea, [contenteditable='true']").last
        if not box.count():
            raise CreativeFlowUnavailable("Không tìm thấy ô Prompt của Google Flow.")
        box.fill(prompt_text)
        if self._captcha_visible(page):
            raise CreativeCaptchaError("Google Flow yêu cầu CAPTCHA.")
        # Submit through an accessible button. We do not guess/bypass challenge controls.
        buttons = page.get_by_role("button")
        submitted = False
        for label in ("Generate", "Create", "Run", "Tạo", "Generate image"):
            try:
                btn = buttons.filter(has_text=label).last
                if btn.count() and btn.is_enabled():
                    btn.click()
                    submitted = True
                    break
            except Exception:
                continue
        if not submitted:
            raise CreativeFlowUnavailable("Không tìm thấy nút tạo ảnh của Google Flow.")
        _creative_heartbeat(worker_id, job["job_id"])
        # Result download is intentionally event-driven, not a fake sleep(180).
        download = None
        try:
            with page.expect_download(timeout=180000) as dl_info:
                # Some Flow builds download immediately after Generate; if not, the event times out.
                page.wait_for_timeout(1000)
            download = dl_info.value
        except Exception:
            if self._captcha_visible(page):
                raise CreativeCaptchaError("Google Flow yêu cầu CAPTCHA.")
            raise CreativeFlowUnavailable("Google Flow chưa phát Download trong thời gian chờ.")
        out_dir = project_dir(job["project_id"]) / "Hình ảnh"
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"{job['scene_id']}.png"
        download.save_as(str(out))
        return _creative_output_contract(job, out, CREATIVE_ENGINE_FLOW, account_id, worker_id)


class CreativeCaptchaError(RuntimeError):
    pass


class CreativeFlowUnavailable(RuntimeError):
    pass


class CreativeInvalidRequest(RuntimeError):
    pass


def _creative_adapter_for(engine):
    if engine == CREATIVE_ENGINE_FLOW:
        return GoogleFlowAdapter()
    if engine == CREATIVE_ENGINE_ZIMAGE:
        return ZImageAdapter()
    raise RuntimeError(f"Engine không hỗ trợ: {engine}")


_CREATIVE_FLOW_ADAPTER = None
_CREATIVE_ZIMAGE_ADAPTER = ZImageAdapter()


def _creative_dispatch_worker(worker_id, job):
    """Execute a claimed job. Scheduler only dispatches; worker chooses actual engine."""
    def runner():
        global _CREATIVE_FLOW_ADAPTER
        account_id = None
        try:
            _creative_heartbeat(worker_id, job["job_id"])
            engine, reason = _creative_choose_engine_for_worker(job["requested_engine"])
            if not engine:
                _creative_retry(job["job_id"], "UNAVAILABLE", reason or "Không có Engine khả dụng.")
                _creative_release_worker(worker_id, "IDLE", reason)
                return
            _creative_set_actual_engine(job["job_id"], engine, reason)
            if engine == CREATIVE_ENGINE_FLOW:
                account_id = _creative_atomic_claim_account(worker_id, job["job_id"], "GOOGLE_FLOW")
                if not account_id:
                    # Keep AUTO/Flow job queued rather than pretending Flow was used.
                    _creative_retry(job["job_id"], "UNAVAILABLE", "Không claim được Account Flow.")
                    _creative_release_worker(worker_id, "IDLE", "Không có Account Flow.")
                    return
                if _CREATIVE_FLOW_ADAPTER is None:
                    _CREATIVE_FLOW_ADAPTER = GoogleFlowAdapter()
                adapter = _CREATIVE_FLOW_ADAPTER
            else:
                adapter = _CREATIVE_ZIMAGE_ADAPTER
            _creative_heartbeat(worker_id, job["job_id"])
            contract = adapter.run(dict(job), worker_id, account_id)
            ok, reason = _creative_mark_output_done(dict(job), contract)
            if not ok:
                _creative_retry(job["job_id"], "OUTPUT_INVALID", reason, account_id)
                _creative_release_worker(worker_id, "IDLE", reason)
                return
            print(f"[CREATIVE] DONE | {job['job_id']} | {contract.get('engine')}", flush=True)
        except CreativeCaptchaError as exc:
            _creative_record_captcha(account_id, job["job_id"], job["scene_id"], str(exc))
            _creative_release_worker(worker_id, "IDLE", str(exc))
            if account_id:
                _creative_set_account_state(account_id, "CAPTCHA", str(exc))
            print(f"[CREATIVE] CAPTCHA | {job['job_id']} | Account={account_id}", flush=True)
        except (CreativeFlowUnavailable, TimeoutError) as exc:
            if job.get("requested_engine") == CREATIVE_ENGINE_AUTO and _creative_fallback_allowed("UNAVAILABLE"):
                # Explicit fallback: only AUTO may fall back. Do not overwrite Flow history.
                _creative_set_actual_engine(job["job_id"], CREATIVE_ENGINE_ZIMAGE, f"Flow unavailable: {exc}")
                try:
                    contract = _CREATIVE_ZIMAGE_ADAPTER.run(dict(job), worker_id, None)
                    ok, reason = _creative_mark_output_done(dict(job), contract)
                    if not ok:
                        raise RuntimeError(reason)
                except Exception as zexc:
                    _creative_retry(job["job_id"], "UNAVAILABLE", f"Flow lỗi, Z Image cũng lỗi: {zexc}")
                    _creative_release_worker(worker_id, "IDLE", str(zexc))
            else:
                _creative_retry(job["job_id"], "UNAVAILABLE", str(exc), account_id)
                _creative_release_worker(worker_id, "IDLE", str(exc))
        except CreativeInvalidRequest as exc:
            _creative_retry(job["job_id"], "INVALID_REQUEST", str(exc), account_id)
            _creative_release_worker(worker_id, "IDLE", str(exc))
        except Exception as exc:
            _creative_retry(job["job_id"], "NETWORK", str(exc), account_id)
            _creative_release_worker(worker_id, "IDLE", str(exc))
        finally:
            if account_id:
                # Preserve CAPTCHA/COOLDOWN/AUTH_ERROR states; only release BUSY.
                with _batch_db() as db:
                    row = db.execute("SELECT status FROM creative_accounts WHERE account_id=?", (account_id,)).fetchone()
                if row and row[0] == "BUSY":
                    _creative_release_account(account_id, success=False)
    with _CREATIVE_RUNTIME_LOCK:
        old = _CREATIVE_WORKER_THREADS.get(worker_id)
        if old and old.is_alive():
            return old
        th = threading.Thread(target=runner, name=f"CreativeWorker-{worker_id}", daemon=True)
        _CREATIVE_WORKER_THREADS[worker_id] = th
        th.start()
        return th


def _creative_start_workers(project, count=2, worker_type="FLOW"):
    _creative_db_init()
    count = max(1, int(count or 1))
    ids = [f"FLOW-W{i:02d}" for i in range(1, count + 1)] if str(worker_type).upper() == "FLOW" else [f"{worker_type.upper()}-W{i:02d}" for i in range(1, count + 1)]
    for wid in ids:
        _creative_worker_register(wid, str(worker_type).upper())
    _creative_start_scheduler(project)
    return ids



# ============================================================
# CREATIVE STUDIO V3 — RELIABILITY + REAL CONTROL PLANE
# ============================================================
# V3 rules:
# - CAPTCHA is never a retry.
# - Retry policy is classified and bounded to max_retry=3.
# - Cooldown expires automatically; accounts do not die forever.
# - Quota is UNKNOWN unless the provider reports a real value.
# - PAUSE is cooperative: running jobs finish; queued work waits.
# - STOP is durable and cooperative: no new jobs are claimed, running workers
#   receive a stop request, output is reconciled before a job is requeued.
# - Recovery checks output before re-running a stale job.
# - Scheduler checks worker/session/account health before dispatch.
# - AUTO fallback is policy-driven. CAPTCHA never silently becomes Z Image.

CREATIVE_V3_VERSION = 3
CREATIVE_ERROR_CLASSES = (
    "NETWORK_ERROR", "TIMEOUT", "FLOW_UNAVAILABLE", "RATE_LIMITED",
    "QUOTA", "AUTH_ERROR", "CAPTCHA", "INVALID_REQUEST",
    "OUTPUT_INVALID", "WORKER_CRASH", "BROWSER_CRASH", "SERVER_BUSY",
)
CREATIVE_RETRYABLE_ERRORS = {"NETWORK_ERROR", "TIMEOUT", "FLOW_UNAVAILABLE", "SERVER_BUSY", "OUTPUT_INVALID", "WORKER_CRASH", "BROWSER_CRASH"}
CREATIVE_FALLBACK_ERRORS = set()
CREATIVE_COOLDOWN_SECONDS = int(os.getenv("VHUNG_FLOW_COOLDOWN_SECONDS", "300") or 300)

class CreativeRateLimited(RuntimeError): pass
class CreativeQuotaExceeded(RuntimeError): pass
class CreativeAuthError(RuntimeError): pass
class CreativeNetworkError(RuntimeError): pass
class CreativeFlowTimeout(TimeoutError): pass


def _creative_v3_migrate():
    """Idempotent SQLite migration for V3-V6 durable state."""
    _creative_db_init()
    columns = {
        "creative_accounts": {
            "quota": "TEXT DEFAULT 'UNKNOWN'",
            "quota_reset_at": "TEXT",
            "session_health": "TEXT DEFAULT 'UNKNOWN'",
            "last_health_check": "TEXT",
        },
        "creative_workers": {
            "stop_requested": "INTEGER NOT NULL DEFAULT 0",
            "heartbeat_timeout": "INTEGER NOT NULL DEFAULT 60",
            "health": "TEXT DEFAULT 'UNKNOWN'",
        },
        "creative_sessions": {
            "last_health_check": "TEXT",
            "health": "TEXT DEFAULT 'UNKNOWN'",
            "page_url": "TEXT",
            "browser_alive": "INTEGER NOT NULL DEFAULT 0",
        },
        "creative_jobs": {
            "fallback_allowed": "INTEGER NOT NULL DEFAULT 1",
            "fallback_reason": "TEXT",
            "control_state": "TEXT DEFAULT 'RUN'",
            "cancel_requested": "INTEGER NOT NULL DEFAULT 0",
            "attempt_started_at": "TEXT",
            "last_output_check": "TEXT",
            "last_output_error": "TEXT",
        },
    }
    with _batch_db() as db:
        for table, wanted in columns.items():
            existing = {r[1] for r in db.execute(f"PRAGMA table_info({table})").fetchall()}
            for col, typ in wanted.items():
                if col not in existing:
                    db.execute(f"ALTER TABLE {table} ADD COLUMN {col} {typ}")
        db.executescript("""
        CREATE TABLE IF NOT EXISTS creative_control (
            project_id TEXT PRIMARY KEY,
            state TEXT NOT NULL DEFAULT 'RUNNING',
            reason TEXT,
            requested_by TEXT,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS creative_recovery (
            recovery_id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id TEXT NOT NULL,
            job_id TEXT NOT NULL,
            worker_id TEXT,
            reason TEXT,
            worker_alive INTEGER NOT NULL DEFAULT 0,
            browser_alive INTEGER NOT NULL DEFAULT 0,
            output_valid INTEGER NOT NULL DEFAULT 0,
            action TEXT,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS creative_health (
            health_id INTEGER PRIMARY KEY AUTOINCREMENT,
            target_type TEXT NOT NULL,
            target_id TEXT NOT NULL,
            status TEXT NOT NULL,
            detail TEXT,
            checked_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_creative_recovery_job ON creative_recovery(job_id,created_at);
        CREATE INDEX IF NOT EXISTS idx_creative_health_target ON creative_health(target_type,target_id,checked_at);
        """)


def _creative_control_state(project):
    _creative_v3_migrate()
    project = _batch_project_name(project)
    with _batch_db() as db:
        row = db.execute("SELECT state FROM creative_control WHERE project_id=?", (project,)).fetchone()
    return str(row[0]) if row else "RUNNING"


def _creative_set_control(project, state, reason="", requested_by="UI"):
    _creative_v3_migrate()
    project = _batch_project_name(project)
    state = str(state or "RUNNING").upper()
    now = _creative_now()
    with _batch_db() as db:
        db.execute("""INSERT INTO creative_control(project_id,state,reason,requested_by,updated_at)
                     VALUES(?,?,?,?,?)
                     ON CONFLICT(project_id) DO UPDATE SET state=excluded.state,reason=excluded.reason,
                     requested_by=excluded.requested_by,updated_at=excluded.updated_at""",
                   (project,state,str(reason or ""),requested_by,now))
        db.execute("UPDATE creative_jobs SET control_state=? WHERE project_id=? AND status IN ('PENDING','READY','RETRY','WAITING_DEPENDENCY')",
                   (state,project))


def _creative_refresh_cooldowns():
    """Move expired COOLDOWN accounts back to ACTIVE, never permanently dead."""
    _creative_v3_migrate()
    now = _creative_now()
    changed = 0
    with _batch_db() as db:
        rows = db.execute("SELECT account_id FROM creative_accounts WHERE status='COOLDOWN' AND cooldown_until IS NOT NULL AND cooldown_until<=?", (now,)).fetchall()
        for row in rows:
            db.execute("UPDATE creative_accounts SET status='ACTIVE',cooldown_until=NULL,last_error=NULL,updated_at=? WHERE account_id=? AND status='COOLDOWN'", (now,row[0]))
            changed += 1
    return changed


def _creative_set_cooldown(account_id, seconds=CREATIVE_COOLDOWN_SECONDS, error=""):
    _creative_v3_migrate()
    from datetime import datetime, timedelta, timezone
    until = (datetime.now(timezone.utc) + timedelta(seconds=max(1,int(seconds or 1)))).isoformat()
    _creative_set_account_state(account_id,"COOLDOWN",error,until)
    return until


def _creative_quota_update(account_id, usage_delta=0, quota=None, quota_reset_at=None, success=False, error=None):
    """Quota is provider-reported only. If unknown, persist UNKNOWN, never invent a limit."""
    _creative_v3_migrate()
    now = _creative_now()
    with _batch_db() as db:
        if quota is None:
            quota_value = "UNKNOWN"
        else:
            quota_value = str(quota)
        db.execute("""UPDATE creative_accounts SET usage=MAX(0,usage+?),quota=?,quota_reset_at=?,
                     last_success=CASE WHEN ? THEN ? ELSE last_success END,
                     last_error=CASE WHEN ? IS NULL THEN last_error ELSE ? END,updated_at=?
                     WHERE account_id=?""",
                   (int(usage_delta or 0),quota_value,quota_reset_at,1 if success else 0,now,error,error,now,account_id))


def _creative_get_account(account_id):
    _creative_v3_migrate()
    with _batch_db() as db:
        row=db.execute("SELECT * FROM creative_accounts WHERE account_id=?",(account_id,)).fetchone()
    return dict(row) if row else None


def _creative_health_log(target_type,target_id,status,detail=""):
    _creative_v3_migrate()
    with _batch_db() as db:
        db.execute("INSERT INTO creative_health(target_type,target_id,status,detail,checked_at) VALUES(?,?,?,?,?)",
                   (target_type,target_id,status,str(detail or "")[:1500],_creative_now()))


def _creative_session_health(account_id, worker_id=None, force=False):
    """Health-check a registered Flow profile in its own worker thread.

    Browser/CDP connectivity alone is not enough: the Flow page must also be
    outside Google's login endpoint and must not show a dedicated sign-in
    screen or CAPTCHA.
    """
    _creative_v3_migrate()
    aid=str(account_id or "").strip()
    result_box={}
    def _health_worker():
        global _CREATIVE_FLOW_ADAPTER
        pw=None
        try:
            if _CREATIVE_FLOW_ADAPTER is None:
                _CREATIVE_FLOW_ADAPTER=GoogleFlowAdapter()
            adapter=_CREATIVE_FLOW_ADAPTER
            pw,browser,page=adapter._context(aid)
            if not browser.is_connected() or page.is_closed():
                result_box["ok"]=False
                result_box["detail"]="Browser/Page không còn hoạt động."
                return
            if adapter._captcha_visible(page):
                result_box["ok"]=False
                result_box["detail"]="CAPTCHA đang yêu cầu xử lý."
                return

            url=str(page.url or "").lower()
            if "labs.google" not in url:
                try:
                    page.goto(adapter.FLOW_URL,wait_until="domcontentloaded",timeout=60000)
                    url=str(page.url or "").lower()
                except Exception as exc:
                    result_box["ok"]=False
                    result_box["detail"]=f"Không mở được Flow: {exc}"
                    return

            if adapter._captcha_visible(page):
                result_box["ok"]=False
                result_box["detail"]="CAPTCHA đang yêu cầu xử lý."
                return

            body=""
            try:
                body=(page.locator("body").inner_text(timeout=4000) or "").lower()
            except Exception:
                pass

            login_url=("accounts.google.com" in url and
                       ("signin" in url or "login" in url))
            dedicated_login=(
                any(x in body for x in (
                    "sign in to continue","choose an account","use another account",
                    "đăng nhập để tiếp tục"
                ))
                and "labs.google" not in url
                and "flow" not in url
            )
            if login_url or dedicated_login:
                result_box["ok"]=False
                result_box["detail"]="Flow session chưa đăng nhập Google."
                return

            result_box["ok"]=True
            result_box["detail"]="Session Flow HEALTHY · Chrome/CDP + Google Flow authenticated."
        except Exception as exc:
            result_box["ok"]=False
            result_box["detail"]=str(exc)
        finally:
            if pw is not None:
                try: pw.stop()
                except Exception: pass

    th=threading.Thread(target=_health_worker,name=f"VHUNG-FlowHealth-{aid}",daemon=True)
    th.start()
    th.join(timeout=90)
    ok=bool(result_box.get("ok")) if not th.is_alive() else False
    detail=str(result_box.get("detail") or ("Flow health-check timeout." if th.is_alive() else "Unknown"))
    now=_creative_now()
    with _batch_db() as db:
        db.execute(
            "UPDATE creative_accounts SET session_health=?,last_health_check=?,updated_at=? WHERE account_id=?",
            ("HEALTHY" if ok else "UNHEALTHY",now,now,aid)
        )
    _creative_health_log("ACCOUNT",aid,"HEALTHY" if ok else "UNHEALTHY",detail)
    return ok,detail

def _creative_claim_account_checked_v7(worker_id, job_id, provider=CREATIVE_ENGINE_FLOW):
    """Claim a real Flow account, auto-activating an already authenticated profile."""
    _creative_refresh_cooldowns()
    account_id = _creative_atomic_claim_account(worker_id, job_id, provider)
    if account_id:
        return account_id, "OK"

    _creative_db_init()
    candidates=[]
    now=_creative_now()
    try:
        with _batch_db() as db:
            rows=db.execute(
                "SELECT account_id,status FROM creative_accounts "
                "WHERE provider=? AND status IN ('AUTH_ERROR','DISABLED','COOLDOWN') "
                "AND (cooldown_until IS NULL OR cooldown_until<=?) ORDER BY updated_at",
                (provider,now),
            ).fetchall()
            candidates=[str(row["account_id"]) for row in rows]
    except Exception:
        candidates=[]

    for aid in candidates:
        try:
            ok,detail=_creative_session_health(aid,worker_id,force=True)
        except Exception as exc:
            ok,detail=False,str(exc)
        if not ok:
            continue
        with _batch_db() as db:
            db.execute(
                "UPDATE creative_accounts SET status='ACTIVE',session_health='HEALTHY',"
                "last_success=?,last_error=NULL,cooldown_until=NULL,updated_at=? "
                "WHERE account_id=? AND provider=?",
                (now,now,aid,provider),
            )
        workflow_cmd_log(
            "FLOW_POOL","AUTO_VERIFY_ACTIVE",account_id=aid,
            detail="Chrome/CDP session HEALTHY; tự ACTIVE trước khi claim job.",
        )
        account_id=_creative_atomic_claim_account(worker_id,job_id,provider)
        if account_id:
            return account_id,"OK"

    return None,"NO_ACTIVE_ACCOUNT"

_creative_claim_account_checked = _creative_claim_account_checked_v7
# ---------- Real image postprocess / Clean MAX VIP / QC ----------

def _creative_image_input_path(job):
    contract = _creative_job_output_from_db(job)
    if contract:
        p = Path(str((contract.get("output") or {}).get("path") or ""))
        if p.is_file():
            return p
    return _creative_find_existing_scene_output(job["project_id"], job["scene_id"])

def _creative_postprocess_image(job, worker_id):
    src = _creative_image_input_path(job)
    if not src or not src.is_file():
        raise CreativeFlowUnavailable(f"POSTPROCESS thiếu IMAGE output: {job['scene_id']}")
    try:
        from PIL import Image
        out_dir = _project_image_dir(project_dir(job["project_id"])) / "Postprocess"
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"{job['scene_id']}_post.png"
        with Image.open(src) as im:
            im.load()
            # Normalize to a standard PNG while preserving the requested frame size.
            if im.size != (CREATIVE_IMAGE_WIDTH, CREATIVE_IMAGE_HEIGHT):
                im = im.resize((CREATIVE_IMAGE_WIDTH, CREATIVE_IMAGE_HEIGHT), Image.Resampling.LANCZOS)
            if im.mode not in ("RGB", "RGBA"):
                im = im.convert("RGBA" if "A" in im.getbands() else "RGB")
            im.save(out, format="PNG", optimize=True)
        return _creative_output_contract(job, out, "POSTPROCESS", None, worker_id)
    except Exception as exc:
        raise CreativeFlowUnavailable(f"POSTPROCESS ảnh lỗi: {exc}")

def _creative_clean_max_vip(job, worker_id):
    src = _creative_image_input_path(job)
    if not src or not src.is_file():
        # For CLEAN, prefer the preceding POSTPROCESS contract explicitly.
        with _batch_db() as db:
            row = db.execute(
                "SELECT * FROM creative_jobs WHERE project_id=? AND scene_id=? AND stage='POSTPROCESS'",
                (job["project_id"], job["scene_id"]),
            ).fetchone()
        if row:
            c = _creative_job_output_from_db(row)
            src = Path(str((c or {}).get("output", {}).get("path") or ""))
    if not src or not src.is_file():
        raise CreativeFlowUnavailable(f"CLEAN MAX VIP thiếu POSTPROCESS output: {job['scene_id']}")
    try:
        from PIL import Image
        out_dir = _project_image_dir(project_dir(job["project_id"])) / "Cleaned"
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"{job['scene_id']}.png"
        with Image.open(src) as im:
            im.load()
            if im.size != (CREATIVE_IMAGE_WIDTH, CREATIVE_IMAGE_HEIGHT):
                im = im.resize((CREATIVE_IMAGE_WIDTH, CREATIVE_IMAGE_HEIGHT), Image.Resampling.LANCZOS)
            if im.mode not in ("RGB", "RGBA"):
                im = im.convert("RGBA" if "A" in im.getbands() else "RGB")
            # Re-save without carrying EXIF/text/ICC chunks from the source.
            clean = Image.new(im.mode, im.size)
            clean.paste(im)
            clean.save(out, format="PNG", optimize=True)
        audit = out_dir / f"{job['scene_id']}.clean.json"
        write_json(audit, {
            "scene_id": job["scene_id"],
            "source": str(src),
            "output": str(out),
            "metadata_stripped": True,
            "width": CREATIVE_IMAGE_WIDTH,
            "height": CREATIVE_IMAGE_HEIGHT,
            "timestamp": _creative_now(),
        })
        return _creative_output_contract(job, out, "CLEAN_MAX_VIP", None, worker_id)
    except Exception as exc:
        raise CreativeFlowUnavailable(f"CLEAN MAX VIP lỗi: {exc}")

class _V7PostProcessAdapter(CreativeStageAdapter):
    stage = "POSTPROCESS"
    def run_job(self, job, worker_id):
        return _creative_postprocess_image(job, worker_id)

class _V7CleanAdapter(CreativeStageAdapter):
    stage = "CLEAN_MAX_VIP"
    def run_job(self, job, worker_id):
        return _creative_clean_max_vip(job, worker_id)

class _V7QCAdapter(CreativeStageAdapter):
    stage = "QC"
    def run_job(self, job, worker_id):
        with _batch_db() as db:
            row = db.execute(
                "SELECT * FROM creative_jobs WHERE project_id=? AND scene_id=? AND stage='CLEAN_MAX_VIP'",
                (job["project_id"], job["scene_id"]),
            ).fetchone()
        if not row:
            raise CreativeFlowUnavailable("QC thiếu CLEAN_MAX_VIP job.")
        contract = _creative_job_output_from_db(row)
        if not contract:
            raise CreativeFlowUnavailable("QC thiếu Clean Output Contract.")
        ok, reason = _creative_verify_output_contract(
            contract, job["project_id"], job["scene_id"]
        )
        if not ok:
            raise CreativeFlowUnavailable(f"QC FAIL: {reason}")
        out = Path(str((contract.get("output") or {}).get("path") or ""))
        if not out.is_file():
            raise CreativeFlowUnavailable("QC output không tồn tại.")
        try:
            from PIL import Image
            with Image.open(out) as im:
                im.load()
                if im.size != (CREATIVE_IMAGE_WIDTH, CREATIVE_IMAGE_HEIGHT):
                    raise CreativeFlowUnavailable(
                        f"QC sai kích thước {im.size}; yêu cầu 1024x576."
                    )
        except CreativeFlowUnavailable:
            raise
        except Exception as exc:
            raise CreativeFlowUnavailable(f"QC ảnh lỗi: {exc}")
        return _creative_output_contract(job, out, "QC", None, worker_id)

_CREATIVE_STAGE_ADAPTERS["POSTPROCESS"] = _V7PostProcessAdapter()
_CREATIVE_STAGE_ADAPTERS["CLEAN_MAX_VIP"] = _V7CleanAdapter()
_CREATIVE_STAGE_ADAPTERS["QC"] = _V7QCAdapter()

# ---------- Canonical dispatcher ----------

def _creative_wait_for_account(job_id, worker_id, reason="NO_ACTIVE_ACCOUNT"):
    now = _creative_now()
    ready = (datetime.now(timezone.utc) + timedelta(seconds=5)).isoformat()
    with _batch_db() as db:
        db.execute(
            "UPDATE creative_jobs SET status='RETRY',error_class='SERVER_BUSY',error=?,"
            "ready_at=?,worker_id=NULL,account_id=NULL,updated_at=? "
            "WHERE job_id=? AND status='RUNNING'",
            (reason, ready, now, job_id),
        )
        db.execute(
            "UPDATE creative_workers SET status='IDLE',job_id=NULL,account_id=NULL,updated_at=? WHERE worker_id=?",
            (now, worker_id),
        )

def _creative_dispatch_worker_v7(worker_id, job):
    def runner():
        account_id = None
        try:
            _creative_heartbeat(worker_id, job["job_id"])
            if _creative_worker_stop_requested(worker_id):
                _creative_retry_v3(job["job_id"], "SERVER_BUSY", "STOP trước khi worker bắt đầu.", None, allow_fallback=False)
                _creative_release_worker(worker_id, "IDLE", "STOP")
                return
            stage = str(job.get("stage") or "IMAGE").upper()

            if stage == "IMAGE":
                requested = str(job.get("requested_engine") or CREATIVE_ENGINE_AUTO).upper()
                # The UI selection is authoritative. Never resurrect a stale
                # fallback_reason from an older job and route a Flow selection to Z Image.
                engine, reason = _creative_choose_engine_for_worker(requested)
                if not engine:
                    _creative_wait_for_account(job["job_id"], worker_id, reason or "NO_ENGINE")
                    return
                _creative_set_actual_engine(job["job_id"], engine, reason)
                if engine == CREATIVE_ENGINE_FLOW:
                    # Create the thread-owned adapter BEFORE session health/claim.
                    # This is what actually reopens the user's saved Chrome profile
                    # when Chrome was closed.
                    adapter = _flow_thread_adapter()
                    account_id, detail = _creative_claim_account_checked(worker_id, job["job_id"], "GOOGLE_FLOW")
                    if not account_id:
                        if detail == "NO_ACTIVE_ACCOUNT":
                            _creative_wait_for_account(job["job_id"], worker_id, detail)
                        else:
                            _creative_release_worker(worker_id, "IDLE", detail)
                        return
                else:
                    adapter = _CREATIVE_ZIMAGE_ADAPTER
                contract = adapter.run(dict(job), worker_id, account_id)
            elif stage in ("DEPTH", "MOTION"):
                # No fake video backend: these stages stay retry/failed until a real
                # backend is installed. Image-only projects never enqueue them.
                raise CreativeFlowUnavailable(
                    f"{stage} backend chưa được cấu hình cho media_type={job.get('metadata_json')}"
                )
            elif stage in ("POSTPROCESS", "CLEAN_MAX_VIP", "QC"):
                contract = _creative_stage_worker_run(dict(job), worker_id)
                _creative_cmd("CREATIVE", "DONE", job=job["job_id"], stage=stage)
                return
            elif stage == "TIMELINE":
                raise CreativeFlowUnavailable("TIMELINE backend chưa được cấu hình.")
            else:
                raise CreativeInvalidRequest(f"Stage không hỗ trợ: {stage}")

            ok, reason = _creative_verify_output_contract(
                contract, job["project_id"], job["scene_id"]
            )
            if not ok:
                raise CreativeFlowUnavailable(reason)
            _creative_mark_done(
                job["job_id"], _creative_json(contract, "{}"), contract.get("engine")
            )
            if account_id:
                _creative_quota_update(account_id, usage_delta=1, quota=None, success=True)
            _creative_release_worker(worker_id, "IDLE")
            if account_id:
                _creative_release_account(account_id, success=True)
            _creative_cmd("CREATIVE", "DONE", job=job["job_id"], engine=contract.get("engine"))
        except CreativeCaptchaError as exc:
            if account_id:
                _creative_record_captcha(account_id, job["job_id"], job["scene_id"], str(exc))
            else:
                _creative_retry_v3(job["job_id"], "CAPTCHA", str(exc), None, allow_fallback=False)
            _creative_release_worker(worker_id, "IDLE", str(exc))
        except CreativeRateLimited as exc:
            _creative_retry_v3(job["job_id"], "RATE_LIMITED", str(exc), account_id)
            _creative_release_worker(worker_id, "IDLE", str(exc))
        except CreativeQuotaExceeded as exc:
            _creative_retry_v3(job["job_id"], "QUOTA", str(exc), account_id)
            _creative_release_worker(worker_id, "IDLE", str(exc))
        except CreativeAuthError as exc:
            _creative_retry_v3(job["job_id"], "AUTH_ERROR", str(exc), account_id)
            _creative_release_worker(worker_id, "IDLE", str(exc))
        except CreativeInvalidRequest as exc:
            _creative_retry_v3(job["job_id"], "INVALID_REQUEST", str(exc), account_id, allow_fallback=False)
            _creative_release_worker(worker_id, "IDLE", str(exc))
        except CreativeFlowTimeout as exc:
            _creative_retry_v3(job["job_id"], "TIMEOUT", str(exc), account_id)
            _creative_release_worker(worker_id, "IDLE", str(exc))
        except CreativeNetworkError as exc:
            _creative_retry_v3(job["job_id"], "NETWORK_ERROR", str(exc), account_id)
            _creative_release_worker(worker_id, "IDLE", str(exc))
        except CreativeFlowUnavailable as exc:
            _creative_retry_v3(job["job_id"], "FLOW_UNAVAILABLE", str(exc), account_id)
            _creative_release_worker(worker_id, "IDLE", str(exc))
        except Exception as exc:
            _creative_retry_v3(job["job_id"], "NETWORK_ERROR", str(exc), account_id)
            _creative_release_worker(worker_id, "IDLE", str(exc))
        finally:
            if account_id:
                with _batch_db() as db:
                    row = db.execute(
                        "SELECT status FROM creative_accounts WHERE account_id=?",
                        (account_id,),
                    ).fetchone()
                if row and row[0] == "BUSY":
                    _creative_release_account(account_id, success=False)
    with _CREATIVE_RUNTIME_LOCK:
        old = _CREATIVE_WORKER_THREADS.get(worker_id)
        if old and old.is_alive():
            return old
        th = threading.Thread(
            target=runner, name=f"CreativeWorker-{worker_id}", daemon=True
        )
        _CREATIVE_WORKER_THREADS[worker_id] = th
        th.start()
        return th

_creative_dispatch_worker = _creative_dispatch_worker_v7

# ---------- Canonical scheduler / workers ----------

def _creative_scheduler_dispatch_v7(project):
    project = _batch_project_name(project)
    _creative_v3_migrate()
    _creative_refresh_cooldowns()
    state = _creative_control_state(project)
    _creative_recover_v3(project)
    if state in {"PAUSED", "STOPPING", "STOPPED"}:
        return False
    _creative_reconcile_dependencies(project)
    with _batch_db() as db:
        workers = db.execute(
            "SELECT worker_id,status FROM creative_workers "
            "WHERE status='IDLE' AND stop_requested=0 ORDER BY worker_id"
        ).fetchall()
    for w in workers:
        if _creative_control_state(project) != "RUNNING":
            break
        picked = _creative_scheduler_pick(project)
        if not picked:
            break
        job = _creative_atomic_claim_job(project, w["worker_id"])
        if job:
            _creative_dispatch_worker(w["worker_id"], job)
    return True

_creative_scheduler_dispatch = _creative_scheduler_dispatch_v7

def _creative_start_workers_v7(project, count=2, worker_type="FLOW"):
    _creative_v3_migrate()
    count = max(1, int(count or 1))
    ids = [
        f"{str(worker_type).upper()}-W{i:02d}"
        for i in range(1, count + 1)
    ]
    for wid in ids:
        _creative_worker_register(wid, str(worker_type).upper())
    _creative_start_scheduler(project)
    return ids

_creative_start_workers = _creative_start_workers_v7

# ---------- Project-scoped stop ----------

def _creative_request_stop_workers_v7(project):
    _creative_v3_migrate()
    project = _batch_project_name(project)
    now = _creative_now()
    with _batch_db() as db:
        # Do not globally stop idle workers belonging to another Project.
        db.execute(
            """UPDATE creative_jobs
               SET cancel_requested=1,control_state='STOPPING',updated_at=?
               WHERE project_id=? AND status='RUNNING'""",
            (now, project),
        )
        db.execute(
            """UPDATE creative_workers
               SET stop_requested=1,updated_at=?
               WHERE worker_id IN (
                   SELECT worker_id FROM creative_jobs
                   WHERE project_id=? AND status='RUNNING' AND worker_id IS NOT NULL
               )""",
            (now, project),
        )

_creative_request_stop_workers = _creative_request_stop_workers_v7

def _creative_stop_v7(project):
    project = _batch_project_name(project)
    _creative_set_control(project, "STOPPING", "Người dùng dừng hàng đợi.")
    _creative_request_stop_workers(project)
    _creative_stop_scheduler(project)
    return True

_creative_stop = _creative_stop_v7

def _creative_resume_v7(project):
    project = _batch_project_name(project)
    _creative_clear_stop_workers()
    _creative_set_control(project, "RUNNING", "Tiếp tục sau STOP/PAUSE.")
    _creative_reconcile_dependencies(project)
    _creative_start_scheduler(project)
    return True

_creative_resume = _creative_resume_v7

# ---------- Master Full waits for the complete image-only creative chain ----------

def _master_zimage_after_prompt_ui_v7(project, handoff_status):
    marker_text = str(handoff_status or "")
    if "MASTER_Z-IMAGE-TURBO_GATE_OK" not in marker_text:
        yield _workflow_progress_card(
            "⏳ ĐANG CHỜ GEMINI VERIFY — chưa tạo Job hình ảnh.", 100, "gemini"
        )
        return
    if _master_is_stopped():
        yield _workflow_progress_card(
            "⏹ MASTER đã dừng — chưa dispatch Job.", 100, "zimage"
        )
        return
    project = _batch_project_name(project)
    scenes = load_manifest(project)
    if not scenes:
        yield _workflow_progress_card("⛔ Không có Scene để tạo Job.", 100, "zimage")
        return

    _creative_master_full(project, scenes, CREATIVE_ENGINE_AUTO, start=True)
    workflow_cmd_log(
        "MASTER", "CREATIVE_JOBS_CREATED",
        project=project, scene_count=len(scenes), requested_engine="AUTO"
    )

    expected = []
    for scene in scenes:
        if scene_media_type(scene, "image") == "image":
            expected.append(str(scene.get("scene_id") or "").strip())
    expected = [x for x in expected if x]
    total = len(expected)

    deadline = time.monotonic() + max(
        300, int(os.getenv("VHUNG_MASTER_IMAGE_TIMEOUT", "86400") or 86400)
    )
    while time.monotonic() < deadline:
        if _master_is_stopped():
            _creative_stop(project)
            yield _workflow_progress_card(
                "⏹ MASTER đã dừng — Job Engine giữ nguyên checkpoint.", 0, "zimage"
            )
            return
        with _batch_db() as db:
            rows = db.execute(
                """SELECT scene_id,stage,status FROM creative_jobs
                   WHERE project_id=? AND scene_id IN (%s)"""
                % (",".join("?" for _ in expected) or "''"),
                [project, *expected],
            ).fetchall() if expected else []
        done = sum(
            1 for r in rows
            if r["stage"] in CREATIVE_IMAGE_ONLY_STAGES and r["status"] == "DONE"
        )
        failed = sum(
            1 for r in rows
            if r["stage"] in CREATIVE_IMAGE_ONLY_STAGES and r["status"] == "FAILED"
        )
        captcha = sum(
            1 for r in rows
            if r["stage"] in CREATIVE_IMAGE_ONLY_STAGES and r["status"] == "CAPTCHA"
        )
        running = sum(
            1 for r in rows
            if r["stage"] in CREATIVE_IMAGE_ONLY_STAGES and r["status"] == "RUNNING"
        )
        active_total = total * len(CREATIVE_IMAGE_ONLY_STAGES)
        pct = int(min(99, (done / active_total) * 100)) if active_total else 100
        yield _workflow_progress_card(
            f"CREATIVE: DONE {done}/{active_total} · RUN {running} · CAPTCHA {captcha} · FAIL {failed}",
            pct, "zimage"
        )
        if active_total and done >= active_total:
            workflow_cmd_log(
                "MASTER", "CREATIVE_COMPLETE_QC_UNLOCK",
                project=project, scene_count=total
            )
            yield _workflow_progress_card(
                f"🟢 IMAGE → POSTPROCESS → CLEAN → QC hoàn tất · {total} Scene.",
                100, "zimage"
            ) + "\n\n[MASTER_ZIMAGE_OK]"
            return
        if failed:
            workflow_cmd_log(
                "MASTER", "CREATIVE_PIPELINE_FAILED",
                project=project, failed=failed, level="ERROR"
            )
            yield _workflow_progress_card(
                f"⛔ Creative pipeline có {failed} Job FAILED.", 100, "zimage"
            )
            return
        time.sleep(1.0)

    yield _workflow_progress_card(
        "⛔ Hết thời gian chờ Creative Job Engine.", 100, "zimage"
    )

_master_zimage_after_prompt_ui = _master_zimage_after_prompt_ui_v7

# Clean UI text: production path is Hình ảnh/ZImage, legacy FluxJS only exists
# in migration code and is never used as an engine.


def _creative_account_pool_summary():
    """Compact account-pool summary for large Flow pools (1000+ accounts)."""
    _creative_v3_migrate()
    _creative_refresh_cooldowns()
    with _batch_db() as db:
        rows = db.execute(
            "SELECT status, COUNT(*) AS n FROM creative_accounts "
            "WHERE provider='GOOGLE_FLOW' GROUP BY status ORDER BY status"
        ).fetchall()
        total = db.execute(
            "SELECT COUNT(*) FROM creative_accounts WHERE provider='GOOGLE_FLOW'"
        ).fetchone()[0]
    counts = {str(row["status"]): int(row["n"]) for row in rows}
    return {
        "total": int(total or 0),
        "active": counts.get("ACTIVE", 0),
        "busy": counts.get("BUSY", 0),
        "cooldown": counts.get("COOLDOWN", 0) + counts.get("RATE_LIMITED", 0),
        "captcha": counts.get("CAPTCHA", 0),
        "quota": counts.get("QUOTA", 0),
        "auth_error": counts.get("AUTH_ERROR", 0),
        "disabled": counts.get("DISABLED", 0),
    }


def _flow_reconcile_profile_registry():
    """Remove only DB rows whose durable profile folder is actually gone.

    Canonical runtime never reads or caches Playwright objects from another
    thread. Chrome/CDP may be closed while the verified profile remains valid.
    """
    _flow_account_schema()
    stale = []
    with _batch_db() as db:
        rows = db.execute(
            "SELECT account_id,session_profile FROM creative_accounts "
            "WHERE provider='GOOGLE_FLOW' AND account_id LIKE 'FLOW-%'"
        ).fetchall()
        for row in rows:
            aid = str(row["account_id"] or "").strip()
            profile = str(row["session_profile"] or "").strip()
            if aid and profile and not Path(profile).expanduser().exists():
                stale.append(aid)
        if stale:
            placeholders = ",".join("?" for _ in stale)
            db.execute(
                f"DELETE FROM creative_accounts WHERE provider='GOOGLE_FLOW' "
                f"AND account_id IN ({placeholders})",
                stale,
            )
    if stale:
        workflow_cmd_log(
            "FLOW_POOL", "STALE_PROFILES_RECONCILED",
            removed=len(stale), accounts=",".join(stale)
        )
    return stale


def _flow_verified_profiles_report():
    """Report every durable Flow profile already registered and VERIFIED/ACTIVE.
    This is read-only: it never creates a profile, opens Chrome, or changes login state.
    """
    _flow_reconcile_profile_registry()
    _flow_account_schema()
    with _batch_db() as db:
        rows = db.execute(
            "SELECT account_id,status,session_health,account_label,session_profile,"
            "cookie_file,last_success,last_health_check,debug_port "
            "FROM creative_accounts WHERE provider='GOOGLE_FLOW' "
            "ORDER BY account_id"
        ).fetchall()

    verified = []
    for row in rows:
        status = str(row["status"] or "").upper()
        health = str(row["session_health"] or "").upper()
        profile = str(row["session_profile"] or "").strip()
        # ACTIVE + HEALTHY is the canonical verified state. Keep ACTIVE rows
        # visible even if an older DB did not populate session_health.
        if status == "ACTIVE" and (health in ("", "HEALTHY", "VERIFIED", "OK")):
            verified.append(row)

    parts = [
        "<div class='batch-center-card'>",
        "<b>🔎 FLOW PROFILE CHECK</b>",
        f"<div>ĐÃ VERIFY / ACTIVE: <b>{len(verified)}</b> · "
        f"TỔNG ACCOUNT ĐÃ ĐĂNG KÝ: <b>{len(rows)}</b></div>",
        "<hr><div style='font-family:monospace;font-size:12px;line-height:1.7'>"
    ]
    if not verified:
        parts.append("🟡 Chưa có Profile nào được VERIFY thành công.<br>")
    else:
        for row in verified:
            aid = html.escape(str(row["account_id"] or ""))
            label = html.escape(str(row["account_label"] or "") or "-")
            profile = html.escape(str(row["session_profile"] or "") or "-")
            health = html.escape(str(row["session_health"] or "HEALTHY"))
            parts.append(
                f"<b>🟢 {aid}</b> | SESSION={health} | LABEL={label}<br>"
                f"&nbsp;&nbsp;📁 {profile}<br>"
            )
    parts.append("</div></div>")
    workflow_cmd_log("FLOW_POOL", "VERIFIED_PROFILE_CHECK", verified=len(verified), total=len(rows))
    return "".join(parts), (
        f"🟢 Đã kiểm tra Profile: {len(verified)} Profile đã VERIFY/ACTIVE. "
        "Không tự mở Chrome và không tạo Profile mới."
        if verified else
        "🟡 Chưa có Profile nào ở trạng thái VERIFY/ACTIVE."
    )


def _creative_account_pool_html(limit=100):
    """Render a bounded account table; never dump 1000+ rows into the Gradio DOM."""
    _creative_v3_migrate()
    _flow_reconcile_profile_registry()
    summary = _creative_account_pool_summary()
    limit = max(10, min(int(limit or 100), 500))
    with _batch_db() as db:
        rows = db.execute(
            "SELECT account_id,status,usage,quota,cooldown_until,session_health,last_health_check "
            "FROM creative_accounts WHERE provider='GOOGLE_FLOW' "
            "ORDER BY account_id LIMIT ?", (limit,)
        ).fetchall()
    parts = [
        "<div class='batch-center-card'>",
        "<b>🌐 GOOGLE FLOW ACCOUNT POOL</b>",
        f"<div>TỔNG <b>{summary['total']}</b> · 🟢 ACTIVE <b>{summary['active']}</b> · "
        f"🟡 BUSY <b>{summary['busy']}</b> · 🟠 COOLDOWN <b>{summary['cooldown']}</b> · "
        f"🔴 CAPTCHA <b>{summary['captcha']}</b> · ⚫ AUTH <b>{summary['auth_error']}</b></div>",
        "<hr><div>HIỂN THỊ TỐI ĐA 100 ACCOUNT ĐẦU. Pool vẫn quản lý toàn bộ account.</div>"
    ]
    if not rows:
        parts.append("<div>🟡 Chưa có Account Google Flow.</div>")
    else:
        parts.append(
            "<div style='font-family:monospace;font-size:12px;line-height:1.65'>"
            "<b>ACCOUNT | STATUS | USAGE | QUOTA | SESSION</b><br>"
        )
        for row in rows:
            parts.append(
                f"{html.escape(str(row['account_id']))} | "
                f"{html.escape(str(row['status']))} | "
                f"{int(row['usage'] or 0)} | "
                f"{html.escape(str(row['quota'] or 'UNKNOWN'))} | "
                f"{html.escape(str(row['session_health'] or 'UNKNOWN'))}<br>"
            )
        parts.append("</div>")
    parts.append("</div>")
    return "".join(parts)


def _creative_import_flow_accounts(raw_text, replace=False):
    """Bulk-register existing Flow accounts. New accounts start DISABLED."""
    _creative_v3_migrate()
    raw = str(raw_text or "").strip()
    if not raw:
        return _creative_account_pool_html(), "🔴 Chưa có danh sách Account để import."

    records = []
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            parsed = parsed.get("accounts", [])
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, str):
                    records.append((item.strip(), ""))
                elif isinstance(item, dict):
                    aid = str(item.get("account_id") or item.get("email") or item.get("id") or "").strip()
                    profile = str(item.get("profile_path") or item.get("session_profile") or "").strip()
                    if aid:
                        records.append((aid, profile))
    except Exception:
        pass

    if not records:
        for line in raw.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "|" in line:
                aid, profile = line.split("|", 1)
            elif "," in line:
                aid, profile = line.split(",", 1)
            else:
                aid, profile = line, ""
            aid = str(aid).strip()
            profile = str(profile).strip()
            if aid:
                records.append((aid, profile))

    unique = {}
    for aid, profile in records:
        key = safe_name(aid)
        if key:
            unique[key] = (aid, profile)
    records = list(unique.values())[:5000]
    now = _creative_now()
    added = 0
    updated = 0
    with _batch_db() as db:
        if replace:
            db.execute("DELETE FROM creative_accounts WHERE provider='GOOGLE_FLOW'")
        for aid, profile in records:
            existing = db.execute(
                "SELECT account_id FROM creative_accounts WHERE account_id=? AND provider='GOOGLE_FLOW'",
                (aid,)
            ).fetchone()
            profile_path = str(Path(profile).expanduser().resolve()) if profile else str(
                APP_STATE_DIR / "flow_profiles" / safe_name(aid)
            )
            if existing:
                db.execute(
                    "UPDATE creative_accounts SET session_profile=?,updated_at=? WHERE account_id=?",
                    (profile_path, now, aid)
                )
                updated += 1
            else:
                db.execute(
                    "INSERT INTO creative_accounts(account_id,provider,status,session_profile,created_at,updated_at) "
                    "VALUES(?,?,?,?,?,?)",
                    (aid, "GOOGLE_FLOW", "DISABLED", profile_path, now, now)
                )
                added += 1
    workflow_cmd_log("FLOW_POOL", "BULK_IMPORT", total=len(records), added=added, updated=updated, replace=bool(replace))
    return _creative_account_pool_html(), (
        f"🟢 Import Pool hoàn tất · {len(records)} dòng · ➕ {added} mới · ♻️ {updated} cập nhật · "
        "Account mới ở DISABLED cho tới khi session thật được VERIFY."
    )


def _creative_flow_account_open(account_id):
    """Open/re-open the SAME registered Flow profile; UI path uses no Playwright API."""
    aid = str(account_id or "").strip()
    if not aid:
        return _creative_account_pool_html(), "🔴 Chưa chọn Account."
    row = _creative_get_account(aid)
    if not row:
        return _creative_account_pool_html(), f"🔴 Không tìm thấy Account: {aid}"
    try:
        profile = str(row.get("session_profile") or "").strip()
        if not profile:
            raise RuntimeError(f"{aid} chưa có session_profile.")
        p = Path(profile).expanduser().resolve()
        p.mkdir(parents=True, exist_ok=True)
        port = int(row.get("debug_port") or _flow_debug_port(aid))
        if not _flow_cdp_ready(port, timeout=1.0):
            _flow_launch_stable_chrome(aid, str(p), GoogleFlowAdapter.FLOW_URL)
            if not _flow_cdp_ready(port, timeout=15):
                raise RuntimeError(f"Chrome Stable đã mở nhưng CDP chưa sẵn sàng: {aid}")
        workflow_cmd_log("FLOW_POOL", "SESSION_OPEN_OR_REOPEN", account_id=aid, profile=str(p), port=port)
        return _creative_account_pool_html(), (
            f"🟡 Đã mở/re-open {aid} bằng đúng Profile cũ.\n"
            "Đăng nhập Google/Flow hoặc xử lý CAPTCHA thủ công, sau đó bấm VERIFY PROFILE."
        )
    except Exception as exc:
        _creative_set_account_state(aid, "AUTH_ERROR", str(exc)[:1500])
        workflow_cmd_log("FLOW_POOL", "SESSION_OPEN_ERROR", account_id=aid, detail=str(exc)[:1000], level="ERROR")
        return _creative_account_pool_html(), f"🔴 Không mở/re-open được session {aid}: {str(exc)[:700]}"


def _creative_flow_account_verify(account_id):
    """Verify one real Playwright Flow session and promote it to ACTIVE only if healthy."""
    aid = str(account_id or "").strip()
    if not aid:
        return _creative_account_pool_html(), "🔴 Chưa chọn Account."
    if not _creative_get_account(aid):
        return _creative_account_pool_html(), f"🔴 Không tìm thấy Account: {aid}"
    try:
        global _CREATIVE_FLOW_ADAPTER
        if _CREATIVE_FLOW_ADAPTER is None:
            _CREATIVE_FLOW_ADAPTER = GoogleFlowAdapter()
        result = _CREATIVE_FLOW_ADAPTER.health_check(aid)
        ok = bool(result.get("ok")) if isinstance(result, dict) else bool(result)
        detail = result.get("detail", "") if isinstance(result, dict) else str(result)
        if ok:
            _creative_set_account_state(aid, "ACTIVE", None, None)
            with _batch_db() as db:
                db.execute(
                    "UPDATE creative_accounts SET session_health='HEALTHY',last_health_check=?,updated_at=? WHERE account_id=?",
                    (_creative_now(), _creative_now(), aid)
                )
            workflow_cmd_log("FLOW_POOL", "ACCOUNT_ACTIVE", account_id=aid, detail=detail)
            return _creative_account_pool_html(), f"🟢 {aid} ACTIVE · {detail}"
        status = "CAPTCHA" if "captcha" in str(detail).lower() else "AUTH_ERROR"
        _creative_set_account_state(aid, status, detail[:1500])
        return _creative_account_pool_html(), f"🔴 {aid} chưa ACTIVE · {detail}"
    except Exception as exc:
        _creative_set_account_state(aid, "AUTH_ERROR", str(exc)[:1500])
        return _creative_account_pool_html(), f"🔴 VERIFY lỗi {aid}: {str(exc)[:700]}"


def _creative_flow_pool_verify_registered():
    """Verify only sessions already open in the adapter; do not spawn 1000 browsers."""
    _creative_v3_migrate()
    global _CREATIVE_FLOW_ADAPTER
    if _CREATIVE_FLOW_ADAPTER is None:
        return _creative_account_pool_html(), (
            "🟡 Chưa có Playwright Flow session đang mở. Dùng MỞ SESSION cho Account cần kích hoạt."
        )
    checked = 0
    active = 0
    for aid in list(_CREATIVE_FLOW_ADAPTER._contexts.keys()):
        checked += 1
        result = _CREATIVE_FLOW_ADAPTER.health_check(aid)
        ok = bool(result.get("ok")) if isinstance(result, dict) else bool(result)
        detail = result.get("detail", "") if isinstance(result, dict) else str(result)
        if ok:
            _creative_set_account_state(aid, "ACTIVE", None, None)
            active += 1
        else:
            _creative_set_account_state(aid, "CAPTCHA" if "captcha" in str(detail).lower() else "AUTH_ERROR", detail[:1500])
    return _creative_account_pool_html(), (
        f"🟢 VERIFY session đang mở: {checked} · ACTIVE: {active}. Không tự mở 1000 browser cùng lúc."
    )


# ---------- Flow account onboarding: profile -> manual login -> persist session ----------
FLOW_ACCOUNT_ROOT = APP_STATE_DIR / "flow_accounts"
FLOW_ACCOUNT_ROOT.mkdir(parents=True, exist_ok=True)
FLOW_PROFILE_ROOT = APP_STATE_DIR / "flow_profiles"
FLOW_PROFILE_ROOT.mkdir(parents=True, exist_ok=True)

def _flow_account_schema():
    """Add local-only identity/session fields without breaking existing V9 databases."""
    _creative_v3_migrate()
    with _batch_db() as db:
        cols = {str(r[1]) for r in db.execute("PRAGMA table_info(creative_accounts)").fetchall()}
        if "account_label" not in cols:
            db.execute("ALTER TABLE creative_accounts ADD COLUMN account_label TEXT")
        if "cookie_file" not in cols:
            db.execute("ALTER TABLE creative_accounts ADD COLUMN cookie_file TEXT")
        if "debug_port" not in cols:
            db.execute("ALTER TABLE creative_accounts ADD COLUMN debug_port INTEGER")

def _flow_next_profile_id():
    """Return the lowest reusable FLOW id and reconcile deleted local profiles.

    If the user deletes FLOW-0001's profile folder from LocalAIStudio_Data,
    that old DB row must not permanently reserve the number. We remove stale
    rows whose registered profile path no longer exists, then choose the
    smallest free FLOW-NNNN instead of max(existing)+1.
    """
    _flow_account_schema()
    global _CREATIVE_FLOW_ADAPTER
    with _batch_db() as db:
        rows = db.execute(
            "SELECT account_id,session_profile,status FROM creative_accounts "
            "WHERE provider='GOOGLE_FLOW' AND account_id LIKE 'FLOW-%'"
        ).fetchall()

        stale = []
        used = set()
        for row in rows:
            aid = str(row["account_id"] or "")
            m = re.fullmatch(r"FLOW-(\d+)", aid)
            if not m:
                continue
            n = int(m.group(1))
            profile = str(row["session_profile"] or "").strip()

            # A manually/custom supplied profile is considered deleted when
            # its registered directory is gone. Do not remove a live session
            # that is still held by the in-memory adapter.
            live = False
            try:
                if _CREATIVE_FLOW_ADAPTER is not None:
                    item = _CREATIVE_FLOW_ADAPTER._contexts.get(aid)
                    if item:
                        _pw, browser, page = item
                        live = bool(browser.is_connected()) and not page.is_closed()
            except Exception:
                live = False

            if profile and not Path(profile).expanduser().exists() and not live:
                stale.append(aid)
            else:
                used.add(n)

        if stale:
            placeholders = ",".join("?" for _ in stale)
            db.execute(
                f"DELETE FROM creative_accounts WHERE provider='GOOGLE_FLOW' "
                f"AND account_id IN ({placeholders})",
                stale,
            )

        n = 1
        while n in used:
            n += 1
        return f"FLOW-{n:04d}"

def _flow_storage_paths(account_id):
    safe = safe_name(account_id)
    root = FLOW_ACCOUNT_ROOT / safe
    root.mkdir(parents=True, exist_ok=True)
    return root, root / "storage_state.json"

def _flow_detect_account_identity(page):
    """Best-effort identity label from the signed-in Flow/Google UI."""
    try:
        body = page.locator("body").inner_text(timeout=3000) or ""
    except Exception:
        body = ""
    emails = re.findall(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", body, flags=re.I)
    return emails[0].strip().lower() if emails else ""

def _flow_persist_storage_state(account_id, browser=None, page=None):
    """Persist a Playwright storage_state snapshot under the tool-owned account folder."""
    root, cookie_file = _flow_storage_paths(account_id)
    try:
        if browser is not None:
            browser.storage_state(path=str(cookie_file))
        elif page is not None:
            page.context.storage_state(path=str(cookie_file))
        else:
            return False, "Không có browser/page để lưu session."
        return True, str(cookie_file)
    except Exception as exc:
        return False, str(exc)

def _flow_debug_port(account_id):
    """Stable private CDP port per tool-owned Flow profile."""
    n = int(hashlib.sha256(str(account_id).encode("utf-8")).hexdigest()[:8], 16)
    return 20000 + (n % 18000)

def _flow_chrome_executable():
    candidates = [
        Path(os.environ.get("PROGRAMFILES", "")) / "Google/Chrome/Application/chrome.exe",
        Path(os.environ.get("PROGRAMFILES(X86)", "")) / "Google/Chrome/Application/chrome.exe",
        Path(os.environ.get("LOCALAPPDATA", "")) / "Google/Chrome/Application/chrome.exe",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None

def _flow_launch_stable_chrome(account_id, profile, url):
    """Launch real installed Chrome with a tool-owned profile; never Chrome Testing."""
    exe = _flow_chrome_executable()
    if not exe:
        raise RuntimeError("Không tìm thấy Google Chrome Stable. Tool không cài Chrome Testing.")
    port = _flow_debug_port(account_id)
    # Chrome 136+ requires remote debugging to target a non-default user-data-dir.
    Path(profile).mkdir(parents=True, exist_ok=True)
    args = [
        str(exe),
        f"--user-data-dir={str(Path(profile).resolve())}",
        f"--remote-debugging-port={port}",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-background-networking",
        "--new-window",
        str(url),
    ]
    flags = 0
    if os.name == "nt":
        flags = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    proc = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            creationflags=flags)
    with _batch_db() as db:
        db.execute("UPDATE creative_accounts SET debug_port=?,updated_at=? WHERE account_id=?",
                   (port, _creative_now(), account_id))
    workflow_cmd_log("FLOW_POOL", "CHROME_STABLE_LAUNCHED", account_id=account_id, port=port, pid=proc.pid)
    return port, proc.pid

def _flow_cdp_ready(port, timeout=8):
    deadline = time.monotonic() + float(timeout)
    url = f"http://127.0.0.1:{int(port)}/json/version"
    while time.monotonic() < deadline:
        try:
            r = requests.get(url, timeout=0.6)
            if r.ok:
                return True
        except Exception:
            pass
        time.sleep(0.2)
    return False

def _flow_create_manual_profile(profile_path=""):
    """Create/register exactly one NEW profile; re-opening uses account_open instead."""
    _flow_account_schema()
    aid = _flow_next_profile_id()
    p = Path(str(profile_path).strip()).expanduser().resolve() if str(profile_path or "").strip() else FLOW_PROFILE_ROOT / safe_name(aid)
    p.mkdir(parents=True, exist_ok=True)
    now = _creative_now()
    with _batch_db() as db:
        port = _flow_debug_port(aid)
        db.execute(
            "INSERT INTO creative_accounts(account_id,provider,status,session_profile,account_label,cookie_file,debug_port,created_at,updated_at) "
            "VALUES(?,?,?,?,?,?,?,?,?)",
            (aid, "GOOGLE_FLOW", "DISABLED", str(p), "", str(_flow_storage_paths(aid)[1]), port, now, now)
        )
    workflow_cmd_log("FLOW_POOL", "PROFILE_REGISTERED_NEW", account_id=aid, profile=str(p))
    return aid, str(p)

def _creative_flow_manual_profile_open(profile_path=""):
    """Register a profile immediately; open Chrome in a background worker."""
    try:
        ok, detail = _playwright_chrome_runtime_check()
        if not ok:
            return _creative_account_pool_html(), detail, ""
        aid, profile = _flow_create_manual_profile(profile_path)
        def _open_worker():
            try:
                port, pid = _flow_launch_stable_chrome(aid, profile, GoogleFlowAdapter.FLOW_URL)
                if _flow_cdp_ready(port):
                    workflow_cmd_log("FLOW_POOL", "PROFILE_BROWSER_READY", account_id=aid, profile=profile, port=port, pid=pid)
                else:
                    _creative_set_account_state(aid, "AUTH_ERROR", "Chrome đã mở nhưng CDP chưa sẵn sàng.")
            except Exception as exc:
                _creative_set_account_state(aid, "AUTH_ERROR", str(exc)[:1500])
                workflow_cmd_log("FLOW_POOL", "PROFILE_BROWSER_ERROR", account_id=aid, detail=str(exc)[:1000], level="ERROR")
        threading.Thread(target=_open_worker, name=f"VHUNG-FlowProfile-{aid}", daemon=True).start()
        return _creative_account_pool_html(), (
            f"🟡 Đã tạo {aid} · Profile: {profile}\n\n"
            "⏳ Chrome đang mở ở nền, giao diện không bị khóa. "
            "Khi cửa sổ Chrome hiện ra → tự đăng nhập Google/Flow → bấm VERIFY PROFILE."
        ), aid
    except Exception as exc:
        return _creative_account_pool_html(), f"🔴 Không mở được Profile: {str(exc)[:1000]}", ""

async def _creative_flow_manual_profile_verify_async_impl(account_id):
    """VERIFY Flow bằng Playwright ASYNC API, tuyệt đối không dùng Sync API trong callback."""
    aid = str(account_id or "").strip()
    if not aid:
        return _creative_account_pool_html(), "🔴 Chưa có Profile/Account đang kiểm tra."
    _flow_account_schema()
    row = _creative_get_account(aid)
    if not row:
        return _creative_account_pool_html(), f"🔴 Không tìm thấy {aid}."

    profile = str((row or {}).get("session_profile") or "").strip()
    if not profile:
        return _creative_account_pool_html(), f"🔴 {aid} chưa có thư mục Profile."

    port = int((row or {}).get("debug_port") or _flow_debug_port(aid))
    try:
        if not _flow_cdp_ready(port, timeout=3):
            p = Path(profile).expanduser().resolve()
            p.mkdir(parents=True, exist_ok=True)
            _flow_launch_stable_chrome(aid, str(p), GoogleFlowAdapter.FLOW_URL)
            if not _flow_cdp_ready(port, timeout=15):
                raise RuntimeError(f"Chrome đã được gọi mở lại nhưng CDP chưa sẵn sàng cho {aid}.")

        # IMPORTANT: VERIFY uses ONLY Playwright Async API.
        # It is intentionally independent from any Sync Playwright objects
        # stored in GoogleFlowAdapter._contexts. This removes the
        # "Sync API inside the asyncio loop" failure completely.
        from playwright.async_api import async_playwright

        async with async_playwright() as pw:
            browser = await pw.chromium.connect_over_cdp(
                f"http://127.0.0.1:{port}", timeout=15000
            )
            contexts = browser.contexts
            if not contexts:
                raise RuntimeError(f"Chrome CDP connected nhưng không có BrowserContext: {aid}")
            context = contexts[0]
            pages = context.pages
            page = pages[0] if pages else await context.new_page()

            if not page.url or "flow" not in page.url.lower():
                try:
                    await page.goto(
                        GoogleFlowAdapter.FLOW_URL,
                        wait_until="domcontentloaded",
                        timeout=60000,
                    )
                except Exception:
                    pass

            body = ""
            try:
                body = (await page.locator("body").inner_text(timeout=5000) or "").lower()
            except Exception:
                pass

            captcha_markers = (
                "captcha", "verify you are human", "unusual traffic", "security check"
            )
            if any(x in body for x in captcha_markers):
                _creative_set_account_state(
                    aid, "CAPTCHA", "Google Flow yêu cầu CAPTCHA."
                )
                return _creative_account_pool_html(), (
                    f"🔴 {aid} chưa ACTIVE · CAPTCHA đang yêu cầu xử lý."
                )

            # Do not classify a Flow page as AUTH_ERROR just because the UI
            # contains generic words such as "Sign in". Flow can render account
            # menus/dialogs containing those words while the actual session is
            # authenticated. Only treat it as logged out when we are clearly on
            # Google's login endpoint or a page whose visible content is a
            # dedicated sign-in screen and is NOT the Flow app.
            current_url = str(page.url or "").lower()
            on_google_login = (
                "accounts.google.com" in current_url
                and ("signin" in current_url or "login" in current_url)
            )
            dedicated_login = (
                any(x in body for x in (
                    "sign in to continue", "choose an account", "use another account"
                ))
                and "labs.google" not in current_url
                and "flow" not in current_url
            )
            if on_google_login or dedicated_login:
                detail = "Flow session chưa đăng nhập Google/Flow."
                _creative_set_account_state(aid, "AUTH_ERROR", detail)
                return _creative_account_pool_html(), f"🔴 {aid} chưa ACTIVE · {detail}"

            identity = ""
            try:
                text_body = await page.locator("body").inner_text(timeout=3000) or ""
                emails = re.findall(
                    r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
                    text_body,
                    flags=re.I,
                )
                identity = emails[0].strip().lower() if emails else ""
            except Exception:
                pass

            root, cookie_file = _flow_storage_paths(aid)
            try:
                await context.storage_state(path=str(cookie_file))
            except Exception as exc:
                _creative_set_account_state(
                    aid, "AUTH_ERROR", f"Không lưu được session: {exc}"
                )
                return _creative_account_pool_html(), (
                    f"🔴 Session mở được nhưng không lưu được cookie/session: {exc}"
                )

            now = _creative_now()
            label = identity or str((row or {}).get("account_label") or aid)
            # Persist the browser-backed session as the source of truth. The
            # Chrome user-data-dir remains the durable login store; the
            # storage_state file is an additional recovery snapshot.
            profile_path = str(Path(profile).expanduser().resolve())
            if not Path(profile_path).exists():
                raise RuntimeError(f"Profile folder không tồn tại sau VERIFY: {profile_path}")
            with _batch_db() as db:
                db.execute(
                    "UPDATE creative_accounts SET status='ACTIVE',account_label=?,"
                    "cookie_file=?,session_health='HEALTHY',last_health_check=?,"
                    "last_success=?,last_error=NULL,updated_at=? WHERE account_id=?",
                    (label, str(cookie_file), now, now, now, aid),
                )

            workflow_cmd_log(
                "FLOW_POOL", "PROFILE_VERIFIED_ACTIVE",
                account_id=aid, identity=label, cookie_file=str(cookie_file)
            )
            return _creative_account_pool_html(), (
                f"🟢 {aid} → ACTIVE · Account: {label}\\n"
                f"🍪 Session/cookie snapshot đã lưu trong tool: {cookie_file}"
            )
    except Exception as exc:
        detail = str(exc)[:1500]
        _creative_set_account_state(aid, "AUTH_ERROR", detail)
        return _creative_account_pool_html(), f"🔴 VERIFY PROFILE lỗi {aid}: {detail[:1000]}"

async def _creative_flow_manual_profile_verify_async(*args, **kwargs):
    _result = await _creative_flow_manual_profile_verify_async_impl(*args, **kwargs)
    try:
        import inspect as _inspect
        _sig = _inspect.signature(_creative_flow_manual_profile_verify_async_impl)
        _bound = _sig.bind_partial(*args, **kwargs)
        _account_id = (
            _bound.arguments.get("account_id")
            or _bound.arguments.get("account")
            or _bound.arguments.get("account_name")
        )
        _profile = (
            _bound.arguments.get("profile")
            or _bound.arguments.get("profile_path")
            or _bound.arguments.get("session_profile")
        )
        _worker = _bound.arguments.get("worker_id")
        _ok = bool(_result)
        if isinstance(_result, dict):
            _ok = bool(_result.get("ok") or _result.get("verified") or _result.get("success"))
            _account_id = _account_id or _result.get("account_id")
            _profile = _profile or _result.get("profile_path")
            _worker = _worker or _result.get("worker_id")
        elif isinstance(_result, (tuple, list)):
            _ok = bool(_result[0]) if _result else False
        if _ok:
            _creative_activate_verified_flow_account(
                account_id=_account_id,
                profile_path=_profile,
                worker_id=_worker,
            )
    except Exception as _exc:
        try:
            workflow_cmd_log(
                "FLOW",
                "PROFILE_VERIFY_WRAPPER_ERROR",
                error=str(_exc)[:500],
            )
        except Exception:
            pass
    return _result

async def _creative_flow_manual_profile_verify_core(account_id):
    """Stable UI-facing implementation; owns the real async Playwright verifier."""
    return await _creative_flow_manual_profile_verify_async_impl(account_id)

async def _creative_flow_manual_profile_verify_impl(account_id):
    """Compatibility entrypoint. Never self-calls; delegates to the unique core."""
    return await _creative_flow_manual_profile_verify_core(account_id)

async def _creative_flow_manual_profile_verify(*args, **kwargs):
    _result = await _creative_flow_manual_profile_verify_impl(*args, **kwargs)
    try:
        import inspect as _inspect
        _sig = _inspect.signature(_creative_flow_manual_profile_verify_impl)
        _bound = _sig.bind_partial(*args, **kwargs)
        _account_id = (
            _bound.arguments.get("account_id")
            or _bound.arguments.get("account")
            or _bound.arguments.get("account_name")
        )
        _profile = (
            _bound.arguments.get("profile")
            or _bound.arguments.get("profile_path")
            or _bound.arguments.get("session_profile")
        )
        _worker = _bound.arguments.get("worker_id")
        _ok = bool(_result)
        if isinstance(_result, dict):
            _ok = bool(_result.get("ok") or _result.get("verified") or _result.get("success"))
            _account_id = _account_id or _result.get("account_id")
            _profile = _profile or _result.get("profile_path")
            _worker = _worker or _result.get("worker_id")
        elif isinstance(_result, (tuple, list)):
            _ok = bool(_result[0]) if _result else False
        if _ok:
            _creative_activate_verified_flow_account(
                account_id=_account_id,
                profile_path=_profile,
                worker_id=_worker,
            )
    except Exception as _exc:
        try:
            workflow_cmd_log(
                "FLOW",
                "PROFILE_VERIFY_WRAPPER_ERROR",
                error=str(_exc)[:500],
            )
        except Exception:
            pass
    return _result

def _creative_account_pool_html(limit=100):
    """Render account pool with local profile/identity state."""
    _flow_account_schema()
    summary = _creative_account_pool_summary()
    limit = max(10, min(int(limit or 100), 500))
    with _batch_db() as db:
        rows = db.execute(
            "SELECT account_id,status,usage,quota,session_health,account_label,session_profile,debug_port "
            "FROM creative_accounts WHERE provider='GOOGLE_FLOW' ORDER BY account_id LIMIT ?", (limit,)
        ).fetchall()
    parts = [
        "<div class='batch-center-card'>",
        "<b>🌐 GOOGLE FLOW ACCOUNT POOL</b>",
        f"<div>TỔNG <b>{summary['total']}</b> · 🟢 ACTIVE <b>{summary['active']}</b> · "
        f"🟡 BUSY <b>{summary['busy']} · 🟠 COOLDOWN <b>{summary['cooldown']} · "
        f"🔴 CAPTCHA <b>{summary['captcha']} · ⚫ AUTH <b>{summary['auth_error']}</b></div>",
        "<hr><div>Profile được mở bằng Playwright. Đăng nhập Google/Flow thủ công. "
        "Sau VERIFY, tool lưu session/cookie snapshot trong thư mục tool.</div>"
    ]
    if not rows:
        parts.append("<div>🟡 Chưa có Profile Google Flow.</div>")
    else:
        parts.append("<div style='font-family:monospace;font-size:12px;line-height:1.65'><b>PROFILE | ACCOUNT | STATUS | SESSION</b><br>")
        for row in rows:
            parts.append(
                f"{html.escape(str(row['account_id']))} | {html.escape(str(row['account_label'] or 'chưa nhận diện'))} | "
                f"{html.escape(str(row['status']))} | {html.escape(str(row['session_health'] or 'UNKNOWN'))}<br>"
            )
        parts.append("</div>")
    parts.append("</div>")
    return "".join(parts)


# ============================================================
# PLAYWRIGHT / GOOGLE CHROME RUNTIME V11
# ============================================================
# Flow uses the user's installed stable Chrome. We deliberately do NOT install
# or launch Playwright's bundled Chromium ("Chrome for Testing").
# Package installation is explicit and runs in a background thread so Gradio
# never freezes while pip is working.

_PLAYWRIGHT_INSTALL_LOCK = threading.RLock()
_PLAYWRIGHT_INSTALL_RUNNING = False
_PLAYWRIGHT_INSTALL_MESSAGE = "⚪ Chưa kiểm tra Playwright / Google Chrome."

def _playwright_runtime_candidates():
    """Return Playwright Python package roots bundled with this tool/legacy tool.

    A valid candidate is the directory named playwright containing
    driver/package/cli.js. The parent of that directory is added to
    PYTHONPATH/sys.path so Playwright uses its matching driver package.
    """
    candidates = []
    env = str(os.getenv("VHUNG_PLAYWRIGHT_RUNTIME") or "").strip()
    if env:
        candidates.append(Path(env).expanduser())
    candidates.extend([
        BASE / "playwright",
        BASE / "tool" / "playwright",
        BASE / "runtime" / "playwright",
        BASE / "tool" / "runtime" / "playwright",
    ])
    # Legacy CloneVoice portable runtime commonly lives under the Windows
    # user's Downloads folder; detect it without hard-coding a username.
    try:
        home = Path.home()
        candidates.extend([
            home / "Downloads" / "CloneVoice-1.1.18-win64" / "playwright",
            home / "Downloads" / "CloneVoice-1.1.18-win64" / "CloneVoice-1.1.18-win64" / "playwright",
        ])
    except Exception:
        pass
    for root in (BASE, *list(BASE.parents)[:4]):
        candidates.extend([
            root / "CloneVoice-1.1.18-win64" / "playwright",
            root / "CloneVoice-1.1.18-win64" / "CloneVoice-1.1.18-win64" / "playwright",
        ])
    out = []
    seen = set()
    for raw in candidates:
        try:
            p = Path(raw).resolve()
            if p.name.lower() == "package" and p.parent.name.lower() == "driver":
                p = p.parent.parent
            elif p.name.lower() == "driver":
                p = p.parent
            key = str(p).lower()
            if key in seen:
                continue
            seen.add(key)
            if (p / "driver" / "package" / "cli.js").is_file() and (p / "__init__.py").is_file():
                out.append(p)
        except Exception:
            continue
    return out


def _playwright_prepare_runtime():
    """Prefer a portable Playwright runtime when present."""
    roots = _playwright_runtime_candidates()
    if roots:
        root = roots[0]
        parent = str(root.parent)
        if parent not in sys.path:
            sys.path.insert(0, parent)
        os.environ["VHUNG_PLAYWRIGHT_ACTIVE_ROOT"] = str(root)
        return str(root)
    return ""


def _playwright_chrome_runtime_check():
    """Probe Playwright outside Gradio asyncio and support bundled old-tool driver."""
    try:
        runtime_root = _playwright_prepare_runtime()
        import importlib.util
        spec = importlib.util.find_spec("playwright")
        if spec is None:
            return False, (
                "🔴 Không tìm thấy Playwright. Đặt thư mục playwright cũ vào "
                "tool\\playwright hoặc đặt VHUNG_PLAYWRIGHT_RUNTIME tới thư mục playwright."
            )

        probe_code = (
            "from playwright.sync_api import sync_playwright\n"
            "pw=sync_playwright().start()\n"
            "pw.stop()\n"
            "print('PLAYWRIGHT_SYNC_OK')\n"
        )
        env = os.environ.copy()
        if runtime_root:
            parent = str(Path(runtime_root).parent)
            old_pp = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = parent + (os.pathsep + old_pp if old_pp else "")
        try:
            proc = subprocess.run(
                [sys.executable, "-c", probe_code],
                capture_output=True,
                text=True,
                timeout=30,
                env=env,
                creationflags=(subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0),
            )
        except subprocess.TimeoutExpired:
            return False, "🔴 Playwright runtime probe timeout (>30s)."

        if proc.returncode != 0 or "PLAYWRIGHT_SYNC_OK" not in (proc.stdout or ""):
            err = (proc.stderr or proc.stdout or "unknown error").strip()
            return False, f"🔴 Playwright runtime lỗi: {err[:900]}"

        chrome = _flow_chrome_executable() if callable(globals().get("_flow_chrome_executable")) else None
        if chrome is None:
            return False, "🔴 Không tìm thấy Google Chrome Stable."

        source = f"PORTABLE {runtime_root}" if runtime_root else "PYTHON PACKAGE"
        return True, f"🟢 Playwright OK · {source} · Chrome Stable OK · {chrome}"
    except Exception as exc:
        return False, f"🔴 Playwright runtime lỗi: {str(exc)[:900]}"

def _playwright_install_worker():
    global _PLAYWRIGHT_INSTALL_RUNNING, _PLAYWRIGHT_INSTALL_MESSAGE
    with _PLAYWRIGHT_INSTALL_LOCK:
        _PLAYWRIGHT_INSTALL_RUNNING = True
        _PLAYWRIGHT_INSTALL_MESSAGE = "🟡 Đang cài/sửa Python Playwright ở nền..."
    try:
        workflow_cmd_log("PLAYWRIGHT", "INSTALL_START")
        proc = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--upgrade", "playwright"],
            capture_output=True, text=True, timeout=900,
            creationflags=(subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0),
        )
        if proc.returncode != 0:
            msg = (proc.stderr or proc.stdout or "pip install failed")[-1800:]
            _PLAYWRIGHT_INSTALL_MESSAGE = f"🔴 Cài Playwright thất bại: {msg}"
            workflow_cmd_log("PLAYWRIGHT", "INSTALL_FAILED", detail=msg, level="ERROR")
            return
        ok, detail = _playwright_chrome_runtime_check()
        if not ok:
            _PLAYWRIGHT_INSTALL_MESSAGE = (
                "🟠 Đã cài Python Playwright nhưng Chrome Stable chưa sẵn sàng. "
                "Tool KHÔNG cài Chromium/Chrome Testing.\n\n" + detail
            )
            workflow_cmd_log("PLAYWRIGHT", "PACKAGE_READY_BROWSER_MISSING", detail=detail, level="WARN")
        else:
            _PLAYWRIGHT_INSTALL_MESSAGE = detail
            workflow_cmd_log("PLAYWRIGHT", "READY", detail=detail)
    except Exception as exc:
        _PLAYWRIGHT_INSTALL_MESSAGE = f"🔴 Lỗi cài Playwright: {str(exc)[:1200]}"
        workflow_cmd_log("PLAYWRIGHT", "INSTALL_EXCEPTION", detail=str(exc)[:1200], level="ERROR")
    finally:
        with _PLAYWRIGHT_INSTALL_LOCK:
            _PLAYWRIGHT_INSTALL_RUNNING = False

def _playwright_install_ui():
    global _PLAYWRIGHT_INSTALL_RUNNING
    with _PLAYWRIGHT_INSTALL_LOCK:
        if _PLAYWRIGHT_INSTALL_RUNNING:
            return "🟡 Playwright đang được cài/sửa ở nền — giao diện không bị khóa."
        _PLAYWRIGHT_INSTALL_RUNNING = True
    threading.Thread(
        target=_playwright_install_worker,
        name="VHUNG-Playwright-Installer",
        daemon=True,
    ).start()
    return "🟡 Đã bắt đầu cài/sửa Playwright ở nền. Bấm KIỂM TRA RUNTIME sau khi xong."

def _playwright_runtime_ui():
    if _PLAYWRIGHT_INSTALL_RUNNING:
        return "🟡 Playwright installer đang chạy ở nền..."
    ok, detail = _playwright_chrome_runtime_check()
    return detail

_V9_FLOW_CONTEXT = GoogleFlowAdapter._context
def _flow_context_registered_profile(self, account_id):
    """Attach to the registered Stable Chrome profile from a CreativeWorker thread.
    Sync Playwright objects are never cached/shared across worker threads.
    """
    aid = str(account_id or "").strip()
    row = _creative_get_account(aid)
    profile = str((row or {}).get("session_profile") or "").strip()
    if not profile:
        raise RuntimeError(f"{aid} chưa có session_profile.")
    profile_path = Path(profile).expanduser().resolve()
    if not profile_path.exists():
        raise RuntimeError(f"{aid} Profile không tồn tại: {profile_path}")

    # Hard guard: this sync API is only legal in the dedicated worker thread.
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        raise RuntimeError(
            "Flow Playwright Sync API phải chạy trong CreativeWorker thread, "
            "không được chạy trong asyncio/Gradio callback."
        )

    ok, detail = _playwright_chrome_runtime_check()
    if not ok:
        raise RuntimeError(detail)

    port = int((row or {}).get("debug_port") or _flow_debug_port(aid))
    if not _flow_cdp_ready(port, timeout=2):
        _flow_launch_stable_chrome(aid, str(profile_path), self.FLOW_URL)
        if not _flow_cdp_ready(port, timeout=15):
            raise RuntimeError(
                f"Chrome đã mở lại nhưng CDP chưa sẵn sàng cho {aid} (port {port})."
            )
        workflow_cmd_log(
            "FLOW_POOL", "WORKER_PROFILE_REOPENED",
            account_id=aid, profile=str(profile_path), port=port
        )
    else:
        workflow_cmd_log(
            "FLOW_POOL", "WORKER_PROFILE_REUSED",
            account_id=aid, profile=str(profile_path), port=port, level="DEBUG"
        )

    from playwright.sync_api import sync_playwright
    pw = sync_playwright().start()
    try:
        browser = pw.chromium.connect_over_cdp(
            f"http://127.0.0.1:{port}", timeout=15000
        )
        contexts = browser.contexts
        if not contexts:
            raise RuntimeError(f"Chrome CDP không có BrowserContext: {aid}")
        context = contexts[0]
        pages = context.pages
        page = pages[0] if pages else context.new_page()
        if not page.url or "flow" not in page.url.lower():
            page.goto(self.FLOW_URL, wait_until="domcontentloaded", timeout=60000)
        workflow_cmd_log(
            "FLOW", "WORKER_SESSION_ATTACHED",
            account_id=aid, port=port, page_url=str(page.url)[:300]
        )
        return pw, browser, page
    except Exception:
        try:
            pw.stop()
        except Exception:
            pass
        raise

GoogleFlowAdapter._context = _flow_context_registered_profile

def launch_ui():
    _cleanup_gradio_temp(force=True)
    workflow_cmd_log('SUPERVISOR', 'STARTUP_HEALTH', project='', health=background_health_snapshot())
    hud_ok = start_workflow_status_server()
    # Selected engine is authoritative. Do not bootstrap/download Z Image when Google Flow is selected.
    if _selected_image_engine() == CREATIVE_ENGINE_ZIMAGE:
        threading.Thread(target=_zimage_startup_preflight,name="VHUNG-ZImage-StartupPreflight",daemon=True).start()
    """Full UI: Voice, Images, Model, Environment, Script and production workflow.

    Project storage is session-gated: opening the app never activates the last
    Project. The user must explicitly connect a Project before any project
    folders/files are initialized.
    """
    global PROJECT_CONNECTED
    PROJECT_CONNECTED.clear()
    need_setup = False
    projects_now = list_projects()
    # Do not auto-connect/recreate the last project on startup.
    # The user must explicitly press "🔗 Chọn & Kết nối Project".
    default_project = ''

    # Script-free workflow HUD: status is rendered by Python into gr.HTML.
    with gr.Blocks(title=APP_DISPLAY_NAME) as ui:
        if not hud_ok:
            gr.Markdown('⚠️ **WORKFLOW HUD OFFLINE:** cổng trạng thái cục bộ đang bị chiếm hoặc không thể mở. Workflow chính vẫn hoạt động; HUD sẽ không hiển thị cho tới khi cổng được giải phóng.', elem_id='workflow-hud-warning')
        # Create the QC project selector before any workflow event wiring.
        # The component is rendered later inside the QC tab, but must exist now
        # because Script / Voice callbacks reference it while launch_ui() builds.
        qc_project = gr.Dropdown(
            choices=projects_now,
            value=default_project or None,
            allow_custom_value=True,
            label='📁 Project đang kết nối',
            render=False,
        )
        if need_setup:
            gr.HTML(f'<div id="brand"><div class="brand">✦ {APP_DISPLAY_NAME}</div><div class="sub">Thiết lập lần đầu</div><div class="badge">● API KEY REQUIRED</div></div>')
            setup_key_box = gr.Textbox(type='password', label='Gemini API Key', placeholder='AIza...')
            with gr.Row():
                open_key_btn = gr.Button('🔑 Mở Google AI Studio')
                test_key_btn = gr.Button('✓ Kiểm tra')
                save_key_btn = gr.Button('💾 Lưu')
                enter_btn = gr.Button('🚀 Vào Workspace', variant='primary')
            setup_status = gr.Markdown()
            open_key_btn.click(_wrap_gradio_callback(_ui_open_key), outputs=[])
            test_key_btn.click(_wrap_gradio_callback(test_gemini_key), setup_key_box, setup_status)
            save_key_btn.click(_wrap_gradio_callback(setup_key), setup_key_box, setup_status)
            enter_btn.click(lambda key: enter_workspace(key), setup_key_box, setup_status, js="""(result) => { if (String(result || '').startsWith('✓')) setTimeout(() => window.location.reload(), 400); }""")
            return ui

        gr.HTML(f'''<div id="vhung-vn-clock-overlay" aria-label="Giờ Việt Nam">
  <div class="vhung-vn-clock-card">
    <div class="vhung-clock-scale">
    <div class="vhung-vn-clock-toolbar">
      <span class="vhung-clock-toolbar-title">🇻🇳 GIỜ VIỆT NAM</span>
      <button class="vhung-clock-logo-toggle" type="button" title="Ẩn / hiện logo">◉ LOGO</button>
      <button class="vhung-clock-hide" type="button" title="Ẩn đồng hồ">×</button>
    </div>
    <div class="vhung-vn-clock-logo-wrap" id="vhung-vn-clock-logo-wrap">
      <img class="vhung-vn-clock-logo" alt="VHƯNG PRO MAX" src="data:image/jpeg;base64,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">
    </div>
    <div class="vhung-vn-clock-brand">✦ VHƯNG PRO MAX</div>
    <div class="vhung-vn-clock-label">UTC+07:00 · Asia/Ho_Chi_Minh</div>
    <div class="vhung-vn-clock-time" id="vhung-vn-clock-time">00:00:00</div>
    <div class="vhung-vn-clock-date" id="vhung-vn-clock-date">30/09/2026</div>
    <div class="vhung-vn-clock-status" id="vhung-vn-clock-status"><span class="status-dot"></span><span class="status-text">Gemini API SẴN SÀNG · 0 key đã lưu</span></div>
    </div>
    <div class="vhung-vn-clock-resize" aria-label="Kéo để thay đổi kích thước toàn bộ bảng" title="Kéo góc để co giãn TOÀN BỘ bảng"></div>
  </div>
  <button id="vhung-vn-clock-reopen" type="button" title="Hiện lại đồng hồ">🕐</button>
</div>''')
        gr.HTML('''<style>
/* VHƯNG PRO MAX compact overlay: full-width frame, small working footprint, always fixed */
#vhung-brand-notice-wrap{
  position:fixed !important; inset:0 0 auto 0 !important; width:100vw !important;
  max-width:none !important; z-index:2147483647 !important; pointer-events:none !important;
}
#vhung-brand-notice-wrap .vhung-brand-shell{
  width:100vw !important; max-width:none !important; box-sizing:border-box !important;
  padding:7px 12px 9px !important; margin:0 !important;
  background:linear-gradient(180deg,rgba(255,255,255,.985),rgba(248,250,252,.96)) !important;
  border:2px solid rgba(14,165,233,.72) !important; border-left:0 !important; border-right:0 !important;
  border-radius:0 0 14px 14px !important;
  box-shadow:0 0 18px rgba(14,165,233,.18),0 5px 20px rgba(15,23,42,.12) !important;
  backdrop-filter:blur(12px) saturate(1.08) !important;
  animation:vhung-frame-pulse-compact 2.1s ease-in-out infinite !important;
}
#vhung-brand-notice-wrap .vhung-brand-title-row{
  width:100% !important; display:flex !important; justify-content:center !important;
  align-items:center !important; gap:7px !important; flex-wrap:nowrap !important;
  margin:0 0 5px !important;
}
#vhung-brand-notice-wrap .vhung-brand-title{margin:0 !important; font-size:18px !important; line-height:1 !important; font-weight:950 !important;}
#vhung-brand-notice-wrap .vhung-brand-title span{
  display:inline-flex !important; align-items:center !important; padding:5px 11px !important;
  border:2px solid transparent !important; border-radius:10px !important;
  background:linear-gradient(135deg,#ffffff,#eef9ff) padding-box,conic-gradient(from 20deg,#8bdcff,#c4b5fd,#a7f3d0,#bae6fd,#8bdcff) border-box !important; color:#172033 !important;
  box-shadow:0 0 0 1px rgba(255,255,255,.95) inset,0 0 13px rgba(14,165,233,.24) !important;
  animation:vhung-title-pulse-compact 1.55s ease-in-out infinite !important;
  white-space:nowrap !important;
}
#vhung-brand-notice-wrap .vhung-author{
  display:inline-flex !important; align-items:center !important; padding:4px 8px !important;
  border:2px solid transparent !important; border-radius:8px !important;
  background:linear-gradient(135deg,#fffdf5,#fff7df) padding-box,conic-gradient(from 40deg,#ffd6a5,#fbcfe8,#ddd6fe,#fde68a,#ffd6a5) border-box !important; color:#7c2d12 !important;
  font-size:10px !important; line-height:1.1 !important; font-weight:900 !important;
  box-shadow:0 0 10px rgba(245,158,11,.20) !important;
  animation:vhung-author-pulse-compact 1.9s ease-in-out infinite !important;
  white-space:nowrap !important;
}
/* Slogan: transparent inside — ONLY the travelling premium border is visible */
#vhung-brand-notice-wrap .vhung-slogan-shell{
  position:relative !important; width:min(980px,calc(100vw - 24px)) !important;
  box-sizing:border-box !important; margin:0 auto 6px !important; padding:5px !important;
  border:0 !important; border-radius:11px !important; overflow:hidden !important; pointer-events:none !important;
  background:
    linear-gradient(transparent,transparent) padding-box,
    conic-gradient(from var(--vhung-border-angle,0deg),
      #8fdfff 0deg,#b9c7ff 70deg,#dfc5ff 145deg,
      #b6eee3 225deg,#ffffff 270deg,#8fdfff 360deg) border-box !important;
  box-shadow:0 0 12px rgba(14,165,233,.16),0 0 24px rgba(167,139,250,.10) !important;
  animation:vhung_border_rotate 3.2s linear infinite !important;
}
#vhung-brand-notice-wrap .vhung-slogan-shell::before{
  content:'' !important; position:absolute !important; inset:-160% !important; z-index:0 !important;
  background:conic-gradient(from 0deg,transparent 0 20deg,rgba(255,255,255,.72) 26deg,transparent 34deg,transparent 360deg) !important;
  animation:vhung_slogan_shine 2.8s linear infinite !important; pointer-events:none !important;
}
#vhung-brand-notice-wrap .vhung-slogan{
  position:relative !important; z-index:1 !important; border-radius:7px !important;
  padding:4px 10px !important; background:transparent !important;
  color:#4b5563 !important; text-align:center !important; font-size:15px !important;
  line-height:1.32 !important; font-weight:800 !important; text-shadow:none !important; font-style:normal !important;
}
#vhung-brand-notice-wrap .vhung-slogan-label{
  font-size:11px !important; font-style:normal !important; font-weight:800 !important;
  color:#64748b !important; letter-spacing:.15px !important;
}
#vhung-brand-notice-wrap .vhung-slogan b{
  color:#8b5cf6 !important; font-size:17px !important; font-weight:950 !important;
  font-style:italic !important; text-shadow:0 1px 0 rgba(255,255,255,.9),0 0 9px rgba(139,92,246,.14) !important;
}
/* Reading notice: wide but compact so the workspace remains usable */
#vhung-brand-notice-wrap .vhung-notice{
  pointer-events:auto !important; position:relative !important; width:min(1040px,calc(100vw - 28px)) !important;
  box-sizing:border-box !important; margin:0 auto !important;
  max-height:min(37vh,330px) !important; overflow:auto !important;
  border:2px solid transparent !important; border-radius:14px !important;
  padding:10px 16px 9px !important;
  background:linear-gradient(180deg,rgba(255,253,245,.985),rgba(255,255,255,.975)) padding-box,conic-gradient(from 70deg,#ffd6a5,#f9c2d4,#fde68a,#c7d2fe,#ffd6a5) border-box !important;
  color:#1f2937 !important; box-shadow:0 10px 30px rgba(15,23,42,.16),0 0 18px rgba(245,158,11,.11) !important;
}
#vhung-brand-notice-wrap .vhung-notice-head{font-size:18px !important; line-height:1.15 !important; font-weight:950 !important; color:#9a3412 !important; margin-bottom:5px !important;}
#vhung-brand-notice-wrap .vhung-notice-body{font-size:14px !important; line-height:1.42 !important; font-weight:750 !important;}
#vhung-brand-notice-wrap .vhung-money-head{font-size:15px !important; font-weight:950 !important; color:#c62828 !important;}
#vhung-brand-notice-wrap .vhung-notice-foot{font-size:15px !important; line-height:1.25 !important; font-weight:950 !important; color:#075985 !important; margin-top:6px !important; padding-top:5px !important; border-top:1px dashed rgba(14,165,233,.35) !important;}
#vhung-brand-notice-wrap .vhung-timer{font-size:12px !important; font-weight:850 !important; color:#475569 !important; margin-top:3px !important;}
#vhung-brand-notice-wrap .vhung-close{display:none !important; margin-top:5px !important; padding:4px 10px !important; border:2px solid transparent !important; border-radius:8px !important; background:#fff !important; color:#9f3f35 !important; font-size:14px !important; font-weight:900 !important; cursor:pointer !important; background:linear-gradient(#fff,#fffaf5) padding-box,conic-gradient(from 10deg,#fbcfe8,#fed7aa,#fde68a,#fecdd3,#fbcfe8) border-box !important; box-shadow:0 0 10px rgba(244,114,182,.15) !important;}
#vhung-brand-notice-wrap.expired .vhung-close{display:inline-block !important;}
#vhung-brand-notice-wrap.closed .vhung-notice{display:none !important;}
#vhung-brand-notice-wrap.closed .vhung-slogan-shell{display:block !important;}
@keyframes vhung-frame-pulse-compact{0%,100%{box-shadow:0 0 12px rgba(14,165,233,.14),0 4px 16px rgba(15,23,42,.10);border-color:rgba(14,165,233,.62)}50%{box-shadow:0 0 28px rgba(14,165,233,.30),0 4px 22px rgba(15,23,42,.14);border-color:rgba(14,165,233,.95)}}
@keyframes vhung-title-pulse-compact{0%,100%{box-shadow:0 0 9px rgba(14,165,233,.18)}50%{box-shadow:0 0 20px rgba(14,165,233,.42),0 0 0 2px rgba(14,165,233,.10) inset;transform:scale(1.008)}}
@keyframes vhung-author-pulse-compact{0%,100%{box-shadow:0 0 7px rgba(245,158,11,.12)}50%{box-shadow:0 0 16px rgba(245,158,11,.38)}}
@keyframes vhung-fire-border-strong{0%{filter:hue-rotate(0deg) brightness(1)}50%{filter:hue-rotate(15deg) brightness(1.18)}100%{filter:hue-rotate(0deg) brightness(1)}}
@keyframes vhung-fire-sweep-fast{from{transform:rotate(0deg)}to{transform:rotate(360deg)}}
@media (max-width:900px){
 #vhung-brand-notice-wrap .vhung-brand-title{font-size:15px !important}
 #vhung-brand-notice-wrap .vhung-brand-title span{padding:4px 8px !important}
 #vhung-brand-notice-wrap .vhung-author{font-size:8px !important;padding:3px 6px !important}
 #vhung-brand-notice-wrap .vhung-slogan{font-size:13px !important;padding:4px 7px !important}
 #vhung-brand-notice-wrap .vhung-notice{width:calc(100vw - 18px) !important;max-height:34vh !important;padding:8px 11px !important}
 #vhung-brand-notice-wrap .vhung-notice-head{font-size:15px !important}
 #vhung-brand-notice-wrap .vhung-notice-body{font-size:12px !important}
 #vhung-brand-notice-wrap .vhung-money-head{font-size:13px !important}
 #vhung-brand-notice-wrap .vhung-notice-foot{font-size:13px !important}
}


</style>''')

        gr.HTML('''<style>
#workflow-return-status { position:fixed !important; right:18px !important; left:auto !important; bottom:18px !important; top:auto !important; width:560px !important; max-width:calc(100vw - 24px) !important; z-index:99999 !important; pointer-events:none !important; }
#workflow-return-status:empty { display:block !important; min-height:1px !important; }
#workflow-return-status .las-hud { box-sizing:border-box !important; width:calc(100% - 9px) !important; min-height:190px !important; max-height:330px !important; margin-right:9px !important; position:sticky !important; top:0 !important; z-index:9999 !important; padding:14px 16px 15px !important; border:9px solid rgba(59,130,246,.10) !important; border-radius:10px !important; background:rgba(255,255,255,.96) !important; color:#0f172a !important; box-shadow:0 7px 18px rgba(15,23,42,.15),0 0 0 1px rgba(255,255,255,.8) inset !important; backdrop-filter:blur(8px) !important; font-family:Inter,'Segoe UI',Arial,sans-serif !important; }
#workflow-return-status .las-hud-top { display:flex !important; flex-wrap:nowrap !important; align-items:center !important; gap:7px !important; min-height:24px !important; font-size:14px !important; white-space:nowrap !important; overflow:hidden !important; }
#workflow-return-status .las-hud-top b { flex:0 0 auto !important; font-size:24px !important; line-height:1 !important; letter-spacing:.2px !important; white-space:nowrap !important; }
#workflow-return-status .las-hud-top .las-hud-version { flex:0 0 auto !important; display:inline-flex !important; align-items:center !important; padding:2px 6px !important; border:1px solid rgba(14,165,233,.38) !important; border-radius:5px !important; color:#0369a1 !important; background:rgba(240,249,255,.92) !important; font-size:8px !important; font-weight:800 !important; letter-spacing:.25px !important; line-height:1 !important; }
#workflow-return-status .las-hud-top strong { flex:0 0 auto !important; font-size:28px !important; color:#0284c7 !important; }
#workflow-return-status .las-hud-state { flex:0 1 auto !important; font-size:12px !important; font-weight:800 !important; white-space:nowrap !important; overflow:hidden !important; text-overflow:ellipsis !important; }
#workflow-return-status .las-hud-stage { margin-top:4px !important; font-size:12px !important; color:#475569 !important; white-space:nowrap !important; overflow:hidden !important; text-overflow:ellipsis !important; }
#workflow-return-status .las-hud-status-line { margin-top:4px !important; font-size:12px !important; line-height:1.25 !important; color:#0f172a !important; white-space:nowrap !important; overflow:hidden !important; text-overflow:ellipsis !important; }
#workflow-return-status .las-hud-status-line .status-label { color:#64748b !important; font-weight:800 !important; margin-right:4px !important; }
#workflow-return-status .las-hud-row1 { display:none !important; }
#workflow-return-status .las-hud-row1 b { font-size:11px !important; letter-spacing:.25px !important; }
#workflow-return-status .las-hud-row1 span { color:#2563eb !important; overflow:hidden !important; white-space:nowrap !important; text-overflow:ellipsis !important; font-weight:700 !important; }
#workflow-return-status .las-hud-row1 strong { font-size:15px !important; color:#0284c7 !important; }
#workflow-return-status .las-hud-row2 { margin-top:5px !important; }
#workflow-return-status .las-hud-track { height:6px !important; overflow:hidden !important; border-radius:999px !important; background:#e2e8f0 !important; border:1px solid rgba(148,163,184,.14) !important; }
#workflow-return-status .las-hud-fill { height:100% !important; border-radius:999px !important; background:linear-gradient(90deg,#22c55e,#06b6d4,#3b82f6,#8b5cf6) !important; box-shadow:0 0 7px rgba(34,211,238,.35) !important; transition:width .22s ease !important; position:relative !important; overflow:hidden !important; }
#workflow-return-status .las-hud-fill::after { content:'' !important; position:absolute !important; inset:0 !important; background:linear-gradient(90deg,transparent,rgba(255,255,255,.35),transparent) !important; animation:las-hud-shimmer 1.15s linear infinite !important; }
@keyframes las-hud-shimmer { from { transform:translateX(-100%); } to { transform:translateX(100%); } }
#workflow-return-status .las-hud-current { margin-top:5px !important; font-size:13px !important; line-height:1.25 !important; }\n#workflow-return-status .las-hud-history { margin-top:9px !important; max-height:132px !important; overflow:auto !important; border-top:1px solid rgba(148,163,184,.25) !important; padding-top:6px !important; }
#workflow-return-status .las-hud-history-row { display:grid !important; grid-template-columns:56px 48px minmax(0,1fr) !important; gap:6px !important; align-items:center !important; font:700 10px/1.25 ui-monospace,SFMono-Regular,Consolas,monospace !important; padding:2px 0 !important; }
#workflow-return-status .las-hud-history-row span { color:#64748b !important; }
#workflow-return-status .las-hud-history-row b { color:#0284c7 !important; }
#workflow-return-status .las-hud-history-row em { color:#334155 !important; font-style:normal !important; overflow:hidden !important; text-overflow:ellipsis !important; white-space:nowrap !important; }
#workflow-return-status .las-hud-next { display:none !important; margin-top:4px !important; font-size:12px !important; color:#64748b !important; white-space:nowrap !important; overflow:hidden !important; text-overflow:ellipsis !important; }
#workflow-return-status .las-hud-next b { color:#334155 !important; }
#workflow-return-status .las-hud-row3 { display:grid !important; grid-template-columns:auto minmax(120px,1fr) !important; gap:10px !important; align-items:start !important; margin-top:11px !important; min-height:70px !important; }
#workflow-return-status .las-hud-chips { display:none !important; }
 #workflow-return-status .las-hud-live { color:#16a34a !important; font-size:10px !important; animation:las-live-pulse 1s ease-in-out infinite; }
 @keyframes las-live-pulse { 50% { opacity:.35; transform:scale(.78); } }
#workflow-return-status .las-hud-chip { display:inline-flex !important; align-items:center !important; height:17px !important; padding:0 5px !important; border-radius:5px !important; background:#111827 !important; border:1px solid #334155 !important; color:#94a3b8 !important; font-size:8px !important; font-weight:700 !important; white-space:nowrap !important; }
#workflow-return-status .las-hud-chip.active { color:#e0f2fe !important; background:#075985 !important; border-color:#38bdf8 !important; box-shadow:0 0 8px rgba(56,189,248,.22) !important; }
#workflow-return-status .las-hud-chip.done { color:#bbf7d0 !important; border-color:#166534 !important; background:#052e16 !important; }
#workflow-return-status .las-hud-detail { min-width:0 !important; overflow:hidden !important; color:#334155 !important; font-size:10px !important; line-height:1.2 !important; }
#workflow-return-status .las-hud-next-title { margin:4px 0 2px !important; font-size:10px !important; font-weight:850 !important; color:#64748b !important; text-transform:uppercase !important; letter-spacing:.35px !important; }
#workflow-return-status .las-hud-next-item { display:flex !important; align-items:flex-start !important; gap:5px !important; margin:2px 0 !important; font-size:10px !important; line-height:1.2 !important; }
#workflow-return-status .las-hud-next-item span { flex:0 0 auto !important; display:inline-grid !important; place-items:center !important; width:18px !important; height:18px !important; border-radius:999px !important; background:#e0f2fe !important; color:#0369a1 !important; font-size:10px !important; font-weight:900 !important; }
@media (max-width:900px) { #workflow-return-status { right:10px !important; bottom:10px !important; width:calc(100vw - 20px) !important; max-width:none !important; } }
#vhung-brand-notice-wrap {
  position:fixed !important; top:0 !important; left:0 !important; width:100vw !important; max-width:none !important;
  margin:0 !important; padding:0 !important; z-index:9000 !important; pointer-events:none !important;
}
#vhung-brand-notice-wrap .vhung-brand-shell {
  width:100vw !important; box-sizing:border-box !important; padding:8px 14px 10px !important;
  background:linear-gradient(180deg,rgba(255,255,255,.985),rgba(248,250,252,.94)) !important;
  border:2px solid rgba(14,165,233,.42) !important; border-left:0 !important; border-right:0 !important;
  border-radius:0 0 18px 18px !important; box-shadow:0 5px 24px rgba(15,23,42,.16) !important;
  backdrop-filter:blur(12px) !important; animation:vhung-frame-pulse 2.2s ease-in-out infinite !important;
}
#vhung-brand-notice-wrap .vhung-brand-title-row {
  width:100% !important; display:flex !important; justify-content:center !important; align-items:center !important;
  gap:8px !important; flex-wrap:wrap !important; margin-bottom:7px !important;
}
#vhung-brand-notice-wrap .vhung-brand-title { position:relative !important; margin:0 !important; font-size:25px !important; font-weight:950 !important; letter-spacing:.55px !important; }
#vhung-brand-notice-wrap .vhung-brand-title span {
  display:inline-block !important; padding:5px 14px !important; border:2px solid #22a6f2 !important; border-radius:12px !important;
  background:linear-gradient(135deg,#ffffff,#eaf7ff) !important; color:#1f2937 !important;
  box-shadow:0 0 0 1px rgba(255,255,255,.95) inset,0 0 16px rgba(14,165,233,.24) !important;
  animation:vhung-title-pulse 1.65s ease-in-out infinite !important;
}
#vhung-brand-notice-wrap .vhung-author {
  display:inline-flex !important; align-items:center !important; padding:5px 11px !important; border:2px solid #f59e0b !important;
  border-radius:10px !important; background:linear-gradient(135deg,#fffaf0,#fff7ed) !important; color:#7c2d12 !important;
  font-size:12px !important; font-weight:900 !important; letter-spacing:.25px !important;
  box-shadow:0 0 12px rgba(245,158,11,.20) !important; animation:vhung-author-pulse 1.9s ease-in-out infinite !important;
}
#vhung-brand-notice-wrap .vhung-slogan-shell {
  position:relative !important; width:100% !important; box-sizing:border-box !important; padding:3px !important;
  border-radius:14px !important; overflow:hidden !important; pointer-events:none !important;
  background:conic-gradient(from 0deg,#ff4d00,#ffd166,#fff3a3,#ff7a00,#ff2d00,#ffe066,#ff4d00) !important;
  animation:vhung-fire-border 1.05s linear infinite !important;
  box-shadow:0 0 12px rgba(255,105,0,.35),0 0 28px rgba(255,180,0,.18) !important;
}
#vhung-brand-notice-wrap .vhung-slogan-shell::before {
  content:'' !important; position:absolute !important; inset:-100% !important;
  background:conic-gradient(from 45deg,transparent 0deg,rgba(255,255,255,.9) 22deg,transparent 48deg,rgba(255,80,0,.9) 90deg,transparent 125deg,rgba(255,235,80,.95) 180deg,transparent 220deg,rgba(255,120,0,.85) 290deg,transparent 330deg) !important;
  animation:vhung-fire-sweep .9s linear infinite !important;
}
#vhung-brand-notice-wrap .vhung-slogan {
  position:relative !important; z-index:1 !important; border-radius:11px !important; padding:7px 14px !important;
  background:linear-gradient(90deg,#fff8e7,#fff1c2,#fff8e7) !important; color:#5b2c06 !important;
  text-align:center !important; font-size:13px !important; line-height:1.35 !important; font-weight:800 !important;
  text-shadow:none !important;
}
#vhung-brand-notice-wrap .vhung-slogan b { color:#8a3b00 !important; }
#vhung-brand-notice-wrap .vhung-notice {
  pointer-events:auto !important; position:relative !important; width:100% !important; box-sizing:border-box !important;
  margin-top:8px !important; max-height:calc(100vh - 150px) !important; overflow:auto !important;
  border:3px solid rgba(245,158,11,.72) !important; border-radius:16px !important; padding:15px 22px !important;
  background:linear-gradient(180deg,rgba(255,253,245,.985),rgba(255,255,255,.965)) !important; color:#1f2937 !important;
  box-shadow:0 14px 42px rgba(15,23,42,.22),0 0 24px rgba(245,158,11,.13) !important;
}
#vhung-brand-notice-wrap .vhung-notice-head { font-size:23px !important; line-height:1.2 !important; font-weight:950 !important; color:#9a3412 !important; margin-bottom:7px !important; }
#vhung-brand-notice-wrap .vhung-notice-body { font-size:17px !important; line-height:1.48 !important; font-weight:750 !important; }
#vhung-brand-notice-wrap .vhung-notice-foot { font-size:18px !important; line-height:1.35 !important; font-weight:950 !important; color:#075985 !important; margin-top:9px !important; padding-top:8px !important; border-top:1px dashed rgba(14,165,233,.35) !important; }
#vhung-brand-notice-wrap .vhung-timer { font-size:14px !important; font-weight:850 !important; color:#475569 !important; margin-top:5px !important; }
#vhung-brand-notice-wrap .vhung-close { display:none !important; margin-top:7px !important; padding:6px 13px !important; border:2px solid #ef4444 !important; border-radius:9px !important; background:#fff !important; color:#991b1b !important; font-weight:900 !important; cursor:pointer !important; }
#vhung-brand-notice-wrap.expired .vhung-close { display:inline-block !important; }
#vhung-brand-notice-wrap.closed .vhung-notice { display:none !important; }
#vhung-brand-notice-wrap.closed .vhung-slogan-shell { display:block !important; }
#vhung-brand-notice-wrap.closed .vhung-brand-shell { padding:5px 10px !important; border-radius:0 0 12px 12px !important; }
#vhung-brand-notice-wrap.closed .vhung-brand-title-row { margin:0 !important; }
#vhung-brand-notice-wrap.closed .vhung-brand-title span { font-size:18px !important; padding:3px 10px !important; }
#vhung-brand-notice-wrap.closed .vhung-author { font-size:9px !important; padding:3px 7px !important; }
@keyframes vhung-frame-pulse { 50% { box-shadow:0 0 0 1px rgba(255,255,255,.85) inset,0 0 30px rgba(14,165,233,.28),0 5px 26px rgba(15,23,42,.18); border-color:rgba(14,165,233,.72); } }
@keyframes vhung-title-pulse { 50% { box-shadow:0 0 25px rgba(14,165,233,.38),0 0 0 3px rgba(14,165,233,.08) inset; transform:scale(1.012); } }
@keyframes vhung-author-pulse { 50% { box-shadow:0 0 18px rgba(245,158,11,.36),0 0 0 2px rgba(255,255,255,.65) inset; } }
@keyframes vhung-fire-border { 0% { filter:hue-rotate(0deg) brightness(1); } 50% { filter:hue-rotate(20deg) brightness(1.22); } 100% { filter:hue-rotate(0deg) brightness(1); } }
@keyframes vhung-fire-sweep { 0% { transform:rotate(0deg); } 100% { transform:rotate(360deg); } }
@media (max-width:900px) {
  #vhung-brand-notice-wrap .vhung-brand-title { font-size:16px !important; }
  #vhung-brand-notice-wrap .vhung-slogan { font-size:11px !important; }
  #vhung-brand-notice-wrap .vhung-notice { max-height:calc(100vh - 125px) !important; padding:12px 15px !important; }
  #vhung-brand-notice-wrap .vhung-notice-head { font-size:19px !important; }
  #vhung-brand-notice-wrap .vhung-notice-body { font-size:15px !important; }
  #vhung-brand-notice-wrap .vhung-notice-foot { font-size:16px !important; }
}
@media (max-width:900px) { .flow-error-grid { grid-template-columns:1fr; } #vhung-brand-notice-wrap .vhung-notice { width:calc(100vw - 30px) !important; top:70px !important; } #vhung-brand-notice-wrap .vhung-notice-body { font-size:18px !important; } }
#master-control-btn:hover, #qc-full-btn:hover { transform: translateY(-1px) scale(1.012); filter: brightness(1.06); box-shadow: 0 0 0 1px rgba(0,194,168,.34), 0 10px 28px rgba(0,0,0,.25); }
#master-control-btn:active, #qc-full-btn:active { transform: translateY(1px) scale(.992); }
</style>''')
        # Gradio blocks <script> tags inside HTML values. Use js_on_load so the HUD
        # runtime actually executes after the component is mounted, without warnings.

        gr.HTML(r'''<style>
/* FINAL VHƯNG PRO MAX: full-width premium frame; workspace starts below it. */
#vhung-brand-notice-wrap{position:fixed !important;top:0 !important;left:0 !important;right:0 !important;width:100vw !important;z-index:2147483647 !important;pointer-events:none !important;}
#vhung-brand-notice-wrap .vhung-brand-shell{width:100vw !important;box-sizing:border-box !important;padding:10px 18px 14px !important;background:linear-gradient(180deg,rgba(255,255,255,.99),rgba(247,251,255,.97)) !important;border:0 !important;border-bottom:3px solid #22a6f2 !important;border-radius:0 0 18px 18px !important;box-shadow:0 5px 28px rgba(14,165,233,.18),0 8px 30px rgba(15,23,42,.10) !important;backdrop-filter:blur(14px) saturate(1.08) !important;animation:vhung-shell-pulse 2.7s ease-in-out infinite !important;}
#vhung-brand-notice-wrap .vhung-brand-title-row{width:100% !important;display:flex !important;justify-content:center !important;align-items:center !important;gap:10px !important;flex-wrap:nowrap !important;margin:0 0 8px !important;}
#vhung-brand-notice-wrap .vhung-brand-title{margin:0 !important;font-size:24px !important;line-height:1 !important;font-weight:950 !important;}
#vhung-brand-notice-wrap .vhung-brand-title span{display:inline-flex !important;align-items:center !important;padding:8px 18px !important;border:3px solid #0ea5e9 !important;border-radius:13px !important;background:linear-gradient(135deg,#fff,#e8f7ff) !important;color:#111827 !important;box-shadow:0 0 0 1px #fff inset,0 0 17px rgba(14,165,233,.28) !important;animation:vhung-brand-electric 1.45s ease-in-out infinite !important;white-space:nowrap !important;}
#vhung-brand-notice-wrap .vhung-author{display:inline-flex !important;align-items:center !important;padding:8px 13px !important;border:3px solid #e11d48 !important;border-radius:11px !important;background:linear-gradient(135deg,#fff1f2,#fff7ed) !important;color:#be123c !important;font-size:13px !important;line-height:1.1 !important;font-weight:950 !important;box-shadow:0 0 13px rgba(225,29,72,.22) !important;animation:vhung-author-neon 1.8s ease-in-out infinite !important;white-space:nowrap !important;}
#vhung-brand-notice-wrap .vhung-slogan-shell{position:relative !important;width:100% !important;box-sizing:border-box !important;margin:0 0 9px !important;padding:4px !important;border-radius:15px !important;overflow:hidden !important;pointer-events:none !important;background:conic-gradient(from 0deg,#ff8a65,#f6b73c,#ffe8a3,#ffd166,#f97316,#ffb86b,#ffe4a8,#ff8a65) !important;box-shadow:0 7px 22px rgba(249,115,22,.15),0 0 18px rgba(251,191,36,.12) !important;animation:vhung-fire-frame 1.05s linear infinite !important;}
#vhung-brand-notice-wrap .vhung-slogan-shell::before{content:'🔥  🔥  🔥' !important;position:absolute !important;z-index:4 !important;right:14px !important;top:-2px !important;font-size:19px !important;filter:drop-shadow(0 0 4px rgba(249,115,22,.45)) !important;opacity:.78 !important;animation:vhung-flame-flicker .42s ease-in-out infinite alternate !important;}
#vhung-brand-notice-wrap .vhung-slogan-shell::after{content:'' !important;position:absolute !important;inset:-160% !important;background:conic-gradient(from 0deg,transparent 0 18deg,rgba(255,255,255,.98) 24deg,transparent 34deg,rgba(255,64,0,.95) 58deg,transparent 74deg,rgba(255,232,75,.98) 105deg,transparent 125deg,rgba(255,105,0,.95) 160deg,transparent 188deg,rgba(255,248,170,.98) 220deg,transparent 255deg,rgba(255,52,0,.95) 305deg,transparent 330deg) !important;animation:vhung-fire-sweep 1s linear infinite !important;}
#vhung-brand-notice-wrap .vhung-slogan{position:relative !important;z-index:3 !important;border-radius:11px !important;padding:9px 20px !important;background:linear-gradient(90deg,#fffaf0,#fff1b8,#fffaf0) !important;color:#5b2500 !important;text-align:center !important;font-size:16px !important;line-height:1.35 !important;font-weight:900 !important;text-shadow:0 1px #fff !important;}
#vhung-brand-notice-wrap .vhung-slogan b{color:#8f2d00 !important;font-weight:950 !important;}
#vhung-brand-notice-wrap .vhung-notice{pointer-events:auto !important;position:relative !important;width:100% !important;box-sizing:border-box !important;margin:0 !important;max-height:285px !important;overflow:auto !important;border:3px solid transparent !important;border-radius:17px !important;padding:15px 30px 13px !important;background:linear-gradient(180deg,rgba(255,253,245,.985),rgba(255,255,255,.975)) padding-box,linear-gradient(90deg,#ff6b35,#f59e0b,#ffd166,#f97316,#ff6b35) border-box !important;background-size:100% 100%,300% 100% !important;box-shadow:0 8px 24px rgba(15,23,42,.12),0 0 18px rgba(249,115,22,.16) !important;}
#vhung-brand-notice-wrap .vhung-notice-head{text-align:center !important;font-size:23px !important;line-height:1.15 !important;font-weight:950 !important;color:#9a3412 !important;margin-bottom:8px !important;}
#vhung-brand-notice-wrap .vhung-notice-body{text-align:center !important;font-size:21px !important;line-height:1.48 !important;font-weight:780 !important;}
#vhung-brand-notice-wrap .vhung-money-head{font-size:18px !important;font-weight:950 !important;color:#c62828 !important;display:inline-block !important;margin-bottom:4px !important;}
#vhung-brand-notice-wrap .vhung-notice-foot-row{display:flex !important;justify-content:center !important;align-items:center !important;gap:18px !important;flex-wrap:wrap !important;margin-top:9px !important;padding-top:8px !important;border-top:1px dashed rgba(14,165,233,.35) !important;}
#vhung-brand-notice-wrap .vhung-notice-foot{text-align:center !important;font-size:17px !important;line-height:1.25 !important;font-weight:950 !important;color:#075985 !important;margin:0 !important;padding:0 !important;border:0 !important;}
#vhung-brand-notice-wrap .vhung-timer{text-align:center !important;font-size:15px !important;line-height:1.25 !important;font-weight:950 !important;color:#334155 !important;margin:0 !important;white-space:nowrap !important;}
#vhung-brand-notice-wrap .vhung-close{display:inline-block !important;margin:5px auto 0 !important;padding:6px 13px !important;border:2px solid transparent !important;border-radius:9px !important;background:linear-gradient(#fff,#fffaf5) padding-box,linear-gradient(90deg,#ff8a65,#f6b73c,#ffd98a,#ff8a65) border-box !important;color:#9a3412 !important;font-size:16px !important;font-weight:900 !important;cursor:pointer !important;box-shadow:0 5px 14px rgba(249,115,22,.13) !important;}
#vhung-brand-notice-wrap.closed .vhung-notice{display:none !important;}
#vhung-brand-notice-wrap.closed .vhung-slogan-shell{display:block !important;}
#vhung-brand-notice-wrap.closed .vhung-brand-shell{padding:6px 10px 8px !important;}
#vhung-brand-notice-wrap.closed .vhung-brand-title-row{margin:0 0 5px !important;}
#vhung-brand-notice-wrap.closed .vhung-slogan{font-size:14px !important;padding:6px 14px !important;}
@keyframes vhung-shell-pulse{0%,100%{box-shadow:0 5px 20px rgba(14,165,233,.13),0 8px 26px rgba(15,23,42,.08)}50%{box-shadow:0 5px 30px rgba(14,165,233,.30),0 8px 32px rgba(15,23,42,.13)}}
@keyframes vhung-brand-electric{0%,100%{box-shadow:0 0 12px rgba(14,165,233,.22),0 0 0 1px #fff inset;filter:saturate(1)}50%{box-shadow:0 0 28px rgba(14,165,233,.58),0 0 0 3px rgba(14,165,233,.10) inset;filter:saturate(1.25) brightness(1.03)}}
@keyframes vhung-author-neon{0%,100%{box-shadow:0 0 10px rgba(225,29,72,.18)}50%{box-shadow:0 0 25px rgba(225,29,72,.48),0 0 0 2px rgba(251,113,133,.20) inset}}
@keyframes vhung-fire-frame{0%{filter:hue-rotate(0deg) brightness(1)}50%{filter:hue-rotate(14deg) brightness(1.18)}100%{filter:hue-rotate(0deg) brightness(1)}}
@keyframes vhung-fire-sweep{from{transform:rotate(0deg)}to{transform:rotate(360deg)}}
@keyframes vhung-flame-flicker{0%{transform:translateY(2px) scale(.92) rotate(-2deg);opacity:.72}100%{transform:translateY(-2px) scale(1.1) rotate(3deg);opacity:1}}
/* FINAL VHƯNG PRO MAX BRAND PATCH — last visual layer. */
#vhung-brand-notice-wrap .vhung-brand-title span,
#vhung-brand-notice-wrap .vhung-author,
#vhung-brand-notice-wrap .vhung-slogan-shell{position:relative !important;transform:none !important;overflow:hidden !important;background-size:300% 100% !important;box-shadow:none !important;}
#vhung-brand-notice-wrap .vhung-brand-title span{border:3px solid transparent !important;border-radius:13px !important;background:linear-gradient(135deg,#ffffff,#f4fbff) padding-box,linear-gradient(90deg,#8ed8ff,#a7b9ff,#d6b7ff,#9be7ef,#8ed8ff) border-box !important;background-size:100% 100%,300% 100% !important;box-shadow:0 7px 20px rgba(96,165,250,.12),0 0 14px rgba(147,197,253,.13) !important;animation:vhung_brand_clean 2.8s linear infinite !important;}
#vhung-brand-notice-wrap .vhung-author{border:3px solid transparent !important;background:linear-gradient(135deg,#fffaff,#ffffff) padding-box,linear-gradient(90deg,#f6b6d0,#f8c6a8,#d9b8f5,#a9c7f7,#f6b6d0) border-box !important;background-size:100% 100%,300% 100% !important;color:#9b6a92 !important;font-style:italic !important;font-weight:500 !important;box-shadow:0 6px 16px rgba(168,85,247,.10),0 0 12px rgba(236,72,153,.08) !important;animation:vhung_author_clean 2.35s linear infinite !important;}
/* RESTORE ORIGINAL SLOGAN BORDER ANIMATION FROM app.py — keep the exact fire/conic border motion. */
#vhung-brand-notice-wrap .vhung-slogan-shell{position:relative !important;width:100% !important;box-sizing:border-box !important;margin:0 0 9px !important;padding:4px !important;border-radius:15px !important;overflow:hidden !important;pointer-events:none !important;background:conic-gradient(from 0deg,#ff2400,#ff7a00,#ffd21f,#fff1a8,#ff5a00,#e51b00,#ffd21f,#ff2400) !important;box-shadow:0 0 14px rgba(255,76,0,.45),0 0 30px rgba(255,179,0,.24) !important;animation:vhung-fire-frame 1.05s linear infinite !important;}
#vhung-brand-notice-wrap .vhung-slogan-shell::before{content:'🔥  🔥  🔥' !important;position:absolute !important;z-index:4 !important;right:14px !important;top:-2px !important;font-size:19px !important;filter:drop-shadow(0 0 5px #ff5a00) !important;animation:vhung-flame-flicker .42s ease-in-out infinite alternate !important;}
#vhung-brand-notice-wrap .vhung-slogan-shell::after{content:'' !important;position:absolute !important;inset:-160% !important;background:conic-gradient(from 0deg,transparent 0 18deg,rgba(255,255,255,.98) 24deg,transparent 34deg,rgba(255,64,0,.95) 58deg,transparent 74deg,rgba(255,232,75,.98) 105deg,transparent 125deg,rgba(255,105,0,.95) 160deg,transparent 188deg,rgba(255,248,170,.98) 220deg,transparent 255deg,rgba(255,52,0,.95) 305deg,transparent 330deg) !important;animation:vhung-fire-sweep 1s linear infinite !important;}
#vhung-brand-notice-wrap .vhung-slogan{position:relative !important;z-index:3 !important;border-radius:11px !important;padding:9px 20px !important;background:linear-gradient(90deg,#fffaf0,#fff1b8,#fffaf0) !important;color:#5b2500 !important;text-align:center !important;font-size:16px !important;line-height:1.35 !important;font-weight:900 !important;text-shadow:0 1px #fff !important;}
#vhung-brand-notice-wrap .vhung-slogan b{color:#8f2d00 !important;font-weight:950 !important;}
@keyframes vhung-fire-frame{0%{filter:hue-rotate(0deg) brightness(1)}50%{filter:hue-rotate(14deg) brightness(1.18)}100%{filter:hue-rotate(0deg) brightness(1)}}
@keyframes vhung-fire-sweep{from{transform:rotate(0deg)}to{transform:rotate(360deg)}}
@keyframes vhung-flame-flicker{0%{transform:translateY(2px) scale(.92) rotate(-2deg);opacity:.72}100%{transform:translateY(-2px) scale(1.1) rotate(3deg);opacity:1}}
#vhung-brand-notice-wrap .vhung-slogan{position:relative !important;z-index:2 !important;animation:none !important;transform:none !important;}
#vhung-brand-notice-wrap .vhung-reopen{display:none !important;position:fixed !important;top:10px !important;right:14px !important;z-index:2147483648 !important;pointer-events:auto !important;cursor:pointer !important;padding:8px 13px !important;border:3px solid transparent !important;border-radius:11px !important;background:linear-gradient(#fff,#fff7ed) padding-box,linear-gradient(90deg,#ef4444,#f59e0b,#ec4899,#ef4444) border-box !important;color:#9f1239 !important;font-size:13px !important;font-weight:900 !important;animation:vhung_reopen_clean 1.8s linear infinite !important;}
#vhung-brand-notice-wrap.closed .vhung-reopen{display:block !important;}
@keyframes vhung_brand_clean{0%{background-position:0 0,0 0}100%{background-position:0 0,300% 0}}
@keyframes vhung_author_clean{0%{background-position:0 0,300% 0}100%{background-position:0 0,0 0}}
@keyframes vhung_slogan_clean{0%{background-position:0 0,0 0}100%{background-position:0 0,300% 0}}
@keyframes vhung_reopen_clean{0%,100%{box-shadow:0 0 5px rgba(239,68,68,.25)}50%{box-shadow:0 0 16px rgba(239,68,68,.55),0 0 25px rgba(236,72,153,.18)}}


/* ============================================================
   FINAL VHƯNG PRO MAX — SOFT CONIC BORDER SYSTEM
   Visual-only patch: no workflow / logic changes.
   Slogan is the visual reference; every visible brand border
   uses the same soft conic-gradient + gentle glow language.
   ============================================================ */
#vhung-brand-notice-wrap .vhung-brand-title span{
  border:2px solid transparent !important;
  border-radius:13px !important;
  background:
    linear-gradient(135deg,#ffffff,#f5fbff) padding-box,
    conic-gradient(from 0deg,#a7e8ff,#b9c7ff,#d9c2ff,#b8f0e4,#a7e8ff) border-box !important;
  background-size:100% 100%,100% 100% !important;
  color:#172033 !important;
  box-shadow:0 0 0 1px rgba(255,255,255,.85) inset,
             0 0 12px rgba(125,211,252,.18),
             0 0 20px rgba(167,139,250,.10) !important;
  animation:vhung_conic_blue 2.8s linear infinite !important;
}

#vhung-brand-notice-wrap .vhung-author{
  border:2px solid transparent !important;
  border-radius:10px !important;
  background:
    linear-gradient(135deg,#fffdfd,#fffaf7) padding-box,
    conic-gradient(from 35deg,#f5b8cf,#f8d0b0,#ddd0f6,#b9d5f7,#f5b8cf) border-box !important;
  background-size:100% 100%,100% 100% !important;
  color:#8f617f !important;
  font-style:italic !important;
  font-weight:650 !important;
  box-shadow:0 0 0 1px rgba(255,255,255,.8) inset,
             0 0 11px rgba(236,72,153,.10),
             0 0 18px rgba(168,85,247,.08) !important;
  animation:vhung_conic_pink 2.7s linear infinite !important;
}

#vhung-brand-notice-wrap .vhung-slogan-shell{
  position:relative !important;
  width:100% !important;
  box-sizing:border-box !important;
  margin:0 0 9px !important;
  padding:1px !important;
  border-radius:15px !important;
  overflow:hidden !important;
  pointer-events:none !important;
  background:
    conic-gradient(from 0deg,#f7b08d,#f6c98e,#f7dda2,#ffe8b8,#efad88,#f5b992,#f9d9a0,#f7b08d) !important;
  box-shadow:0 0 8px rgba(245,135,72,.18),
             0 0 16px rgba(246,181,91,.10) !important;
  animation:vhung_conic_fire 2.2s linear infinite !important;
}

#vhung-brand-notice-wrap .vhung-slogan-shell::before{
  content:'🔥  🔥  🔥' !important;
  position:absolute !important;
  z-index:4 !important;
  right:14px !important;
  top:-2px !important;
  font-size:19px !important;
  filter:drop-shadow(0 0 4px rgba(239,125,74,.28)) !important;
  opacity:.58 !important;
  animation:vhung-flame-soft .75s ease-in-out infinite alternate !important;
}

#vhung-brand-notice-wrap .vhung-slogan-shell::after{
  content:'' !important;
  position:absolute !important;
  inset:-160% !important;
  background:conic-gradient(
    from 0deg,
    transparent 0 22deg,
    rgba(255,255,255,.58) 28deg,
    transparent 38deg,
    rgba(242,132,92,.38) 62deg,
    transparent 80deg,
    rgba(255,226,151,.48) 112deg,
    transparent 132deg,
    rgba(244,151,105,.34) 170deg,
    transparent 194deg,
    rgba(255,241,187,.45) 225deg,
    transparent 260deg,
    rgba(239,118,87,.32) 305deg,
    transparent 330deg
  ) !important;
  animation:vhung_fire_sweep_soft 2.4s linear infinite !important;
}

#vhung-brand-notice-wrap .vhung-slogan{
  position:relative !important;
  z-index:3 !important;
  border-radius:11px !important;
  padding:9px 20px !important;
  background:linear-gradient(90deg,#fffdf7,#fff6d8,#fffdf7) !important;
  color:#70452f !important;
  text-align:center !important;
  font-size:20px !important;
  line-height:1.35 !important;
  font-weight:900 !important;
  text-shadow:0 1px rgba(255,255,255,.95) !important;
  animation:none !important;
  transform:none !important;
}

#vhung-brand-notice-wrap .vhung-slogan b{
  color:#a45e45 !important;
  font-weight:950 !important;
  font-style:italic !important;
}

#vhung-brand-notice-wrap .vhung-notice{
  border:2px solid transparent !important;
  border-radius:17px !important;
  background:
    linear-gradient(180deg,rgba(255,253,245,.985),rgba(255,255,255,.975)) padding-box,
    conic-gradient(from 0deg,#f6b08d,#f7d39b,#f3b8a8,#e9b7c9,#f6b08d) border-box !important;
  background-size:100% 100%,100% 100% !important;
  box-shadow:0 8px 24px rgba(15,23,42,.10),
             0 0 15px rgba(245,135,72,.11) !important;
  animation:vhung_conic_notice 3s linear infinite !important;
}

#vhung-brand-notice-wrap .vhung-close{
  border:2px solid transparent !important;
  border-radius:9px !important;
  background:
    linear-gradient(#fff,#fffaf5) padding-box,
    conic-gradient(from 20deg,#b8e8ff,#c9c1f7,#e9c7f2,#b8e8ff) border-box !important;
  color:#695b83 !important;
  font-size:16px !important;
  box-shadow:0 0 9px rgba(129,140,248,.11) !important;
  animation:vhung_conic_close 2.6s linear infinite !important;
}

#vhung-brand-notice-wrap .vhung-reopen{
  border:2px solid transparent !important;
  border-radius:11px !important;
  background:
    linear-gradient(#fff,#fffaf5) padding-box,
    conic-gradient(from 25deg,#f6b8c9,#f6d7a7,#d7c4f3,#b9d8f4,#f6b8c9) border-box !important;
  color:#875d78 !important;
  box-shadow:0 0 10px rgba(236,72,153,.10) !important;
  animation:vhung_conic_reopen 2.8s linear infinite !important;
}

#vhung-brand-notice-wrap .vhung-brand-shell{
  border-bottom:2px solid transparent !important;
  border-image:none !important;
  background:
    linear-gradient(180deg,rgba(255,255,255,.99),rgba(247,251,255,.97)) padding-box,
    conic-gradient(from 0deg,#b8e8ff,#c9c1f7,#bcebdd,#f7d3bd,#b8e8ff) border-box !important;
}

@keyframes vhung_conic_blue{
  0%{filter:hue-rotate(0deg) brightness(1)}
  50%{filter:hue-rotate(8deg) brightness(1.025)}
  100%{filter:hue-rotate(0deg) brightness(1)}
}
@keyframes vhung_conic_pink{
  0%{filter:hue-rotate(0deg) brightness(1)}
  50%{filter:hue-rotate(-7deg) brightness(1.02)}
  100%{filter:hue-rotate(0deg) brightness(1)}
}
@keyframes vhung_conic_fire{
  0%{filter:hue-rotate(0deg) brightness(1)}
  50%{filter:hue-rotate(7deg) brightness(1.035)}
  100%{filter:hue-rotate(0deg) brightness(1)}
}
@keyframes vhung_conic_notice{
  0%{filter:hue-rotate(0deg) brightness(1)}
  50%{filter:hue-rotate(-5deg) brightness(1.02)}
  100%{filter:hue-rotate(0deg) brightness(1)}
}
@keyframes vhung_conic_close{
  0%{filter:hue-rotate(0deg)}
  50%{filter:hue-rotate(8deg) brightness(1.02)}
  100%{filter:hue-rotate(0deg)}
}
@keyframes vhung_conic_reopen{
  0%{filter:hue-rotate(0deg)}
  50%{filter:hue-rotate(-8deg) brightness(1.02)}
  100%{filter:hue-rotate(0deg)}
}
@keyframes vhung_fire_sweep_soft{
  from{transform:rotate(0deg)}
  to{transform:rotate(360deg)}
}
@keyframes vhung-flame-soft{
  0%{transform:translateY(1px) scale(.96) rotate(-1deg);opacity:.45}
  100%{transform:translateY(-1px) scale(1.04) rotate(1deg);opacity:.68}
}

@media (max-width:900px){
  #vhung-brand-notice-wrap .vhung-slogan{font-size:16px !important;padding:7px 10px !important;}
  #vhung-brand-notice-wrap .vhung-slogan-shell{padding:1px !important;}
}



/* ============================================================
   VIDEO-MATCH MOTION OVERRIDE
   The reference video uses a visible travelling light around
   the border, not just hue pulsing.  This keeps the same soft
   palettes while continuously rotating the conic highlight.
   ============================================================ */
@property --vhung-border-angle {
  syntax: "<angle>";
  inherits: false;
  initial-value: 0deg;
}

#vhung-brand-notice-wrap .vhung-brand-title span,
#vhung-brand-notice-wrap .vhung-author,
#vhung-brand-notice-wrap .vhung-notice,
#vhung-brand-notice-wrap .vhung-close,
#vhung-brand-notice-wrap .vhung-reopen,
#vhung-brand-notice-wrap .vhung-brand-shell{
  --vhung-border-angle:0deg;
  animation:vhung_border_rotate 3.2s linear infinite !important;
}

/* Each element gets a real rotating conic-gradient border. */
#vhung-brand-notice-wrap .vhung-brand-title span{
  background:
    linear-gradient(135deg,#ffffff,#f5fbff) padding-box,
    conic-gradient(
      from var(--vhung-border-angle),
      #8fdfff 0deg,
      #b9c7ff 70deg,
      #dfc5ff 145deg,
      #b6eee3 225deg,
      #ffffff 270deg,
      #8fdfff 360deg
    ) border-box !important;
}

#vhung-brand-notice-wrap .vhung-author{
  background:
    linear-gradient(135deg,#fffdfd,#fffaf7) padding-box,
    conic-gradient(
      from var(--vhung-border-angle),
      #f2aec9 0deg,
      #f7d0b1 75deg,
      #d8c8f4 150deg,
      #b9d8f5 225deg,
      #ffffff 275deg,
      #f2aec9 360deg
    ) border-box !important;
}

#vhung-brand-notice-wrap .vhung-notice{
  background:
    linear-gradient(180deg,rgba(255,253,245,.985),rgba(255,255,255,.975)) padding-box,
    conic-gradient(
      from var(--vhung-border-angle),
      #f29d78 0deg,
      #f6c978 65deg,
      #ffe6a9 120deg,
      #f4b2a4 190deg,
      #e7b5c9 250deg,
      #fff4cf 300deg,
      #f29d78 360deg
    ) border-box !important;
}

#vhung-brand-notice-wrap .vhung-close{
  background:
    linear-gradient(#fff,#fffaf5) padding-box,
    conic-gradient(
      from var(--vhung-border-angle),
      #aee5ff 0deg,
      #c4bcf6 90deg,
      #edc5ee 175deg,
      #ffffff 250deg,
      #aee5ff 360deg
    ) border-box !important;
}

#vhung-brand-notice-wrap .vhung-reopen{
  background:
    linear-gradient(#fff,#fffaf5) padding-box,
    conic-gradient(
      from var(--vhung-border-angle),
      #f1afc6 0deg,
      #f7d49f 75deg,
      #d4c1f2 160deg,
      #b8d9f3 240deg,
      #ffffff 295deg,
      #f1afc6 360deg
    ) border-box !important;
}

/* The slogan is the reference animation: warm border + a soft
   travelling white-gold shine. */
#vhung-brand-notice-wrap .vhung-slogan-shell{
  --vhung-border-angle:0deg;
  background:
    conic-gradient(
      from var(--vhung-border-angle),
      #f2a27c 0deg,
      #f5c47e 55deg,
      #ffe6a8 105deg,
      #fff4c9 150deg,
      #efaa84 210deg,
      #f6bd8f 270deg,
      #fff0bb 320deg,
      #f2a27c 360deg
    ) !important;
  animation:vhung_border_rotate_fire 2.7s linear infinite !important;
}

#vhung-brand-notice-wrap .vhung-slogan-shell::after{
  content:'' !important;
  position:absolute !important;
  z-index:2 !important;
  top:-25% !important;
  bottom:-25% !important;
  left:-75% !important;
  right:auto !important;
  bottom:auto !important;
  width:55% !important;
  height:150% !important;
  background:linear-gradient(
    112deg,
    transparent 0%,
    transparent 34%,
    rgba(255,255,255,.12) 43%,
    rgba(255,255,255,.72) 50%,
    rgba(255,244,200,.28) 57%,
    transparent 68%,
    transparent 100%
  ) !important;
  filter:blur(1px) !important;
  transform:skewX(-8deg) translateX(0) !important;
  animation:vhung_video_shine 2.7s ease-in-out infinite !important;
  pointer-events:none !important;
}

#vhung-brand-notice-wrap .vhung-slogan{
  z-index:3 !important;
}

/* A gentle brightness pulse is layered on top of the rotation,
   matching the subtle breathing seen in the reference video. */
@keyframes vhung_border_rotate{
  0%{
    --vhung-border-angle:0deg;
    filter:brightness(1);
  }
  50%{
    --vhung-border-angle:180deg;
    filter:brightness(1.025);
  }
  100%{
    --vhung-border-angle:360deg;
    filter:brightness(1);
  }
}

@keyframes vhung_border_rotate_fire{
  0%{
    --vhung-border-angle:0deg;
    filter:brightness(1);
  }
  50%{
    --vhung-border-angle:180deg;
    filter:brightness(1.035);
  }
  100%{
    --vhung-border-angle:360deg;
    filter:brightness(1);
  }
}

@keyframes vhung_video_shine{
  0%{transform:skewX(-8deg) translateX(-20%);opacity:0}
  12%{opacity:.15}
  42%{opacity:.85}
  55%{opacity:.35}
  72%{opacity:0}
  100%{transform:skewX(-8deg) translateX(285%);opacity:0}
}

@media (prefers-reduced-motion: reduce){
  #vhung-brand-notice-wrap .vhung-brand-title span,
  #vhung-brand-notice-wrap .vhung-author,
  #vhung-brand-notice-wrap .vhung-notice,
  #vhung-brand-notice-wrap .vhung-close,
  #vhung-brand-notice-wrap .vhung-reopen,
  #vhung-brand-notice-wrap .vhung-slogan-shell,
  #vhung-brand-notice-wrap .vhung-slogan-shell::after{
    animation-duration:6s !important;
  }
}


/* ============================================================
   FINAL USER VISUAL PATCH — transparent slogan + thicker travelling border
   ============================================================ */
/* FINAL REQUESTED BORDER SIZES — only these three items are changed. */
#vhung-brand-notice-wrap .vhung-brand-title span{
  border-width:3.6px !important;
}
#vhung-brand-notice-wrap .vhung-author{
  border-width:4.1px !important;
}
#vhung-brand-notice-wrap .vhung-notice,
#vhung-brand-notice-wrap .vhung-reopen{
  border-width:4.6px !important;
}
#vhung-brand-notice-wrap .vhung-close{
  border-width:4.4px !important;
}
#vhung-brand-notice-wrap .vhung-brand-shell{
  border-bottom-width:4.6px !important;
}

/* Password hide/show: collapse only the notice area; never cover the workspace. */
#vhung-brand-notice-wrap .vhung-brand-shell{
  max-height:460px !important;
  overflow:hidden !important;
  transition:max-height .48s cubic-bezier(.22,.61,.36,1),
             padding .48s cubic-bezier(.22,.61,.36,1),
             box-shadow .35s ease !important;
}
#vhung-brand-notice-wrap .vhung-slogan-shell,
#vhung-brand-notice-wrap .vhung-notice{
  opacity:1 !important;
  transform:translateY(0) !important;
  max-height:420px !important;
  transition:opacity .28s ease,
             transform .48s cubic-bezier(.22,.61,.36,1),
             max-height .48s cubic-bezier(.22,.61,.36,1),
             margin .48s cubic-bezier(.22,.61,.36,1) !important;
}
#vhung-brand-notice-wrap.closed .vhung-brand-shell{
  max-height:58px !important;
  padding-top:5px !important;
  padding-bottom:5px !important;
  box-shadow:0 4px 13px rgba(15,23,42,.09) !important;
}
#vhung-brand-notice-wrap.closed .vhung-slogan-shell,
#vhung-brand-notice-wrap.closed .vhung-notice{
  opacity:0 !important;
  transform:translateY(-18px) !important;
  max-height:0 !important;
  margin-top:0 !important;
  margin-bottom:0 !important;
  padding-top:0 !important;
  padding-bottom:0 !important;
  border-width:0 !important;
  overflow:hidden !important;
  pointer-events:none !important;
}
#vhung-brand-notice-wrap.closed .vhung-brand-title-row{
  margin-bottom:0 !important;
}
#vhung-brand-notice-wrap.closed .vhung-brand-title span{
  font-size:18px !important;
  padding:5px 11px !important;
}
#vhung-brand-notice-wrap.closed .vhung-author{
  font-size:10px !important;
  padding:4px 8px !important;
}
#vhung-brand-notice-wrap .vhung-close{
  display:inline-flex !important;
  align-items:center !important;
  justify-content:center !important;
  visibility:visible !important;
  opacity:1 !important;
}
#vhung-brand-notice-wrap .vhung-reopen{
  transform:translateY(-4px);
  opacity:0;
  transition:opacity .25s ease, transform .48s cubic-bezier(.22,.61,.36,1) !important;
}
#vhung-brand-notice-wrap.closed .vhung-reopen{
  opacity:1 !important;
  transform:translateY(0) !important;
}
#vhung-brand-notice-wrap .vhung-slogan-shell{
  padding:5px !important;
  background:
    conic-gradient(
      from var(--vhung-border-angle,0deg),
      #f19a72 0deg,
      #f5c06f 48deg,
      #ffe39b 96deg,
      #fff7cf 142deg,
      #ef9d78 205deg,
      #f5b87e 264deg,
      #fff0b6 320deg,
      #f19a72 360deg
    ) !important;
  animation:vhung_border_rotate_fire 2.55s linear infinite !important;
  box-shadow:0 0 10px rgba(245,135,72,.20),0 0 20px rgba(246,181,91,.11) !important;
}

#vhung-brand-notice-wrap .vhung-slogan{
  background:transparent !important;
  color:#70452f !important;
  font-style:italic !important;
  font-weight:900 !important;
  text-shadow:0 1px 0 rgba(255,255,255,.85),0 0 10px rgba(255,196,110,.14) !important;
}

#vhung-brand-notice-wrap .vhung-slogan,
#vhung-brand-notice-wrap .vhung-slogan b{
  font-style:normal !important;
}
#vhung-brand-notice-wrap .vhung-slogan b{
  color:#b24e24 !important;
  font-style:italic !important;
  font-weight:950 !important;
  text-shadow:0 1px 0 rgba(255,255,255,.9),0 0 9px rgba(255,151,80,.16) !important;
}

#vhung-brand-notice-wrap .vhung-slogan-shell::after{
  z-index:2 !important;
  opacity:.82 !important;
}

@media (max-width:900px){
  #vhung-brand-notice-wrap .vhung-slogan-shell{padding:5px !important;}
  #vhung-brand-notice-wrap .vhung-slogan{background:transparent !important;}
  #vhung-brand-notice-wrap .vhung-slogan b{font-size:16px !important;}
  #vhung-brand-notice-wrap .vhung-slogan-label{font-size:10px !important;}
}
/* FINAL SLOGAN CLEAN OVERRIDE: transparent interior, animated border only */
#vhung-brand-notice-wrap .vhung-slogan-shell{
  padding:5px !important; border:0 !important;
  background:linear-gradient(transparent,transparent) padding-box,
             conic-gradient(from var(--vhung-border-angle,0deg),#8fdfff 0deg,#b9c7ff 70deg,#dfc5ff 145deg,#b6eee3 225deg,#fff 270deg,#8fdfff 360deg) border-box !important;
  animation:vhung_border_rotate 3.2s linear infinite !important;
}
#vhung-brand-notice-wrap .vhung-slogan{background:transparent !important; font-style:normal !important;}
/* Keep the backend commit button mounted in DOM; JS triggers it reliably. */
#new-project-commit{display:none !important;}
#vhung-brand-notice-wrap .vhung-slogan b{font-style:italic !important;}
@keyframes vhung_slogan_shine{from{transform:rotate(0deg)}to{transform:rotate(360deg)}}
/* ============================================================
   FINAL CLEAN BORDER / PROJECT BUTTON PATCH
   ============================================================ */
#vhung-brand-notice-wrap .vhung-slogan-shell{
  position:relative !important;
  box-sizing:border-box !important;
  padding:5px !important;
  border:0 !important;
  border-radius:13px !important;
  overflow:hidden !important;
  background:transparent !important;
  box-shadow:0 0 12px rgba(14,165,233,.12),0 0 24px rgba(139,92,246,.08) !important;
  animation:none !important;
}
/* One and ONLY one animated layer: the border ring. Nothing is painted inside. */
#vhung-brand-notice-wrap .vhung-slogan-shell::before{
  content:'' !important;
  position:absolute !important;
  inset:0 !important;
  z-index:0 !important;
  padding:4px !important;
  box-sizing:border-box !important;
  border-radius:inherit !important;
  background:conic-gradient(from var(--vhung-border-angle,0deg),
    #8fdfff 0deg,#b9c7ff 70deg,#dfc5ff 145deg,
    #b6eee3 225deg,#ffffff 270deg,#8fdfff 360deg) !important;
  -webkit-mask:linear-gradient(#000 0 0) content-box,linear-gradient(#000 0 0) !important;
  -webkit-mask-composite:xor !important;
  mask-composite:exclude !important;
  animation:vhung_border_rotate 3.2s linear infinite !important;
  pointer-events:none !important;
}
/* Kill every old shine/fire layer that used to sweep across the whole box. */
#vhung-brand-notice-wrap .vhung-slogan-shell::after{
  content:none !important;
  display:none !important;
  animation:none !important;
  background:none !important;
}
#vhung-brand-notice-wrap .vhung-slogan{
  position:relative !important;
  z-index:1 !important;
  box-sizing:border-box !important;
  border:0 !important;
  background:transparent !important;
  padding:4px 10px !important;
}
#vhung-brand-notice-wrap .vhung-slogan-label{
  font-style:normal !important;
  font-size:11px !important;
  font-weight:800 !important;
}
#vhung-brand-notice-wrap .vhung-slogan b{
  font-size:17px !important;
  font-style:italic !important;
  font-weight:950 !important;
}
/* Softer notice frame: 0.5px thinner than the previous 2px frame. */
#vhung-brand-notice-wrap .vhung-notice{
  border-width:1.5px !important;
  box-shadow:0 7px 22px rgba(15,23,42,.10),0 0 14px rgba(245,158,11,.08) !important;
}
#new-project-commit{display:none !important;}
@media (max-width:900px){
  #vhung-brand-notice-wrap .vhung-slogan-shell{padding:5px !important;}
  #vhung-brand-notice-wrap .vhung-slogan{background:transparent !important;}
}
</style>''')

        with gr.Row():
            project = gr.Dropdown(choices=projects_now, value=default_project or None, allow_custom_value=True, label='📁 Project hiện tại / Mở Project cũ', scale=4)
            new_project_btn = gr.Button('＋ Tạo & Kết nối Project', variant='primary', scale=2, elem_id='new-project-confirm-trigger')
            new_project_commit_btn = gr.Button('Tạo Project mới', visible=False, elem_id='new-project-commit')
            open_project_btn = gr.Button('📂 Mở folder có sẵn', scale=2)
        project_active_path = gr.Textbox(value=SETTINGS.get('selected_project_path',''), label='📌 Project đang dùng', interactive=False)
        # Internal root state only. The user no longer needs a separate folder
        # picker: typing a folder name and pressing the button creates it in
        # Downloads and connects it immediately.
        project_output_dir = gr.Textbox(value=SETTINGS.get('project_output_dir',''), visible=False, interactive=False)
        project_root_status = gr.Markdown('💡 **＋ Tạo & Kết nối Project** luôn tạo Project mới. Muốn mở Project cũ dùng **📂 Mở folder có sẵn**.')

        with gr.Accordion('⚙ Cài đặt', open=False):
            with gr.Row():
                ui_lang = gr.Dropdown(choices=[('🇻🇳 Tiếng Việt', 'vi'), ('🇺🇸 English', 'en')], value=SETTINGS.get('language', 'vi'), label='Ngôn ngữ giao diện')
                appearance = gr.Dropdown(choices=[('Tối', 'dark'), ('Sáng', 'light'), ('Theo hệ thống', 'system')], value=SETTINGS.get('appearance', 'dark'), label='Giao diện')
                accent = gr.Dropdown(choices=list(ACCENTS.keys()), value=SETTINGS.get('accent', 'teal'), label='Màu nhấn')
            settings_key = gr.Textbox(value=SETTINGS.get('gemini_api_key', ''), type='password', label='Gemini API Key')
            with gr.Row():
                save_gemini_key_btn = gr.Button('🔑 Lưu API Gemini riêng', variant='primary')
                gemini_key_status = gr.Markdown(
                    f"**{GEMINI_KEY_UX_BUILD}** · " +
                    (f"🟢 Gemini API SẴN SÀNG · {len(_load_gemini_keys())} key đã lưu." if _load_gemini_keys() else "🟠 Chưa có Gemini API Key."),
                    elem_id='gemini-key-runtime-status'
                )
            save_gemini_key_btn.click(_wrap_gradio_callback(save_gemini_key_ui), [settings_key], gemini_key_status)
            # Use a bounded dropdown instead of Slider for this persisted setting.
            # Some Gradio/Browser states can submit null for a Slider during a rerender;
            # Slider preprocessing rejects null before save_settings_ui can sanitize it.
            # Dropdown accepts None, and save_settings_ui safely falls back to the persisted value.
            _saved_vram = SETTINGS.get('max_vram_percent', 82)
            try:
                _saved_vram = max(50, min(95, int(_saved_vram)))
            except (TypeError, ValueError):
                _saved_vram = 82
            max_vram = gr.Dropdown(choices=list(range(50, 96)), value=_saved_vram,
                                   label='Ngưỡng VRAM mục tiêu (%)',
                                   info='Chọn từ 50% đến 95%. Nếu UI gửi rỗng, hệ thống giữ giá trị đã lưu.')
            save_settings_btn = gr.Button('💾 Lưu & Tải lại', variant='primary')
            settings_status = gr.Markdown()
            save_settings_btn.click(_wrap_gradio_callback(save_settings_ui), [ui_lang, appearance, accent, settings_key, max_vram], settings_status, js="""() => { setTimeout(() => window.location.reload(), 700); }""")

        # Internal Gradio sink only. The visible LIVE HUB is a single body-level HUD
        # created by JS below, so tab switching/re-rendering can never duplicate it.
        workflow_return_status = gr.HTML("", elem_id='workflow-return-status')
        gr.HTML('''<style>
/* REAL VIETNAM CLOCK OVERLAY — draggable, hideable, logo toggle; white frame remains. */
#vhung-vn-clock-overlay{
  position:fixed !important; top:16px !important; right:16px !important; left:auto !important;
  width:460px !important; height:420px !important; min-width:320px !important; min-height:260px !important;
  max-width:calc(100vw - 20px) !important; max-height:calc(100vh - 20px) !important;
  z-index:2147483646 !important; pointer-events:auto !important; box-sizing:border-box !important;
  resize:none !important; overflow:visible !important;
}
#vhung-vn-clock-overlay .vhung-vn-clock-resize{
  position:absolute !important; right:4px !important; bottom:4px !important;
  width:26px !important; height:26px !important; z-index:20 !important;
  cursor:nwse-resize !important; touch-action:none !important; user-select:none !important;
  border-right:3px solid rgba(56,189,248,.45) !important;
  border-bottom:3px solid rgba(129,140,248,.45) !important;
  border-bottom-right-radius:7px !important;
  background:linear-gradient(135deg,transparent 0 48%,rgba(56,189,248,.18) 49% 54%,transparent 55% 66%,rgba(129,140,248,.22) 67% 72%,transparent 73%) !important;
}
#vhung-vn-clock-overlay .vhung-vn-clock-resize:hover{
  border-right-color:rgba(14,165,233,.85) !important; border-bottom-color:rgba(99,102,241,.85) !important;
  filter:brightness(1.08) !important;
}
#vhung-vn-clock-overlay .vhung-vn-clock-card{
  box-sizing:border-box !important; width:100% !important; height:100% !important; min-height:0 !important; padding:6px 10px 8px !important;
  container-type:size !important; overflow:hidden !important;
  border-radius:20px !important; background:rgba(255,255,255,.985) !important; color:#0f172a !important;
  border:2px solid transparent !important;
  background-image:linear-gradient(#fff,#f8fafc),linear-gradient(90deg,#38bdf8,#818cf8,#a78bfa,#2dd4bf,#38bdf8) !important;
  background-origin:border-box !important; background-clip:padding-box,border-box !important;
  background-size:100% 100%,300% 100% !important;
  box-shadow:0 10px 32px rgba(15,23,42,.16),0 0 24px rgba(56,189,248,.16) !important;
  backdrop-filter:blur(12px) saturate(1.08) !important;
  animation:vhung-vn-clock-border 3.2s linear infinite !important; text-align:center !important;
  pointer-events:auto !important; user-select:none !important; position:relative !important;
}
#vhung-vn-clock-overlay .vhung-clock-toolbar{
  display:flex !important; align-items:center !important; justify-content:space-between !important;
  gap:6px !important; margin-bottom:4px !important;
}
#vhung-vn-clock-overlay .vhung-clock-toolbar-title{
  flex:1 !important; text-align:center !important; font-size:11px !important;
  font-weight:950 !important; letter-spacing:.55px !important; color:#0369a1 !important;
}
#vhung-vn-clock-overlay button{
  appearance:none !important; border:1px solid rgba(148,163,184,.42) !important;
  background:#fff !important; color:#334155 !important; border-radius:8px !important;
  padding:4px 7px !important; font-size:10px !important; font-weight:900 !important;
  cursor:pointer !important; box-shadow:0 2px 7px rgba(15,23,42,.08) !important;
}
#vhung-vn-clock-overlay .vhung-vn-clock-logo-wrap{cursor:grab !important;touch-action:none !important;user-select:none !important;}
#vhung-vn-clock-overlay .vhung-vn-clock-logo-wrap:active{cursor:grabbing !important;}
#vhung-vn-clock-overlay .vhung-vn-clock-logo{cursor:grab !important;user-select:none !important;-webkit-user-drag:none !important;touch-action:none !important;}
#vhung-vn-clock-overlay .vhung-clock-hide{font-size:15px !important;line-height:12px !important;padding:3px 7px !important;}
#vhung-vn-clock-overlay .vhung-vn-clock-logo-wrap{
  width:min(18cqw,20cqh) !important; height:min(18cqw,20cqh) !important;
  min-width:58px !important; min-height:58px !important;
  margin:1cqh auto 1cqh !important;
  border-radius:50% !important; padding:4px !important; background:#fff !important;
  border:2px solid rgba(148,163,184,.32) !important;
  box-shadow:0 4px 14px rgba(15,23,42,.14),0 0 0 4px rgba(255,255,255,.95) !important;
  overflow:hidden !important;
}
#vhung-vn-clock-overlay .vhung-vn-clock-logo{
  display:block !important; width:100% !important; height:100% !important;
  object-fit:cover !important; border-radius:50% !important;
}
#vhung-vn-clock-overlay.logo-hidden .vhung-vn-clock-logo-wrap{display:none !important;}
#vhung-vn-clock-overlay .vhung-vn-clock-brand{
  font-size:clamp(26px,min(7.2cqw,7.2cqh),76px) !important;
  line-height:1 !important; font-weight:950 !important;
  letter-spacing:.2px !important; color:#172033 !important; white-space:nowrap !important;
}
#vhung-vn-clock-overlay .vhung-vn-clock-label{
  margin-top:.5cqh !important; font-size:clamp(10px,min(2.5cqw,2.7cqh),28px) !important; font-weight:900 !important;
  letter-spacing:.55px !important; color:#0369a1 !important;
}
#vhung-vn-clock-overlay .vhung-vn-clock-time{
  margin-top:.3cqh !important; font-size:clamp(42px,min(13.5cqw,16cqh),120px) !important; line-height:1.02 !important;
  font-weight:950 !important; font-variant-numeric:tabular-nums !important;
  letter-spacing:1.4px !important; color:#0f766e !important;
  text-shadow:0 2px 12px rgba(20,184,166,.16) !important;
}
#vhung-vn-clock-overlay .vhung-vn-clock-date{
  margin-top:.2cqh !important; font-size:clamp(13px,min(3.2cqw,3.8cqh),34px) !important; font-weight:850 !important;
  color:#475569 !important; font-variant-numeric:tabular-nums !important;
}
#vhung-vn-clock-overlay .vhung-vn-clock-status{
  display:inline-flex !important; align-items:center !important; justify-content:center !important; gap:5px !important;
  margin:1cqh auto 0 !important; padding:.7cqh 1.3cqw !important; max-width:92% !important; box-sizing:border-box !important;
  border:1px solid rgba(34,197,94,.30) !important; border-radius:999px !important;
  background:rgba(240,253,244,.96) !important; color:#166534 !important; font-size:11px !important; line-height:1.15 !important;
  font-weight:900 !important; white-space:nowrap !important; overflow:hidden !important; text-overflow:ellipsis !important;
  font-size:clamp(10px,min(2.4cqw,2.8cqh),25px) !important;
}
#vhung-vn-clock-overlay .vhung-vn-clock-status .status-dot{
  width:7px !important; height:7px !important; flex:0 0 7px !important; border-radius:50% !important;
  background:#22c55e !important; box-shadow:0 0 0 2px rgba(34,197,94,.12) !important;
}
#vhung-vn-clock-overlay .vhung-clock-scale{
  width:100% !important; height:100% !important;
  min-height:100% !important; box-sizing:border-box !important;
  position:absolute !important; inset:0 !important; width:100% !important; height:100% !important;
  min-height:0 !important; box-sizing:border-box !important;
  display:flex !important; flex-direction:column !important; align-items:center !important;
  justify-content:center !important; gap:0 !important;
  transform:none !important; transform-origin:center center !important;
  container-type:size !important; will-change:auto !important;
}
#vhung-vn-clock-overlay .vhung-clock-scale > *{flex:0 0 auto !important;}
#vhung-vn-clock-overlay .vhung-clock-scale{
  position:absolute !important; inset:0 !important; width:100% !important; height:100% !important;
  box-sizing:border-box !important; padding:1.5% 2.5% !important; margin:0 !important;
  display:grid !important; grid-template-columns:1fr !important;
  grid-template-rows:8% 22% 16% 7% 28% 7% 9% 3% !important;
  align-items:center !important; justify-items:center !important;
  transform:none !important; transform-origin:center center !important;
  container-type:size !important; overflow:hidden !important;
}
#vhung-vn-clock-overlay .vhung-clock-scale > *{
  box-sizing:border-box !important; min-width:0 !important; min-height:0 !important;
  width:auto !important; height:auto !important; margin:0 !important;
}
#vhung-vn-clock-overlay .vhung-clock-scale > .vhung-clock-toolbar{
  grid-row:1 !important; width:100% !important; height:100% !important; margin:0 !important;
}
#vhung-vn-clock-overlay .vhung-vn-clock-logo-wrap{
  grid-row:2 !important; width:auto !important; height:100% !important; aspect-ratio:1 / 1 !important;
  min-width:0 !important; min-height:0 !important; max-width:22cqw !important; max-height:100% !important;
  padding:4px !important; margin:0 !important;
}
#vhung-vn-clock-overlay .vhung-vn-clock-brand{
  grid-row:3 !important; width:100% !important;
  font-size:clamp(26px,8.6cqw,96px) !important; line-height:.95 !important;
  font-weight:950 !important; letter-spacing:.2px !important; white-space:nowrap !important;
}
#vhung-vn-clock-overlay .vhung-vn-clock-label{
  grid-row:4 !important; margin:0 !important; width:100% !important;
  font-size:clamp(10px,2.6cqw,30px) !important; line-height:1 !important;
}
#vhung-vn-clock-overlay .vhung-vn-clock-time{
  grid-row:5 !important; width:100% !important; margin:0 !important;
  font-size:clamp(44px,16cqw,160px) !important; line-height:.9 !important;
  letter-spacing:1.2px !important;
}
#vhung-vn-clock-overlay .vhung-vn-clock-date{
  grid-row:6 !important; margin:0 !important; width:100% !important;
  font-size:clamp(14px,3.5cqw,38px) !important; line-height:1 !important;
}
#vhung-vn-clock-overlay .vhung-vn-clock-status{
  grid-row:7 !important; margin:0 !important; max-width:92% !important;
  font-size:clamp(11px,2.5cqw,26px) !important; line-height:1 !important;
  padding:.8% 1.6% !important;
}
#vhung-vn-clock-overlay .vhung-vn-clock-resize{grid-row:8 !important; justify-self:end !important; align-self:end !important;}
#vhung-vn-clock-reopen{
  display:none !important; position:fixed !important; top:16px !important; right:16px !important;
  width:42px !important; height:42px !important; padding:0 !important; border-radius:50% !important;
  font-size:20px !important; z-index:2147483647 !important; pointer-events:auto !important;
}
#vhung-vn-clock-overlay.clock-hidden .vhung-vn-clock-card{display:none !important;}
#vhung-vn-clock-overlay.clock-hidden #vhung-vn-clock-reopen{display:block !important;}
@keyframes vhung-vn-clock-border{
  0%{background-position:0 0,0% 0;box-shadow:0 10px 32px rgba(15,23,42,.16),0 0 18px rgba(56,189,248,.12)}
  50%{background-position:0 0,150% 0;box-shadow:0 10px 36px rgba(15,23,42,.18),0 0 28px rgba(129,140,248,.22)}
  100%{background-position:0 0,300% 0;box-shadow:0 10px 32px rgba(15,23,42,.16),0 0 18px rgba(45,212,191,.12)}
}
@media (max-width:900px){
  #vhung-vn-clock-overlay{top:10px !important;right:10px !important;width:300px !important;max-width:calc(100vw - 20px) !important;}
}
</style>''')
        gr.HTML('''<style>
/* FINAL HUD: always-on-top, always visible, large and shared with CMD progress. */
#workflow-return-status .las-hud { min-height:240px !important; width:760px !important; max-width:calc(100vw - 24px) !important; }
#workflow-return-status .las-hud-top strong[data-live-pct] { font-size:28px !important; min-width:70px !important; }
#workflow-return-status .las-hud-track { height:12px !important; }
#workflow-return-status .las-hud-history { max-height:145px !important; overflow:auto !important; }
#workflow-return-status .las-hud-history-row { min-height:25px !important; }
/* FINAL HUD: always-on-top, never hidden by Gradio tab switching. */
#workflow-return-status,
#workflow-return-status > div,
#workflow-return-status .las-hud {
  visibility:visible !important;
  opacity:1 !important;
}
#workflow-return-status {
  position:fixed !important;
  display:block !important;
  pointer-events:none !important;
  z-index:2147483000 !important;
}
#workflow-return-status .las-hud {
  position:relative !important;
  display:block !important;
  overflow:hidden !important;
}
#workflow-return-status .las-hud-top {
  grid-template-columns:minmax(0,1fr) auto auto auto !important;
}
#workflow-return-status .las-hud-clock {
  min-width:58px !important;
  text-align:right !important;
  color:#0f766e !important;
  font-size:14px !important;
  font-weight:900 !important;
  font-variant-numeric:tabular-nums !important;
  white-space:nowrap !important;
}
#workflow-return-status .las-hud-clock-idle { color:#64748b !important; }
#workflow-return-status .las-hud-vnclock { min-width:92px !important; text-align:right !important; color:#0f766e !important; font-size:14px !important; font-weight:950 !important; font-variant-numeric:tabular-nums !important; white-space:nowrap !important; }
#vhung-brand-notice-wrap ~ * { box-sizing:border-box !important; }
/* Legacy Gradio HUD is a data sink only. Never render a second visible copy. */
#workflow-return-status { display:none !important; width:0 !important; height:0 !important; min-height:0 !important; overflow:hidden !important; }
/* Single body-level LIVE HUB */
#vhung-live-hub-body{position:fixed;right:18px;bottom:18px;width:430px;max-width:calc(100vw - 20px);z-index:2147483001;box-sizing:border-box;pointer-events:auto;font-family:Inter,'Segoe UI',Arial,sans-serif;}
#vhung-live-hub-body .vhung-hub-card{box-sizing:border-box;background:rgba(255,255,255,.97);border:1px solid rgba(56,189,248,.55);border-radius:16px;padding:9px 11px;box-shadow:0 12px 32px rgba(15,23,42,.16),0 0 0 1px rgba(255,255,255,.7) inset;backdrop-filter:blur(10px);}
#vhung-live-hub-body .hub-top{display:flex;align-items:center;gap:8px;white-space:nowrap;overflow:hidden;}
#vhung-live-hub-body .hub-title{font-size:16px;font-weight:900;color:#172033;}
#vhung-live-hub-body .hub-stage{font-size:11px;color:#475569;margin-top:3px;font-weight:800;}
#vhung-live-hub-body .hub-status{font-size:11px;color:#0f172a;margin-top:3px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
#vhung-live-hub-body .hub-live-detail{font-size:11px;color:#0f172a;margin-top:5px;padding:5px 7px;border-radius:9px;background:rgba(14,165,233,.07);border:1px solid rgba(14,165,233,.16);}
#vhung-live-hub-body .hub-counters{display:flex;flex-wrap:wrap;gap:12px;margin-top:7px;font:700 10px/1.3 ui-monospace,Consolas,monospace;color:#475569;}
#vhung-live-hub-body .hub-counters b{color:#0f172a;font-variant-numeric:tabular-nums;}
#vhung-live-hub-body .hub-history{margin-top:7px;max-height:110px;overflow:auto;border-top:1px solid rgba(148,163,184,.22);padding-top:5px;}
#vhung-live-hub-body .hub-row{display:grid;grid-template-columns:55px minmax(0,1fr);gap:5px;font:700 10px/1.35 ui-monospace,Consolas,monospace;padding:2px 0;}
#vhung-live-hub-body .hub-row span{color:#64748b}.hub-row em{font-style:normal;color:#334155;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
@keyframes vhung-hub-shimmer{from{transform:translateX(-100%)}to{transform:translateX(100%)}}
@media(max-width:900px){#vhung-live-hub-body{right:10px;bottom:10px;width:calc(100vw - 20px)}#vhung-live-hub-body .hub-title{font-size:17px}}
</style>''')
        with gr.Row(elem_classes=['master-control-row']):
            master_control_btn = gr.Button('⏹ Dừng toàn bộ', variant='stop', elem_id='master-control-btn', scale=2)
        master_control_status = gr.Markdown('🟢 **ON** — Workflow sẵn sàng.', elem_id='master-control-status')
        # STOP binding is registered after master_pipeline_event is created below,
        # so pressing Dừng toàn bộ can cancel the queued MASTER chain as well as
        # setting the Python stop flags.

        ui.load(None, js="""() => {
          document.title = 'VHƯNG PRO MAX';
          if (window.__vhungVietnamClockStarted) return;
          window.__vhungVietnamClockStarted = true;
          const formatTime = new Intl.DateTimeFormat('vi-VN', {timeZone:'Asia/Ho_Chi_Minh',hour:'2-digit',minute:'2-digit',second:'2-digit',hour12:false});
          const formatDate = new Intl.DateTimeFormat('vi-VN', {timeZone:'Asia/Ho_Chi_Minh',day:'2-digit',month:'2-digit',year:'numeric'});
          const paint = () => {
            const now = new Date();
            const t = formatTime.format(now);
            const d = formatDate.format(now);
            const timeEl = document.getElementById('vhung-vn-clock-time');
            const dateEl = document.getElementById('vhung-vn-clock-date');
            if (timeEl) timeEl.textContent = t;
            if (dateEl) dateEl.textContent = d;
            document.querySelectorAll('.las-hud-vnclock').forEach(el => {el.textContent = '🕐 ' + t; el.title = 'Giờ Việt Nam · Asia/Ho_Chi_Minh';});
          };
          paint();
          setInterval(paint, 250);
        }""")

        ui.load(None, js="""() => {
          /* Draggable Vietnam clock + independent logo visibility. Position is local to this browser only. */
          if (window.__vhungClockControlsStarted) return;
          window.__vhungClockControlsStarted = true;
          const root = document.getElementById('vhung-vn-clock-overlay');
          if (!root) return;
          const card = root.querySelector('.vhung-vn-clock-card');
          const drag = root.querySelector('.vhung-vn-clock-logo-wrap') || card;
          const logoImg = root.querySelector('.vhung-vn-clock-logo');
          const logoBtn = root.querySelector('.vhung-clock-logo-toggle');
          const hideBtn = root.querySelector('.vhung-clock-hide');
          const reopen = root.querySelector('#vhung-vn-clock-reopen');
          const posKey = 'vhung_vn_clock_position_v3';
          const logoKey = 'vhung_vn_clock_logo_visible_v2';
          try {
            const logoVisible = localStorage.getItem(logoKey);
            if (logoVisible === '0') root.classList.add('logo-hidden');
          } catch(e) {}
          logoBtn?.addEventListener('click', (ev) => {
            ev.preventDefault(); ev.stopPropagation();
            root.classList.toggle('logo-hidden');
            try { localStorage.setItem(logoKey, root.classList.contains('logo-hidden') ? '0' : '1'); } catch(e) {}
          });
          hideBtn?.addEventListener('click', (ev) => {
            ev.preventDefault(); ev.stopPropagation();
            root.classList.add('clock-hidden');
          });
          reopen?.addEventListener('click', (ev) => {
            ev.preventDefault(); ev.stopPropagation();
            root.classList.remove('clock-hidden');
          });
          const resizeHandle = root.querySelector('.vhung-vn-clock-resize');
          const scaleBox = root.querySelector('.vhung-clock-scale');
          const statusEl = root.querySelector('.vhung-vn-clock-status .status-text');
          const statusDot = root.querySelector('.vhung-vn-clock-status .status-dot');
          const px = (v) => `${Math.round(v)}px`;
          const syncClockScale = () => {
            if (!scaleBox || !card) return;
            /* Border = canvas duy nhất; nội dung không bị co vào hộp 900x330. */
            scaleBox.style.position = 'absolute';
            scaleBox.style.inset = '0';
            scaleBox.style.width = '100%';
            scaleBox.style.height = '100%';
            scaleBox.style.minWidth = '0';
            scaleBox.style.minHeight = '0';
            scaleBox.style.margin = '0';
            scaleBox.style.transform = 'none';
            scaleBox.style.transformOrigin = 'center center';
          };
          const refreshClockStatus = () => {
            if (!statusEl) return;
            const m = document.querySelector('#gemini-key-runtime-status');
            const txt = String(m?.textContent || '');
            const mm = txt.match(/([0-9]+) key/i);
            const count = mm ? Number(mm[1]) : (/Runtime đã có Gemini API Key/i.test(txt) ? 1 : 0);
            const ready = count > 0 || /Runtime đã có Gemini API Key/i.test(txt);
            statusEl.textContent = `Gemini API ${ready ? 'SẴN SÀNG' : 'CHƯA SẴN SÀNG'} · ${count} key${count===1?'':'s'} đã lưu`;
            if (statusDot) statusDot.style.background = ready ? '#22c55e' : '#f59e0b';
          };
          const setImportant = (prop, value) => root.style.setProperty(prop, px(value), 'important');
          const setAutoImportant = (prop, value) => root.style.setProperty(prop, value, 'important');
          const applyGeometry = (left, top, width, height) => {
            const maxW = Math.max(320, window.innerWidth - 20);
            const maxH = Math.max(260, window.innerHeight - 20);
            const w = Math.max(320, Math.min(maxW, width));
            const h = Math.max(260, Math.min(maxH, height));
            const l = Math.max(0, Math.min(Math.max(0, window.innerWidth - w), left));
            const t = Math.max(0, Math.min(Math.max(0, window.innerHeight - h), top));
            setImportant('width', w); setImportant('height', h);
            setImportant('left', l); setImportant('top', t); setAutoImportant('right','auto');
            requestAnimationFrame(syncClockScale);
          };
          const loadPos = () => {
            try {
              const pos = JSON.parse(localStorage.getItem(posKey) || 'null');
              if (pos && [pos.left,pos.top,pos.width,pos.height].every(Number.isFinite)) {
                applyGeometry(pos.left, pos.top, pos.width, pos.height);
              } else {
                const r = root.getBoundingClientRect();
                applyGeometry(r.left, r.top, r.width || 460, r.height || 420);
              }
            } catch(e) {
              const r = root.getBoundingClientRect();
              applyGeometry(r.left || 16, r.top || 16, r.width || 460, r.height || 420);
            }
          };
          const savePos = () => {
            try { const r=root.getBoundingClientRect(); localStorage.setItem(posKey, JSON.stringify({left:Math.round(r.left),top:Math.round(r.top),width:Math.round(r.width),height:Math.round(r.height)})); } catch(e) {}
          };
          let dragging=false, resizing=false, startX=0, startY=0, baseL=0, baseT=0, baseW=0, baseH=0;
          const startDrag = (ev) => {
            if (!card || ev.button === 2 || resizing) return;
            if (ev.target?.closest?.('button,.vhung-vn-clock-resize')) return;
            const rect=root.getBoundingClientRect();
            dragging=true; startX=ev.clientX; startY=ev.clientY; baseL=rect.left; baseT=rect.top;
            setImportant('left',rect.left); setImportant('top',rect.top); setAutoImportant('right','auto');
            try {(ev.currentTarget||card).setPointerCapture(ev.pointerId);} catch(e) {}
            document.body.style.userSelect='none'; ev.preventDefault(); ev.stopPropagation();
          };
          const moveDrag = (ev) => {
            if (!dragging) return;
            const maxL=Math.max(0,window.innerWidth-root.offsetWidth), maxT=Math.max(0,window.innerHeight-root.offsetHeight);
            setImportant('left',Math.max(0,Math.min(maxL,baseL+ev.clientX-startX)));
            setImportant('top',Math.max(0,Math.min(maxT,baseT+ev.clientY-startY)));
            ev.preventDefault();
          };
          const stopPointer = () => {
            if (dragging || resizing) savePos();
            dragging=false; resizing=false; document.body.style.userSelect='';
          };
          const startResize = (ev) => {
            if (ev.button !== 0) return;
            const rect=root.getBoundingClientRect();
            resizing=true; dragging=false; startX=ev.clientX; startY=ev.clientY; baseW=rect.width; baseH=rect.height; baseL=rect.left; baseT=rect.top;
            try { resizeHandle.setPointerCapture(ev.pointerId); } catch(e) {}
            document.body.style.userSelect='none'; ev.preventDefault(); ev.stopPropagation();
          };
          const moveResize = (ev) => {
            if (!resizing) return;
            const maxW=Math.max(320,window.innerWidth-baseL-4), maxH=Math.max(260,window.innerHeight-baseT-4);
            const w=Math.max(320,Math.min(maxW,baseW+ev.clientX-startX));
            const h=Math.max(260,Math.min(maxH,baseH+ev.clientY-startY));
            setImportant('width',w); setImportant('height',h);
            syncClockScale();
            ev.preventDefault();
          };
          card?.addEventListener('pointerdown', startDrag, {passive:false});
          resizeHandle?.addEventListener('pointerdown', startResize, {passive:false});
          window.addEventListener('pointermove', moveDrag, {passive:false});
          window.addEventListener('pointermove', moveResize, {passive:false});
          window.addEventListener('pointerup', stopPointer, {passive:true});
          window.addEventListener('pointercancel', stopPointer, {passive:true});
          let resizeTimer=0;
          if (window.ResizeObserver) {
            const ro=new ResizeObserver(()=>{ syncClockScale(); clearTimeout(resizeTimer); resizeTimer=setTimeout(savePos,120); });
            ro.observe(root);
          }
          window.addEventListener('resize', () => { if (!root.classList.contains('clock-hidden')) loadPos(); });
          setTimeout(loadPos, 80);
          setTimeout(syncClockScale, 120);
          refreshClockStatus();
          setInterval(refreshClockStatus, 1200);
        }""")

        ui.load(None, js="""() => {
          /* Real Vietnam time in the workflow HUD; progress/state are untouched. */
          if (window.__vhungHudVietnamClockStarted) return;
          window.__vhungHudVietnamClockStarted = true;
          const formatTime = new Intl.DateTimeFormat('vi-VN', {timeZone:'Asia/Ho_Chi_Minh',hour:'2-digit',minute:'2-digit',second:'2-digit',hour12:false});
          const tick = () => {
            const now = formatTime.format(new Date());
            document.querySelectorAll('#workflow-return-status .las-hud-vnclock').forEach(el => {el.textContent = `🕐 ${now}`;});
          };
          tick();
          setInterval(tick, 250);
        }""")


        ui.load(None, js="""() => {
          /* ONE body-level LIVE HUB. Data comes only from the real local workflow telemetry server. */
          if (window.__vhungSingleBodyHubStarted) return;
          window.__vhungSingleBodyHubStarted = true;
          const esc=v=>String(v??'').replace(/[&<>\"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;',"'":'&#39;'}[c]));
          const fmtSec=s=>{s=Math.max(0,Math.floor(Number(s)||0)); const h=Math.floor(s/3600),m=Math.floor((s%3600)/60),x=s%60; return `${String(h).padStart(2,'0')}:${String(m).padStart(2,'0')}:${String(x).padStart(2,'0')}`;};
          let root=document.getElementById('vhung-live-hub-body');
          if(!root){
            root=document.createElement('div'); root.id='vhung-live-hub-body';
            root.innerHTML=`<div class="vhung-hub-card">
              <div class="hub-top"><span class="hub-title">⚡ HƯNG PRO WORKFLOW</span><span class="hub-state">⚪ SẴN SÀNG</span></div>
              <div class="hub-stage">🚀 Quy trình</div>
              <div class="hub-status">Sẵn sàng</div>
              <div class="hub-live-detail"><span class="hub-operation">Chưa có</span></div>
              <div class="hub-counters"><span>Tiến trình: <b class="hub-pct">0%</b></span><span>Thời gian: <b class="hub-elapsed">00:00:00</b></span></div>
            </div>`;
            document.body.appendChild(root);
          }
          const tick=async()=>{
            try{
              if(!document.body.contains(root)) document.body.appendChild(root);
              const r=await fetch('http://127.0.0.1:8765/workflow/status?limit=30&t='+Date.now(),{cache:'no-store'});
              if(!r.ok) throw new Error('HTTP '+r.status);
              const d=await r.json();
              const pct=Math.max(0,Math.min(100,Number(d.pct||0)));
              const status=String(d.status||'Sẵn sàng');
              const done=/hoàn tất|đã xong|thành công/i.test(status) || Number(d.running||0)===0 && pct>=100;
              root.querySelector('.hub-state').textContent=d.level==='ERROR'?'🔴 LỖI':done?'🟢 HOÀN TẤT':d.running?'🔵 ĐANG CHẠY':'⚪ SẴN SÀNG';
              root.querySelector('.hub-stage').textContent=( {script:'📝 Kịch bản',voice:'🎙 Voice Full',gemini:'🤖 Gemini',zimage:'⚡ Tạo ảnh',qc:'✅ QC',master:'🚀 MASTER',b1:'🤖 Gemini B1',b2:'🎙 Voice / B2',b3:'🎨 Tạo ảnh B3'}[String(d.stage||'').toLowerCase()]||('🚀 '+String(d.stage||'Workflow')));
              root.querySelector('.hub-status').textContent=status.slice(0,120);
              const viOp=(v)=>{
                let x=String(v||'Chưa có');
                x=x.replace(/startup health(?: check)?(?: start| ok| fail)?/ig,'kiểm tra hệ thống');
                x=x.replace(/focus(?:ed|ing)?/ig,'đang xử lý');
                x=x.replace(/health check/ig,'kiểm tra hệ thống');
                x=x.replace(/system stats/ig,'thông số hệ thống');
                x=x.replace(/browser ready/ig,'đã mở trình duyệt');
                return x;
              };
              root.querySelector('.hub-operation').textContent=viOp(d.operation||d.status||'Chưa có').slice(0,120);
              root.querySelector('.hub-pct').textContent=String(pct)+'%';
              root.querySelector('.hub-elapsed').textContent=fmtSec(d.elapsed_s);
            }catch(err){
              root.querySelector('.hub-state').textContent='🟠 MẤT TELEMETRY';
              root.querySelector('.hub-status').textContent='Trạng thái: Không đọc được trạng thái workflow — không hiển thị %';
            }
          };
          tick(); setInterval(tick,700);
        }""")

        ui.load(None, js="""() => {
          if (window.__vhungProjectButtonGateInstalled) return;
          window.__vhungProjectButtonGateInstalled = true;
          document.addEventListener('click', (ev) => {
            const target = ev.target instanceof Element
              ? ev.target.closest('#new-project-confirm-trigger, #new-project-confirm-trigger button')
              : null;
            if (!target) return;
            const trigger = document.getElementById('new-project-confirm-trigger');
            if (!trigger) return;
            /* Click #1: do NOT cancel/stop the event. Gradio's real handler runs normally. */
            if (trigger.dataset.firstProjectCreated !== '1') {
              trigger.dataset.firstProjectCreated = '1';
              return;
            }
            /* Click #2+: only allow Gradio to proceed after confirmation. */
            const ok = window.confirm(`⚠️ TẠO PROJECT MỚI\n\nProject hiện tại sẽ được giữ nguyên trong folder cũ. Tool sẽ tạo + kết nối một Project mới và reset toàn bộ trạng thái giao diện về trắng tinh.\n\nBạn có chắc muốn tiếp tục?`);
            if (!ok) {
              ev.preventDefault();
              ev.stopImmediatePropagation();
            }
          }, true);
        }""")

        
# ============================================================
# CREATIVE STUDIO UI + IMAGE-ONLY MASTER CHAIN
# ============================================================
CREATIVE_IMAGE_ONLY_STAGES = ("IMAGE", "POSTPROCESS", "CLEAN_MAX_VIP", "QC")
CREATIVE_STUDIO_VERSION = "V11.6.2"

def _creative_image_engine_ui_value(value=None):
    value = str(value or _selected_image_engine() or CREATIVE_ENGINE_FLOW).upper().strip()
    return value if value in (CREATIVE_ENGINE_FLOW, CREATIVE_ENGINE_ZIMAGE) else CREATIVE_ENGINE_FLOW

def _creative_master_full(project, scenes, requested_engine=None, start=True):
    """Canonical image-only Master: IMAGE -> POSTPROCESS -> CLEAN -> QC."""
    project = _batch_project_name(project)
    engine = _creative_image_engine_ui_value(requested_engine)
    now = _creative_now()
    _creative_db_init()
    with _batch_db() as db:
        for scene in scenes or []:
            sid = str(scene.get("scene_id") or "").strip()
            if not sid or scene_media_type(scene, "image") != "image":
                continue
            for stage in CREATIVE_IMAGE_ONLY_STAGES:
                deps = [] if stage == "IMAGE" else (
                    ["IMAGE"] if stage == "POSTPROCESS" else
                    ["POSTPROCESS"] if stage == "CLEAN_MAX_VIP" else
                    ["CLEAN_MAX_VIP"]
                )
                jid = _creative_job_id(project, sid, stage)
                db.execute(
                    """INSERT INTO creative_jobs(
                        job_id,project_id,scene_id,stage,status,priority,max_retry,ready_at,
                        requested_engine,dependency_json,created_at,updated_at,control_state
                    ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(project_id,scene_id,stage) DO UPDATE SET
                        requested_engine=excluded.requested_engine,
                        dependency_json=excluded.dependency_json,
                        updated_at=excluded.updated_at,
                        control_state='RUN'
                    """,
                    (jid, project, sid, stage, "PENDING",
                     _creative_priority("NORMAL"), CREATIVE_MAX_RETRY, now,
                     engine, _creative_json(deps), now, now, "RUN"),
                )
    if start:
        _creative_set_control(project, "RUNNING", "Creative Studio Master")
        _creative_start_workers(project, count=1, worker_type="FLOW" if engine == CREATIVE_ENGINE_FLOW else "Z_IMAGE")
        _creative_reconcile_dependencies(project)
        _creative_start_scheduler(project)
    return True

def _creative_studio_queue_html(project=""):
    project = _batch_project_name(project or SETTINGS.get("selected_project", ""))
    if not project:
        return "<div class='creative-empty'>Chưa chọn Project.</div>"
    _creative_db_init()
    with _batch_db() as db:
        rows = db.execute(
            """SELECT scene_id,stage,status,retry_count,actual_engine,error,worker_id,account_id
               FROM creative_jobs WHERE project_id=?
               ORDER BY CAST(REPLACE(scene_id,'Scene_','') AS INTEGER),stage""",
            (project,),
        ).fetchall()
    if not rows:
        return "<div class='creative-empty'>Chưa có Job. Bấm TẠO / CHẠY để khởi tạo.</div>"
    icons = {"DONE":"🟢","RUNNING":"🔵","READY":"🟡","PENDING":"⚪",
             "WAITING_DEPENDENCY":"⏳","RETRY":"🟠","FAILED":"🔴",
             "CAPTCHA":"🟥","SKIPPED":"⏭️"}
    html_rows = []
    for r in rows:
        status = str(r["status"] or "")
        err = str(r["error"] or "")
        html_rows.append(
            "<tr>"
            f"<td>{html.escape(str(r['scene_id'] or ''))}</td>"
            f"<td>{html.escape(str(r['stage'] or ''))}</td>"
            f"<td>{icons.get(status,'⚪')} {html.escape(status)}</td>"
            f"<td>{html.escape(str(r['actual_engine'] or r['account_id'] or ''))}</td>"
            f"<td>{html.escape(str(r['retry_count'] or 0))}</td>"
            f"<td>{html.escape(str(r['worker_id'] or ''))}</td>"
            f"<td title='{html.escape(err)}'>{html.escape(err[:160])}</td>"
            "</tr>"
        )
    return (
        "<div class='creative-queue-wrap'><table class='creative-queue'>"
        "<thead><tr><th>Scene</th><th>Stage</th><th>Status</th><th>Engine/Account</th>"
        "<th>Retry</th><th>Worker</th><th>Error</th></tr></thead>"
        "<tbody>" + "".join(html_rows) + "</tbody></table></div>"
    )

def _creative_studio_status_ui(project=""):
    project = _batch_project_name(project or SETTINGS.get("selected_project", ""))
    if not project:
        return "⚪ Chưa có Project."
    with _batch_db() as db:
        rows = db.execute(
            "SELECT status,COUNT(*) n FROM creative_jobs WHERE project_id=? GROUP BY status",
            (project,),
        ).fetchall()
    counts = {str(r["status"]): int(r["n"]) for r in rows}
    return (
        f"**{project}** · 🟢 DONE {counts.get('DONE',0)} · "
        f"🔵 RUN {counts.get('RUNNING',0)} · 🟠 RETRY {counts.get('RETRY',0)} · "
        f"🔴 FAIL {counts.get('FAILED',0)} · 🟥 CAPTCHA {counts.get('CAPTCHA',0)}"
    )

def _creative_studio_create_run_ui(project, engine):
    project = _batch_project_name(project or SETTINGS.get("selected_project", ""))
    if not project:
        return "🔴 Chưa chọn Project.", _creative_studio_queue_html("")
    scenes = load_manifest(project)
    if not scenes:
        return "🔴 Project chưa có Scene/manifest.", _creative_studio_queue_html(project)
    engine = _creative_image_engine_ui_value(engine)
    if engine == CREATIVE_ENGINE_FLOW and not _creative_engine_available(CREATIVE_ENGINE_FLOW):
        return "🔴 Google Flow chưa CONNECTED/ACTIVE. Hãy VERIFY Profile trước.", _creative_studio_queue_html(project)
    try:
        _creative_master_full(project, scenes, engine, start=True)
        workflow_cmd_log("CREATIVE_STUDIO", "MASTER_STARTED", project=project, scenes=len(scenes), engine=engine)
        return f"🟢 Đã đưa {len(scenes)} Scene vào queue · Engine={engine}", _creative_studio_queue_html(project)
    except Exception as exc:
        return f"🔴 Không khởi tạo Creative Studio: {str(exc)[:1200]}", _creative_studio_queue_html(project)

def _creative_studio_refresh_ui(project):
    return _creative_studio_status_ui(project), _creative_studio_queue_html(project)

def _creative_studio_pause_ui(project):
    project = _batch_project_name(project or SETTINGS.get("selected_project", ""))
    if not project: return "🔴 Chưa chọn Project."
    _creative_set_control(project, "PAUSED", "Creative Studio PAUSE")
    return "⏸ Đã PAUSE queue."

def _creative_studio_resume_ui(project):
    project = _batch_project_name(project or SETTINGS.get("selected_project", ""))
    if not project: return "🔴 Chưa chọn Project."
    _creative_resume(project)
    return "▶️ Đã RESUME queue."

def _creative_studio_stop_ui(project):
    project = _batch_project_name(project or SETTINGS.get("selected_project", ""))
    if not project: return "🔴 Chưa chọn Project."
    _creative_stop(project)
    return "⏹ Đã STOP queue của Project này."

def _creative_studio_depth_ui(project):
    project = _batch_project_name(project or SETTINGS.get("selected_project", ""))
    scenes = load_manifest(project)
    if not scenes: return "🔴 Không có Scene."
    try:
        from PIL import Image, ImageOps
        out_dir = _project_image_dir(project_dir(project)) / "Depth"
        out_dir.mkdir(parents=True, exist_ok=True)
        done = 0
        for sc in scenes:
            sid = str(sc.get("scene_id") or "").strip()
            src = _creative_find_existing_scene_output(project, sid)
            if not src: continue
            out = out_dir / f"{sid}_depth.png"
            with Image.open(src) as im:
                im = ImageOps.autocontrast(im.convert("L").resize(
                    (CREATIVE_IMAGE_WIDTH, CREATIVE_IMAGE_HEIGHT), Image.Resampling.LANCZOS
                ))
                im.save(out, "PNG", optimize=True)
            done += 1
        return f"🌀 Depth batch hoàn tất: {done}/{len(scenes)} Scene → Hình ảnh/Depth/"
    except Exception as exc:
        return f"🔴 Depth batch lỗi: {exc}"

def _creative_studio_motion_ui(project):
    project = _batch_project_name(project or SETTINGS.get("selected_project", ""))
    scenes = load_manifest(project)
    if not scenes: return "🔴 Không có Scene."
    out_dir = _project_image_dir(project_dir(project)) / "MotionControl"
    out_dir.mkdir(parents=True, exist_ok=True)
    done = 0
    for sc in scenes:
        sid = str(sc.get("scene_id") or "").strip()
        if not sid: continue
        write_json(out_dir / f"{sid}.json", {
            "scene_id": sid, "type": "motion_control",
            "source_image": str(_creative_find_existing_scene_output(project, sid) or ""),
            "depth": str(out_dir.parent / "Depth" / f"{sid}_depth.png"),
            "camera": {"pan_x":0.0,"pan_y":0.0,"zoom":0.0,"rotation":0.0},
            "generated_at": _creative_now(),
        })
        done += 1
    return f"🎛 Motion Control batch hoàn tất: {done}/{len(scenes)} Scene → Hình ảnh/MotionControl/"

def _creative_studio_timeline_ui(project):
    project = _batch_project_name(project or SETTINGS.get("selected_project", ""))
    scenes = load_manifest(project)
    if not scenes: return "🔴 Không có Scene."
    out = project_dir(project) / "Dòng thời gian" / "creative_timeline.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    items=[]; t=0.0
    for sc in scenes:
        sid=str(sc.get("scene_id") or "").strip()
        if not sid: continue
        try: dur=float(sc.get("duration") or 0)
        except Exception: dur=0.0
        items.append({"scene_id":sid,"start":t,"duration":dur,"end":t+dur}); t+=dur
    write_json(out, {"project":project,"fps":30,"duration":t,"scenes":items,"generated_at":_creative_now()})
    return f"🧭 Timeline đã tạo: {out}"


with gr.Tabs(elem_id='main-workflow-tabs') as main_tabs:
            # UI SOURCE OF TRUTH: top-level Gradio tabs; Độc Lập is the dedicated standalone workspace. Every direct child of gr.Tabs MUST be gr.Tab/gr.TabItem.
            # =================================================
            # VOICE
            # =================================================
            # =================================================
            # ENVIRONMENT + VOICE SAMPLE / CLONE
            # Tab đầu tiên: chỉ kiểm tra server có hoạt động hay không.
            # Không cài/update/model/start/stop server từ UI ở đây.
            with gr.Tab('⚙ Môi trường'):
                gr.Markdown('### 💾 Cấu hình workflow — lưu nhiều cấu hình')
                cfg_preview = gr.Markdown(
                    profile_preview_rows(current_workflow_profile()),
                    elem_id='cfg-profile-preview',
                    elem_classes=['cfg-profile-preview']
                )
                with gr.Row():
                    cfg_select = gr.Dropdown(
                        choices=workflow_profile_choices(),
                        value=current_workflow_profile() or None,
                        label='Cấu hình đã lưu',
                        scale=3
                    )
                    cfg_name = gr.Textbox(value=current_workflow_profile() or None, label='Tên cấu hình', scale=2)
                    cfg_gemini_directive = gr.Dropdown(
                        choices=GEMINI_VISUAL_DIRECTIVE_CHOICES,
                        value=workflow_profile_directive_mode(current_workflow_profile()),
                        label="🎯 Directive hình ảnh cho Gemini",
                        info="Lưu theo Workflow Profile → workflow_config.json.",
                        interactive=True,
                    )
                    cfg_gemini_directive.change(
                        _set_gemini_visual_directive,
                        inputs=[cfg_gemini_directive],
                        outputs=[],
                    )
                    cfg_save = gr.Button('💾 Lưu cấu hình & Cập nhật', variant='primary')
                    cfg_use = gr.Button('▶ Sử dụng', variant='primary')
                    cfg_delete = gr.Button('🗑 Xóa')
                gr.Markdown("### 🎨 Môi trường tạo ảnh")
                image_engine = gr.Radio(
                    choices=CREATIVE_IMAGE_ENGINE_CHOICES,
                    value=_selected_image_engine(),
                    label="Chọn engine Master",
                    info="Chỉ chạy engine được chọn. Không tự fallback sang engine khác."
                )
                with gr.Row():
                    image_engine_check = gr.Button("🔌 KIỂM TRA KẾT NỐI", variant="primary")
                    image_engine_status = gr.Markdown("🟡 Chưa VERIFY kết nối engine.")
                image_engine.change(
                    _wrap_gradio_callback(_save_image_engine_setting),
                    image_engine, image_engine_status, show_progress="minimal"
                )
                image_engine_check.click(
                    _wrap_gradio_callback(_check_image_engine_ui),
                    image_engine, image_engine_status, show_progress="minimal"
                )

                gr.Markdown("### 🧩 Playwright Runtime — Google Chrome Stable")
                with gr.Row():
                    playwright_check_btn = gr.Button("🔌 KIỂM TRA RUNTIME")
                    playwright_install_btn = gr.Button("🛠 CÀI / SỬA PLAYWRIGHT", variant="primary")
                playwright_status = gr.Markdown(_playwright_runtime_ui())
                playwright_check_btn.click(
                    _wrap_gradio_callback(_playwright_runtime_ui),
                    None, playwright_status, show_progress="hidden"
                )
                playwright_install_btn.click(
                    _wrap_gradio_callback(_playwright_install_ui),
                    None, playwright_status, show_progress="hidden"
                )

                gr.Markdown("### 👥 Google Flow Account Pool — đăng nhập Profile thủ công")
                flow_pool_profile_path = gr.Textbox(
                    label="📁 Profile folder (để trống = tool tự tạo Profile mới)",
                    placeholder="Ví dụ: F:\\FlowProfiles\\Profile001",
                    info="Mỗi lần MỞ PROFILE MỚI sẽ tạo một Profile/Account record riêng. Bạn tự đăng nhập Google/Flow."
                )
                with gr.Row():
                    flow_pool_new = gr.Button("➕ MỞ PROFILE MỚI", variant="primary")
                    flow_pool_reopen = gr.Button("🔓 MỞ LẠI PROFILE CŨ", variant="primary")
                    flow_pool_verify = gr.Button("🔌 VERIFY PROFILE", variant="primary")
                    flow_pool_check = gr.Button("🔎 KIỂM TRA PROFILE ĐÃ VERIFY", variant="secondary")
                    flow_pool_refresh = gr.Button("🔄 REFRESH POOL")
                flow_pool_account = gr.Textbox(label="Profile ID đang mở", placeholder="FLOW-0001", value="")
                flow_pool_status = gr.Markdown(_creative_account_pool_html())
                flow_pool_hint = gr.Markdown(
                    "🟡 Lần đầu: MỞ PROFILE MỚI → đăng nhập Google/Flow → VERIFY. "
                    "Lần sau: chọn FLOW-xxxx → MỞ LẠI PROFILE CŨ; tool dùng đúng thư mục profile đã lưu, không tạo account mới."
                )
                flow_pool_new.click(
                    _wrap_gradio_callback(_creative_flow_manual_profile_open),
                    flow_pool_profile_path,
                    [flow_pool_status, flow_pool_hint, flow_pool_account],
                    show_progress="minimal"
                )
                flow_pool_reopen.click(
                    _wrap_gradio_callback(_creative_flow_account_open),
                    flow_pool_account,
                    [flow_pool_status, flow_pool_hint],
                    show_progress="minimal"
                )
                flow_pool_verify.click(
                    _wrap_gradio_callback(_creative_flow_manual_profile_verify),
                    flow_pool_account,
                    [flow_pool_status, flow_pool_hint],
                    show_progress="minimal"
                )
                flow_pool_check.click(
                    _wrap_gradio_callback(_flow_verified_profiles_report),
                    None, [flow_pool_status, flow_pool_hint], show_progress="minimal"
                )
                flow_pool_refresh.click(
                    _wrap_gradio_callback(lambda: _creative_account_pool_html()),
                    None, flow_pool_status, show_progress="hidden"
                )

                with gr.Row():
                    cfg_language = gr.Dropdown(choices=[('🤖 Auto','auto')]+[(language_label(k),k) for k in LANGUAGE_CATALOG], value=SETTINGS.get('selected_language','auto'), label='Ngôn ngữ')
                    _cfg_lang0 = SETTINGS.get('selected_language','auto') or 'auto'
                    _cfg_eng0 = SETTINGS.get('selected_engine','auto') or 'auto'
                    _cfg0_boot = workflow_profile_snapshot(current_workflow_profile())
                    _cfg_actual_lang0 = _effective_voice_language(_cfg_lang0, str(_cfg0_boot.get('script') or SETTINGS.get('workflow_script','') or ''))
                    _cfg_voice_choices0 = all_voice_choices(_cfg_actual_lang0, _cfg_eng0)
                    _cfg_voice0 = normalize_voice_value(SETTINGS.get('selected_voice') or '')
                    if _cfg_voice0 and _cfg_voice0 not in {v for _, v in _cfg_voice_choices0}:
                        _cfg_voice_choices0.insert(0, (Path(_cfg_voice0).name if Path(_cfg_voice0).expanduser().is_file() else _cfg_voice0, _cfg_voice0))
                    if not _cfg_voice0:
                        _cfg_voice0 = _cfg_voice_choices0[0][1] if _cfg_voice_choices0 else None
                    cfg_engine = gr.Dropdown(choices=engine_choices(_cfg_actual_lang0), value=_cfg_eng0, label='Server / Engine')
                    cfg_voice = gr.Dropdown(choices=_cfg_voice_choices0, value=_cfg_voice0, label='Voice', filterable=True)
                with gr.Row():
                    _cfg_project0 = workflow_profile_snapshot(current_workflow_profile()).get('project') or default_project or None
                    cfg_project = gr.Dropdown(choices=projects_now, value=_cfg_project0 if _cfg_project0 in projects_now else default_project or None, allow_custom_value=True, label='📁 Project đang kết nối', scale=3)
                    # Legacy profile compatibility only. Scene density is controlled by
                    # Gemini Scene Director semantic beats; this value is never exposed
                    # as a user-facing slider anymore.
                    cfg_target = gr.State(value=_safe_int(SETTINGS.get('scene_target_chars',320),320,180,550))
                gr.Markdown('### 🎨 Phân tích phong cách')
                _cfg0 = workflow_profile_snapshot(current_workflow_profile())
                cfg_competitor_urls = gr.Textbox(
                    label='🔗 Link style (mỗi link một dòng)',
                    lines=5,
                    value=str(_cfg0.get('gemini_competitor_urls') or ''),
                    placeholder='https://www.youtube.com/@KenhDoiThu\nhttps://www.youtube.com/watch?v=...',
                    info='Phân tích Style từ nguồn tham chiếu và ghép Global Visual Style.'
                )
                cfg_status = gr.Markdown(profile_preview_rows(current_workflow_profile()))
                with gr.Group(elem_classes=['style-reference-group']):
                    gr.Markdown('#### 🖼 Ảnh tham chiếu Nhân vật → Gemini + Z-IMAGE-TURBO (tối đa 3)')
                    zimage_ref_enabled_env = gr.Checkbox(
                        label='Bật dùng ảnh tham chiếu cho Gemini + Z-IMAGE-TURBO',
                        value=bool(SETTINGS.get('zimage_reference_enabled', True)),
                        info='Tối đa 3 ảnh. Cùng một bộ MASTER trong Project/Nhân vật/ sẽ được gửi thật cho Gemini và dùng cho Z-IMAGE-TURBO.'
                    )
                    zimage_ref_env_images = gr.File(
                        label='🖼 Ảnh mẫu nhân vật — Gemini + Z-IMAGE-TURBO — tối đa 3 ảnh',
                        file_count='multiple',
                        type='filepath',
                    )
                    zimage_ref_env_status = gr.Markdown(
                        '⚪ Chưa có ảnh mẫu. Gemini và Z-IMAGE-TURBO sẽ chạy không có Character Reference.'
                    )
                    def _save_zimage_ref_setting(enabled):
                        SETTINGS['zimage_reference_enabled'] = bool(enabled)
                        save_settings()
                        msg = '🟢 Z-IMAGE-TURBO Reference đã bật — tối đa 3 ảnh.' if enabled else '⚪ Z-IMAGE-TURBO Reference đã tắt — Z-IMAGE-TURBO chỉ dùng Prompt.'
                        workflow_cmd_log('Z-IMAGE-TURBO', 'REFERENCE_SETTING', enabled=bool(enabled), max_images=3)
                        return msg
                    zimage_ref_enabled_env.change(_wrap_gradio_callback(_save_zimage_ref_setting), zimage_ref_enabled_env, zimage_ref_env_status, show_progress='hidden')
                    def _save_zimage_env_refs(files, project_name):
                        project_name = safe_name(project_name or SETTINGS.get("selected_project", ""))
                        stored = _zimage_store_master_references(files, project_name=project_name)
                        if stored:
                            msg = f'🟢 Đã nạp {len(stored)}/3 ảnh → `{project_name}/Nhân vật/` → MASTER sẽ gửi ảnh thật cho Gemini B1 và dùng cùng ảnh cho Z-IMAGE-TURBO.'
                        else:
                            msg = f'⚪ Không có ảnh tham chiếu trong `{project_name}/Nhân vật/` — Gemini + Z-IMAGE-TURBO chỉ dùng Prompt.'
                        workflow_cmd_log('Z-IMAGE-TURBO', 'REFERENCE_ENV_READY', project=project_name, count=len(stored), max_images=3)
                        return msg
                    zimage_ref_env_images.change(_wrap_gradio_callback(_save_zimage_env_refs), [zimage_ref_env_images, cfg_project], zimage_ref_env_status, show_progress='minimal')

                def _cfg_view(name):
                    c = workflow_profile_snapshot(name)
                    lang = c.get('language','auto') or 'auto'
                    eng = c.get('engine','auto') or 'auto'
                    voice = normalize_voice_value(c.get('voice','') or '')
                    project_name = c.get('project','Project_001') or 'Project_001'
                    target = _safe_int(c.get('target',320), 320, 180, 550)
                    competitor_urls = str(c.get('gemini_competitor_urls') or '')
                    # workflow_config.json is the source of truth for the selected profile.
                    gemini_directive_mode = workflow_profile_directive_mode(name)
                    actual_lang = lang if lang != 'auto' else detect_script_language(c.get('script','') or SETTINGS.get('workflow_script','') or '')
                    eng_choices = engine_choices(actual_lang)
                    if eng not in {v for _,v in eng_choices}:
                        eng = 'auto'
                    voice_choices = all_voice_choices(actual_lang, eng)
                    if voice and voice not in {v for _,v in voice_choices}:
                        voice_choices.insert(0, (Path(str(voice)).name if Path(str(voice)).expanduser().is_file() else str(voice), str(voice)))
                    if not voice:
                        voice = voice_choices[0][1] if voice_choices else None
                    return (
                        profile_preview_rows(name), lang, gr.Dropdown(choices=eng_choices, value=eng),
                        gr.Dropdown(choices=voice_choices, value=voice),
                        project_name, target, competitor_urls,
                        f'👁 Đang xem cấu hình **{name}** — chưa áp dụng vào workflow.\n\n📌 Cấu hình được lưu riêng trong `LocalAIStudio_Data/workflow_config.json`.',
                        gemini_directive_mode
                    )

                def _cfg_language_changed(lang, current_engine, current_voice):
                    # Changing language must NOT silently discard a manually selected
                    # server/voice. Keep the explicit engine when it remains valid;
                    # otherwise fall back to Auto. Keep the selected reference voice
                    # whenever that voice is valid for the new engine.
                    actual_lang = _effective_voice_language(lang, SETTINGS.get('workflow_script','') or '')
                    choices = engine_choices(actual_lang)
                    valid_engines = {v for _, v in choices}
                    eng = current_engine or SETTINGS.get('selected_engine','auto') or 'auto'
                    eng = eng if eng in valid_engines else 'auto'
                    voices = all_voice_choices(actual_lang, eng)
                    valid_voices = {v for _, v in voices}
                    saved_voice = current_voice or SETTINGS.get('selected_voice','') or ''
                    value = saved_voice if saved_voice in valid_voices else (voices[0][1] if voices else None)
                    return gr.Dropdown(choices=choices, value=eng), gr.Dropdown(choices=voices, value=value)

                def _cfg_engine_changed(lang, eng, current_voice):
                    actual_lang = _effective_voice_language(lang, SETTINGS.get('workflow_script','') or '')
                    eng = eng or 'auto'
                    voices = all_voice_choices(actual_lang, eng)
                    valid_voices = {v for _, v in voices}
                    saved_voice = current_voice or SETTINGS.get('selected_voice','') or ''
                    value = saved_voice if saved_voice in valid_voices else (voices[0][1] if voices else None)
                    return gr.Dropdown(choices=voices, value=value)

                cfg_language.change(_wrap_gradio_callback(_cfg_language_changed), [cfg_language, cfg_engine, cfg_voice], [cfg_engine, cfg_voice], show_progress='minimal')
                cfg_engine.change(_wrap_gradio_callback(_cfg_engine_changed), [cfg_language, cfg_engine, cfg_voice], cfg_voice, show_progress='minimal')

                def _save_cfg(name, lang, eng, voice, project_name, target, competitor_urls, gemini_directive_mode):
                    SETTINGS['gemini_reference_images'] = []
                    SETTINGS['gemini_competitor_urls'] = str(competitor_urls or '').strip()
                    saved_name = safe_name(name or '')
                    existing_cfg = load_workflow_profiles().get(saved_name, {}) if saved_name else {}
                    # Directive is an explicit Workflow Profile setting.
                    gemini_directive_mode = normalize_gemini_visual_directive_mode(
                        gemini_directive_mode
                        or existing_cfg.get('gemini_visual_directive_mode')
                        or SETTINGS.get('gemini_visual_directive_mode', 'auto')
                    )
                    SETTINGS['gemini_visual_directive_mode'] = gemini_directive_mode
                    script_for_profile = SETTINGS.get('workflow_script','') or ''
                    _, eng, resolved, voice = _canonical_voice_for_engine(lang, eng, normalize_voice_value(voice), script_for_profile)
                    msg = save_workflow_profile_values(name, lang, eng, voice, project_name, target, [], competitor_urls, gemini_directive_mode)
                    if saved_name in load_workflow_profiles():
                        load_workflow_profile(saved_name)
                        SETTINGS['gemini_visual_directive_mode'] = gemini_directive_mode
                        SETTINGS['workflow_profile'] = saved_name
                        save_workflow_profiles(load_workflow_profiles())
                    workflow_cmd_log(
                        'CONFIG', 'GEMINI_VISUAL_DIRECTIVE_SAVED',
                        profile=saved_name, mode=gemini_directive_mode, directive=dict(GEMINI_VISUAL_DIRECTIVE_CHOICES).get(gemini_directive_mode, gemini_directive_mode), project=safe_name(project_name or ''),
                    )
                    competitor_link_count = sum(1 for u in str(competitor_urls or '').splitlines() if u.strip())
                    # Re-read workflow_config.json so the visible dropdown is refreshed from the same store that was written.
                    fresh_choices = workflow_profile_choices()
                    fresh_preview = profile_preview_rows(saved_name)
                    return (
                        gr.update(choices=fresh_choices, value=saved_name),
                        fresh_preview,
                        msg + f"\n\n📌 Gemini Style source: **{competitor_link_count} link đối thủ + toàn bộ Kịch bản**"
                        + "\n\n💾 Đã lưu riêng trong LocalAIStudio_Data/workflow_config.json."
                    )

                def _live_competitor_urls(profile_name, project_name, competitor_urls):
                    name = safe_name(profile_name or current_workflow_profile() or 'Default')
                    project_name = safe_name(project_name or SETTINGS.get('selected_project',''))
                    SETTINGS['selected_project'] = project_name or SETTINGS.get('selected_project','')
                    SETTINGS['gemini_reference_images'] = []
                    SETTINGS['gemini_competitor_urls'] = str(competitor_urls or '').strip()
                    SETTINGS['workflow_profile'] = name if name in load_workflow_profiles() else SETTINGS.get('workflow_profile','')
                    save_settings()
                    competitor_link_count = sum(1 for u in str(competitor_urls or '').splitlines() if u.strip())
                    return f"🟢 Gemini Style source đã lưu: **{competitor_link_count} link đối thủ + toàn bộ Kịch bản**"

                def _delete_cfg():
                    """Delete the currently active profile without using the Dropdown as an input.

                    Gradio can preprocess an empty Dropdown value before the callback runs;
                    when the last profile has been deleted its choices are ``[]`` and the
                    browser may still submit ``""``. Using Settings as the source here avoids
                    the ``Value:  is not in the list of choices: []`` crash entirely.
                    """
                    name = str(SETTINGS.get("workflow_profile", "") or "").strip()
                    msg, _ = delete_workflow_profile(name)
                    choices = workflow_profile_choices()
                    # Never invent a selected profile after deletion.
                    value = None
                    return gr.update(choices=choices, value=value), profile_preview_rows(""), msg + "\n\n📌 Cấu hình được lưu riêng trong `LocalAIStudio_Data/workflow_config.json`."

                def _sync_saved_profile_to_workflow(name):
                    name = str(name or '').strip() or current_workflow_profile()
                    data = apply_workflow_profile(name)
                    lang = data.get('language','auto') or SETTINGS.get('selected_language','auto') or 'auto'
                    if lang not in LANGUAGE_CATALOG and lang != 'auto': lang = 'auto'
                    eng = data.get('engine','auto') or SETTINGS.get('selected_engine','auto') or 'auto'
                    voice = data.get('voice','') or SETTINGS.get('selected_voice','') or ''
                    script = data.get('script','') or SETTINGS.get('workflow_script','') or ''
                    actual, eng_value, resolved_value, voice_value = _canonical_voice_for_engine(
                        lang, eng, voice, script
                    )
                    ecs = engine_choices(actual)
                    eng_value = eng_value if eng_value in {v for _,v in ecs} else 'auto'
                    vcs = all_voice_choices(actual, eng_value)
                    if voice_value and Path(str(voice_value)).expanduser().is_file() and voice_value not in {v for _, v in vcs}:
                        vcs.insert(0, (Path(str(voice_value)).name, voice_value))
                    if voice_value and voice_value not in {v for _, v in vcs}:
                        vcs.insert(0, (Path(str(voice_value)).name if Path(str(voice_value)).expanduser().is_file() else str(voice_value), str(voice_value)))
                    if not voice_value:
                        voice_value = vcs[0][1] if vcs else None
                    SETTINGS['selected_engine'] = eng_value
                    SETTINGS['selected_voice'] = voice_value or ''
                    project_name = (safe_name(SETTINGS.get('selected_project','')) if PROJECT_CONNECTED and SETTINGS.get('selected_project_path') else (data.get('project','') or SETTINGS.get('selected_project','') or ''))
                    # The selected workflow profile owns the Gemini directive.
                    gemini_directive_mode = workflow_profile_directive_mode(name)
                    SETTINGS['gemini_visual_directive_mode'] = gemini_directive_mode
                    workflow_cmd_log('CONFIG', 'WORKFLOW_PROFILE_SELECTED', profile=name, gemini_directive_mode=gemini_directive_mode, directive=dict(GEMINI_VISUAL_DIRECTIVE_CHOICES).get(gemini_directive_mode, gemini_directive_mode))
                    save_settings()
                    return (data.get('message',''), _voice_selection_status(lang, eng_value, voice_value), script, data.get('target',320),
                            _project_dropdown_update(project_name),
                            gr.update(choices=[('🤖 Tự động nhận dạng từ Kịch bản','auto')] + [(language_label(k),k) for k in LANGUAGE_CATALOG], value=lang),
                            gr.Dropdown(choices=ecs, value=eng_value), gr.Dropdown(choices=vcs, value=voice_value),
                            name, project_name,
                            gr.update(choices=workflow_profile_choices(), value=name))

                cfg_competitor_urls.change(_wrap_gradio_callback(_live_competitor_urls),[cfg_select,cfg_project,cfg_competitor_urls],cfg_status,show_progress='minimal')
                # cfg_project/cfg_target are ordinary inputs; no zero-output handlers needed.

                gr.Markdown('### 🟢 GPU / CUDA')
                gpu_start_status=gr.Markdown('🟦 GPU/CUDA chưa tự chạy khi mở Tool — bấm **🔄 Kiểm tra lại GPU/CUDA** khi cần.')
                gpu_refresh_btn=gr.Button('🔄 Kiểm tra lại GPU/CUDA')
                gpu_refresh_btn.click(lambda *args:gpu_report(),outputs=gpu_start_status,show_progress='minimal')

                gr.Markdown('### 🖥 Điều khiển Server thủ công')
                with gr.Row():
                    env_server=gr.Dropdown(choices=[(ENGINE_INFO[k]['title'],k) for k in ENGINE_INFO],value='vieneu',label='Server',scale=3)
                    env_start_btn=gr.Button('▶ Bật Server',variant='primary')
                    env_stop_btn=gr.Button('■ Tắt Server')
                    env_stop_all_btn=gr.Button('⏹ DỪNG TẤT CẢ', variant='stop')
                    env_check=gr.Button('🔎 Test Server',variant='secondary')
                env_manual_status=gr.Markdown('Có thể bật/tắt server thủ công để debug. Voice Full vẫn tự chuyển server theo cấu hình.')
                env_start_btn.click(lambda n:MANAGER.start(n or 'vieneu'),env_server,env_manual_status)
                env_stop_btn.click(lambda n:MANAGER.stop(n or 'vieneu'),env_server,env_manual_status)
                env_stop_all_btn.click(_wrap_gradio_callback(stop_all_runtime_ui), outputs=env_manual_status, queue=False, show_progress='hidden')

                gr.Markdown('### 🖥 Kiểm tra Voice Server')
                with gr.Row():
                    env_check = gr.Button('🔎 Test Server', variant='primary')
                env_progress = gr.Slider(0,100,0,step=1,label='Tiến trình',interactive=False)
                env_status = gr.Markdown('Chỉ kiểm tra trạng thái server hiện tại. Không tự mở, không tự tắt, không cài/update.')
                env_table = gr.Dataframe(
                    headers=['Server','Environment','Port','Health'],
                    value=[[ENGINE_INFO[k]['short'],
                            '✅ Có' if MANAGER.installed(k) else '❌ Thiếu',
                            ENGINE_INFO[k]['port'],
                            '🟢 Online' if MANAGER.running(k) and MANAGER.health(k) else '🔴 Offline']
                           for k in ENGINE_INFO],
                    interactive=False, wrap=True
                )

                def _environment_server_test(name):
                    name = name or 'vieneu'
                    info = ENGINE_INFO[name]
                    installed = MANAGER.installed(name)
                    running = MANAGER.running(name)
                    healthy = MANAGER.health(name) if running else False
                    if not installed:
                        msg = f"❌ **{info['short']}**: chưa có environment."
                        state = '❌ Environment thiếu'
                    elif healthy:
                        msg = f"✅ **{info['short']}** đang hoạt động bình thường — `{info['url']}/health` trả OK."
                        state = '🟢 Online'
                    elif running:
                        msg = f"⚠ **{info['short']}** có process nhưng health chưa phản hồi. Kiểm tra log server."
                        state = '🟠 Process chạy / Health lỗi'
                    else:
                        msg = f"🔴 **{info['short']}** đang tắt. Không tự mở server."
                        state = '🔴 Offline'
                    rows = []
                    for k, i in ENGINE_INFO.items():
                        r = MANAGER.running(k)
                        h = MANAGER.health(k) if r else False
                        rows.append([i['short'], '✅ Có' if MANAGER.installed(k) else '❌ Thiếu', i['port'], '🟢 Online' if h else '🔴 Offline'])
                    return 100, msg, rows

                env_check.click(_wrap_gradio_callback(_environment_server_test), env_server, [env_progress, env_status, env_table], show_progress='minimal')
                # Do not run GPU/CUDA probes during page load: each virtualenv check can take up to 30s
                # and would leave the Gradio browser on the Loading screen. The check remains manual.

                gr.Markdown('---')
                gr.Markdown('### 🎧 Voice có sẵn — nghe giọng mẫu')
                with gr.Row():
                    available_lang = gr.Dropdown(choices=[('🤖 Auto','auto')] + [(language_label(k), k) for k in LANGUAGE_CATALOG], value='vi', label='Ngôn ngữ')
                    available_engine = gr.Dropdown(choices=engine_choices('vi'), value='auto', label='Server / Engine')
                    _initial_available_choices = all_voice_choices('vi', 'auto')
                    _initial_available_value = (SETTINGS.get('selected_voice') if SETTINGS.get('selected_voice') in {v for _, v in _initial_available_choices} else (_initial_available_choices[0][1] if _initial_available_choices else None))
                    available_voice = gr.Dropdown(choices=_initial_available_choices, value=_initial_available_value, label='Voice', filterable=True)
                available_text = gr.Textbox(value='Xin chào, đây là câu nghe thử cho giọng hiện tại.', lines=2, label='Câu nghe thử')
                with gr.Row():
                    available_preview_file_btn = gr.Button('▶ Nghe file')
                    available_preview_ai_btn = gr.Button('🎧 Nghe thử bằng AI', variant='primary')
                    available_use_btn = gr.Button('✓ Dùng giọng này')
                    available_refresh_btn = gr.Button('🔄 Làm mới')
                    voice_refresh_all = gr.Button('🔄 Làm mới Voice/Clone')
                available_progress = gr.Slider(0,100,0,step=1,label='Tiến trình',interactive=False, elem_classes=['las-progress-compact'])
                available_audio = gr.Audio(type='filepath', label='Audio mẫu / nghe thử', interactive=False)
                available_status = gr.Markdown()
                voice_refresh_status = gr.Markdown()

                def _refresh_available(lang, eng):
                    choices = all_voice_choices(lang, eng)
                    configured = normalize_voice_value(SETTINGS.get("selected_voice", "") or "")
                    val = configured if configured in {v for _, v in choices} else (choices[0][1] if choices else None)
                    return gr.Dropdown(choices=engine_choices(lang), value='auto'), gr.Dropdown(choices=choices, value=val)

                available_lang.change(_wrap_gradio_callback(_refresh_available), [available_lang, available_engine], [available_engine, available_voice])
                available_engine.change(lambda l,e: gr.Dropdown(choices=all_voice_choices(l,e), value=(str(SETTINGS.get("selected_voice", "") or "").strip() if str(SETTINGS.get("selected_voice", "") or "").strip() in {v for _,v in all_voice_choices(l,e)} else (all_voice_choices(l,e)[0][1] if all_voice_choices(l,e) else None))), [available_lang,available_engine], available_voice)
                available_refresh_btn.click(lambda l,e: gr.Dropdown(choices=all_voice_choices(l,e), value=(str(SETTINGS.get("selected_voice", "") or "").strip() if str(SETTINGS.get("selected_voice", "") or "").strip() in {v for _,v in all_voice_choices(l,e)} else (all_voice_choices(l,e)[0][1] if all_voice_choices(l,e) else None))), [available_lang,available_engine], available_voice)
                available_preview_file_btn.click(lambda v: (100,)+available_voice_preview(v,''), available_voice, [available_progress,available_audio,available_status])
                available_preview_ai_btn.click(lambda l,e,v,t: preview_tts(l,e,v,t), [available_lang,available_engine,available_voice,available_text], [available_audio,available_status])
                available_use_btn.click(lambda v: save_selected_voice(v), available_voice, available_status)

                clone_modal_toggle = gr.Button('🧬 Clone Voice', variant='secondary')
                with gr.Column(visible=False) as clone_modal:
                    gr.Markdown('#### 🧬 Clone Voice — tạo WAV trực tiếp')
                    with gr.Row():
                        clone_lang = gr.Dropdown(choices=[('🤖 Auto','auto')] + [(language_label(k), k) for k in LANGUAGE_CATALOG], value='vi', label='Ngôn ngữ')
                        clone_engine = gr.Dropdown(choices=engine_choices('vi'), value='auto', label='Server / Engine')
                        clone_name = gr.Textbox(value='My_Voice', label='Tên Voice')
                    with gr.Row():
                        clone_audio = gr.File(type='filepath', label='File voice mẫu')
                        clone_transcript = gr.Textbox(lines=3, label='Transcript', placeholder='Để trống để Clone Prep tự nhận dạng.')
                    clone_voice_search = gr.Textbox(label='🔎 Tìm Voice', placeholder='Gõ tên voice để lọc...', interactive=True)
                    _clone_voices0 = audio_files_for_ui() or [('— Chưa có Voice —', '__NONE__')]
                    clone_profile_select = gr.Dropdown(choices=_clone_voices0, value=_clone_voices0[0][1], label='Voice đã lưu', filterable=True)
                    clone_preview_text = gr.Textbox(value='Xin chào, đây là câu nghe thử cho Clone Voice.', lines=2, label='Câu nghe thử')
                    with gr.Row():
                        clone_create = gr.Button('🧬 Tạo Voice', variant='primary')
                        clone_listen = gr.Button('🎧 Nghe Clone bằng AI')
                        clone_refresh = gr.Button('🔄 Làm mới')
                        clone_use = gr.Button('✓ Dùng Voice này')
                        clone_close = gr.Button('✕ Đóng')
                    clone_audio_preview = gr.Audio(type='filepath', label='Audio Clone', interactive=False)
                    clone_transcript_out = gr.Textbox(lines=3, label='Transcript', interactive=False)
                    clone_status = gr.Markdown()

                clone_modal_toggle.click(lambda: gr.Column(visible=True), outputs=clone_modal)
                clone_close.click(lambda: gr.Column(visible=False), outputs=clone_modal)
                clone_lang.change(lambda l,e: gr.Dropdown(choices=engine_choices(l if l != 'auto' else 'vi'), value=e if e in {v for _,v in engine_choices(l if l != 'auto' else 'vi')} else 'auto'), [clone_lang,clone_engine], clone_engine)
                def _clone_create_ui(n,a,l,t,e):
                    # CREATE VOICE only creates the new WAV. It must NOT mutate the
                    # workflow's persisted selected_voice. User chooses a Voice for
                    # workflow execution separately via "Dùng Voice này" / Voice dropdown.
                    voice_path, transcript, status = create_voice(n,a,l,t)
                    resolved = resolve_engine(l if l != 'auto' else 'vi', e)
                    choices = audio_files_for_ui() or [('— Chưa có Voice —', '__NONE__')]
                    return gr.Dropdown(choices=choices, value=voice_path), transcript, f"{status} · Server/Model TTS: **{ENGINE_INFO[resolved]['short']}**"
                clone_create.click(_wrap_gradio_callback(_clone_create_ui), [clone_name,clone_audio,clone_lang,clone_transcript,clone_engine], [clone_profile_select,clone_transcript_out,clone_status])
                clone_listen.click(lambda l,e,p,t: preview_tts(l if l != 'auto' else 'vi', e, p, t), [clone_lang,clone_engine,clone_profile_select,clone_preview_text], [clone_audio_preview,clone_status])
                clone_refresh.click(lambda: (lambda c: gr.Dropdown(choices=(c or [('— Chưa có Voice —', '__NONE__')]), value=(c[0][1] if c else '__NONE__')))(audio_files_for_ui()), outputs=clone_profile_select)
                clone_voice_search.input(lambda q: (lambda c: gr.Dropdown(choices=(c or [('— Không tìm thấy Voice —', '__NONE__')]), value=(c[0][1] if c else '__NONE__')))(filter_voice_choices(q)), clone_voice_search, clone_profile_select)
                clone_profile_select.change(lambda p: audio_preview_path(p), clone_profile_select, clone_audio_preview)
                clone_use.click(lambda p,e: save_selected_voice_and_engine(p,e), [clone_profile_select,clone_engine], clone_status)
            with gr.Tab('📝 Kịch bản', id='script') as script_tab:
                _script_profile_choices = workflow_profile_choices()
                _script_profile_value = current_workflow_profile() or (_script_profile_choices[0] if _script_profile_choices else None)
                script_profile_select = gr.Dropdown(choices=_script_profile_choices, value=_script_profile_value, label='⚙ Cấu hình bắt buộc', interactive=True)
                script_profile_preview = gr.Markdown(_active_voice_config_status(), elem_id='script-voice-config-status')
                script_profile_status = gr.Markdown('Chọn cấu hình ở đây. 1 Click / Thực hiện thao tác chỉ chạy theo cấu hình đã chọn.')
                script_box = gr.Textbox(lines=20, value=workflow_profile_snapshot(current_workflow_profile()).get('script','') or SETTINGS.get('workflow_script',''), label='Nội dung kịch bản', interactive=True)
                with gr.Row():
                    auto_pipeline_btn = gr.Button('⚡ RUN FAST · SMART', variant='primary')
                    resume_project_btn = gr.Button('♻ RESUME PROJECT', variant='secondary')
                    manual_execute_btn = gr.Button('▶ Thực hiện thao tác', variant='secondary')
                auto_pipeline_status = gr.Markdown('**⚡ RUN FAST · SMART** = Gemini Scene Director + checkpoint thông minh; chỉ dựng lại phần thiếu/stale. **♻ RESUME PROJECT** = đọc checkpoint và bỏ qua mọi bước đã VERIFY; nếu Prompt hợp lệ thì nhảy thẳng Z-Image-Turbo. **🖼 IMAGE ONLY** = chỉ render ảnh, tuyệt đối không gọi Gemini.')
                scene_target = gr.Slider(180,550,value=_safe_int(SETTINGS.get('scene_target_chars',320),320,180,550),step=10,label='Độ dài mục tiêu mỗi Scene (ký tự)', visible=False)
                media_mode = gr.Dropdown(
                    choices=[(MEDIA_MODE_OPTIONS['pure_images'], 'pure_images'), (MEDIA_MODE_OPTIONS['mixed_media'], 'mixed_media')],
                    value=normalize_media_mode(SETTINGS.get('media_mode','pure_images')),
                    label='🎞 Chế độ tạo hình ảnh',
                    info='Ảnh tĩnh full = mọi Scene chạy Z-IMAGE-TURBO. Kết hợp = Gemini tự quyết IMAGE/VIDEO; Scene VIDEO không gọi Z-IMAGE-TURBO vì Z-IMAGE-TURBO chỉ tạo ảnh tĩnh.'
                )
                media_mode_status = gr.Markdown(
                    '🖼 **Ảnh tĩnh full** — mọi Scene đều là IMAGE và được phép chạy Z-IMAGE-TURBO.' if normalize_media_mode(SETTINGS.get('media_mode','pure_images')) == 'pure_images' else '🎬 **Kết hợp Ảnh + Video linh hoạt** — Gemini quyết định từng Scene; VIDEO không gọi Z-IMAGE-TURBO.',
                    elem_id='media-mode-status'
                )
                scene_target.change(_wrap_gradio_callback(_ui_save_scene_target), scene_target, outputs=[])
                media_mode.change(_wrap_gradio_callback(_ui_save_media_mode), media_mode, media_mode_status, show_progress='hidden')
                script_box.change(_wrap_gradio_callback(_ui_save_script_text), script_box, outputs=[])
                script_progress = gr.Slider(0,100,0,step=1,label='Tiến trình',interactive=False, visible=False, elem_id='las-script-progress', elem_classes=['las-progress-compact'])
                scene_table = gr.Dataframe(headers=['Scene','Narration'],interactive=False,wrap=True)
                script_status = gr.Markdown('Tự viết kịch bản → bấm Tạo Scenes. Không qua bước làm mượt.')
                manual_execute_btn.click(
                    _wrap_gradio_callback(_script_progress_ui),
                    [project, script_box, scene_target, media_mode],
                    [script_progress, scene_table, script_status, workflow_return_status],
                    show_progress='minimal'
                ).then(
                    _wrap_gradio_callback(save_scenes_snapshot),
                    [project],
                    [script_status, scene_table],
                    show_progress='minimal'
                ).then(
                    _wrap_gradio_callback(_show_voice_full_tab), outputs=[main_tabs], queue=False, show_progress='hidden'
                )
            with gr.Tab('🎚 Tạo Voice Full', id='voice_full') as voice_full_tab:
                full_config_status = gr.Markdown(
                    _active_voice_config_status(),
                    elem_id='full-active-config-status',
                    elem_classes=['full-active-config-status']
                )
                _full_lang0 = SETTINGS.get('selected_language','auto') or 'auto'
                _full_eng0 = SETTINGS.get('selected_engine','auto') or 'auto'
                _full_actual_lang0 = _full_lang0 if _full_lang0 != 'auto' else detect_script_language(SETTINGS.get('workflow_script','') or '')
                _initial_full_engine_choices = engine_choices(_full_actual_lang0)
                _full_eng0 = _full_eng0 if _full_eng0 in {v for _, v in _initial_full_engine_choices} else 'auto'
                full_language = gr.Dropdown(choices=[("🤖 Tự động nhận dạng từ Kịch bản", "auto")] + [(language_label(k), k) for k in LANGUAGE_CATALOG], value=_full_lang0, label='Ngôn ngữ')
                full_engine = gr.Dropdown(choices=_initial_full_engine_choices, value=_full_eng0, label='Server / Engine — đồng bộ theo Cấu hình workflow')
                saved_full_voice = normalize_voice_value(SETTINGS.get('selected_voice', ''))
                _initial_full_choices = all_voice_choices(_full_actual_lang0, _full_eng0)
                if saved_full_voice and Path(str(saved_full_voice)).expanduser().is_file() and saved_full_voice not in {v for _, v in _initial_full_choices}:
                    _initial_full_choices.insert(0, (Path(str(saved_full_voice)).name, saved_full_voice))
                _initial_full_value = (saved_full_voice if saved_full_voice in {v for _, v in _initial_full_choices} else (_initial_full_choices[0][1] if _initial_full_choices else None))
                full_voice = gr.Dropdown(choices=_initial_full_choices, value=_initial_full_value, label='Voice', filterable=True)
                full_recommendation = gr.Markdown('**Khuyến nghị:** Tiếng Việt → VieNeu-TTS • English/ngôn ngữ Chatterbox → Chatterbox V3 • còn lại → OmniVoice.')
                with gr.Row():
                    full_voice_speed = gr.Slider(0.75, 1.50, value=float(SETTINGS.get('voice_speed', 1.0)), step=0.05, label='🎚 Tốc độ đọc', info='0.75× chậm → 1.50× nhanh; xử lý local để không phụ thuộc engine.')
                    full_voice_pause = gr.Slider(0, 1200, value=int(SETTINGS.get('voice_pause_ms', 180)), step=20, label='⏸ Khoảng lặng cuối Scene (ms)', info='Chèn pause giữa các đoạn Voice Scene khi gộp Voice Full.')
                full_voice_tuning_status = gr.Markdown(f"🎚 Voice tuning hiện tại: **{float(SETTINGS.get('voice_speed',1.0)):.2f}×** · pause **{int(SETTINGS.get('voice_pause_ms',180))} ms**")
                full_voice_speed.change(_wrap_gradio_callback(_ui_save_voice_tuning), [full_voice_speed, full_voice_pause], full_voice_tuning_status, show_progress='hidden')
                full_voice_pause.change(_wrap_gradio_callback(_ui_save_voice_tuning), [full_voice_speed, full_voice_pause], full_voice_tuning_status, show_progress='hidden')
                _startup_engine_status = '⏸ Chưa bật TTS server. Khi chạy Voice Full, app sẽ nhận dạng ngôn ngữ Kịch bản và tự bật đúng server GPU.'
                full_engine_status = gr.Markdown(_startup_engine_status)
                with gr.Row():
                    full_server_start = gr.Button('▶ Bật Server', variant='secondary')
                    full_release_vram = gr.Button('♻ Giải phóng VRAM', variant='secondary')
                    full_generate = gr.Button('🎙 Tạo từng Voice Scene', variant='primary')
                    full_stop = gr.Button('⏹ Dừng tab Voice Full', variant='stop')
                # Standalone Voice controls are moved to the dedicated ĐỘC LẬP tab below.
                gr.Markdown('### 🎙 Khung Voice Scene đã tạo')
                full_voice_loader_box = gr.HTML(voice_live_loader('Voice Full', 0, 'Sẵn sàng', False), visible=True, elem_classes=['voice-progress-host'])
                full_status = gr.Markdown('Sẵn sàng tạo Voice Scene từ Project đang chọn.')
                # DUY NHẤT một khung Scene của Project hiện tại:
                # Scene | Kịch bản | Trạng thái | File đã tạo (nghe được)
                full_output_dir = gr.Textbox(value=SETTINGS.get('project_output_dir',''), visible=False, interactive=False)
                full_scene_table = gr.HTML(voice_scene_player_html(default_project), show_label=False, elem_classes=['voice-scene-frame'])
                gr.Markdown('### 🎙 Đang tạo Voice Scene')
                full_voice_progress_table = gr.HTML(voice_progress_table_html(default_project), show_label=False, elem_classes=['voice-progress-frame'])
                full_scene_players = gr.HTML('', visible=False)
                full_combine = gr.Button('🔗 Gộp Voice Full', variant='secondary', visible=False)
                full_combine_loader = gr.HTML('', visible=False, elem_classes=['voice-progress-host'])
                full_combine_status = gr.Markdown()
                full_audio = gr.HTML(voice_full_player_html(default_project))
                full_download = gr.File(label='', visible=False)

                gr.Markdown('### 🤖 Lấy Prompt từ Gemini — chỉ chạy 1 lần cho tới khi hoàn tất')
                with gr.Row():
                    gemini_profile_voice = gr.State(current_workflow_profile() or 'Default')
                    get_prompt_btn = gr.Button('🤖 Lấy Prompt', variant='primary', visible=False, scale=2)
                    prompt_retry_btn_voice = gr.Button('🔄 Tạo lại Prompt', variant='secondary', visible=False, scale=2)
                    # Z-Image is intentionally NOT opened here. It is opened only when the user
                    # clicks 🚀 Bắt đầu tạo ảnh in the Hình ảnh tab.
                    go_zimage_btn_voice = gr.Button('💾 Lưu prompt → sang Hình ảnh', visible=False, elem_id='prompt-handoff-btn')
                prompt_progress = gr.Slider(0,100,0,step=1,label='Gemini Prompt — Tiến trình',interactive=False, elem_id='las-prompt-progress', elem_classes=['las-progress-compact'])
                prompt_status = gr.Markdown('**Workflow 2 bước:** (1) Gemini API phân tích FULL Kịch bản + references → Topic + Global Style và VERIFY checkpoint; (2) sau Voice Full + release VRAM, Gemini API dùng checkpoint đó để tạo **Scene + Style Prompt** theo batch → VERIFY đủ **N/N** → chuyển sang Z-IMAGE-TURBO Local. Tối đa 3 ảnh tham chiếu được upload trực tiếp qua Gemini File API.')
                prompt_output = gr.Dataframe(headers=['Image Scene','Kịch bản','Prompt đã tạo'], value=prompt_scene_rows(load_manifest(default_project)), interactive=False, wrap=True, label='🖼 Prompt theo Scene')
                with gr.Row():
                    prompt_refresh_all = gr.Button('🔄 Làm mới Prompt', variant='secondary', elem_id='prompt-refresh-all')
                    prompt_refresh_status = gr.Markdown('B1 sẽ cập nhật Topic + Global Style tại đây trước khi B2 được phép chạy.', elem_id='prompt-refresh-status')
                prompt_attempt_token = gr.State(0)
                # Internal event gates: B1 and B2 are intentionally separate browser events.
                # Gradio `.then()` chains run server-side without guaranteeing that the B1
                # DOM update has painted before B2 starts, so B2 is triggered only by a
                # client-side click AFTER the Prompt checkpoint refresh has completed.
                manual_step2_trigger_btn = gr.Button('internal-step2-manual', visible=False, elem_id='prompt-step2-trigger')
                master_step2_trigger_btn = gr.Button('internal-step2-master', visible=False, elem_id='master-step2-trigger')


                def _sync_all_scene_views(project_name):
                    scenes = load_manifest(project_name or default_project)
                    rows = prompt_scene_rows(scenes)
                    players = voice_scene_player_html(project_name or default_project)
                    return (
                        players, '', rows,
                        f'🔄 Đã đồng bộ dữ liệu mới: **{len(scenes)} Scene**.' if scenes else '🔄 Chưa có Scene.',
                        f'**{len(scenes)} scenes**' if scenes else '**0 scenes** — trắng.',
                        rows,
                        gr.update(visible=bool(scenes)),
                        gr.update(visible=False),
                        gr.update(visible=False),
                    )

                def _refresh_prompt_checkpoint_ui(p):
                    scenes = load_manifest(p or default_project)
                    root = project_dir(p or default_project)
                    obj = read_json(_project_content_dir(root) / 'Prompt' / 'topic_style.json', {}) or {}
                    topic = str(obj.get('topic') or '').strip()
                    style = re.sub(r'\s+', ' ', str(obj.get('style') or obj.get('visual_style') or '')).strip()
                    ok = bool(topic and style and obj.get('b1_verified') and obj.get('api_response_verified') and obj.get('api_response_sha256'))
                    if ok:
                        # Real UI checkpoint: the Scene table is updated with the exact
                        # verified B1 context before B2 is permitted to send anything.
                        preview = []
                        for sc in scenes:
                            mt = scene_media_type(sc, 'image')
                            body = '🎬 VIDEO — B2 sẽ bỏ qua Z-IMAGE-TURBO' if mt == 'video' else '⏳ Chờ Gemini B2 tạo Prompt'
                            preview.append([sc.get('scene_id',''), sc.get('text',''), f'GLOBAL TOPIC: {topic} | GLOBAL VISUAL STYLE: {style} | SCENE: {body}'])
                        return preview, ('[PROMPT_REFRESH_OK:%d] B1 đã nạp Topic + Global Style vào Prompt theo Scene — API VERIFY OK.' % int(time.time()*1000))
                    return prompt_scene_rows(scenes), ('[PROMPT_REFRESH_FAIL:%d] Chưa có B1 API VERIFY.' % int(time.time()*1000))

                prompt_refresh_all.click(
                    _wrap_gradio_callback(_refresh_prompt_checkpoint_ui), project, [prompt_output, prompt_refresh_status],
                    show_progress='hidden'
                )

                def _auto_engine_for_script(script_text, current_engine):
                    lang, selected = detect_script_and_engine(script_text or '', current_engine or 'auto')
                    choices = engine_choices(lang)
                    valid_eng = {v for _, v in choices}
                    engine_value = current_engine if current_engine in valid_eng else 'auto'
                    voice_choices = all_voice_choices(lang, engine_value)
                    voice_value = str(SETTINGS.get('selected_voice') or '').strip()
                    if voice_value and voice_value not in {v for _, v in voice_choices}:
                        voice_choices.insert(0, (Path(voice_value).name if Path(voice_value).expanduser().is_file() else voice_value, voice_value))
                    if not voice_value:
                        voice_value = voice_choices[0][1] if voice_choices else None
                    language_choices = [("🤖 Tự động nhận dạng từ Kịch bản", "auto")] + [(language_label(k), k) for k in LANGUAGE_CATALOG]
                    language_value = lang if lang == "auto" or lang in LANGUAGE_CATALOG else "auto"
                    status = f"🔎 Kịch bản: **{language_label(lang)}** → {'Tự động → ' + ENGINE_INFO[selected]['short'] if engine_value == 'auto' else ENGINE_INFO[selected]['short']} (chưa bật server)"
                    return gr.Dropdown(choices=choices, value=engine_value), gr.Dropdown(choices=language_choices, value=language_value), gr.Dropdown(choices=voice_choices, value=voice_value), status

                # Script is the source of truth for automatic language/model routing.
                def _script_changed_for_setup(text):
                    SETTINGS['workflow_script'] = text or ''
                    save_settings()
                    return _auto_engine_for_script(text, SETTINGS.get('selected_engine','auto'))
                script_box.change(_wrap_gradio_callback(_script_changed_for_setup), script_box, [full_engine, full_language, full_voice, full_engine_status])
                def _full_language_changed(l, e, current_voice):
                    lang = l if l and l != 'auto' else detect_script_language(SETTINGS.get('workflow_script','') or '')
                    e = e or 'auto'
                    choices = all_voice_choices(lang, e)
                    valid = {v for _, v in choices}
                    # Preserve the Voice currently selected in this UI first.
                    # Persisted workflow Voice is only a fallback when the current
                    # selection is missing/invalid for the newly selected language.
                    current = normalize_voice_value(current_voice or "")
                    configured = normalize_voice_value(SETTINGS.get("selected_voice", "") or "")
                    value = (current if current in valid else
                             configured if configured in valid else
                             (choices[0][1] if choices else None))
                    return gr.Dropdown(choices=choices, value=value)

                def _switch_voice_engine(l, e, current_voice):
                    # UI selection must be side-effect free. Do NOT start/stop a GPU
                    # TTS server merely because the user changes the dropdown. The
                    # actual server is activated only when Voice Full generation starts
                    # or when the explicit "Bật Server" button is pressed.
                    lang = l if l and l != 'auto' else detect_script_language(SETTINGS.get('workflow_script','') or '')
                    e = e or 'auto'
                    choices = all_voice_choices(lang, e)
                    valid = {v for _, v in choices}
                    current = normalize_voice_value(current_voice or "")
                    configured = normalize_voice_value(SETTINGS.get("selected_voice", "") or "")
                    value = (current if current in valid else
                             configured if configured in valid else
                             (choices[0][1] if choices else None))
                    selected = resolve_engine(lang, e)
                    voice_label = next((label for label, val in choices if val == value), str(value or '(chưa chọn)'))
                    status = f"✓ Đã chọn: **{voice_label}** · **{ENGINE_INFO[selected]['title']}** · GPU chưa bật"
                    if configured and configured in valid:
                        status = f"✓ Theo cấu hình: **{voice_label}** · **{ENGINE_INFO[selected]['title']}** · GPU chưa bật"
                    return gr.Dropdown(choices=choices, value=value), status

                full_language.change(_wrap_gradio_callback(_full_language_changed), [full_language, full_engine, full_voice], full_voice, show_progress='minimal').then(
                    _wrap_gradio_callback(_active_voice_config_status), [full_language, full_engine, full_voice], full_config_status, show_progress='minimal'
                )

                # Selecting a server immediately switches the GPU process: the old
                # server is terminated first, then the selected server is started.
                full_engine.change(_wrap_gradio_callback(_switch_voice_engine), [full_language,full_engine,full_voice], [full_voice,full_engine_status]).then(
                    _wrap_gradio_callback(_active_voice_config_status), [full_language, full_engine, full_voice], full_config_status, show_progress='minimal'
                )
                full_voice.change(_wrap_gradio_callback(_active_voice_config_status), [full_language, full_engine, full_voice], full_config_status, show_progress='minimal')
                full_server_start.click(_wrap_gradio_callback(start_voice_server_ui),[full_language,full_engine],[full_engine,full_engine_status])
                full_release_vram.click(_wrap_gradio_callback(release_voice_vram),[full_language,full_engine],[full_voice_loader_box,full_status])
                full_stop.click(_wrap_gradio_callback(stop_full_voice), outputs=full_status)

                def _standalone_script_changed(text):
                    lang = detect_script_language(str(text or '')) if str(text or '').strip() else 'auto'
                    lang_choices = [("🤖 Tự động nhận dạng", "auto")] + [(language_label(k), k) for k in LANGUAGE_CATALOG]
                    actual = lang if lang in LANGUAGE_CATALOG else (_full_actual_lang0 if _full_actual_lang0 in LANGUAGE_CATALOG else 'en')
                    choices = engine_choices(actual)
                    eng = 'auto' if 'auto' in {v for _, v in choices} else choices[0][1]
                    voices = all_voice_choices(actual, eng)
                    return gr.Dropdown(choices=lang_choices, value='auto'), gr.Dropdown(choices=choices, value=eng), gr.Dropdown(choices=voices, value=(voices[0][1] if voices else None))

                project_output_dir.change(lambda p: p, project_output_dir, full_output_dir)
                full_combine.click(_wrap_gradio_callback(_combine_full_voice_final),[project,full_output_dir],[full_combine_loader,full_combine_status,full_audio,full_download])

                def _gemini_reference_info(profile_name, run_project=None):
                    # MASTER uses the canonical Project/Nhân vật/ images for BOTH Gemini and Z-IMAGE-TURBO.
                    # Never fall back to an old tool-level reference cache.
                    profile_name = str(profile_name or '').strip()
                    if profile_name not in load_workflow_profiles():
                        profile_name = current_workflow_profile() or 'Default'
                    c = workflow_profile_snapshot(profile_name)
                    urls = [u.strip() for u in str(c.get('gemini_competitor_urls') or '').splitlines() if u.strip()]
                    if not urls:
                        urls = [u.strip() for u in str(SETTINGS.get('gemini_competitor_urls') or '').splitlines() if u.strip()]
                    try:
                        refs = _project_character_reference_paths(run_project)
                    except Exception:
                        refs = []
                    refs = [str(x) for x in refs[:3] if x and Path(str(x)).is_file()]
                    return refs, list(dict.fromkeys(urls))[:10]

                def _parse_topic_style_result_local(raw_text):
                    """Tolerant Gemini STEP-1 parser.

                    Accepts clean JSON, fenced JSON, JSON embedded in prose,
                    Markdown tables, and plain ``Topic:`` / ``Global Style:``
                    text. Gemini is a browser response, so formatting must never
                    be a hard dependency for the Topic + Style gate.
                    """
                    text = str(raw_text or '').replace('\ufeff', '').strip()
                    if not text:
                        raise RuntimeError('Gemini không trả nội dung Topic + Global Style.')

                    def clean(v):
                        return re.sub(r'\s+', ' ', str(v or '')).strip(' `*_\t\r\n|')

                    def obj_values(obj):
                        if not isinstance(obj, dict):
                            return '', ''
                        topic = clean(obj.get('topic') or obj.get('main_topic') or obj.get('global_topic') or obj.get('video_topic'))
                        style = clean(obj.get('style') or obj.get('visual_style') or obj.get('global_visual_style') or obj.get('master_style_prompt') or obj.get('global_style'))
                        return topic, style

                    # 1) JSON: use JSONDecoder.raw_decode instead of greedy \{.*\}.
                    candidates = []
                    candidates.append(text)
                    candidates.extend(m.group(1).strip() for m in re.finditer(r'```(?:json|javascript|text)?\s*(.*?)\s*```', text, re.I | re.S))
                    dec = json.JSONDecoder()
                    for c in list(candidates):
                        try:
                            obj = json.loads(c)
                            topic, style = obj_values(obj)
                            if topic and style:
                                return topic, style
                        except Exception:
                            pass
                        for m in re.finditer(r'\{', c):
                            try:
                                obj, _ = dec.raw_decode(c[m.start():])
                                topic, style = obj_values(obj)
                                if topic and style:
                                    return topic, style
                            except Exception:
                                continue

                    # 2) Quoted key/value recovery, including single quotes.
                    key_topic = r'(?:topic|main[_ ]topic|global[_ ]topic|video[_ ]topic)'
                    key_style = r'(?:style|visual[_ ]style|global[_ ]visual[_ ]style|master[_ ]style(?:_prompt)?|global[_ ]style)'
                    mt = re.search(r'["\']' + key_topic + r'["\']\s*:\s*["\'](.*?)["\']', text, re.I | re.S)
                    ms = re.search(r'["\']' + key_style + r'["\']\s*:\s*["\'](.*?)["\']', text, re.I | re.S)
                    if mt and ms:
                        return clean(mt.group(1)), clean(ms.group(1))

                    # 3) Markdown table: | Topic | ... | / | Global Visual Style | ... |
                    table = []
                    for line in text.splitlines():
                        cells = [clean(x) for x in line.strip().strip('|').split('|')]
                        if len(cells) >= 2 and not all(re.fullmatch(r'[-: ]+', x or '-') for x in cells):
                            table.append(cells)
                    topic = style = ''
                    for cells in table:
                        label = clean(cells[0]).lower()
                        value = clean(' | '.join(cells[1:]))
                        if not value:
                            continue
                        if ('global visual style' in label or label in {'visual style','style','global style','master style','master visual style'}):
                            style = value
                        elif label in {'topic','main topic','global topic','video topic','main video topic'}:
                            topic = value
                    if topic and style:
                        return topic, style

                    # 4) Plain text headings / labels. Accept common Markdown emphasis.
                    def line_value(patterns):
                        for line in text.splitlines():
                            raw = re.sub(r'^\s*[-*#>]+\s*', '', line).strip()
                            for pat in patterns:
                                m = re.match(pat, raw, re.I)
                                if m and clean(m.group(1)):
                                    return clean(m.group(1))
                        return ''
                    topic = line_value([
                        r'(?:main\s+video\s+topic|main\s+topic|global\s+topic|video\s+topic|topic)\s*[:\-–]\s*(.+)$',
                    ])
                    style = line_value([
                        r'(?:global\s+visual\s+style|global\s+style|visual\s+style|master\s+visual\s+style|master\s+style|style)\s*[:\-–]\s*(.+)$',
                    ])
                    if topic and style:
                        return topic, style

                    # 5) Last-resort same-line extraction when Gemini collapses labels.
                    mt = re.search(r'(?:main\s+video\s+topic|global\s+topic|topic)\s*[:\-]\s*(.+?)(?=\s+(?:global\s+visual\s+style|visual\s+style|global\s+style|style)\s*[:\-])', text, re.I | re.S)
                    ms = re.search(r'(?:global\s+visual\s+style|visual\s+style|global\s+style|style)\s*[:\-]\s*(.+)$', text, re.I | re.S)
                    if mt and ms:
                        return clean(mt.group(1)), clean(ms.group(1))
                    raise RuntimeError('Gemini không trả đủ Topic + Global Style ở dạng JSON, bảng hoặc text.')

                def _validate_scene_prompts_strict(scenes, topic, visual_style):
                    """Hard Gate: every Scene must have a non-empty ONE-LINE prompt with exact global DNA."""
                    errors = []
                    t_clean = re.sub(r'\s+', ' ', str(topic or '')).strip()
                    s_clean = re.sub(r'\s+', ' ', str(visual_style or '')).strip()
                    if not t_clean or not s_clean:
                        return False, ['Global Topic hoặc Global Visual Style đang rỗng.']
                    for scene in scenes or []:
                        sid = str(scene.get('scene_id') or 'Unknown')
                        media_type = scene_media_type(scene, 'image')
                        prompt = str(scene.get('prompt') or '').strip()
                        if media_type == 'video':
                            if prompt: errors.append(f'{sid}: VIDEO không được chứa Z-IMAGE-TURBO Prompt.')
                            continue
                        if not prompt:
                            errors.append(f'{sid}: Prompt bị rỗng.')
                            continue
                        if '\n' in prompt or '\r' in prompt:
                            errors.append(f'{sid}: Prompt chứa xuống dòng; yêu cầu đúng 1 dòng liên tục.')
                        if f'GLOBAL TOPIC: {t_clean}' not in prompt:
                            errors.append(f'{sid}: thiếu GLOBAL TOPIC.')
                        if f'GLOBAL VISUAL STYLE: {s_clean}' not in prompt:
                            errors.append(f'{sid}: thiếu GLOBAL VISUAL STYLE.')
                    return (len(errors) == 0, errors)

                def _canonical_scene_prompt(topic, visual_style, scene_prompt, visual_type='', text_in_image=None, text_style=''):
                    """Build ONE-LINE, self-contained Flow prompt with mandatory global DNA."""
                    topic_clean = re.sub(r'\s+', ' ', str(topic or '')).strip()
                    style_clean = re.sub(r'\s+', ' ', str(visual_style or '')).strip()
                    scene_clean = re.sub(r'\s+', ' ', str(scene_prompt or '')).strip()
                    visual_type_clean = re.sub(r'\s+', ' ', str(visual_type or '')).strip()
                    if not topic_clean or not style_clean or not scene_clean:
                        raise RuntimeError('Không thể tạo Prompt hoàn chỉnh: thiếu Topic, Global Style hoặc Scene Prompt.')
                    # Strip accidental duplicated global headers from Gemini's scene body.
                    scene_clean = re.sub(r'^(?:GLOBAL\s+TOPIC|VIDEO\s+TOPIC)\s*:\s*.*?(?:\s*\|\s*|$)', '', scene_clean, flags=re.I)
                    scene_clean = re.sub(r'^(?:GLOBAL\s+VISUAL\s+STYLE|VISUAL\s+STYLE|STYLE)\s*:\s*.*?(?:\s*\|\s*|$)', '', scene_clean, flags=re.I)
                    scene_clean = re.sub(r'\s+', ' ', scene_clean).strip(' |')
                    parts = [f'GLOBAL TOPIC: {topic_clean}', f'GLOBAL VISUAL STYLE: {style_clean}']
                    if visual_type_clean:
                        parts.append(f'IMAGE TYPE: {visual_type_clean}')
                    # Keep image-text as a per-scene decision. Gemini's scene body is authoritative.
                    # Precompute this value: Python 3.11 rejects backslashes inside f-string expressions.
                    text_style_clean = re.sub(r"\s+", " ", str(text_style or "")).strip()
                    if bool(text_in_image):
                        if text_style_clean:
                            parts.append(f'TEXT STYLE RULE: {text_style_clean}')
                        parts.append('SCENE TEXT: render only the exact short in-scene wording already specified by the scene state; no subtitles, captions, logos or watermarks')
                    else:
                        parts.append('TEXT RULE: no visible text for this scene; do not add subtitles, captions, labels, logos, watermarks or decorative typography')
                    parts.append(f'SCENE DIRECTION: {scene_clean}')
                    return ' | '.join(parts).strip()

                def _run_prompt_from_voice(project_name, profile_name, attempt_token=0, retry=False, phase="auto"):
                    # Gemini stage: (1) FULL SCRIPT -> topic + one global style;
                    # (2) scenes are processed in small batches; every returned record is one line.
                    project_key = safe_name(project_name)
                    try:
                        workflow_cmd_log("GEMINI", "START", project=project_name, retry=retry)
                        attempt_token = int(attempt_token or 0)
                    except Exception:
                        attempt_token = 0
                    with PROMPT_BATCH_STATE_LOCK:
                        expected_token = int(PROMPT_UI_TOKENS.get(project_key, 0))
                        if attempt_token != expected_token:
                            yield 0, '⚠ Lệnh này đã được xử lý/đang chờ. Không gửi thêm lệnh Gemini.', prompt_scene_rows(load_manifest(project_name)), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), expected_token, ''
                            return
                        PROMPT_UI_TOKENS[project_key] = expected_token + 1
                        state = PROMPT_BATCH_STATE.get(project_key, 'idle')
                        if state == 'running':
                            yield 0, '⏳ Gemini đang xử lý. Không gửi thêm lệnh.', prompt_scene_rows(load_manifest(project_name)), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), int(attempt_token) + 1, ''
                            return
                        if phase == 'batch':
                            if state not in ('style_ready', 'style_ready_checkpoint', 'await_batch'):
                                yield 0, '⚠ BƯỚC 2 bị khóa: chưa có Topic + Global Style đã xác nhận.', prompt_scene_rows(load_manifest(project_name)), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), int(attempt_token) + 1, ''
                                return
                        elif not retry and state != 'idle':
                            yield 0, '⚠ Lượt **Lấy Prompt** này đã được gửi rồi. Không gửi lại. Nếu thất bại hãy bấm **🔄 Tạo lại Prompt**.', prompt_scene_rows(load_manifest(project_name)), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), int(attempt_token) + 1, ''
                            return
                        PROMPT_BATCH_STATE[project_key] = 'running'
                    if not PROMPT_BATCH_LOCK.acquire(blocking=False):
                        with PROMPT_BATCH_STATE_LOCK:
                            PROMPT_BATCH_STATE[project_key] = 'await_retry'
                        yield 0, '⏳ Gemini đang xử lý. Không gửi thêm lệnh.', prompt_scene_rows(load_manifest(project_name)), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), int(attempt_token) + 1, ''
                        return
                    try:
                        if FULL_STOP_EVENT.is_set() or PROMPT_STOP_EVENT.is_set() or SCRIPT_STOP_EVENT.is_set():
                            raise RuntimeError("Đã dừng workflow Prompt trong tab 🎚 Tạo Voice Full.")
                        scenes = load_manifest(project_name)
                        if not scenes:
                            raise RuntimeError("Chưa có Scene.")
                        if phase == 'master':
                            b1_root = project_dir(project_name)
                            b1_style = read_json(_project_content_dir(b1_root) / 'Prompt' / 'topic_style.json', {}) or {}
                            b1_scene_ok, _ = _project_scene_checkpoint(project_name, infer=True, script_text=_canonical_script_for_project(project_name))
                            current_style_hash = _project_current_script_hash(project_name)
                            b1_style_hash = str(b1_style.get('script_sha256') or '').strip()
                            b1_ready = bool(
                                b1_scene_ok and
                                current_style_hash and b1_style_hash == current_style_hash and
                                str(b1_style.get('topic') or '').strip() and
                                str(b1_style.get('style') or b1_style.get('visual_style') or '').strip() and
                                bool(b1_style.get('b1_verified') or b1_style.get('api_response_verified'))
                            )
                            if b1_style and b1_style_hash and current_style_hash and b1_style_hash != current_style_hash:
                                workflow_cmd_log('MASTER', 'RESUME_STYLE_SCRIPT_MISMATCH', project=project_name, stored_script_sha256=b1_style_hash[:16], current_script_sha256=current_style_hash[:16], level='WARN')
                            if b1_ready:
                                phase = 'batch'
                                with PROMPT_BATCH_STATE_LOCK:
                                    PROMPT_BATCH_STATE[project_key] = 'style_ready'
                                workflow_cmd_log('MASTER', 'RESUME_SKIP_GEMINI_B1', project=project_name)
                            else:
                                _clear_prompt_verified(project_name)
                                _clear_gemini_chunk_checkpoints(project_name)
                        elif phase != 'batch':
                            _clear_prompt_verified(project_name)
                        profile_name = (profile_name or current_workflow_profile() or SETTINGS.get('workflow_profile') or 'Default').strip()
                        if profile_name not in load_workflow_profiles():
                            profile_name = current_workflow_profile() or 'Default'
                        root = project_dir(project_name)
                        (_project_content_dir(root) / 'Kịch bản').mkdir(parents=True, exist_ok=True)
                        (_project_content_dir(root) / 'Prompt').mkdir(parents=True, exist_ok=True)
                        yield 5, f'🤖 Gemini API đang xử lý B1/B2 — cấu hình workflow: **{profile_name}**.', prompt_scene_rows(scenes), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), int(attempt_token) + 1, ''

                        if phase == 'style':
                            # Every fresh Style run must execute B1 again. Never let a stale
                            # topic_style.json from an earlier run silently unlock B2.
                            stale_files = (
                                'Prompt/topic_style.json',
                                'Prompt/topic_style_raw.txt',
                                'Prompt/style_bible.json',
                                'Prompt/gemini_chunks.json',
                                'Prompt/prompt_verified.json',
                                'Prompt/topic_style_api_verified.txt',
                                'Prompt/topic_style_verified.json',
                                'Prompt/pipeline.json',
                                'project_bible.json',
                                'scene_states.json',
                            )
                            removed += _clear_gemini_chunk_checkpoints(project_name)
                            removed = 0
                            for rel in stale_files:
                                try:
                                    fp = root / rel
                                    if fp.exists():
                                        fp.unlink()
                                        removed += 1
                                except Exception:
                                    pass
                            workflow_cmd_log('GEMINI', 'STEP1_FRESH_GATE_RESET', project=project_name, removed=removed, retry=retry)

                        if phase == 'batch':
                            # RESUME FIX: B1 was already verified, so the batch path must
                            # reconstruct the same reference/competitor context that the
                            # fresh B1 path normally initializes. Otherwise reference_images
                            # is a local variable that has never been assigned.
                            reference_images, competitor_urls = _gemini_reference_info(profile_name, project_name)
                            reference_images = list(reference_images or [])[:3]
                            competitor_urls = list(dict.fromkeys([u.strip() for u in (competitor_urls or []) if u.strip()]))[:10]
                            ref_names = ' | '.join(Path(x).name for x in reference_images) if reference_images else '(không có)'

                            style_obj = read_json(_project_content_dir(root) / 'Prompt' / 'topic_style.json', {}) or {}
                            # Canonical source for new runs; legacy style_bible is read only for old Projects.
                            style_bible = read_json(_project_content_dir(root) / 'Prompt' / 'style_bible.json', {}) or {}
                            character_bible = style_obj.get('character_bible') or style_bible.get('character_bible') or {}
                            topic = str(style_obj.get('topic') or style_bible.get('topic') or '').strip()
                            visual_style = re.sub(
                                r'\s+', ' ',
                                str(
                                    style_obj.get('style')
                                    or style_obj.get('visual_style')
                                    or style_bible.get('master_style_prompt')
                                    or style_bible.get('style')
                                    or ''
                                )
                            ).strip()
                            visual_type = str(style_obj.get('visual_type') or style_bible.get('visual_type') or '').strip()
                            text_in_image = bool(style_obj.get('text_in_image', style_bible.get('text_in_image', False)))
                            text_style = str(style_obj.get('text_style') or style_bible.get('text_style') or '').strip()
                            competitor_style_confirmed = bool(style_obj.get('competitor_style_confirmed', True))
                            competitor_evidence = str(style_obj.get('competitor_evidence') or '').strip()
                            _step2_urls, _step2_confirmed, _step2_competitor_ok = _competitor_style_gate(style_obj)
                            if not topic or not visual_style:
                                raise RuntimeError('BƯỚC 2 BỊ KHÓA: Project chưa có Topic + Global Style đã xác nhận.')
                            if not _step2_competitor_ok:
                                raise RuntimeError('BƯỚC 2 BỊ KHÓA: Gemini chưa xác nhận style đối thủ và cũng không có bằng chứng fallback do URL không truy cập được.')
                            if _step2_urls and not _step2_confirmed:
                                workflow_cmd_log('GEMINI', 'COMPETITOR_STYLE_FALLBACK', project=project_name, reason='competitor URL inaccessible', evidence=competitor_evidence)
                            style_msg = (
                                f'✅ BƯỚC 1 ĐÃ XÁC NHẬN. **Topic:** {topic}\n'
                                f'**Global Style:** {visual_style}\n'
                                f'**Bằng chứng:** {competitor_evidence or "(không có)"}\n'
                                '↩ Đã lấy Style từ Project. Gemini API chạy BƯỚC 2 — Scene Prompt.'
                            )
                            workflow_cmd_log('GEMINI', 'STYLE_RELOADED_FOR_STEP2', project=project_name, topic=topic, style=visual_style)
                        else:
                            # STEP 1 — FULL SCRIPT -> topic + one global style.
                            # Resolve the FULL script from several authoritative sources.
                            # Older runs could have a valid manifest while script.txt was missing;
                            # that must never block Gemini when every Scene already contains narration.
                            script_candidates = []
                            for fp in (
                                _project_content_dir(root) / 'Kịch bản' / 'script.txt',
                                _project_content_dir(root) / 'Kịch bản' / 'full_script.txt',
                                root / 'script.txt',
                            ):
                                try:
                                    if fp.exists():
                                        val = fp.read_text(encoding='utf-8').strip()
                                        if val:
                                            script_candidates.append(val)
                                except Exception:
                                    pass
                            settings_script = str(SETTINGS.get('workflow_script', '') or '').strip()
                            if settings_script:
                                script_candidates.append(settings_script)
                            # The Scene manifest is always the final recovery source because
                            # create_scenes_progress writes the exact narration into every Scene.
                            scene_script = '\n\n'.join(
                                (s.get('text') or s.get('narration') or '').strip()
                                for s in scenes
                                if (s.get('text') or s.get('narration') or '').strip()
                            ).strip()
                            if scene_script:
                                script_candidates.append(scene_script)
                            full_script = next((x for x in script_candidates if x), '')
                            if not full_script:
                                raise RuntimeError(
                                    f'Không tìm thấy FULL Kịch bản trong Project `{project_name}`: '
                                    'script.txt và toàn bộ Scene đều rỗng.'
                                )
                            script_sha256 = _script_fingerprint(full_script)
                            workflow_cmd_log('GEMINI', 'STYLE_SOURCE_SCRIPT_FINGERPRINT', project=project_name, script_sha256=script_sha256[:16], chars=len(full_script))
                            # Repair the canonical script file before sending to Gemini so the
                            # next retry uses the same source of truth.
                            script_file = _project_content_dir(root) / 'Kịch bản' / 'script.txt'
                            script_file.parent.mkdir(parents=True, exist_ok=True)
                            if not script_file.exists() or script_file.read_text(encoding='utf-8').strip() != full_script:
                                script_file.write_text(full_script, encoding='utf-8')
                            SETTINGS['workflow_script'] = full_script
                            save_settings()
                            workflow_cmd_log('GEMINI', 'FULL_SCRIPT_READY', project=project_name, chars=len(full_script), source=('script_file' if script_file.exists() else 'scene_manifest'))
                            # STEP 2A — CHARACTER REFERENCE: upload the actual MASTER files to Gemini.
                            # The same Project/Nhân vật/ files remain canonical for Z-IMAGE-TURBO.
                            reference_images, competitor_urls = _gemini_reference_info(profile_name, project_name)
                            competitor_urls = list(dict.fromkeys([u.strip() for u in competitor_urls if u.strip()]))[:10]
                            _channel_urls = [
                                u for u in competitor_urls
                                if ('youtube.com/' in u.lower() or 'youtube-nocookie.com/' in u.lower())
                                and 'youtube.com/watch?' not in u.lower()
                                and 'youtube.com/shorts/' not in u.lower()
                                and 'youtu.be/' not in u.lower()
                            ]
                            _discovered_video_urls = _gemini_channel_search_tool_urls(_channel_urls, project_name) if _channel_urls else []
                            if _discovered_video_urls:
                                competitor_urls = list(dict.fromkeys(competitor_urls + _discovered_video_urls))[:16]
                                workflow_cmd_log(
                                    'GEMINI', 'YOUTUBE_CHANNEL_VIDEOS_READY',
                                    project=project_name, channel_count=len(_channel_urls),
                                    video_count=len(_discovered_video_urls),
                                    videos=' | '.join(_discovered_video_urls), pct=31
                                )
                            ref_lines = '\n'.join(f'- {u}' for u in competitor_urls) or '(none configured)'
                            ref_names = ' | '.join(Path(x).name for x in reference_images) if reference_images else '(không có)'
                            # Initialize before any telemetry can reference it. Gemini fills this
                            # value only after the B1 response has been parsed.
                            character_bible = {}
                            workflow_cmd_log(
                                'GEMINI', 'STYLE_REFERENCES_READY',
                                project=project_name, profile=profile_name,
                                reference_image_count=len(reference_images),
                                reference_images=ref_names,
                                competitor_url_count=len(competitor_urls),
                                competitor_urls=' | '.join(competitor_urls)
                            )
                            if reference_images:
                                yield 10, f'🟡 ĐANG GỬI ẢNH THAM CHIẾU → Gemini · **{len(reference_images)}/3 ảnh** · {ref_names}', prompt_scene_rows(scenes), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), int(attempt_token) + 1, ''
                                workflow_cmd_log('GEMINI', 'REFERENCE_UPLOAD_START', project=project_name, count=len(reference_images), files=ref_names)
                            else:
                                yield 10, '⚪ Không có Ảnh tham chiếu Nhân vật — Gemini B1 chạy không có reference.', prompt_scene_rows(scenes), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), int(attempt_token) + 1, ''
                            if not competitor_urls:
                                workflow_cmd_log('GEMINI', 'COMPETITOR_OPTIONAL_NOT_PROVIDED', project=project_name, level='INFO')
                            reference_instruction = (
                                'If reference images are attached, inspect the actual attached character/reference images, confirm that they were received, and build a concrete CHARACTER BIBLE / CHARACTER LOCK from those images. Do not claim to have seen an image that was not attached. '
                                if reference_images else ''
                            )
                            topic_style_request = (
                                'Analyze the FULL VIDEO SCRIPT below before writing any scene image prompts. '
                                'Do NOT generate images. First identify the domain/subdomain, main topic, narrative angle, visual system appropriate to THIS DOMAIN, all characters/subjects needed by the script, and how the attached references map to them. Do not force one mascot or one visual style across unrelated domains. '
                                + reference_instruction +
                                'COMPETITOR ANALYSIS IS REQUIRED WHEN LINKS ARE PROVIDED: open the exact YouTube channel/video URLs, inspect the accessible channel/video material and compare multiple relevant videos when a channel URL is supplied. '
                                'Study the competitor visual language: character design, rendering style, composition, camera/framing tendencies, color palette, contrast, lighting, environment treatment, typography/text-in-image treatment, pacing/presentation and overall image/thumbnail aesthetic. '
                                'Then define the PROJECT VISUAL SYSTEM for this video and store the reusable domain-specific rules. Keep continuity inside this video, while allowing a different domain to use a different visual system on the next video. Do not copy logos, characters, trademarks or exact frames. '
                                f'GEMINI VISUAL DIRECTIVE MODE: {normalize_gemini_visual_directive_mode(SETTINGS.get("gemini_visual_directive_mode", "auto"))}. GEMINI VISUAL DIRECTIVE COMMAND: {gemini_visual_directive_text(SETTINGS.get("gemini_visual_directive_mode", "auto"))}. Treat this as an authoritative workflow instruction. Gemini + Skill Stage 3 must follow it across related Scenes and must never silently override a user-locked mode. '
                                'YouTube watch/shorts links are supplied to Gemini as native video inputs; analyze those actual videos when attached. YouTube channel URLs are NOT URL-Context documents: the workflow uses Google Search to discover public videos from the exact channel, then supplies those discovered videos as native video inputs. Analyze those discovered videos when present. If no exact public video can be discovered/analyzed, explicitly set competitor_style_confirmed to false and explain that limitation in competitor_evidence; never pretend a channel page or video was inspected. '
                                'If a non-YouTube URL cannot actually be accessed, explicitly set competitor_style_confirmed to false and explain the limitation in competitor_evidence; never pretend that a source was inspected. '
                                'Return ONLY JSON matching the enforced B1 schema. Required fields: topic, topic_id, domain, subtopic, narrative_angle, visual_type, text_in_image, text_style, style, style_bible, character_bible, visual_rules, continuity_rules, competitor_style_confirmed, competitor_evidence. topic_id must be a stable lowercase library slug and must describe the domain/topic (for example history, health, science, technology); do NOT create a project-specific id. visual_rules and continuity_rules contain reusable topic/domain rules only. style_bible and character_bible belong to THIS PROJECT and must never be treated as global memory. '
                                'IMPORTANT: topic <= 12 words. The style field must be an explicit style lock, not a generic mood phrase. style_bible is the AUTHORITATIVE reusable visual DNA and must be detailed, concrete and production-usable. '
                                'The style_bible must describe visual characteristics actually observed from the accessible competitor material, not scene narration, plot events or generic filler adjectives. Do not collapse it into a generic phrase such as cinematic, realistic, beautiful lighting. '
                                'STYLE LOCK IS CRITICAL: identify the actual rendering mode first (for example 2D hand-drawn cartoon, flat illustration, 3D animation, photorealistic photography, painterly). Preserve that rendering mode across every IMAGE scene. Never infer photorealism merely because the story is serious, cinematic, emotional or dramatic. If the competitor is a hand-drawn/cartoon channel, explicitly say so in rendering_mode, shape_language, outline_style and character_rendering. Do not add 35mm film, photorealistic skin, cinematic photography, volumetric lighting or shallow depth of field unless those traits are actually observed in the competitor material. '
                                'competitor_evidence should be one short factual sentence naming what was actually observed from the accessible competitor material. '
                                'text_in_image is a GLOBAL preference only; each scene will decide its own image text later.\n\n'
                                'COMPETITOR CHANNEL / VIDEO LINKS TO CHECK NOW:\n' + ref_lines + '\n\n'
                                'FULL VIDEO SCRIPT:\n' + full_script
                            )
                            yield 12, f'🟡 ĐANG CHỜ ⏳ Gemini B1 — lượt 1/3 · 00:00 · Character Reference: {len(reference_images)}/3 ảnh', prompt_scene_rows(scenes), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), int(attempt_token) + 1, ''
                            if FULL_STOP_EVENT.is_set() or PROMPT_STOP_EVENT.is_set() or SCRIPT_STOP_EVENT.is_set():
                                raise RuntimeError('Đã dừng trước khi gửi FULL Kịch bản cho Gemini.')
                            workflow_cmd_log("GEMINI", "B1_API_CALL_ENTER", project=project_name, reference_count=len(reference_images), competitor_url_count=len(competitor_urls), pct=13)
                            topic_raw = _gemini_api_text(
                                topic_style_request,
                                reference_images=list(reference_images or [])[:3],
                                response_mode="topic_style",
                                max_attempts=3,
                                url_context_urls=competitor_urls,
                                response_validator=lambda raw: (
                                    _topic_structured_result_valid(((_GEMINI_LAST_STRUCTURED_RESULT or {}).get('topic_style') or {}))
                                    and _competitor_style_gate(
                                        (_GEMINI_LAST_STRUCTURED_RESULT or {}).get('topic_style') or {}
                                    )[2]
                                )
                            ).strip()
                            workflow_cmd_log("GEMINI", "STYLE_RESPONSE_RECEIVED", project=project_name, response_chars=len(topic_raw))
                            if reference_images:
                                workflow_cmd_log('GEMINI', 'REFERENCE_UPLOAD_CONFIRMED', project=project_name, count=len(reference_images), files=ref_names, character_bible=bool(character_bible))
                            structured_ts = {}
                            try:
                                structured_ts = dict((_GEMINI_LAST_STRUCTURED_RESULT or {}).get('topic_style') or {})
                            except Exception:
                                structured_ts = {}
                            style_bible = {}
                            character_bible = {}
                            if _topic_structured_result_valid(structured_ts):
                                topic = str(structured_ts.get('topic')).strip()
                                visual_style = str(structured_ts.get('style')).strip()
                                try:
                                    style_bible = dict(structured_ts.get('style_bible') or {})
                                except Exception:
                                    style_bible = {}
                                try:
                                    character_bible = dict(structured_ts.get('character_bible') or {})
                                except Exception:
                                    character_bible = {}
                                workflow_cmd_log("GEMINI", "STYLE_UI_TEXT_VERIFIED", project=project_name, source="api_structured_response", character_bible=bool(character_bible))
                            else:
                                raise RuntimeError(
                                    'Gemini B1 không trả đúng Structured Topic Profile schema: thiếu topic_id/domain/subtopic/visual_rules/continuity_rules hoặc style_bible. '
                                    'Không dùng parser text fallback vì B1 phải là nguồn phân loại chính thức.'
                                )
                            if not topic or not visual_style:
                                raise RuntimeError("Gemini đã trả response nhưng chưa nhận diện được đủ Topic + Global Style.")
                            visual_style = re.sub(r'\s+', ' ', str(visual_style or '')).strip()
                            # The short `style` field is only a summary. The detailed style_bible is
                            # the authoritative visual DNA sent to B2 and then embedded in every IMAGE prompt.
                            if style_bible:
                                bible_labels = {
                                    'visual_identity':'VISUAL IDENTITY', 'rendering_mode':'RENDERING MODE',
                                    'shape_language':'SHAPE LANGUAGE', 'outline_style':'OUTLINE STYLE',
                                    'composition':'COMPOSITION',
                                    'camera_framing':'CAMERA / FRAMING', 'lens_perspective':'LENS / PERSPECTIVE',
                                    'lighting':'LIGHTING', 'color_palette':'COLOR PALETTE', 'color_grading':'COLOR / GRADING',
                                    'contrast_exposure':'CONTRAST / EXPOSURE',
                                    'environment_production_design':'ENVIRONMENT / PRODUCTION DESIGN',
                                    'texture_detail':'TEXTURE / DETAIL', 'character_rendering':'CHARACTER RENDERING',
                                    'facial_feature_style':'FACIAL FEATURE STYLE', 'depth_of_field':'DEPTH OF FIELD',
                                    'thumbnail_text_layout':'THUMBNAIL TEXT LAYOUT', 'cinematic_language':'CINEMATIC LANGUAGE',
                                    'negative_style_rules':'NEGATIVE STYLE RULES', 'style_lock':'STYLE LOCK',
                                }
                                bible_parts = []
                                for key, label in bible_labels.items():
                                    val = re.sub(r'\s+', ' ', str(style_bible.get(key) or '')).strip()
                                    if val:
                                        bible_parts.append(f'{label}: {val}')
                                if bible_parts:
                                    visual_style = f"{visual_style} || STYLE BIBLE: " + ' | '.join(bible_parts)
                            # Prevent accidental loss of the competitor DNA. Allow a full production
                            # style bible instead of the previous 420-character truncation.
                            if len(visual_style) > 2600:
                                visual_style = visual_style[:2600].rsplit(' | ', 1)[0].strip(' |')
                            competitor_style_confirmed = False
                            competitor_evidence = ''
                            try:
                                # BỔ SUNG: Bóc tách JSON chặt chẽ hơn
                                json_candidate = topic_raw.strip()
                                match_json = re.search(r'\{.*\}', json_candidate, re.DOTALL)
                                if match_json:
                                    json_candidate = match_json.group(0)
                                else:
                                    json_candidate = re.sub(r'^```(?:json)?\s*|\s*```$', '', json_candidate, flags=re.I)
                                
                                _ts_obj = json.loads(json_candidate)
                                if isinstance(_ts_obj, dict):
                                    visual_type = str(_ts_obj.get('visual_type') or '').strip()
                                    text_in_image = bool(_ts_obj.get('text_in_image')) if 'text_in_image' in _ts_obj else False
                                    text_style = str(_ts_obj.get('text_style') or '').strip()
                                    competitor_style_confirmed = bool(_ts_obj.get('competitor_style_confirmed'))
                                    competitor_evidence = str(_ts_obj.get('competitor_evidence') or '').strip()
                                else:
                                    visual_type, text_in_image, text_style = '', False, ''
                            except Exception:
                                visual_type, text_in_image, text_style = '', False, ''
                                # BỔ SUNG: Fallback Regex cứu hộ nếu JSON hỏng nặng
                                if re.search(r'"competitor_style_confirmed"\s*:\s*true', topic_raw, re.I):
                                    competitor_style_confirmed = True
                                m_ev = re.search(r'"competitor_evidence"\s*:\s*"(.*?)"', topic_raw, re.S)
                                if m_ev:
                                    competitor_evidence = m_ev.group(1).strip()
                            competitor_style_fallback = bool(
                                competitor_urls and not competitor_style_confirmed and
                                bool(re.search(
                                    r'(?:cannot|can\'t|could\s+not|couldn\'t|unable|unavailable|inaccessible|not\s+(?:directly\s+)?accessible|access(?:ed|ing)?\s+(?:is\s+)?(?:restricted|denied|blocked)|restricted|connectivity\s+limitations?|client\s+connectivity|network\s+(?:limitations?|error)|not\s+available|cannot\s+be\s+(?:visually\s+)?rendered|không[^.]{0,120}(?:truy cập|tiếp cận|kết nối)|bị hạn chế|không thể truy cập)',
                                    competitor_evidence, re.I
                                ))
                            )
                            # One authoritative competitor gate. This must be the same gate
                            # used by B2/resume, otherwise B1 and B2 can disagree about whether
                            # the competitor source was actually handled.
                            _gate_urls, _gate_confirmed, _gate_ok = _competitor_style_gate({
                                'competitor_urls': competitor_urls,
                                'competitor_style_confirmed': competitor_style_confirmed,
                                'competitor_style_fallback': competitor_style_fallback,
                                'competitor_evidence': competitor_evidence,
                            })
                            if not _gate_ok:
                                workflow_cmd_log(
                                    'GEMINI', 'COMPETITOR_STYLE_GATE_FAIL',
                                    project=project_name,
                                    competitor_url_count=len(_gate_urls),
                                    confirmed=_gate_confirmed,
                                    evidence=competitor_evidence,
                                    level='ERROR',
                                )
                                raise RuntimeError(
                                    'Gemini chưa xác nhận style đối thủ. '
                                    'Không mở Z-IMAGE-TURBO Local; kiểm tra link/video đối thủ hoặc chạy lại Bước 1.'
                                )
                            competitor_style_confirmed = _gate_confirmed
                            if competitor_style_fallback:
                                workflow_cmd_log(
                                    'GEMINI', 'COMPETITOR_STYLE_FALLBACK',
                                    project=project_name, competitor_url_count=len(competitor_urls),
                                    evidence=competitor_evidence
                                )
                            write_json(
                                _project_content_dir(root) / 'Prompt' / 'topic_style.json',
                                {
                                    'topic': topic, 'topic_id': _topic_slug(str(structured_ts.get('topic_id') or structured_ts.get('domain') or topic)),
                                    'domain': str(structured_ts.get('domain') or 'other').strip(),
                                    'subtopic': str(structured_ts.get('subtopic') or structured_ts.get('subdomain') or '').strip(),
                                    'style': visual_style, 'style_summary': str(structured_ts.get('style') or visual_style).strip(),
                                    'style_bible': style_bible,
                                    'visual_rules': structured_ts.get('visual_rules') or {},
                                    'continuity_rules': structured_ts.get('continuity_rules') or {},
                                'style_lock': str(style_bible.get('style_lock') or style_bible.get('rendering_mode') or '').strip(),
                                'visual_type': visual_type,
                                    'text_in_image': text_in_image, 'text_style': text_style,
                                    'reference_images': reference_images, 'reference_image_names': [Path(x).name for x in reference_images], 'character_bible': character_bible, 'competitor_urls': competitor_urls,
                                    'competitor_style_confirmed': competitor_style_confirmed,
                                    'competitor_style_fallback': competitor_style_fallback,
                                    'competitor_evidence': competitor_evidence,
                                    'raw': topic_raw,
                                    'b1_verified': False,
                                    'api_response_verified': False,
                                    'api_response_sha256': '',
                                    'script_sha256': script_sha256,
                                    'script_chars': len(full_script),
                                    'updated': datetime.now().isoformat(),
                                    'source': 'Gemini FULL script + competitor channel/video analysis + Google Search discovery'
                                }
                            )
                            # Persist the complete B1 evidence set before B2 can unlock.
                            (_project_content_dir(root) / 'Prompt').mkdir(parents=True, exist_ok=True)
                            (_project_content_dir(root) / 'Prompt' / 'topic_style_raw.txt').write_text(topic_raw, encoding='utf-8')
                            write_json(_project_content_dir(root) / 'Prompt' / 'style_bible.json', {
                                'topic': topic, 'master_style_prompt': visual_style, 'style': visual_style,
                                'style_summary': str(structured_ts.get('style') or visual_style).strip(),
                                'style_bible': style_bible,
                                'character_bible': character_bible,
                                'reference_images': reference_images,
                                'reference_image_names': [Path(x).name for x in reference_images],
                                'visual_type': visual_type, 'text_in_image': text_in_image, 'text_style': text_style,
                                'competitor_urls': competitor_urls,
                                'competitor_style_confirmed': competitor_style_confirmed,
                                'competitor_style_fallback': competitor_style_fallback,
                                'competitor_evidence': competitor_evidence,
                                'script_sha256': script_sha256,
                                'script_chars': len(full_script),
                                'source': 'Gemini FULL script + competitor analysis + Google Search discovery',
                                'updated': datetime.now().isoformat(),
                            })
                            workflow_cmd_log(
                                'GEMINI', 'COMPETITOR_STYLE_CONFIRMED',
                                project=project_name, competitor_url_count=len(competitor_urls),
                                confirmed=competitor_style_confirmed, evidence=competitor_evidence
                            )
                            workflow_cmd_log("GEMINI", "STYLE_CONFIRMED", project=project_name, topic=topic, style=visual_style)
                            if competitor_style_fallback:
                                style_msg = (
                                    f'⚠️ ĐỐI THỦ KHÔNG TRUY CẬP ĐƯỢC — dùng Global Style fallback. **Chủ đề:** {topic}\n'
                                    f'**Global Style:** {visual_style}\n'
                                    f'**Bằng chứng:** {competitor_evidence or "(Gemini không trả bằng chứng)"}\n'
                                    '🤖 Gemini API Topic/Global Style đã VERIFY → BƯỚC 2.'
                                )
                            else:
                                style_msg = (
                                    f'✅ ĐÃ XÁC NHẬN STYLE ĐỐI THỦ. **Chủ đề:** {topic}\n'
                                    f'**Global Style:** {visual_style}\n'
                                    f'**Bằng chứng:** {competitor_evidence or "(Gemini không trả bằng chứng)"}\n'
                                    '🤖 Gemini API Topic/Global Style đã VERIFY → BƯỚC 2.'
                                )
                            # STEP-1 hard gate: Gemini API response has already passed the validator.
                            # No clipboard, browser focus, tab switching or copy/verify step is used.
                            api_response_verified = bool(topic and visual_style)
                            context_sha256 = hashlib.sha256(
                                f"GLOBAL TOPIC: {topic}\n\nGLOBAL VISUAL STYLE: {visual_style}".encode('utf-8')
                            ).hexdigest()
                            workflow_cmd_log(
                                'GEMINI', 'STYLE_API_RESPONSE_VERIFIED', project=project_name,
                                verified=api_response_verified, sha256=context_sha256[:16], pct=90
                            )
                            try:
                                state_fp = _project_content_dir(root) / 'Prompt' / 'topic_style.json'
                                state_obj = read_json(state_fp, {}) or {}
                                state_obj.update({
                                    'b1_verified': api_response_verified,
                                    'api_response_verified': api_response_verified,
                                    'api_response_sha256': context_sha256,
                                    'script_sha256': script_sha256,
                                    'script_chars': len(full_script),
                                    'verified_at': datetime.now().isoformat(),
                                })
                                write_json(state_fp, state_obj)
                                _topic_id_v4, style_bible, _bible_v4 = _persist_topic_and_bible(
                                    project_name, structured_ts, topic, visual_style, style_bible,
                                    character_bible, visual_type,
                                    ref_names.split(' | ') if ref_names and ref_names != '(không có)' else [],
                                    competitor_urls, competitor_style_confirmed, competitor_evidence,
                                )
                                workflow_cmd_log('GEMINI', 'AUTO_TOPIC_CLASSIFIED', project=project_name,
                                                 topic_id=_topic_id_v4,
                                                 domain=str(structured_ts.get('domain') or 'other'),
                                                 profile=str(_topic_profile_path(_topic_id_v4)))
                            except Exception as persist_bible_exc:
                                workflow_cmd_log('GEMINI', 'PROJECT_BIBLE_PERSIST_WARNING', project=project_name, error=str(persist_bible_exc)[:300], level='WARN')
                            try:
                                write_json(_project_content_dir(root) / 'Prompt' / 'topic_style_verified.json', {
                                    'verified': api_response_verified, 'topic': topic, 'style': visual_style,
                                    'style_summary': str(structured_ts.get('style') or visual_style).strip(),
                                    'style_bible': style_bible,
                                    'visual_type': visual_type, 'text_in_image': text_in_image, 'text_style': text_style,
                                    'competitor_style_confirmed': competitor_style_confirmed,
                                    'competitor_style_fallback': competitor_style_fallback,
                                    'competitor_evidence': competitor_evidence,
                                    'api_response_verified': api_response_verified,
                                    'api_response_sha256': context_sha256,
                                    'script_sha256': script_sha256,
                                    'script_chars': len(full_script),
                                    'transport': 'Gemini API',
                                    'verified_at': datetime.now().isoformat(),
                                })
                            except Exception as persist_exc:
                                workflow_cmd_log('GEMINI', 'STYLE_STATE_PERSIST_WARNING', project=project_name, error=str(persist_exc)[:300], level='WARN')

                            # MASTER FIX: B1 checkpoint stays in the Gemini API stage.
                            # No browser focus, clipboard, or tab switching is used here.
                            workflow_cmd_log(
                                'GEMINI', 'STEP1_API_CHECKPOINT_KEPT',
                                project=project_name,
                                handoff='stay_in_api_stage',
                                checkpoint='topic_style.json',
                            )
                            style_msg = (
                                f'✅ BƯỚC 1 ĐÃ API VERIFY — **Topic:** {topic}\n'
                                f'**Global Style:** {visual_style}\n'
                                f'**Bằng chứng:** {competitor_evidence or "(không có)"}\n'
                                '🤖 API B1 đã xác nhận → chuẩn bị BƯỚC 2.'
                            )
                            workflow_cmd_log('GEMINI', 'STEP1_CHECKPOINT_READY', project=project_name, topic=topic, style=visual_style)
                            yield 25, style_msg, prompt_scene_rows(scenes), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), int(attempt_token) + 1, style_msg
                            workflow_cmd_log(
                                'GEMINI', 'B1_TO_B2_CONTINUE', project=project_name,
                                phase=phase, scene_count=len(scenes), same_api_stage=True,
                            )

                        if phase == 'style':
                            # Manual B1 remains a checkpointed stage. MASTER uses phase='master'
                            # and deliberately falls through into B2 without leaving Gemini.
                            with PROMPT_BATCH_STATE_LOCK:
                                PROMPT_BATCH_STATE[project_key] = 'style_ready'
                            confirm = (
                                f'✅ BƯỚC 1/2 ĐÃ API VERIFY — **Topic:** {topic} · **Global Style:** {visual_style}.\n'
                                '↩ Đã xác nhận checkpoint. Bấm B2 để tiếp tục.'
                            )
                            yield 100, confirm, prompt_scene_rows(scenes), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), int(attempt_token) + 1, confirm
                            return

                        # STEP 2 — Chunked scene prompts. MASTER reaches this point in the
                        # SAME Python callback and SAME Python API stage immediately after B1.
                        # <=35 scenes = keep the legacy single Gemini request.
                        # >35 scenes = automatically split into sequential chunks of 35.
                        # This is intentionally independent of narration length: scene count
                        # is the deterministic chunk gate requested for production.
                        total = len(scenes)
                        target_chunk = total if total <= 35 else 35
                        chunks = [scenes[i:i + target_chunk] for i in range(0, total, target_chunk)]
                        # Durable B2 resume identity: completed chunks are reusable only for
                        # the exact same script + B1 visual context + media mode + chunk plan.
                        script_sha256 = _project_current_script_hash(project_name)
                        media_mode = normalize_media_mode(SETTINGS.get('media_mode', 'pure_images'))
                        _chunk_context_blob = json.dumps({
                            'topic': topic, 'visual_style': visual_style, 'visual_type': visual_type,
                            'text_in_image': bool(text_in_image), 'text_style': text_style,
                            'character_bible': character_bible, 'chunk_size': int(target_chunk),
                            'chunk_count': len(chunks), 'media_mode': media_mode,
                        }, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
                        context_sha256 = hashlib.sha256(_chunk_context_blob.encode('utf-8')).hexdigest()
                        workflow_cmd_log(
                            'GEMINI', 'SCENE_CHUNK_PLAN',
                            project=project_name, total_scenes=total,
                            chunk_size=target_chunk, chunk_count=len(chunks),
                            mode='single_request' if total <= 35 else 'sequential_35_scene_chunks',
                        )
                        all_prompts = {}
                        yield 30, f'🧠 Đã nhớ Topic + Style. Chia **{total} Scene → {len(chunks)} lượt Gemini** (tối đa {target_chunk}/lượt)…', prompt_scene_rows(scenes), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), int(attempt_token) + 1, style_msg
                        for chunk_index, chunk in enumerate(chunks, 1):
                            if FULL_STOP_EVENT.is_set() or PROMPT_STOP_EVENT.is_set() or SCRIPT_STOP_EVENT.is_set():
                                raise RuntimeError('Đã dừng trước khi gửi Chunk Gemini.')
                            first_id = chunk[0].get('scene_id','')
                            last_id = chunk[-1].get('scene_id','')
                            media_mode = normalize_media_mode(SETTINGS.get('media_mode', 'pure_images'))
                            expected_ids = [str(s.get('scene_id') or '') for s in chunk]
                            checkpoint_records = _load_gemini_chunk_checkpoint(
                                project_name, chunk_index, len(chunks), target_chunk, expected_ids,
                                script_sha256, context_sha256, media_mode,
                            )
                            if checkpoint_records is not None:
                                parsed = checkpoint_records
                                raw_chunk = ''
                                workflow_cmd_log(
                                    'GEMINI', 'CHUNK_CHECKPOINT_RESUME', project=project_name,
                                    chunk=chunk_index, chunks=len(chunks), first=first_id, last=last_id,
                                    scene_count=len(chunk), checkpoint=str(_gemini_chunk_checkpoint_path(project_name, chunk_index)),
                                )
                            else:
                                scene_payload = [{
                                    'scene_id': s.get('scene_id',''),
                                    'timestamp_start': s.get('timestamp_start') or s.get('start_timestamp') or s.get('start_time') or s.get('time_start') or '',
                                    'timestamp_end': s.get('timestamp_end') or s.get('end_timestamp') or s.get('end_time') or s.get('time_end') or '',
                                    'narration': (s.get('text') or '').strip(),
                                    'explicit_media_hint': s.get('media_type') if s.get('media_type_source') == 'explicit_tag' else ''
                                } for s in chunk]
                                completion_marker = f"<<<CHUNK_COMPLETE_{chunk_index}_{last_id}>>>"
                                if media_mode == 'mixed_media':
                                    media_rules = ('For EACH scene decide MEDIA TYPE first: IMAGE or VIDEO. Choose VIDEO when motion, demonstration, exercise/action, transformation, process, camera movement, or another dynamic visual is essential; choose IMAGE for explanatory, conceptual, portrait, diagram, establishing or static evidence scenes. Do not choose VIDEO merely because a person is present. Explicit [VIDEO]/[CLIP] forces VIDEO; [IMAGE]/[IMG]/[ẢNH] forces IMAGE. ')
                                else:
                                    media_rules = 'MEDIA MODE is PURE IMAGES: every scene MUST be IMAGE. '
                                _bible_ctx = read_json(_project_content_dir(root) / 'Prompt' / 'project_bible.json', {}) or {}
                                _state_ctx = _recent_scene_state_context(project_name, limit=12)
                                batch_prompt = (
                                    'You are the scene media director and visual prompt writer. Do NOT generate images and do NOT call any image-generation tool. '
                                    'Use the PROJECT BIBLE and PROJECT VISUAL SYSTEM. The visual system is authoritative for THIS VIDEO and must be appropriate to its domain. Do not force one channel-wide mascot/style across unrelated domains. Preserve rendering mode, line/shape language, character/subject identity, palette, lighting, setting and continuity. If the style lock says 2D hand-drawn/cartoon, the IMAGE prompt must explicitly remain 2D hand-drawn/cartoon and must NOT introduce photorealistic skin, photographic lens language, 35mm film, volumetric photography or cinematic-photography wording. Preserve narration meaning, continuity, characters and setting. ' + media_rules +
                                    'STRICT STAGE-3 CONTRACT: produce EXACTLY ONE IMAGE prompt per timestamp/scene record. Never merge two timestamps into one prompt and never invent extra timestamps. Each IMAGE record MUST carry timestamp_start, timestamp_end, style_anchor, visual_action, continuity_from_previous, state_after, end_style_lock. Preserve source timestamps exactly; if no real timestamps exist, leave timestamp fields empty and use scene_id as the timeline key — NEVER fabricate clock times. The prompt order MUST be START TIMESTAMP/TIMELINE KEY -> STYLE ANCHOR -> CONCRETE VISUAL ACTION -> CHARACTER/OBJECT/LOCATION CONTINUITY -> CAMERA/COMPOSITION -> END STYLE LOCK. END STYLE LOCK must preserve the authoritative rendering mode, shape language, palette, lighting and character identity for the next scene. ' +
                                    'OUTPUT CONTRACT: Return ONLY JSON with this exact top-level shape: {"scenes":[{"scene_id":"Scene_001","media_type":"image","timestamp_start":"","timestamp_end":"","prompt":"...","state":{"scene_id":"Scene_001","narration":"...","characters":[],"location":"","objects":[],"action":"","continuity_from_previous":"","style_anchor":"","state_after":"","end_style_lock":"","timestamp_start":"","timestamp_end":"","text_in_image":false,"text_content":"","text_position":""}}]}. Include exactly one record per supplied scene, never merge timestamps/scenes and never invent scene IDs. For VIDEO use media_type=video and prompt=""; for IMAGE use media_type=image and one self-contained production-ready Z-Image-Turbo prompt. Preserve timestamps exactly. state_after must describe the continuity facts that the NEXT scene needs. Do not invent entities not supported by the script or existing Project Bible. '
                                    'For VIDEO use exactly TYPE=VIDEO || PROMPT=NONE. Never write a Z-IMAGE-TURBO image prompt for VIDEO. For IMAGE write one self-contained production-ready Z-Image-Turbo prompt. '
                                    'TEXT POLICY: Decide text_in_image INDEPENDENTLY FOR EACH IMAGE SCENE from that scene narration, dialogue and visual purpose. Do NOT default every scene to false and do NOT copy a global text preference into every scene. Use text_in_image=true when a SHORT piece of visible text materially helps illustrate the scene or directly reinforces the spoken dialogue (for example a short quoted phrase, sign, label or key word that belongs naturally inside the scene). Otherwise use false. When true, text_content MUST be a concise exact phrase taken from or directly matching the scene dialogue/meaning, normally <= 80 characters, NEVER a full subtitle or paragraph; text_position MUST describe a natural in-world position. Never invent decorative captions, subtitles, UI text, logos or watermarks. Never turn dialogue into bottom-screen subtitles. Never add text merely because a character is speaking. After the last scene write exactly: ' + completion_marker + '\n\n'
                                    f'MEDIA MODE: {media_mode}\nPROJECT TOPIC: {topic}\nTOPIC ID: {str(_bible_ctx.get("topic_id") or _topic_slug(_bible_ctx.get("domain") or topic))}\nPROJECT VISUAL SYSTEM: {visual_style}\nPROJECT BIBLE: {json.dumps(_bible_ctx, ensure_ascii=False, separators=(",",":"))}\nRECENT SCENE STATES (continuity memory): {json.dumps(_state_ctx, ensure_ascii=False, separators=(",",":"))}\nREFERENCE LIBRARY: {json.dumps([Path(x).name for x in reference_images], ensure_ascii=False)}\nCHARACTER LOCK: {json.dumps(character_bible, ensure_ascii=False, separators=(",",":")) if character_bible else "NONE — do not invent a reference character"}\nIMAGE TYPE: {visual_type or "infer from the style"}\n'
                                    'PER-SCENE TEXT DECISION: Choose independently for every scene. Global B1 text_in_image is only a hint and MUST NOT force the scene decision.\n'
                                    f'TEXT STYLE: {text_style or "short readable in-world text only when that scene calls for it"}\n\n'
                                    f'SCENES {first_id} → {last_id}:\n' + json.dumps(scene_payload, ensure_ascii=False, separators=(',', ':'))
                                )
                                workflow_cmd_log("GEMINI", "SEND_CHUNK", project=project_name, chunk=chunk_index, chunks=len(chunks), first=first_id, last=last_id, scene_count=len(chunk))
                                pct = 35 + int((chunk_index - 1) / max(1, len(chunks)) * 40)
                                yield pct, f'🟡 ĐANG CHỜ ⏳ ĐANG CHỜ GEMINI VERIFY Prompt + Style · lượt 1/3 · 00:00 · Scene **{first_id} → {last_id}**', prompt_scene_rows(scenes), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), int(attempt_token) + 1, ''
                                raw_chunk = _gemini_api_text(
                                    batch_prompt,
                                    completion_marker=completion_marker, response_mode='batch',
                                    max_attempts=3,
                                    response_validator=lambda raw, _chunk=chunk, _mode=media_mode: (
                                        _parse_batch_scene_media_records(raw, _chunk, _mode)
                                    )
                                )
                                raw_chunk = str(raw_chunk or '').strip()
                                parsed = _parse_batch_scene_media_records(raw_chunk, chunk, media_mode)
                                missing_chunk = [sid for sid in expected_ids if sid not in parsed]
                                if missing_chunk:
                                    raise RuntimeError(f'Gemini Chunk {chunk_index}/{len(chunks)} thiếu Scene: ' + ', '.join(missing_chunk))
                                _save_gemini_chunk_checkpoint(
                                    project_name, chunk_index, len(chunks), target_chunk, expected_ids,
                                    first_id, last_id, script_sha256, context_sha256, media_mode, parsed,
                                )
                                workflow_cmd_log(
                                    'GEMINI', 'CHUNK_CHECKPOINT_SAVED', project=project_name,
                                    chunk=chunk_index, chunks=len(chunks), first=first_id, last=last_id,
                                    scene_count=len(chunk),
                                )

                            for sid in expected_ids:
                                record = parsed[sid]
                                scene_ref = next(s for s in chunk if str(s.get('scene_id')) == sid)
                                scene_ref['media_type'] = record.get('media_type','image')
                                scene_ref['media_type_source'] = 'explicit_tag' if scene_ref.get('media_type_source') == 'explicit_tag' else 'gemini'
                                all_prompts[sid] = record.get('prompt','')
                                scene_ref['reference_ids'] = list(record.get('reference_ids') or [])[:3]
                                _state = record.get('state') or {}
                                if not isinstance(_state, dict):
                                    _state = {}
                                _state.setdefault('scene_id', sid)
                                _state.setdefault('narration', scene_ref.get('text',''))
                                _state.setdefault('media_type', scene_ref.get('media_type','image'))
                                _state['text_in_image'] = bool(_state.get('text_in_image', False))
                                _state['text_content'] = str(_state.get('text_content') or '').strip()
                                _state['text_position'] = str(_state.get('text_position') or '').strip()
                                if not _state['text_in_image']:
                                    _state['text_content'] = ''
                                    _state['text_position'] = ''
                                _state.setdefault('updated_at', datetime.now().isoformat())
                                _all_states = _load_scene_states(project_name)
                                _state_map = _all_states.get('states') if isinstance(_all_states, dict) else {}
                                if not isinstance(_state_map, dict):
                                    _state_map = {}
                                _state_map[sid] = _state
                                _save_scene_states(project_name, _state_map)
                                _update_project_bible_from_states(project_name, [_state])
                            # Chunk text stays in memory until the complete B2 result is verified.
                            # Do not write dozens of intermediate files during production runs.
                            workflow_cmd_log("GEMINI", "CHUNK_RECEIVED", project=project_name, chunk=chunk_index, chunks=len(chunks), scene_count=len(chunk), response_chars=(len(raw_chunk) if raw_chunk else 0), resumed=(checkpoint_records is not None))
                            yield 35 + int(chunk_index / max(1, len(chunks)) * 40), f'✅ Chunk **{chunk_index}/{len(chunks)}** hoàn tất — {len(chunk)}/{len(chunk)} Scene. Đang chuyển lượt tiếp theo…', prompt_scene_rows(scenes), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), int(attempt_token) + 1, ''

                        for scene in scenes:
                            sid = scene.get('scene_id','')
                            if scene_media_type(scene,'image') == 'video':
                                scene['prompt_core']=''
                                scene['prompt']=''
                                scene['prompt_context']={'topic':topic,'style':visual_style,'media_type':'video'}
                                continue
                            scene_core = re.sub(r'\s+', ' ', str(all_prompts.get(sid, '') or '')).strip()
                            scene['prompt_core'] = scene_core
                            _scene_state_doc = _load_scene_states(project_name)
                            _scene_states_map = _scene_state_doc.get('states') if isinstance(_scene_state_doc, dict) else {}
                            _scene_state = _scene_states_map.get(sid, {}) if isinstance(_scene_states_map, dict) else {}
                            _scene_text_flag = bool(_scene_state.get('text_in_image', False))
                            _scene_text_style = text_style if _scene_text_flag else ''
                            scene['text_in_image'] = _scene_text_flag
                            scene['text_content'] = str(_scene_state.get('text_content') or '') if _scene_text_flag else ''
                            scene['text_position'] = str(_scene_state.get('text_position') or '') if _scene_text_flag else ''
                            scene['prompt'] = _canonical_scene_prompt(topic, visual_style, scene_core, visual_type=visual_type, text_in_image=_scene_text_flag, text_style=_scene_text_style)
                            scene['prompt_context'] = {'topic': topic, 'style': visual_style, 'media_type':'image', 'text_in_image': _scene_text_flag}
                        workflow_cmd_log("GEMINI", "SCENE_STATES_PERSISTED", project=project_name,
                                         scene_state_file=str(_scene_states_path(project_name)),
                                         state_count=len((_load_scene_states(project_name).get('states') or {})))
                        workflow_cmd_log("GEMINI", "ALL_CHUNKS_RECEIVED", project=project_name, scene_count=total, chunk_count=len(chunks))
                        # Build and validate the complete Gemini result in memory first.
                        # The order is intentional: Gemini result -> API VERIFY -> persist JSON.
                        # This makes the Gemini API validation the immediate user-visible
                        # completion signal instead of waiting behind disk writes.
                        missing = [s.get('scene_id','Scene') for s in scenes if scene_media_type(s,'image') != 'video' and not (s.get('prompt') or '').strip()]
                        if missing:
                            raise RuntimeError('Gemini chưa trả đủ Prompt theo Scene: ' + ', '.join(missing))
                        valid_prompts, prompt_errors = _validate_scene_prompts_strict(scenes, topic, visual_style)
                        if not valid_prompts:
                            preview_errors = '; '.join(prompt_errors[:20])
                            raise RuntimeError('STRICT PROMPT GATE FAIL: ' + preview_errors)
                        # FAST HARD GATE: the Gemini response is already parsed into the
                        # canonical manifest above. The complete Gemini API response has already been
                        # parsed and validated; persist it directly without clipboard handoff.
                        image_scene_count = sum(1 for sc in scenes if scene_media_type(sc, 'image') != 'video')
                        workflow_cmd_log(
                            'GEMINI', 'PROMPT_VERIFY_FAST', project=project_name,
                            scene_count=len(scenes), image_scene_count=image_scene_count
                        )
                        image_scene_count = sum(1 for sc in scenes if scene_media_type(sc, 'image') != 'video')
                        api_prompt_verified = bool(image_scene_count == 0 or all((sc.get('prompt') or '').strip() for sc in scenes if scene_media_type(sc, 'image') != 'video'))
                        prompt_batch_text = '\n\n'.join(
                            f"{sc.get('scene_id', 'Scene')}:\n{str(sc.get('prompt') or sc.get('image_prompt') or '').strip()}"
                            for sc in scenes if scene_media_type(sc, 'image') != 'video' and str(sc.get('prompt') or sc.get('image_prompt') or '').strip()
                        )
                        prompt_batch_sha256 = hashlib.sha256(prompt_batch_text.encode('utf-8')).hexdigest() if prompt_batch_text else ''
                        workflow_cmd_log(
                            'GEMINI', 'PROMPT_BATCH_API_VERIFIED', project=project_name,
                            scene_count=image_scene_count, verified=api_prompt_verified,
                            sha256=prompt_batch_sha256[:16], pct=95
                        )
                        # B2 API response is verified. Now persist the machine-readable
                        # artifacts used by Z-IMAGE-TURBO and the final gate.
                        write_json(_project_content_dir(root) / 'Cảnh' / 'manifest.json', {'scenes': scenes})
                        write_json(_project_content_dir(root) / 'Prompt' / 'scene_inputs.json', {'topic': topic, 'style': visual_style, 'generated_by': 'Gemini: reference analysis -> chunked scene prompts', 'updated': datetime.now().isoformat(), 'scenes': [{'scene_id': s.get('scene_id',''), 'prompt_core': s.get('prompt_core',''), 'input': s.get('prompt',''), 'topic': topic, 'style': visual_style} for s in scenes]})
                        workflow_cmd_log('GEMINI', 'B2_API_VERIFY_JSON_PERSIST_START', project=project_name, scene_count=len(scenes))
                        yield 94, f'✅ BƯỚC 2 GEMINI API VERIFY THÀNH CÔNG — {len(scenes)}/{len(scenes)} Prompt. Đang lưu JSON ngầm…', prompt_scene_rows(scenes), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), int(attempt_token) + 1, ''
                        # Hard handoff gate: only this point is allowed to unlock Z-IMAGE-TURBO Local.
                        verified_file = _write_prompt_verified(
                            project_name, topic, visual_style, len(scenes),
                            api_response_verified=True
                        )
                        workflow_cmd_log("GEMINI", "PROMPT_VERIFIED", project=project_name, topic=topic, style=visual_style, scene_count=len(scenes), marker=str(verified_file))
                        with PROMPT_BATCH_STATE_LOCK:
                            PROMPT_BATCH_STATE[project_key] = 'success'
                        # Gemini API stage is complete. The MASTER gate now unlocks the
                        # local Z-IMAGE-TURBO image stage; no Gemini browser handoff is needed.
                        return_msg = '⏳ Gemini STEP-2 đã VERIFY xong; MASTER đang chuyển sang Hình ảnh → Z-IMAGE-TURBO.'
                        yield 100, f'✅ HOÀN TẤT: {len(scenes)}/{len(scenes)} Prompt theo Scene.\n🎬 **Chủ đề:** {topic}\n🎨 **Global Style:** {visual_style}\n{return_msg}\n➡ Đang chuyển sang tab **Hình ảnh** tự động.', prompt_scene_rows(scenes), gr.update(visible=True, value='💾 Lưu prompt → sang Hình ảnh'), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), int(attempt_token) + 1, f'✅ HOÀN TẤT: {len(scenes)}/{len(scenes)} Prompt theo Scene. 🎬 Topic: {topic} · 🎨 Style: {visual_style}. {return_msg} ➡ Đang chuyển sang tab Hình ảnh.'
                    except Exception as exc:
                        _clear_prompt_verified(project_name)
                        stopped_now = bool(FULL_STOP_EVENT.is_set() or PROMPT_STOP_EVENT.is_set() or SCRIPT_STOP_EVENT.is_set())
                        with PROMPT_BATCH_STATE_LOCK:
                            PROMPT_BATCH_STATE[project_key] = 'idle' if stopped_now else 'await_retry'
                        try:
                            rows = prompt_scene_rows(load_manifest(project_name))
                        except Exception:
                            rows = []
                        if stopped_now:
                            yield 0, '⏹ Đã dừng — 0%.', rows, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), int(attempt_token) + 1, '⏹ Đã dừng — 0%.'
                        else:
                            workflow_cmd_log(
                                'GEMINI', 'WORKFLOW_ERROR', project=project_name,
                                error=str(exc)[:1200], phase=phase, retry=retry,
                            )
                            yield 100, f'❌ Gemini chỉ báo lỗi sau khi đã hết 3 lượt thử.\n👉 Bấm **🔄 Tạo lại Prompt**.', rows, gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), int(attempt_token) + 1, ''
                    finally:
                        PROMPT_BATCH_LOCK.release()

                # Register the launch_ui-local Gemini runner for the module-level
                # one-click finalizer.  Do this immediately after defining the
                # generator, before any event can invoke the master callback.

                def _reset_prompt_gate(project_name):
                    key = safe_name(project_name)
                    with PROMPT_BATCH_STATE_LOCK:
                        PROMPT_BATCH_STATE[key] = 'idle'
                        new_token = int(PROMPT_UI_TOKENS.get(key, 0)) + 1
                        PROMPT_UI_TOKENS[key] = new_token
                    # This token MUST be passed to the next .then(_get_prompt_ui).
                    # Returning it to no outputs caused Gradio to warn and left the
                    # downstream Prompt stage holding the old token, so Gemini never ran.
                    return new_token

                def _master_gemini_strict_sequence(project_name, profile_name, attempt_token=0):
                                    """MASTER Gemini pipeline: B1 Topic/Style -> B2 Scene chunks in ONE callback.

                                    One Python generator owns the Gemini API stage. B1 is verified, then B2 is sent
                                    without browser/clipboard/tab dependencies. The only UI handoff occurs after all
                                    Gemini artifacts are verified.
                                    """
                                    # RUN FAST is SMART: a verified Prompt checkpoint bypasses Gemini entirely.
                                    # This is the same safety rule as explicit RESUME, but without requiring a
                                    # separate button for a normal rerun of a completed project.
                                    existing_prompt_ok, existing_prompt_obj = _project_prompt_checkpoint(project_name)
                                    if existing_prompt_ok:
                                        scene_count = int(existing_prompt_obj.get("scene_count", 0) or 0)
                                        workflow_cmd_log(
                                            "MASTER", "RUN_FAST_JUMP_IMAGE", project=project_name,
                                            engine=_selected_image_engine(),
                                            scene_count=scene_count,
                                            script_sha256=str(existing_prompt_obj.get("script_sha256") or _project_current_script_hash(project_name))[:16],
                                            reason="verified Prompt checkpoint",
                                        )
                                        msg = (
                                            f"♻ RUN FAST RESUME — Prompt VERIFY OK ({scene_count} Scene). "
                                            f"Bỏ qua B1/B2 Gemini → Engine đã chọn: {_selected_image_engine()}."
                                        )
                                        yield tuple([
                                            100, msg, prompt_scene_rows(load_manifest(project_name)),
                                            gr.update(visible=True, value="💾 Prompt đã VERIFY — sang Hình ảnh"),
                                            gr.update(visible=False), gr.update(visible=True),
                                            int(attempt_token or 0),
                                            _workflow_progress_card(msg, 100, "image") + "\n\n[MASTER_PROMPT_RESUME_OK]"
                                        ])
                                        return
                                    if not _master_voice_gate_ok(project_name):
                                        msg = "⛔ Gemini Prompt bị khóa — Voice Full chưa VERIFY hoàn tất. Không được nhảy sang B3."
                                        workflow_cmd_log("MASTER", "GEMINI_BLOCKED_VOICE_GATE", project=project_name)
                                        yield tuple([
                                            0, msg, prompt_scene_rows(load_manifest(project_name)),
                                            gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
                                            int(attempt_token or 0), _workflow_progress_card(msg, 100, "voice")
                                        ])
                                        return
                                    if not _GEMINI_STAGE_LOCK.acquire(blocking=False):
                                        msg = "⛔ Gemini đang chạy — không gửi thêm lệnh."
                                        workflow_cmd_log("MASTER", "GEMINI_STAGE_ALREADY_RUNNING", project=project_name)
                                        yield tuple([
                                            100, msg, prompt_scene_rows(load_manifest(project_name)),
                                            gr.update(visible=False), gr.update(visible=True), gr.update(visible=False),
                                            int(attempt_token or 0), _workflow_progress_card(msg, 100, "gemini")
                                        ])
                                        return
                                    try:
                                        workflow_cmd_log('B1', 'START — Gemini Topic/Style + Scene Prompt')
                                        project_key = safe_name(project_name)
                                        root = project_dir(project_name)
                                        # RUN FAST POLICY: B1/B2 checkpoints were cleared immediately before this callback.
                                        workflow_cmd_log(
                                            'MASTER', 'GEMINI_B1_B2_SINGLE_CHAIN_START',
                                            project=project_name, handoff='gemini_api',
                                        )

                                        # phase='master' is the key: _run_prompt_from_voice executes B1,
                                        # verifies Topic + Style, then falls through directly into chunked B2.
                                        # It does NOT focus the tool tab between the two requests.
                                        for item in _run_prompt_from_voice(
                                            project_name, profile_name, int(attempt_token or 0), False, 'master'
                                        ):
                                            normalized = _prompt_workflow_item(item)
                                            # The declared Gradio callback has exactly 8 outputs.
                                            # Never allow an accidental 9th value to reach Gradio.
                                            if isinstance(normalized, (tuple, list)):
                                                normalized = tuple(list(normalized)[:8])
                                                if len(normalized) < 8:
                                                    normalized = normalized + tuple("" for _ in range(8 - len(normalized)))
                                            yield normalized

                                        state = PROMPT_BATCH_STATE.get(project_key, '')
                                        verified = _project_content_dir(root) / 'Prompt' / 'prompt_verified.json'
                                        if state != 'success' or not verified.exists():
                                            workflow_cmd_log(
                                                'MASTER', 'GEMINI_B1_B2_NOT_COMPLETE',
                                                project=project_name, state=state, verified=verified.exists(),
                                            )
                                            return

                                        # Gemini B1 + B2 are verified. Z-IMAGE-TURBO keeps its own B3 stage below.
                                        # Keep this as one compact CMD line; do not dump Gemini response text.
                                        workflow_cmd_log(
                                            'MASTER', 'GEMINI_B1_B2_COMPLETE',
                                            project=project_name, scene_count=len(load_manifest(project_name)),
                                            handoff='gemini_api',
                                        )
                                    finally:
                                        # Release only the Python reservation. UI tab switching happens only after B1+B2 verification.
                                        try:
                                            _GEMINI_STAGE_LOCK.release()
                                        except RuntimeError:
                                            pass

                def _image_only_ui(project_name, rerender_all=False):
                                    """IMAGE ONLY: selected engine only; never silently switches to Z Image."""
                                    project_name=safe_name(str(project_name or SETTINGS.get("selected_project","") or ""))
                                    if not project_name:
                                        yield _workflow_progress_card("⛔ IMAGE ONLY: chưa chọn Project.",0,"image"); return
                                    prompt_ok,_=_project_prompt_checkpoint(project_name)
                                    if not prompt_ok:
                                        yield _workflow_progress_card("⛔ IMAGE ONLY bị khóa — Prompt checkpoint chưa hợp lệ.",0,"prompt"); return
                                    engine=_selected_image_engine();ok,detail=_creative_environment_check(engine)
                                    if not ok: yield _workflow_progress_card(detail,0,"image"); return
                                    scenes=load_manifest(project_name)
                                    if not scenes: yield _workflow_progress_card("⛔ Không có Scene.",0,"image"); return
                                    _creative_master_full(project_name,scenes,engine,start=True)
                                    total=sum(1 for s in scenes if scene_media_type(s,"image")=="image");active_total=total*len(CREATIVE_IMAGE_ONLY_STAGES)
                                    while True:
                                        if _master_is_stopped(): _creative_stop(project_name);yield _workflow_progress_card("⏹ IMAGE ONLY đã dừng.",0,"image");return
                                        with _batch_db() as db: rows=db.execute("SELECT scene_id,stage,status FROM creative_jobs WHERE project_id=?",(project_name,)).fetchall()
                                        done=sum(1 for r in rows if r["stage"] in CREATIVE_IMAGE_ONLY_STAGES and r["status"]=="DONE")
                                        failed=sum(1 for r in rows if r["stage"] in CREATIVE_IMAGE_ONLY_STAGES and r["status"]=="FAILED")
                                        captcha=sum(1 for r in rows if r["stage"] in CREATIVE_IMAGE_ONLY_STAGES and r["status"]=="CAPTCHA")
                                        pct=int(min(99,(done/active_total)*100)) if active_total else 100
                                        status=f"{engine}: DONE {done}/{active_total} · CAPTCHA {captcha} · FAIL {failed}"
                                        yield _workflow_progress_card(status,pct,"image")
                                        if failed or (active_total and done>=active_total): return
                                        time.sleep(1.0)
                def _resume_project_ui(project_name):
                                    """Smart RESUME: if Prompt is valid, jump directly to Z-Image; otherwise continue Gemini only when needed."""
                                    project_name = safe_name(str(project_name or SETTINGS.get("selected_project", "") or ""))
                                    if not project_name:
                                        yield _workflow_progress_card("⛔ RESUME: chưa chọn Project.", 0, "gemini")
                                        return
                                    prompt_ok, prompt_obj = _project_prompt_checkpoint(project_name)
                                    if prompt_ok:
                                        workflow_cmd_log(
                                            "MASTER", "RESUME_JUMP_ZIMAGE", project=project_name,
                                            script_sha256=str(prompt_obj.get("script_sha256") or _project_current_script_hash(project_name))[:16],
                                            reason="verified Prompt checkpoint",
                                        )
                                        yield _workflow_progress_card(
                                            "♻ RESUME: Prompt checkpoint hợp lệ → bỏ qua Scene/Voice/Gemini → Z-Image-Turbo.",
                                            5, "zimage"
                                        )
                                        last = ""
                                        for pct, status, hud in run_zimage_txt2img_queue(project_name):
                                            last = str(status or "")
                                            yield hud
                                        return

                                    scene_ok, _ = _project_scene_checkpoint(project_name, infer=True)
                                    if not scene_ok:
                                        workflow_cmd_log("MASTER", "RESUME_BLOCKED_SCENE_CHECKPOINT", project=project_name)
                                        yield _workflow_progress_card(
                                            "⛔ RESUME: Scene checkpoint chưa hợp lệ. Dùng RUN FAST · FRESH để dựng lại Scene.",
                                            0, "gemini"
                                        )
                                        return
                                    voice_ok, _, _ = _project_voice_checkpoint(project_name)
                                    if not voice_ok:
                                        workflow_cmd_log("MASTER", "RESUME_BLOCKED_VOICE_CHECKPOINT", project=project_name)
                                        yield _workflow_progress_card(
                                            "⛔ RESUME: Voice Full checkpoint chưa hợp lệ. Dùng RUN FAST · FRESH để hoàn tất lại Voice rồi Prompt.",
                                            0, "voice"
                                        )
                                        return

                                    workflow_cmd_log("MASTER", "RESUME_GEMINI_PROMPT_NEEDED", project=project_name)
                                    token = _reset_prompt_gate(project_name)
                                    for item in _master_gemini_strict_sequence(project_name, current_workflow_profile() or "Default", token):
                                        try:
                                            yield item[-1] if isinstance(item, (tuple, list)) and len(item) >= 8 else item
                                        except Exception:
                                            continue
                                    prompt_ok, _ = _project_prompt_checkpoint(project_name)
                                    if not prompt_ok:
                                        yield _workflow_progress_card("⛔ RESUME: Gemini Prompt chưa VERIFY xong.", 100, "gemini")
                                        return
                                    workflow_cmd_log("MASTER", "RESUME_GEMINI_COMPLETE_TO_ZIMAGE", project=project_name)
                                    for pct, status, hud in run_zimage_txt2img_queue(project_name):
                                        yield hud


                def _master_reset_prompt_gate(project_name):
                    """RUN FAST luôn reset Prompt; RESUME mới được dùng checkpoint."""
                    token = _reset_prompt_gate(project_name)
                    try:
                        root = project_dir(project_name)
                        removed = 0
                        removed += _clear_gemini_chunk_checkpoints(project_name)
                        for rel in (
                            'Prompt/topic_style.json', 'Prompt/topic_style_raw.txt', 'Prompt/style_bible.json',
                            'Prompt/gemini_chunks.json', 'Prompt/prompt_verified.json',
                            'Prompt/topic_style_api_verified.txt', 'Prompt/topic_style_verified.json',
                            'Prompt/image_handoff_verified.json', 'Prompt/pipeline.json',
                            'Prompt/project_bible.json', 'project_bible.json', 'scene_states.json'
                        ):
                            try:
                                fp = root / rel
                                if fp.exists(): fp.unlink(); removed += 1
                            except Exception: pass
                        workflow_cmd_log('MASTER','RUN_FAST_FRESH_PROMPT_RESET',project=project_name,removed=removed)
                    except Exception as exc:
                        workflow_cmd_log('MASTER','RUN_FAST_PROMPT_RESET_WARNING',project=project_name,error=str(exc)[:300],level='WARN')
                    return token


                # V63: removed the earlier partial auto_pipeline_btn binding.
                # It registered a SECOND click handler for the same MASTER button,
                # causing two Gemini pipelines to start concurrently: one acquired
                # GEMINI_STAGE_LOCK while the other logged GEMINI_STAGE_ALREADY_RUNNING
                # and then incorrectly continued into the image gate.


                # local image is now included in QC Full; legacy button is intentionally unused.
                # QC project is synchronized by the explicit Project events below; no discarded-output callback.

            # YOUTUBE KIT
            # =================================================

            # =================================================
            # PROJECT ARCHIVE
            # =================================================
            # IMAGES — Z-IMAGE-TURBO Local only for the production MASTER path.
            with gr.Tab('🎨 Hình ảnh', id='images') as image_tab:
                with gr.Row():
                    image_refresh_all = gr.Button('🔄 Làm mới Hình ảnh', variant='secondary', elem_id='image-refresh-all')
                    image_refresh_status = gr.Markdown(
                        'Chỉ tải lại dữ liệu của project hiện tại. Nếu chưa có dữ liệu sẽ để trắng.',
                        elem_id='image-refresh-status'
                    )
                gem_scene_count = gr.Markdown(f'**{len(load_manifest(default_project))} scenes**' if load_manifest(default_project) else '**0 scenes** — trắng.')
                image_style_context = gr.Markdown(_image_context_ui(default_project), elem_id='image-style-context')
                gem_status = gr.Markdown('')
                gem_output = gr.Dataframe(headers=['Image Scene','Kịch bản','Prompt đã tạo'], value=prompt_scene_rows(load_manifest(default_project)), interactive=False, wrap=True, label='🖼 Scene + Prompt')
                with gr.Row():
                    image_only_btn = gr.Button('🖼 IMAGE ONLY · Chỉ tạo ảnh', variant='primary', visible=True, elem_id='image-only-btn')
                    start_image_btn = gr.Button('🌐 Tạo ảnh bằng Google Flow' if _selected_image_engine() == CREATIVE_ENGINE_FLOW else '⚡ Tạo ảnh bằng Z Image', variant='secondary', visible=True, elem_id='manual-flow-trigger')
                with gr.Row():
                    zimage_progress = gr.Slider(0,100,0,step=1,label='🖌 Tiến trình vẽ ảnh',interactive=False, elem_id='las-zimage-progress', elem_classes=['las-progress-compact'], scale=4)
                zimage_status = gr.Markdown('🟦 Chưa bắt đầu.')
                comfyui_live_hub = gr.Markdown('**ComfyUI:** chưa kết nối · **MAX GPU:** HighVRAM + PyTorch attention + preview OFF · API graph 1024×576 là nguồn chạy thật. Canvas có thể đang hiển thị workflow cũ.', elem_id='comfyui-live-hub')

                # ComfyUI is API-only for the production workflow.  No browser tab is opened.
                image_refresh_all.click(_wrap_gradio_callback(_refresh_image_data_master_ui), project, [gem_scene_count, gem_output, image_style_context, start_image_btn, image_refresh_status], show_progress='minimal')
                def _selected_engine_manual_ui(project_name):
                    engine=_selected_image_engine()
                    ok,detail=_creative_environment_check(engine)
                    if not ok:
                        yield 0,detail,_workflow_progress_card(detail,0,"image"); return
                    scenes=load_manifest(project_name)
                    if not scenes:
                        yield 0,"⛔ Chưa có Scene.",_workflow_progress_card("⛔ Chưa có Scene.",0,"image"); return
                    _creative_master_full(_batch_project_name(project_name),scenes,engine,start=True)
                    total=sum(1 for s in scenes if scene_media_type(s,"image")=="image")
                    active_total=total*len(CREATIVE_IMAGE_ONLY_STAGES)
                    while True:
                        with _batch_db() as db:
                            rows=db.execute("SELECT scene_id,stage,status FROM creative_jobs WHERE project_id=?",(project_name,)).fetchall()
                        done=sum(1 for r in rows if r["stage"] in CREATIVE_IMAGE_ONLY_STAGES and r["status"]=="DONE")
                        failed=sum(1 for r in rows if r["stage"] in CREATIVE_IMAGE_ONLY_STAGES and r["status"]=="FAILED")
                        captcha=sum(1 for r in rows if r["stage"] in CREATIVE_IMAGE_ONLY_STAGES and r["status"]=="CAPTCHA")
                        pct=int(min(99,(done/active_total)*100)) if active_total else 100
                        status=f"{engine}: DONE {done}/{active_total} · CAPTCHA {captcha} · FAIL {failed}"
                        yield pct,status,_workflow_progress_card(status,pct,"image")
                        if failed or (active_total and done>=active_total): return
                        time.sleep(1.0)

                image_only_btn.click(
                    _wrap_gradio_callback(clear_full_stop), outputs=[], show_progress='minimal'
                ).then(
                    _wrap_gradio_callback(_image_only_ui), [project], workflow_return_status, show_progress='full'
                )

                start_image_btn.click(
                    _wrap_gradio_callback(clear_full_stop), outputs=[], show_progress='minimal'
                ).then(
                    lambda *args: (0, '⚡ Z-Image-Turbo: nạp model + chuẩn bị batch…'),
                    outputs=[zimage_progress, zimage_status]
                ).then(
                    _wrap_gradio_callback(_selected_engine_manual_ui), [project], [zimage_progress, zimage_status, workflow_return_status], show_progress='full'
                )

                project.change(lambda p: f'**{len(load_manifest(p))} scenes**' if load_manifest(p) else '**0 scenes** — trắng.', project, gem_scene_count)
                project.change(lambda p: _image_context_ui(p), project, image_style_context)
                project.change(lambda p: prompt_scene_rows(load_manifest(p)), project, gem_output)

                def _commit_prompt_and_switch(project_name):
                    scenes = load_manifest(project_name)
                    missing = [s.get('scene_id','Scene') for s in scenes if scene_media_type(s,'image') != 'video' and not (s.get('prompt') or '').strip()]
                    if not scenes:
                        return '**0 scenes** — trắng.', [], '❌ Chưa có Scene.', gr.Button(visible=False)
                    if missing:
                        return f'**{len(scenes)} scenes**', prompt_scene_rows(scenes), '❌ Prompt chưa đủ: ' + ', '.join(missing), gr.Button(visible=False)
                    return (
                        f'**{len(scenes)} scenes — Prompt đầy đủ**',
                        prompt_scene_rows(scenes),
                        f'✅ Đã đồng bộ **{len(scenes)}/{len(scenes)} Prompt** từ Gemini. Sẵn sàng Z-IMAGE-TURBO Local.',
                        gr.Button(visible=True, value='▶ Thực hiện thao tác')
                    )

                go_zimage_btn_voice.click(
                    _wrap_gradio_callback(_commit_prompt_and_switch), project,
                    [gem_scene_count, gem_output, gem_status, start_image_btn],
                    show_progress='minimal'
                ).then(
                    _wrap_gradio_callback(_show_images_tab), outputs=[main_tabs], queue=False, show_progress='hidden'
                )

                master_control_btn.click(
                    _wrap_gradio_callback(master_toggle),
                    outputs=[master_control_btn, master_control_status, script_progress, script_status, full_voice_loader_box, full_status, prompt_progress, prompt_status, prompt_output, gem_scene_count, gem_status, get_prompt_btn, prompt_retry_btn_voice, go_zimage_btn_voice, zimage_progress, zimage_status, workflow_return_status, image_refresh_status],
                    queue=False, show_progress='hidden',
                )

                # Duplicate bindings removed: project refresh, Prompt handoff and STOP are registered once above.


            # =================================================
            # ĐỘC LẬP — VOICE FULL + Z-IMAGE THUMBNAIL
            # =================================================
            with gr.Tab('🧰 Độc Lập', id='independent'):
                gr.Markdown('## 🧰 ĐỘC LẬP — Voice Full + Tạo ảnh')
                gr.Markdown('Không cần Project, không cần Scene, không chạy Gemini. Hai công cụ này chạy riêng và không sửa Manifest/Prompt/Voice của Project.')
                # Nested tabs must contain ONLY Tab/TabItem children. Keeping the
                # explanatory Markdown outside the Tabs prevents Gradio from
                # rendering the first nested tab label twice on some builds.
                with gr.Tabs():
                    with gr.Tab('🎙 Voice Full', id='ind_voice'):
                        standalone_voice_script = gr.Textbox(lines=10, label='📝 Kịch bản Voice Full độc lập', placeholder='Dán toàn bộ kịch bản vào đây…')
                        with gr.Row():
                            standalone_language = gr.Dropdown(choices=[("🤖 Tự động nhận dạng", "auto")] + [(language_label(k), k) for k in LANGUAGE_CATALOG], value='auto', label='Ngôn ngữ')
                            standalone_engine = gr.Dropdown(choices=engine_choices(_full_actual_lang0), value='auto', label='Server / Engine')
                            standalone_voice = gr.Dropdown(choices=_initial_full_choices, value=_initial_full_value, label='Voice', filterable=True)
                        with gr.Row():
                            standalone_voice_speed = gr.Slider(0.75, 1.50, value=float(SETTINGS.get('voice_speed', 1.0)), step=0.05, label='🎚 Tốc độ đọc', info='0.75× chậm → 1.50× nhanh')
                        with gr.Row():
                            standalone_generate = gr.Button('🎙 TẠO VOICE FULL ĐỘC LẬP', variant='primary')
                            standalone_stop = gr.Button('⏹ Dừng', variant='stop')
                        standalone_loader = gr.HTML(voice_live_loader('Voice Full độc lập', 0, 'Sẵn sàng', False))
                        standalone_status = gr.Markdown('Chưa tạo. Nếu đã kết nối Project, file sẽ lưu tại `Project/02_MEDIA/Âm thanh/Voice Full/Voice_Full.wav`.')
                        with gr.Row():
                            standalone_audio = gr.Audio(label='🔊 Nghe Voice Full', type='filepath', interactive=False)
                            standalone_download = gr.File(label='📥 File Voice_Full.wav', visible=False)

                        def _independent_script_changed(text):
                            lang = detect_script_language(str(text or '')) if str(text or '').strip() else _full_actual_lang0
                            choices = engine_choices(lang)
                            eng = 'auto' if 'auto' in {v for _, v in choices} else (choices[0][1] if choices else 'auto')
                            voices = all_voice_choices(lang, eng)
                            return gr.Dropdown(choices=choices, value=eng), gr.Dropdown(choices=voices, value=(voices[0][1] if voices else None))

                        standalone_voice_script.change(_wrap_gradio_callback(_independent_script_changed), standalone_voice_script, [standalone_engine, standalone_voice], show_progress='hidden')
                        standalone_language.change(
                            _wrap_gradio_callback(lambda l,e: (
                                gr.Dropdown(choices=engine_choices(l if l != 'auto' else _full_actual_lang0), value='auto'),
                                gr.Dropdown(choices=all_voice_choices(l if l != 'auto' else _full_actual_lang0, 'auto'), value=(all_voice_choices(l if l != 'auto' else _full_actual_lang0, 'auto')[0][1] if all_voice_choices(l if l != 'auto' else _full_actual_lang0, 'auto') else None))
                            )),
                            [standalone_language, standalone_engine], [standalone_engine, standalone_voice], show_progress='hidden'
                        )

                    with gr.Tab('🖼 Tạo ảnh', id='ind_image'):
                        independent_thumb_prompt = gr.Textbox(lines=7, label='✍️ Prompt tạo ảnh', placeholder='Ví dụ: Một chiến binh cổ đại đứng giữa thành phố đổ nát, ánh sáng điện ảnh, chi tiết cao…')
                        with gr.Row():
                            independent_thumb_refs = gr.File(label='🖼 Ảnh tham chiếu — tối đa 3 (không bắt buộc)', file_count='multiple', type='filepath')
                        with gr.Row():
                            independent_thumb_width = gr.Dropdown(choices=[('1024 × 576 — 16:9','1024x576'),('1280 × 720 — 16:9','1280x720'),('1536 × 864 — 16:9','1536x864'),('576 × 1024 — 9:16','576x1024'),('1024 × 1024 — 1:1','1024x1024'),('768 × 1024 — 3:4','768x1024')], value='1024x576', label='📐 Kích thước')
                            independent_thumb_count = gr.Slider(1, 12, value=4, step=1, label='🔢 Số ảnh tạo ra')
                        with gr.Row():
                            independent_thumb_generate = gr.Button('🎨 TẠO ẢNH ĐỘC LẬP', variant='primary')
                            independent_thumb_stop = gr.Button('⏹ Dừng', variant='stop')
                        independent_thumb_progress = gr.Slider(0,100,0,step=1,label='Tiến trình',interactive=False)
                        independent_thumb_status = gr.Markdown('Sẵn sàng — mỗi lần chạy tạo Batch riêng trong `Thumb/`.')
                        independent_thumb_output = gr.Files(label='🖼 Ảnh đã tạo', interactive=False)

                    def _standalone_voice_generate_ui(script, language, engine, voice, speed):
                        # Keep standalone speed local to this run; do not alter Project voice settings.
                        old = SETTINGS.get('voice_speed', 1.0)
                        SETTINGS['voice_speed'] = float(speed or 1.0)
                        try:
                            yield from independent_full_voice_generator(language, engine, voice, script, stop_event=VOICE_STOP_EVENT)
                        finally:
                            SETTINGS['voice_speed'] = old

                    def _ind_thumb_ui(prompt, refs, size, count):
                        try:
                            w,h=[int(x) for x in str(size).split('x',1)]
                            results,out_dir=independent_thumbnail_generator(prompt, refs, w, h, int(count))
                            return 100, f'🟢 Đã tạo **{len(results)} ảnh** → `{out_dir}`', results
                        except Exception as exc:
                            return 0, f'🔴 Tạo ảnh lỗi: {exc}', []

                    def _clear_independent_voice_stop():
                        VOICE_STOP_EVENT.clear()
                        return None

                    def _stop_independent_voice():
                        VOICE_STOP_EVENT.set()
                        return '⏹ Đã yêu cầu dừng Voice Full độc lập.'

                    standalone_generate.click(
                        _wrap_gradio_callback(_clear_independent_voice_stop), outputs=[], queue=False, show_progress='hidden'
                    ).then(
                        _wrap_gradio_callback(_standalone_voice_generate_ui),
                        [standalone_voice_script, standalone_language, standalone_engine, standalone_voice, standalone_voice_speed],
                        [standalone_loader, standalone_status, standalone_audio, standalone_download], show_progress='minimal'
                    )
                    standalone_stop.click(_wrap_gradio_callback(_stop_independent_voice), outputs=standalone_status, queue=False, show_progress='hidden')
                    independent_thumb_generate.click(
                        _wrap_gradio_callback(clear_full_stop), outputs=[], queue=False, show_progress='hidden'
                    ).then(
                        _wrap_gradio_callback(_ind_thumb_ui),
                        [independent_thumb_prompt, independent_thumb_refs, independent_thumb_width, independent_thumb_count],
                        [independent_thumb_progress, independent_thumb_status, independent_thumb_output], show_progress='minimal'
                    )
                    independent_thumb_stop.click(_wrap_gradio_callback(lambda: (SCRIPT_STOP_EVENT.set(), '⏹ Đã yêu cầu dừng tạo ảnh độc lập.')[1]), outputs=[independent_thumb_status], queue=False, show_progress='hidden')
            with gr.Tab('✅ QC / Timeline / Subtitle / CapCut', id='qc'):
                qc_project.render()
                gr.Markdown(
                    '### 🎬 PRODUCTION FINISH GATE\n'
                    '**Asset → Timeline → Subtitle/SRT → CapCut Package → Final QC → Metadata Clean**\n\n'
                    'MASTER 1-CLICK tự chạy chuỗi này sau khi Z-IMAGE-TURBO Verify PASS; nút bên dưới vẫn có thể chạy lại riêng cho Project hiện tại.'
                )
                with gr.Group(elem_classes=['qc-compact']):
                    with gr.Row():
                        qc_full_btn=gr.Button('🔍 THỰC HIỆN TOÀN BỘ QC + TIMELINE + SRT + CAPCUT', variant='primary', elem_id='qc-full-btn', scale=5)
                        qc_clean_btn=gr.Button('🧼 Làm sạch logo + metadata', visible=False,)
                    qc_progress=gr.Slider(0,100,0,step=1,label='Tiến trình Production Finish',interactive=False, elem_classes=['las-progress-compact'])
                    qc_result=gr.Markdown('🟦 Sẵn sàng — chờ Asset / Voice / Z-IMAGE-TURBO.', visible=True, elem_id='qc-result')
                    qc_detail=gr.Markdown('⏳ Khi chạy: Asset QC → Timeline → SRT → CapCut → Final QC → CLEAN.', visible=True, elem_id='qc-detail')

                with gr.Accordion('🧹 Watermark / Metadata Cleaner → PRODUCTION CLEAN', open=True):
                    gr.Markdown(
                        '🧼 **CLEAN MAX VIP** · 1 nút duy nhất cho ảnh Z-IMAGE-TURBO local. '
                        'Tự động cập nhật Cleaner runtime riêng của tool, backup ảnh gốc, làm sạch logo/watermark thuộc media mày sở hữu/được phép chỉnh sửa + EXIF/XMP/software/comment/ICC metadata. '
                        'Ảnh gốc được backup trước khi thay thế.'
                    )
                    with gr.Row():
                        clean_full_btn = gr.Button(
                            '🧼 CLEAN MAX VIP — DỌN ẢNH TOÀN DIỆN',
                            variant='primary',
                            scale=6,
                            elem_id='clean-full-btn'
                        )
                    wm_status = gr.HTML(
                        _master_pipeline_hud(
                            0, 0,
                            '🧼 Sẵn sàng — CLEAN MAX VIP: chỉ làm sạch metadata local → Verify file.'
                        ),
                        elem_id='master-pipeline-status'
                    )
                    wm_files = gr.File(
                        label='📦 Ảnh trong folder đã chọn sau xử lý',
                        file_count='multiple',
                        interactive=False
                    )
                    clean_full_btn.click(
                        _wrap_gradio_callback(_production_image_cleanup_pipeline),
                        [qc_project],
                        [wm_status, wm_files, workflow_return_status],
                        show_progress='full'
                    )

                with gr.Accordion('📋 Chi tiết QC / Timeline / SRT / CapCut', open=True):
                    asset_out=gr.Textbox(lines=3,interactive=False,label='Asset QC')
                    timeline_out=gr.Code(language='json',interactive=False,label='Timeline JSON')
                    srt_file=gr.File(label='SRT')
                    capcut_out=gr.Textbox(interactive=False,label='CapCut Package')
                    final_out=gr.Textbox(lines=4,interactive=False,label='Final QC')
                def _master_resume_preflight(project_name):
                    # RUN FAST is SMART RESUME. It inspects durable checkpoints but never trusts
                    # a checkpoint whose script fingerprint differs from the current script.
                    prompt_ok, prompt_obj = _project_prompt_checkpoint(project_name)
                    if prompt_ok:
                        msg = (
                            f"♻ RUN FAST SMART — Prompt VERIFY OK ({int(prompt_obj.get('scene_count', 0) or 0)} Scene). "
                            "Gemini sẽ bị bỏ qua; tiếp tục Z-Image-Turbo."
                        )
                        workflow_cmd_log("MASTER", "RUN_FAST_SMART_PREFLIGHT_PROMPT_OK", project=project_name)
                        return msg
                    scene_ok, scenes = _project_scene_checkpoint(project_name, infer=True)
                    if scene_ok:
                        msg = (
                            f"♻ RUN FAST SMART — Scene VERIFY OK ({len(scenes)} Scene), Prompt chưa đủ. "
                            "Chỉ chạy các stage còn thiếu; không xóa checkpoint hợp lệ."
                        )
                        workflow_cmd_log("MASTER", "RUN_FAST_SMART_PREFLIGHT_SCENE_OK", project=project_name, scene_count=len(scenes))
                        return msg
                    msg = (
                        "🆕 RUN FAST SMART — chưa có checkpoint hợp lệ hoặc script đã đổi. "
                        "Chạy Scene → Voice → Gemini B1/B2 → Z-Image-Turbo."
                    )
                    workflow_cmd_log("MASTER", "RUN_FAST_SMART_PREFLIGHT_FRESH_REQUIRED", project=project_name)
                    return msg
                resume_project_btn.click(
                    _wrap_gradio_callback(clear_full_stop), outputs=[], show_progress='minimal'
                ).then(
                    _wrap_gradio_callback(_resume_project_ui), [project], workflow_return_status, show_progress='full'
                ).then(
                    _wrap_gradio_callback(_show_images_tab), outputs=[main_tabs], show_progress='hidden'
                )


                master_pipeline_event = auto_pipeline_btn.click(
                     _wrap_gradio_callback(_ui_require_profile_and_start), script_profile_select, outputs=[], show_progress='minimal'
                 ).then(
                     _wrap_gradio_callback(clear_full_stop), outputs=[], show_progress='minimal'
                 ).then(
                     _wrap_gradio_callback(_sync_profile_for_run), [script_profile_select, project, script_box, scene_target],
                     [project, script_box, scene_target, full_language, full_engine, full_voice, gemini_profile_voice, qc_project, script_profile_preview, script_profile_status], show_progress='minimal'
                 ).then(
                     # RUN FAST Scene Director: Gemini is authoritative.
                     # Do NOT run the legacy local character-count splitter here.
                     _wrap_gradio_callback(_master_gemini_split_mixed_media_ui),
                     [project, script_box, scene_target, media_mode, gemini_profile_voice],
                     [script_progress, scene_table, script_status, workflow_return_status], show_progress='minimal'
                 ).then(
                     _wrap_gradio_callback(_show_voice_full_tab_if_scene_ok), [project], [main_tabs], queue=False, show_progress='hidden'
                 ).then(
                     _master_full_voice_progress_ui, [project, full_language, full_engine, full_voice, script_box],
                     [full_voice_loader_box, full_status, full_audio, full_download, full_scene_table, full_voice_progress_table, full_scene_players, workflow_return_status], show_progress='minimal'
                 ).then(
                     _wrap_gradio_callback(_master_combine_full_voice_ui), [project, full_output_dir],
                     [full_combine_loader, full_combine_status, full_audio, full_download, workflow_return_status], show_progress='minimal'
                 ).then(
                     _wrap_gradio_callback(_master_release_voice_vram_ui), [project, full_language, full_engine],
                     [full_combine_loader, full_combine_status, workflow_return_status], show_progress='minimal'
                 ).then(
                     _wrap_gradio_callback(_master_reset_prompt_gate), [project], [prompt_attempt_token], show_progress='minimal'
                 ).then(
                     _wrap_gradio_callback(_master_gemini_strict_sequence), [project, gemini_profile_voice, prompt_attempt_token],
                     [prompt_progress, prompt_status, prompt_output, get_prompt_btn, prompt_retry_btn_voice, go_zimage_btn_voice, prompt_attempt_token, workflow_return_status], show_progress='minimal'
                 ).then(
                     _wrap_gradio_callback(_master_verify_gemini_for_local_zimage_ui), [project],
                     [workflow_return_status], show_progress='minimal'
                 ).then(
                     _wrap_gradio_callback(_refresh_image_data_master_ui), [project],
                     [gem_scene_count, gem_output, image_style_context, start_image_btn, image_refresh_status],
                     show_progress='hidden'
                 ).then(
                     _wrap_gradio_callback(_show_images_tab_if_gemini_ok), [project], [main_tabs], show_progress='hidden'
                 ).then(
                     _wrap_gradio_callback(_master_zimage_after_prompt_ui), [project, workflow_return_status],
                     [workflow_return_status], show_progress='minimal'
                 ).then(
                     _wrap_gradio_callback(_show_qc_tab_if_zimage_ok), [workflow_return_status], [main_tabs],
                     queue=False, show_progress='hidden'
                 )

                # IMPORTANT: QC must start ONLY after the Z-Image stage has emitted
                # MASTER_ZIMAGE_OK. Previously QC was attached to the ROOT
                # master_pipeline_event before the Z-Image stage, so it raced ahead
                # with a stale/empty gate and later reported QC BLOCKED even though
                # Z-Image had completed. The variable now points to the final event
                # in the complete pre-QC chain above.
                master_pipeline_event.then(
                    _wrap_gradio_callback(_master_qc_stage_if_zimage_ok), [project, workflow_return_status],
                    [qc_progress,qc_result,qc_detail,asset_out,timeline_out,srt_file,capcut_out,final_out],
                    show_progress='minimal'
                ).then(
                    _wrap_gradio_callback(_master_clean_stage_if_qc_ok), [project, final_out, workflow_return_status],
                    [wm_status, wm_files, workflow_return_status], show_progress='minimal'
                ).then(
                    _wrap_gradio_callback(_master_finish_if_qc_ok), [workflow_return_status, final_out, wm_status],
                    [master_control_btn, master_control_status], show_progress='minimal'
                )

                qc_full_btn.click(
                    _wrap_gradio_callback(qc_full_ai_run),qc_project,
                    [qc_progress,qc_result,qc_detail,asset_out,timeline_out,srt_file,capcut_out,final_out],
                    show_progress='minimal'
                ).then(
                    lambda p: _workflow_progress_card(f"✅ QC / Timeline / Subtitle / CapCut hoàn tất cho Project **{p}**. Gemini API đã hoàn tất; Z-Image-Turbo chạy qua ComfyUI API.", 100, "zimage"),
                    qc_project, workflow_return_status, show_progress='hidden'
                ).then(
                    _wrap_gradio_callback(master_finish), outputs=[master_control_btn, master_control_status], show_progress='minimal'
                ).then(
                    _wrap_gradio_callback(_auto_clean_after_qc_ui), [qc_project, final_out], [wm_status, wm_files, workflow_return_status], show_progress='minimal'
                )
                # local image is now included in QC Full; legacy button is intentionally unused.
                # QC project is synchronized by the explicit Project events below; no discarded-output callback.

            # =================================================
                with gr.Accordion('🚀 Export / CapCut', open=False):
                    export_resolution=gr.Dropdown(choices=['1080p','1440p','4K'],value='1080p',label='Độ phân giải')
                    export_aspect=gr.Dropdown(choices=['16:9','9:16','1:1','4:5','4:3'],value=str(SETTINGS.get('video_aspect_ratio','16:9')),label='Tỉ lệ khung hình')
                    export_fps=gr.Dropdown(choices=[24,30,60],value=30,label='FPS')
                    with gr.Row():
                        capcut_open_btn=gr.Button('🎬 Mở CapCut PC + Import',variant='secondary')
                        timeline_json_btn=gr.Button('📄 Mở timeline.json')
                        preview_btn=gr.Button('▶ Xem trước',variant='primary')
                        save_mp4_btn=gr.Button('💾 Lưu MP4 vào Project')
                    export_progress=gr.Slider(0,100,0,step=1,label='Tiến trình',interactive=False, elem_classes=['las-progress-compact'])
                    export_video=gr.Video(label='Bản xem trước / MP4 đã lưu')
                    export_status=gr.Markdown('Xem trước không lưu vào Project. Chỉ nút **Lưu MP4 vào Project** mới tạo file trong Project.')

                    export_aspect.change(_wrap_gradio_callback(_set_video_aspect), export_aspect, outputs=[])

                    def preview_ui(p,r,a,f):
                        preview_dir=Path(tempfile.gettempdir()) / 'LocalAIStudioPreview'
                        preview_dir.mkdir(parents=True,exist_ok=True)
                        out=preview_dir / f"{safe_name(p)}_preview_{int(time.time())}.mp4"
                        yield 10,None,'Đang tạo bản xem trước — 10%'
                        result=render_video(p,r,int(f),output_path=out,aspect=a)
                        yield 100,result,'▶ Xem trước hoàn tất — **không lưu vào Project**.'

                    def save_mp4_ui(p,r,a,f):
                        yield 10,None,'Đang render MP4 vào Project — 10%'
                        result=render_video(p,r,int(f),output_path=None,aspect=a)
                        yield 100,result,'💾 MP4 đã được lưu vào `Project/output`. Không tự dọn dữ liệu Project.'

                    preview_btn.click(_wrap_gradio_callback(preview_ui),[project,export_resolution,export_aspect,export_fps],[export_progress,export_video,export_status])
                    save_mp4_btn.click(_wrap_gradio_callback(save_mp4_ui),[project,export_resolution,export_aspect,export_fps],[export_progress,export_video,export_status])
                    capcut_open_btn.click(_wrap_gradio_callback(open_capcut_pc_ui),project,export_status,show_progress='minimal')
                    timeline_json_btn.click(_wrap_gradio_callback(open_timeline_json),project,export_status,show_progress='minimal')

            # YOUTUBE KIT
            # =================================================

            # =================================================
            # BATCH CENTER
            # =================================================
            with gr.Tab('🧩 Batch Center', id='batch_center') as batch_center_tab:
                gr.HTML("""<style>
                #batch-center-root{margin:0 0 12px 0}
                .batch-center-card{border:1px solid rgba(148,163,184,.35);border-radius:16px;padding:16px;background:linear-gradient(145deg,rgba(15,23,42,.96),rgba(30,41,59,.94));color:#e5e7eb;box-shadow:0 10px 30px rgba(0,0,0,.16)}
                .batch-title{font-size:20px;font-weight:950;letter-spacing:.4px}.batch-title span{font-size:12px;opacity:.65}
                .batch-project{margin-top:8px;font-size:14px}.batch-queue-title{margin:18px 0 8px;font-size:13px;display:flex;justify-content:space-between}
                .batch-stage-grid{display:grid;grid-template-columns:repeat(5,minmax(120px,1fr));gap:10px}
                .batch-stage{background:rgba(255,255,255,.045);border:1px solid rgba(255,255,255,.08);border-radius:11px;padding:10px}
                .batch-stage-head{display:flex;justify-content:space-between;font-size:11px;font-weight:900}.batch-stage-head b{opacity:.8}
                .batch-bar{height:8px;background:rgba(148,163,184,.22);border-radius:99px;overflow:hidden;margin-top:8px}.batch-bar i{display:block;height:100%;background:linear-gradient(90deg,#38bdf8,#818cf8);border-radius:99px}
                .batch-stage small{display:block;text-align:right;margin-top:5px;opacity:.65}.batch-runtime{display:grid;grid-template-columns:1fr 2fr 1fr;gap:10px;margin-top:12px}
                .batch-runtime div,.batch-current{padding:9px 11px;border-radius:10px;background:rgba(255,255,255,.035);border:1px solid rgba(255,255,255,.07)}
                .batch-runtime span{display:block;font-size:10px;opacity:.6}.batch-runtime b{font-size:12px}.batch-current{margin-top:10px;font-size:12px}
                .batch-table-wrap{max-height:430px;overflow:auto;border:1px solid rgba(148,163,184,.25);border-radius:12px}
                .batch-table{width:100%;border-collapse:collapse;font-size:12px}.batch-table th,.batch-table td{padding:7px 8px;border-bottom:1px solid rgba(148,163,184,.14);text-align:center}.batch-table th:first-child,.batch-table td:first-child{text-align:left;position:sticky;left:0;background:rgba(15,23,42,.98)}
                .batch-empty{padding:18px;border:1px dashed rgba(148,163,184,.4);border-radius:12px;text-align:center}
                @media(max-width:900px){.batch-stage-grid{grid-template-columns:1fr 1fr}.batch-runtime{grid-template-columns:1fr}.batch-table{font-size:11px}}
                </style>""")
                batch_hud = gr.HTML(_batch_html(default_project), elem_id='batch-center-root')
                with gr.Row():
                    batch_workers = gr.Dropdown(choices=list(range(1,11)), value=5, label='👷 Workers', scale=1)
                    batch_start = gr.Button('▶ START', variant='primary', scale=1)
                    batch_pause = gr.Button('⏸ PAUSE', variant='secondary', scale=1)
                    batch_resume = gr.Button('▶ RESUME', variant='secondary', scale=1)
                    batch_stop = gr.Button('🛑 STOP', variant='stop', scale=1)
                    batch_refresh = gr.Button('🔄 Làm mới', variant='secondary', scale=1)
                batch_open_captcha = gr.Button('🔓 MỞ SESSION', variant='secondary', scale=1)
                batch_resolve_captcha = gr.Button('✅ ĐÃ XỬ LÝ CAPTCHA', variant='secondary', scale=1)
                batch_auto = gr.Checkbox(value=True, label='🔄 Tự cập nhật', scale=1)
                batch_timer = gr.Timer(1.5, active=True)
                batch_status = gr.Markdown('🟦 Batch Center sẵn sàng.')
                gr.Markdown(
                    'ℹ️ **Job Engine:** GOOGLE FLOW là Primary, Z-IMAGE-TURBO là Fallback qua pipeline hiện tại. '
                    'CAPTCHA dừng riêng Account/Job để người dùng xử lý; không bypass và không tính retry. '
                    'DEPTH/MOTION/POSTPROCESS/CLEAN/QC dùng cùng Scheduler, không tạo queue riêng.'
                )
                batch_jobs = gr.HTML(_batch_table_html(default_project))
                batch_start.click(_wrap_gradio_callback(_batch_center_start), [project,batch_workers], [batch_hud,batch_status], show_progress='minimal').then(
                    _wrap_gradio_callback(_batch_table_html), project, batch_jobs, show_progress='hidden')
                batch_pause.click(_wrap_gradio_callback(_batch_center_pause), project, [batch_hud,batch_status], show_progress='minimal').then(
                    _wrap_gradio_callback(_batch_table_html), project, batch_jobs, show_progress='hidden')
                batch_resume.click(_wrap_gradio_callback(_batch_center_resume), [project,batch_workers], [batch_hud,batch_status], show_progress='minimal').then(
                    _wrap_gradio_callback(_batch_table_html), project, batch_jobs, show_progress='hidden')
                batch_stop.click(_wrap_gradio_callback(_batch_center_stop), project, [batch_hud,batch_status], show_progress='minimal').then(
                    _wrap_gradio_callback(_batch_table_html), project, batch_jobs, show_progress='hidden')
                batch_refresh.click(_wrap_gradio_callback(_batch_center_refresh), project, [batch_hud,batch_jobs], show_progress='hidden')
                batch_open_captcha.click(_wrap_gradio_callback(_creative_open_captcha_session), project, [batch_hud,batch_status], show_progress='minimal').then(
                    _wrap_gradio_callback(_batch_table_html), project, batch_jobs, show_progress='hidden')
                batch_resolve_captcha.click(_wrap_gradio_callback(_creative_resolve_captcha_ui), project, [batch_hud,batch_status], show_progress='minimal').then(
                    _wrap_gradio_callback(_batch_table_html), project, batch_jobs, show_progress='hidden')
                batch_center_tab.select(_wrap_gradio_callback(_batch_center_refresh), project, [batch_hud,batch_jobs], queue=False, show_progress='hidden')
                batch_auto.change(
                    lambda enabled: gr.Timer(1.5, active=bool(enabled)),
                    batch_auto, batch_timer, queue=False
                )
                batch_timer.tick(
                    _wrap_gradio_callback(_batch_center_refresh),
                    project, [batch_hud,batch_jobs], queue=False, show_progress='hidden'
                )

            # =================================================
            # PROJECT ARCHIVE
            # =================================================

            with gr.Tab('📋 Project Archive', id='project_archive') as project_archive_tab:
                gr.Markdown('### 📋 Project Archive & Schedule')
                gr.Markdown('🟢 Đã hoàn thành · 🟡 Đang dựng · 🔴 Đang lên ý tưởng. Archive JSON được lưu riêng trong Tool/ProjectArchive; **Thời Gian Xử Lý** là thời gian thực tế của 1-Click.')
                project_archive_table = gr.HTML(project_archive_html(), elem_id='project-archive-table')
                with gr.Row():
                    archive_project = gr.Dropdown(
                        choices=project_archive_choices(),
                        value=(safe_name(SETTINGS.get('selected_project','')) if safe_name(SETTINGS.get('selected_project','')) in project_archive_choices() else None),
                        allow_custom_value=False,
                        label='📁 Project cần quản lý',
                        scale=3,
                    )
                    archive_status = gr.Dropdown(choices=[('🔴 Đang lên ý tưởng','idea'),('🟡 Đang dựng','building'),('🟢 Đã hoàn thành','done')], value='idea', label='Trạng thái', scale=2)
                    archive_done = gr.Checkbox(label='☑ Đã hoàn thành', value=False, scale=1)
                with gr.Row():
                    archive_note = gr.Textbox(label='📝 Note cá nhân', placeholder='Ý tưởng tiếp theo, việc cần làm, hook…', scale=2)
                    archive_publish_date = gr.Textbox(label='📅 Ngày xuất bản', placeholder='2026-10-05')
                    archive_next_date = gr.Textbox(label='🗓 Lịch Project tiếp theo', placeholder='2026-10-07')
                with gr.Row():
                    archive_save_btn = gr.Button('💾 Lưu trạng thái / lịch', variant='primary')
                    archive_refresh_btn = gr.Button('🔄 Quét lại thư mục Project', variant='secondary')
                archive_status_msg = gr.Markdown('')
            # Project Archive events — the table is the overview; the editable
            # checkbox/dropdowns below it operate on the explicitly selected row.
            def _archive_refresh_ui():
                choices = project_archive_choices()
                current = safe_name(SETTINGS.get('selected_project',''))
                value = current if current in choices else (choices[0] if choices else None)
                return gr.Dropdown(choices=choices, value=value), project_archive_html()

            archive_refresh_btn.click(
                _wrap_gradio_callback(_archive_refresh_ui),
                outputs=[archive_project, project_archive_table],
                show_progress='hidden'
            ).then(
                _wrap_gradio_callback(project_archive_load), archive_project,
                [archive_status, archive_done, archive_note, archive_publish_date, archive_next_date, project_archive_table],
                show_progress='hidden'
            )
            archive_project.change(
                _wrap_gradio_callback(project_archive_load), archive_project,
                [archive_status, archive_done, archive_note, archive_publish_date, archive_next_date, project_archive_table],
                show_progress='hidden'
            )
            archive_save_btn.click(
                _wrap_gradio_callback(project_archive_save), [archive_project, archive_status, archive_done, archive_note, archive_publish_date, archive_next_date],
                [archive_status_msg, project_archive_table], show_progress='minimal'
            ).then(
                lambda: gr.Dropdown(choices=project_archive_choices(), value=safe_name(SETTINGS.get('selected_project','')) if safe_name(SETTINGS.get('selected_project','')) in project_archive_choices() else None),
                outputs=archive_project, show_progress='hidden'
            )
            project_archive_tab.select(
                lambda: project_archive_load(safe_name(SETTINGS.get('selected_project',''))),
                outputs=[archive_status, archive_done, archive_note, archive_publish_date, archive_next_date, project_archive_table],
                queue=False, show_progress='hidden'
            )

            # MASTER completion updates the Archive table immediately, even when
            # the user is not viewing the Archive tab.
            master_control_status.change(
                lambda: project_archive_html(),
                outputs=project_archive_table,
                queue=False, show_progress='hidden'
            )

            # -------------------------------------------------
            # HARD TAB SYNC: every time Script / Voice Full / Images is opened,
            # re-read the active profile and current Project from disk. This fixes
            # stale tab state without requiring a manual re-selection.
            _tab_profile_outputs = [script_profile_status, script_profile_preview, script_box, scene_target, project, full_language, full_engine, full_voice, gemini_profile_voice, qc_project, script_profile_select]
            _tab_project_outputs = [script_box, scene_target, script_progress, scene_table, script_status, full_audio, full_download, full_status, full_scene_table, full_voice_progress_table, full_scene_players, gem_scene_count, gem_output, gem_status, zimage_status, start_image_btn, qc_project, full_combine_loader, full_combine_status, prompt_progress, prompt_status, prompt_output, get_prompt_btn, prompt_retry_btn_voice, go_zimage_btn_voice, workflow_return_status, zimage_progress, image_refresh_status]

            def _sync_active_profile_on_tab():
                name = current_workflow_profile()
                result = _script_profile_apply(name) if name else _script_profile_apply('')
                # _tab_profile_outputs has exactly 11 components.
                return tuple(result[:11])

            script_tab.select(_wrap_gradio_callback(_sync_active_profile_on_tab), inputs=[], outputs=_tab_profile_outputs, queue=False, show_progress='hidden').then(
                _wrap_gradio_callback(_active_voice_config_status), [full_language, full_engine, full_voice], full_config_status, queue=False, show_progress='hidden'
            ).then(
                _wrap_gradio_callback(load_project_ui), project, _tab_project_outputs, queue=False, show_progress='hidden'
            )
            voice_full_tab.select(_wrap_gradio_callback(_sync_active_profile_on_tab), inputs=[], outputs=_tab_profile_outputs, queue=False, show_progress='hidden').then(
                _wrap_gradio_callback(_active_voice_config_status), [full_language, full_engine, full_voice], full_config_status, queue=False, show_progress='hidden'
            ).then(
                _wrap_gradio_callback(load_project_ui), project, _tab_project_outputs, queue=False, show_progress='hidden'
            )
            def _image_tab_refresh(p):
                p = p or default_project
                scenes = load_manifest(p)
                count_msg = f'**{len(scenes)} scenes**' if scenes else '**0 scenes** — trắng.'
                rows = prompt_scene_rows(scenes)
                image_count = sum(1 for sc in scenes if scene_media_type(sc, 'image') != 'video')
                video_count = max(0, len(scenes) - image_count)
                status = f'🖼 Z-IMAGE-TURBO Local sẵn sàng — {image_count} Scene IMAGE chạy Z-IMAGE-TURBO' + (f', {video_count} Scene VIDEO bỏ qua Z-IMAGE-TURBO.' if video_count else '.')
                return count_msg, rows, status

            image_tab.select(
                _wrap_gradio_callback(_image_tab_refresh), project,
                [gem_scene_count, gem_output, zimage_status],
                queue=False, show_progress='hidden'
            )

            # Manual refresh remains available, but Voice/Profile catalog is loaded automatically at startup.
            def _refresh_voice_ui(lang='vi', eng='auto'):
                # Refresh Voice chỉ nạp lại catalog/profile khi người dùng bấm nút.
                # Không xóa voice mẫu đã có và không đụng vào project/script hiện tại.
                lang = lang or 'vi'
                eng = eng or 'auto'
                voices = all_voice_choices(lang, eng)
                return (
                    gr.Dropdown(choices=engine_choices(lang), value=eng),
                    gr.Dropdown(choices=voices, value=(voices[0][1] if voices else None)),
                    gr.Dropdown(choices=voices, value=(voices[0][1] if voices else None)),
                    gr.Dropdown(choices=voices, value=(voices[0][1] if voices else None)),
                    '🔄 Đã nạp lại danh sách Voice. Dữ liệu project/script không bị xóa.'
                )

            voice_refresh_all.click(
                _wrap_gradio_callback(_refresh_voice_ui),
                inputs=[available_lang, available_engine],
                outputs=[available_engine, available_voice, clone_profile_select, full_voice, voice_refresh_status],
                show_progress='minimal'
            )


        def _profile_engine_ui(language, engine, voice):
            lang = _effective_voice_language(language, SETTINGS.get('workflow_script','') or '')
            eng = engine or "auto"
            choices = engine_choices(lang)
            valid = {v for _, v in choices}
            if eng not in valid:
                eng = "auto"
            voices = all_voice_choices(lang, eng)
            voice_value = str(voice or '').strip()
            if voice_value and voice_value not in {v for _, v in voices}:
                voices.insert(0, (Path(voice_value).name if Path(voice_value).expanduser().is_file() else voice_value, voice_value))
            if not voice_value:
                voice_value = voices[0][1] if voices else None
            return gr.Dropdown(choices=choices, value=eng), gr.Dropdown(choices=voices, value=voice_value)

        def _script_profile_apply(name):
            name = str(name or '').strip() or current_workflow_profile()
            data = apply_workflow_profile(name)
            if not data.get('ok'):
                return (data.get('message',''), _voice_selection_status(), '', 320,
                        SETTINGS.get('selected_project','Project_001'),
                        gr.update(choices=[('🤖 Tự động nhận dạng từ Kịch bản','auto')] + [(language_label(k),k) for k in LANGUAGE_CATALOG], value='auto'),
                        gr.update(choices=engine_choices(_effective_voice_language('auto', SETTINGS.get('workflow_script','') or '')), value='auto'),
                        gr.update(choices=all_voice_choices(_effective_voice_language('auto', SETTINGS.get('workflow_script','') or ''),'auto'), value=(all_voice_choices(_effective_voice_language('auto', SETTINGS.get('workflow_script','') or ''),'auto')[0][1] if all_voice_choices(_effective_voice_language('auto', SETTINGS.get('workflow_script','') or ''),'auto') else None)),
                        name or current_workflow_profile() or 'Default',
                        SETTINGS.get('selected_project','') or None, gr.update(choices=workflow_profile_choices(), value=None), normalize_gemini_visual_directive_mode(SETTINGS.get('gemini_visual_directive_mode','auto')))
            lang = data.get('language','auto') or 'auto'
            if lang not in LANGUAGE_CATALOG and lang != 'auto': lang = 'auto'
            script = data.get('script','') or ''
            actual, eng, resolved, voice = _canonical_voice_for_engine(
                lang, data.get('engine','auto') or 'auto', data.get('voice','') or '', script
            )
            ecs = engine_choices(actual)
            eng = eng if eng in {v for _,v in ecs} else 'auto'
            vcs = all_voice_choices(actual, eng)
            if voice and Path(str(voice)).expanduser().is_file() and voice not in {v for _, v in vcs}:
                vcs.insert(0, (Path(str(voice)).name, voice))
            if voice and voice not in {v for _, v in vcs}:
                vcs.insert(0, (Path(str(voice)).name if Path(str(voice)).expanduser().is_file() else str(voice), str(voice)))
            if not voice:
                voice = vcs[0][1] if vcs else None
            project_name = (safe_name(SETTINGS.get('selected_project','')) if PROJECT_CONNECTED and SETTINGS.get('selected_project_path') else (data.get('project','') or SETTINGS.get('selected_project','') or ''))
            # The selected workflow profile owns the Gemini directive.
            gemini_directive_mode = workflow_profile_directive_mode(name)
            SETTINGS['gemini_visual_directive_mode'] = gemini_directive_mode
            SETTINGS['workflow_profile'] = name
            directive_label = dict(GEMINI_VISUAL_DIRECTIVE_CHOICES).get(gemini_directive_mode, gemini_directive_mode)
            workflow_cmd_log('CONFIG', 'WORKFLOW_PROFILE_SELECTED', profile=name, gemini_directive_mode=gemini_directive_mode, directive=directive_label)
            save_settings()
            return (data['message'], _voice_selection_status(lang, eng, voice), script, data.get('target',320),
                    _project_dropdown_update(project_name),
                    gr.update(choices=[('🤖 Tự động nhận dạng từ Kịch bản','auto')] + [(language_label(k),k) for k in LANGUAGE_CATALOG], value=lang),
                    gr.update(choices=ecs, value=eng), gr.update(choices=vcs, value=voice),
                    name, project_name, gr.update(choices=workflow_profile_choices(), value=name))

        # Environment profile selection: one authoritative event chain.
        # IMPORTANT: this Dropdown can become choices=[] after deleting the last profile.
        # Use .input (user-only) instead of .change (also fires on programmatic refresh),
        # otherwise Gradio may preprocess stale value "" against choices=[] and crash.
        cfg_select.input(
            _wrap_gradio_callback(_sync_saved_profile_to_workflow), cfg_select,
            [script_profile_status,script_profile_preview,script_box,scene_target,project,full_language,full_engine,full_voice,gemini_profile_voice,qc_project,script_profile_select],
            show_progress='minimal'
        ).then(
            _wrap_gradio_callback(_active_voice_config_status), [full_language, full_engine, full_voice], full_config_status,
            show_progress='hidden'
        ).then(
            _wrap_gradio_callback(_cfg_view), cfg_select,
            [cfg_preview,cfg_language,cfg_engine,cfg_voice,cfg_project,cfg_target,cfg_competitor_urls,cfg_status,cfg_gemini_directive],
            show_progress='minimal'
        )

        # Environment save: persist once, then synchronize the same saved profile.
        cfg_save.click(
            _wrap_gradio_callback(_save_cfg),
            [cfg_name,cfg_language,cfg_engine,cfg_voice,cfg_project,cfg_target,cfg_competitor_urls,cfg_gemini_directive],
            [cfg_select,cfg_preview,cfg_status],
            show_progress='minimal'
        ).then(
            _wrap_gradio_callback(_sync_saved_profile_to_workflow), cfg_name,
            [script_profile_status,script_profile_preview,script_box,scene_target,project,full_language,full_engine,full_voice,gemini_profile_voice,qc_project,script_profile_select],
            show_progress='minimal'
        ).then(
            _wrap_gradio_callback(_active_voice_config_status), [full_language, full_engine, full_voice], full_config_status,
            show_progress='hidden'
        )

        # Environment delete: delete once, then refresh the dependent QC/project UI.
        cfg_delete.click(
            _wrap_gradio_callback(_delete_cfg), None, [cfg_select,cfg_preview,cfg_status], show_progress='minimal'
        ).then(
            lambda: (gr.Dropdown(choices=list_projects(), value=SETTINGS.get('selected_project') or None), 0, '🧹 QC đã làm mới sau khi xóa cấu hình.', '', '', ''),
            outputs=[qc_project, qc_progress, qc_result, qc_detail, asset_out, final_out],
            show_progress='minimal'
        )

        # Selecting a profile in Script is also a real synchronization action.
        script_profile_select.change(
            _wrap_gradio_callback(_script_profile_apply), script_profile_select,
            [script_profile_status, script_profile_preview, script_box, scene_target, project, full_language, full_engine, full_voice, gemini_profile_voice, qc_project, cfg_select],
            show_progress='minimal'
        ).then(
            _wrap_gradio_callback(_active_voice_config_status), [full_language, full_engine, full_voice], full_config_status,
            show_progress='hidden'
        ).then(
            _wrap_gradio_callback(_cfg_view), script_profile_select,
            [cfg_preview,cfg_language,cfg_engine,cfg_voice,cfg_project,cfg_target,cfg_competitor_urls,cfg_status,cfg_gemini_directive],
            show_progress='minimal'
        )
        def _use_profile_and_show_script(name):
            global PROJECT_CONNECTED
            # The selected saved profile is authoritative. A stale Gradio value
            # must never override workflow_config.json.
            result = _script_profile_apply(name)
            project_name = safe_name(result[4] if len(result) > 4 else "")
            root = _project_root_for_name(project_name) if project_name else Path()
            if project_name and root.exists() and root.is_dir():
                root = root.resolve()
                SETTINGS["selected_project"] = project_name
                SETTINGS["selected_project_path"] = str(root)
                SETTINGS["project_output_dir"] = str(root)
                SETTINGS["voice_output_dir"] = str(root)
                PROJECT_CONNECTED.set()
                save_settings()
                project_dir(project_name)
                msg = str(result[0] or "") + f"\n\n🔗 Đã kết nối đúng Project `{project_name}` — không tạo folder mới."
                result = (msg, *result[1:])
            return (*result, gr.update(selected="script"))

        cfg_use.click(
            _wrap_gradio_callback(_use_profile_and_show_script), cfg_select,
            [script_profile_status, script_profile_preview, script_box, scene_target, project, full_language, full_engine, full_voice, gemini_profile_voice, qc_project, script_profile_select, main_tabs],
            show_progress='minimal'
        ).then(
            _wrap_gradio_callback(_active_voice_config_status), [full_language, full_engine, full_voice], full_config_status,
            show_progress='hidden'
        )

        # Mở folder có sẵn: người dùng tự chọn folder, tool dùng đúng folder đó làm Project root.
        open_project_btn.click(
            _wrap_gradio_callback(open_existing_project), None,
            [project, project_output_dir, full_output_dir, project_root_status, qc_project],
            show_progress='minimal'
        ).then(
            lambda name, path, _full, status, qc: (path, status, name),
            [project, project_output_dir, full_output_dir, project_root_status, qc_project],
            [project_active_path, project_root_status, cfg_project],
            show_progress='minimal'
        ).then(
            _wrap_gradio_callback(load_project_ui), project,
            [script_box, scene_target, script_progress, scene_table, script_status, full_audio, full_download, full_status,
             full_scene_table, full_voice_progress_table, full_scene_players, gem_scene_count, gem_output, gem_status, zimage_status, start_image_btn,
             qc_project,
             full_combine_loader, full_combine_status, prompt_progress, prompt_status, prompt_output,
             get_prompt_btn, prompt_retry_btn_voice, go_zimage_btn_voice, workflow_return_status, zimage_progress, image_refresh_status],
            show_progress='minimal'
        )

        # Project mới: nhập tên folder → tạo trực tiếp trong Downloads → kết nối ngay.
        # TẠO PROJECT MỚI = luôn tạo một Project hoàn toàn mới.
        # Không lấy giá trị đang chọn trong Dropdown làm tên Project mới.
        # Project cũ chỉ được mở bằng nút "📂 Mở folder có sẵn".
        new_project_btn.click(
            _wrap_gradio_callback(lambda: new_project('')), None,
            [project, project_output_dir, full_output_dir, project_root_status, qc_project],
            show_progress='minimal'
        ).then(
            lambda name, path, _full, status, qc: (path, status, name),
            [project, project_output_dir, full_output_dir, project_root_status, qc_project],
            [project_active_path, project_root_status, cfg_project],
            show_progress='minimal'
        ).then(
            _wrap_gradio_callback(reset_project_ui), project,
            [script_box, scene_target, script_progress, scene_table, script_status, full_audio, full_download, full_status,
             full_scene_table, full_voice_progress_table, full_scene_players, gem_scene_count, gem_output, gem_status, zimage_status, start_image_btn,
             qc_project,
             full_combine_loader, full_combine_status, prompt_progress, prompt_status, prompt_output,
             get_prompt_btn, prompt_retry_btn_voice, go_zimage_btn_voice, workflow_return_status, zimage_progress, image_refresh_status],
            show_progress='minimal'
        ).then(
            outputs=[qc_result, qc_detail, asset_out, timeline_out, srt_file, capcut_out, final_out],
            show_progress='hidden'
        ).then(
            lambda: ('🧼 Cleaner sẵn sàng.', []), outputs=[wm_status, wm_files], show_progress='hidden'
        )

        def _select_project_live(name):
            global PROJECT_CONNECTED
            name = safe_name(name or '')
            if not name:
                return None
            root = _project_root_for_name(name)
            SETTINGS['selected_project'] = name
            SETTINGS['selected_project_path'] = str(root)
            SETTINGS['project_output_dir'] = str(root)
            SETTINGS['voice_output_dir'] = str(root)
            PROJECT_CONNECTED.set()
            save_settings()
            project_dir(name)
            return name

        # Selecting an existing name updates the connected Project and refreshes only data components.
        # No browser reload and no full-page refresh.
        project.change(_wrap_gradio_callback(_select_project_live), project, project, show_progress='minimal').then(
            _wrap_gradio_callback(load_project_ui), project,
            [script_box, scene_target, script_progress, scene_table, script_status, full_audio, full_download, full_status,
             full_scene_table, full_voice_progress_table, full_scene_players, gem_scene_count, gem_output, gem_status, zimage_status, start_image_btn,
             qc_project,
             full_combine_loader, full_combine_status, prompt_progress, prompt_status, prompt_output,
             get_prompt_btn, prompt_retry_btn_voice, go_zimage_btn_voice, workflow_return_status, zimage_progress, image_refresh_status],
            show_progress='minimal'
        )

        # Batch Center follows the same connected Project; never create a second queue.
        project.change(
            _wrap_gradio_callback(_batch_center_refresh), project,
            [batch_hud, batch_jobs], show_progress='hidden'
        )

        gr.Markdown(f'**{APP_DISPLAY_NAME}:** toàn bộ dữ liệu tạo ra của từng Project được lưu trong thư mục Project bạn đã chọn: Script / Scenes / Prompts / Audio / Images / ZImage / Output / QC / Timeline / CapCut...')

    return ui




def shutdown_runtime_pool():
    """Stop runtime helpers without touching user/AI child processes."""
    try:
        _SESSION_CLEANER_STOP.set()
    except Exception:
        pass


def cleanup_completed_projects_on_shutdown():
    """At shutdown, clean only ephemeral files for projects that already have a final MP4.
    Source script/Scenes/Prompts remain persistent. Images/audio/SRT/timeline/CapCut/MP4 stay.
    """
    try:
        for name in list_projects():
            root = _project_root_for_name(name)
            if not root.exists():
                continue
            mp4s = list(root.glob("*.mp4")) + list((_project_export_dir(root) / "Xuất video").glob("*.mp4"))
            if mp4s:
                cleanup_intermediate_project_files(name)
    except Exception:
        pass

# ============================================================
# V11.5.1 — GOOGLE FLOW REAL WORKER UI CONTROL
# ============================================================

_FLOW_ACCOUNT_LOCKS = {}
_FLOW_ACCOUNT_LOCKS_GUARD = threading.RLock()

def _flow_account_serial_lock(account_id):
    aid = str(account_id or "").strip()
    with _FLOW_ACCOUNT_LOCKS_GUARD:
        return _FLOW_ACCOUNT_LOCKS.setdefault(aid, threading.RLock())

def _flow_page_has_captcha(page):
    try:
        text = (page.locator("body").inner_text(timeout=1500) or "").lower()
        return any(k in text for k in ("captcha","verify you are human","unusual traffic","security check"))
    except Exception:
        return False

def _flow_thread_context(self, account_id):
    """Fresh Sync Playwright connection in the current Creative worker thread."""
    aid = str(account_id or "").strip()
    row = _creative_get_account(aid)
    if not row:
        raise CreativeAuthError(f"Flow Account không tồn tại: {aid}")
    profile = str(row.get("session_profile") or "").strip()
    if not profile:
        raise CreativeAuthError(f"{aid} chưa có session_profile.")
    ok, detail = _playwright_chrome_runtime_check()
    if not ok:
        raise CreativeFlowUnavailable(detail)
    port = int(row.get("debug_port") or _flow_debug_port(aid))
    if not _flow_cdp_ready(port, timeout=2):
        p = Path(profile).expanduser().resolve()
        p.mkdir(parents=True, exist_ok=True)
        workflow_cmd_log("FLOW_POOL","WORKER_PROFILE_REOPEN",account_id=aid,profile=str(p))
        _flow_launch_stable_chrome(aid, str(p), self.FLOW_URL)
        if not _flow_cdp_ready(port, timeout=15):
            raise CreativeFlowUnavailable(f"Chrome Stable đã mở lại nhưng CDP chưa sẵn sàng: {aid}")
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        raise RuntimeError("Flow Sync Playwright phải chạy trong worker thread, không chạy trong asyncio callback.")
    _playwright_prepare_runtime()
    from playwright.sync_api import sync_playwright
    pw = sync_playwright().start()
    try:
        browser = pw.chromium.connect_over_cdp(f"http://127.0.0.1:{port}", timeout=15000)
        contexts = browser.contexts
        if not contexts:
            raise CreativeFlowUnavailable(f"Chrome CDP không có BrowserContext: {aid}")
        context = contexts[0]
        pages = [p for p in context.pages if not p.is_closed()]
        page = next(
            (p for p in pages if "labs.google" in str(p.url or "").lower()
             or "flow" in str(p.url or "").lower()),
            pages[0] if pages else context.new_page(),
        )
        try:
            page.bring_to_front()
        except Exception:
            pass
        if "labs.google" not in str(page.url or "").lower() and "flow" not in str(page.url or "").lower():
            page.goto(self.FLOW_URL, wait_until="domcontentloaded", timeout=60000)
        try:
            page.wait_for_load_state("domcontentloaded", timeout=15000)
        except Exception:
            pass
        workflow_cmd_log("FLOW","WORKER_SESSION_ATTACHED",account_id=aid,port=port,url=str(page.url or "")[:500])
        return pw, browser, page
    except Exception:
        try:
            pw.stop()
        except Exception:
            pass
        raise

GoogleFlowAdapter._context = _flow_thread_context

def _flow_prompt_box_visible(page):
    """Detect the real Flow scene composer across current/legacy UI builds."""
    selectors=("textarea","[contenteditable='true']","[role='textbox']","input[type='text']")
    words=("prompt","describe","image","what do you want","enter a prompt","type a prompt","mô tả","nhập","describe your scene","tell flow")
    for sel in selectors:
        try:
            loc=page.locator(sel)
            for i in range(loc.count()-1,-1,-1):
                el=loc.nth(i)
                if not el.is_visible() or not el.is_enabled(): continue
                attrs=[]
                for name in ("placeholder","aria-label","data-placeholder","name","data-testid","data-test-id","title"):
                    try:
                        v=el.get_attribute(name)
                        if v: attrs.append(str(v))
                    except Exception: pass
                label=" ".join(attrs).lower()
                if any(k in label for k in words): return True
                try:
                    parent=el.locator("xpath=ancestor::*[.//button or .//*[@role='button']][1]")
                    controls=parent.locator("button,[role='button']")
                    labels=[]
                    for j in range(controls.count()-1,-1,-1):
                        ctl=controls.nth(j)
                        if not ctl.is_visible() or not ctl.is_enabled(): continue
                        labels.append(" ".join(filter(None,(ctl.get_attribute("aria-label"),ctl.get_attribute("title"),ctl.get_attribute("data-testid"),ctl.get_attribute("data-tooltip"),ctl.get_attribute("data-tooltip-content"),ctl.inner_text(timeout=150)))))
                    if any(re.search(r"generate|create\s*(image|images)?|submit|run|tạo\s*(ảnh|hình)?",x or "",re.I) for x in labels): return True
                except Exception: pass
        except Exception: continue
    return False

def _flow_create_project(page, timeout=60):
    """Create/reuse a real Google Flow project and wait for the scene composer.

    Current Flow desktop uses "+ New project". Older builds used New/Create
    project variants. Resolve the real clickable control first, then verify
    that the prompt composer is actually visible before returning.
    """
    timeout = max(15, int(timeout))
    deadline = time.monotonic() + timeout
    url_now = str(page.url or "")
    workflow_cmd_log("FLOW", "PROJECT_STAGE_START", url=url_now[:500], timeout=timeout)
    # Current Flow UI (as seen in the verified Chrome profile) labels the
    # homepage project card exactly "Dự án mới". Keep this as a first-class
    # control, not a generic button guess.

    def _ready():
        try:
            return _flow_prompt_box_visible(page)
        except Exception:
            return False

    if _ready():
        workflow_cmd_log("FLOW", "PROJECT_ALREADY_OPEN", url=str(page.url or "")[:500])
        return

    create_patterns = (
        r"^\\+\\s*New project$",
        r"^New project$",
        r"^\\+\\s*Dự án mới$",
        r"^Dự án mới$",
        r"^Create new project$",
        r"^Create project$",
        r"^New$",
        r"^Tạo dự án$",
        r"^Tạo mới$",
        r"^Create new$",
        r"^Start new project$",
    )

    def _click_create_control():
        # 1. Exact accessible button/name.
        for pat in create_patterns:
            try:
                loc = page.get_by_role("button", name=re.compile(pat, re.I))
                for n in range(loc.count() - 1, -1, -1):
                    b = loc.nth(n)
                    if b.is_visible() and b.is_enabled():
                        b.scroll_into_view_if_needed()
                        b.click(timeout=5000)
                        workflow_cmd_log("FLOW", "CREATE_PROJECT_CLICKED", label=pat)
                        return True
            except Exception:
                pass

        # 2. Current Flow may expose the "+ New project" text on a clickable
        #    wrapper rather than the button's accessible name.
        text_patterns = (
            r"\\+\\s*New project",
            r"New project",
            r"\\+\\s*Dự án mới",
            r"Dự án mới",
            r"Tạo dự án",
            r"Tạo mới",
        )
        for pat in text_patterns:
            try:
                loc = page.get_by_text(re.compile(pat, re.I))
                for n in range(loc.count() - 1, -1, -1):
                    item = loc.nth(n)
                    if not item.is_visible():
                        continue
                    # Prefer the nearest clickable ancestor.
                    candidates = [
                        item,
                        item.locator("xpath=ancestor::*[@role='button'][1]"),
                        item.locator("xpath=ancestor::button[1]"),
                        item.locator("xpath=ancestor::*[self::a or @role='link'][1]"),
                    ]
                    for c in candidates:
                        try:
                            if c.count() and c.first.is_visible() and c.first.is_enabled():
                                c.first.scroll_into_view_if_needed()
                                c.first.click(timeout=5000)
                                workflow_cmd_log("FLOW", "CREATE_PROJECT_CLICKED_TEXT", label=pat)
                                return True
                        except Exception:
                            pass
            except Exception:
                pass

        # 3. Data/tooltip/test-id fallback for the current Flow shell.
        try:
            loc = page.locator(
                "[aria-label*='new project' i],"
                "[title*='new project' i],"
                "[data-testid*='new-project' i],"
                "[data-testid*='new_project' i],"
                "[data-test-id*='new-project' i]"
            )
            for n in range(loc.count() - 1, -1, -1):
                b = loc.nth(n)
                if b.is_visible() and b.is_enabled():
                    b.scroll_into_view_if_needed()
                    b.click(timeout=5000)
                    workflow_cmd_log("FLOW", "CREATE_PROJECT_CLICKED_ATTR")
                    return True
        except Exception:
            pass
        return False

    while time.monotonic() < deadline:
        if _flow_page_has_captcha(page):
            raise CreativeCaptchaError("Google Flow yêu cầu CAPTCHA.")

        if _ready():
            workflow_cmd_log("FLOW", "PROJECT_READY", url=str(page.url or "")[:500])
            return

        clicked = _click_create_control()
        if clicked:
            workflow_cmd_log("FLOW", "CREATE_PROJECT_WAIT")

            # Some Flow builds navigate directly; others show a small creation
            # surface first. Never click arbitrary buttons: only accept a
            # composer or an explicitly named project-confirmation control.
            wait_deadline = min(deadline, time.monotonic() + 20)
            while time.monotonic() < wait_deadline:
                if _flow_page_has_captcha(page):
                    raise CreativeCaptchaError("Google Flow yêu cầu CAPTCHA.")
                if _ready():
                    workflow_cmd_log("FLOW", "PROJECT_READY", url=str(page.url or "")[:500])
                    return

                for pat in (
                    r"^Create project$",
                    r"^Create$",
                    r"^Continue$",
                    r"^Tạo$",
                    r"^Tiếp tục$",
                ):
                    try:
                        dialogs = page.get_by_role("dialog")
                        target = (
                            dialogs.get_by_role("button", name=re.compile(pat, re.I))
                            if dialogs.count()
                            else page.get_by_role("button", name=re.compile(pat, re.I))
                        )
                        for n in range(target.count() - 1, -1, -1):
                            b = target.nth(n)
                            if b.is_visible() and b.is_enabled():
                                b.click(timeout=3000)
                                workflow_cmd_log("FLOW", "CREATE_PROJECT_CONFIRMED", label=pat)
                                break
                    except Exception:
                        pass

                if _ready():
                    workflow_cmd_log("FLOW", "PROJECT_READY", url=str(page.url or "")[:500])
                    return
                time.sleep(0.35)

        workflow_cmd_log(
            "FLOW", "PROJECT_STAGE_WAIT",
            url=str(page.url or "")[:500],
            remaining_seconds=max(0, int(deadline - time.monotonic())),
            level="DEBUG",
        )
        time.sleep(0.5)

    try:
        body = (page.locator("body").inner_text(timeout=2000) or "")[:3000]
    except Exception:
        body = ""
    raise CreativeFlowUnavailable(
        f"Không mở/tạo được Project Google Flow hoặc chưa xuất hiện Scene Prompt composer. "
        f"url={str(page.url or '')[:300]} · body={body!r}"
    )

def _flow_find_prompt_box(page, timeout=45):
    """Return the actual Flow scene composer after project creation/reuse."""
    _flow_create_project(page,timeout=max(20,int(timeout)))
    deadline=time.monotonic()+max(10,int(timeout))
    selectors=("textarea","[contenteditable='true']","[role='textbox']","input[type='text']")
    while time.monotonic()<deadline:
        if _flow_page_has_captcha(page): raise CreativeCaptchaError("Google Flow yêu cầu CAPTCHA.")
        for sel in selectors:
            try:
                loc=page.locator(sel)
                for i in range(loc.count()-1,-1,-1):
                    item=loc.nth(i)
                    if not item.is_visible() or not item.is_enabled(): continue
                    attrs=[]
                    for name in ("placeholder","aria-label","data-placeholder","name","data-testid","data-test-id","title"):
                        try:
                            v=item.get_attribute(name)
                            if v: attrs.append(str(v))
                        except Exception: pass
                    label=" ".join(attrs).lower()
                    if any(k in label for k in ("prompt","describe","image","what do you want","enter a prompt","type a prompt","mô tả","nhập","describe your scene","tell flow")): return item
                    try:
                        parent=item.locator("xpath=ancestor::*[.//button or .//*[@role='button']][1]")
                        controls=parent.locator("button,[role='button']")
                        for j in range(controls.count()-1,-1,-1):
                            c=controls.nth(j)
                            if not c.is_visible() or not c.is_enabled(): continue
                            lab=" ".join(filter(None,(c.get_attribute("aria-label"),c.get_attribute("title"),c.get_attribute("data-testid"),c.get_attribute("data-tooltip"),c.get_attribute("data-tooltip-content"),c.inner_text(timeout=150))))
                            if re.search(r"generate|create\s*(image|images)?|submit|run|tạo\s*(ảnh|hình)?",lab,re.I) and not re.search(r"project|new|upload|attach|settings|menu|more|delete|cancel|close",lab,re.I): return item
                    except Exception: pass
            except Exception: pass
        time.sleep(.35)
    try: body=(page.locator("body").inner_text(timeout=2000) or "")[:2200]
    except Exception: body=""
    raise CreativeFlowUnavailable(f"Không tìm thấy Scene Prompt composer sau {timeout}s · url={str(page.url or '')[:300]} · body={body!r}")

def _flow_generation_started(page, before_img_count=0):
    """Best-effort evidence that Flow accepted the submit."""
    try:
        if page.locator("img").count() > int(before_img_count or 0):
            return True
    except Exception:
        pass
    markers = (
        r"generating", r"creating", r"processing", r"rendering",
        r"đang tạo", r"đang xử lý", r"cancel", r"stop"
    )
    for pat in markers:
        try:
            loc = page.get_by_text(re.compile(pat, re.I))
            if loc.count() and any(
                loc.nth(i).is_visible()
                for i in range(max(0, loc.count() - 5), loc.count())
            ):
                return True
        except Exception:
            pass
    try:
        for sel in (
            "[aria-busy='true']",
            "[role='progressbar']",
            "[data-testid*='progress' i]",
            "[data-testid*='generation' i]",
        ):
            loc = page.locator(sel)
            if loc.count() and any(
                loc.nth(i).is_visible()
                for i in range(max(0, loc.count() - 5), loc.count())
            ):
                return True
    except Exception:
        pass
    return False


def _flow_click_generate_image(page, prompt_box=None, before_img_count=0):
    """Submit one scene through the real Google Flow composer.

    Ported from the legacy Creative Studio behavior, but kept safe:
      1) exact accessible Generate/Create/Run controls;
      2) legacy Playwright text-button matching (has_text);
      3) explicit generate/create test-id/aria/title controls;
      4) controls belonging to the same prompt composer;
      5) only then keyboard submit, and only if Flow visibly starts generation.

    Never click an arbitrary page button.
    """
    box = prompt_box or _flow_find_prompt_box(page, 10)

    def _visible_enabled(loc):
        try:
            for i in range(loc.count() - 1, -1, -1):
                b = loc.nth(i)
                if b.is_visible() and b.is_enabled():
                    return b
        except Exception:
            pass
        return None

    def _label(btn):
        vals = []
        for name in (
            "aria-label", "title", "data-testid", "data-test-id",
            "data-tooltip", "data-tooltip-content", "name"
        ):
            try:
                v = btn.get_attribute(name)
                if v:
                    vals.append(str(v))
            except Exception:
                pass
        try:
            vals.append(btn.inner_text(timeout=200) or "")
        except Exception:
            pass
        return " ".join(vals).strip()

    def _safe_click(btn, tag):
        try:
            if not btn or not btn.is_visible() or not btn.is_enabled():
                return False
            btn.scroll_into_view_if_needed()
            btn.click(timeout=5000)
            workflow_cmd_log(
                "FLOW", "GENERATE_CONTROL_CLICKED",
                method=tag, label=_label(btn)[:300]
            )
            return True
        except Exception as exc:
            workflow_cmd_log(
                "FLOW", "GENERATE_CONTROL_CLICK_FAILED",
                method=tag, error=str(exc)[:500], level="DEBUG"
            )
            return False

    # 1) Exact accessible names.
    exact_patterns = (
        r"^Generate Image$",
        r"^Generate images?$",
        r"^Generate$",
        r"^Create Image$",
        r"^Create images?$",
        r"^Create$",
        r"^Run$",
        r"^Submit$",
        r"^Send$",
        r"^Tạo ảnh$",
        r"^Tạo hình$",
        r"^Tạo$",
    )
    for pat in exact_patterns:
        try:
            btn = _visible_enabled(
                page.get_by_role("button", name=re.compile(pat, re.I))
            )
            if _safe_click(btn, "exact-accessible"):
                return True
        except Exception:
            pass

    # 2) Legacy Creative Studio behavior: Flow builds that expose visible
    #    button text but do not expose the same accessible name.
    legacy_labels = (
        "Generate", "Generate Image", "Create", "Create Image",
        "Run", "Submit", "Send", "Tạo", "Tạo ảnh", "Tạo hình"
    )
    for label in legacy_labels:
        try:
            loc = page.get_by_role("button").filter(
                has_text=re.compile(r"^\s*" + re.escape(label) + r"\s*$", re.I)
            )
            btn = _visible_enabled(loc)
            if _safe_click(btn, "legacy-has-text"):
                return True
        except Exception:
            pass

    # 3) Explicit DOM attributes used by Flow UI builds.
    try:
        attr = page.locator(
            "button[aria-label*='generate' i],"
            "button[aria-label*='create image' i],"
            "button[title*='generate' i],"
            "button[title*='create image' i],"
            "[role='button'][aria-label*='generate' i],"
            "[role='button'][aria-label*='create image' i],"
            "[data-testid*='generate' i],"
            "[data-testid*='create-image' i],"
            "[data-testid*='create_image' i],"
            "[data-test-id*='generate' i],"
            "[data-test-id*='create-image' i]"
        )
        btn = _visible_enabled(attr)
        if _safe_click(btn, "explicit-generate-attribute"):
            return True
    except Exception:
        pass

    # 4) Search only inside the prompt composer, never the whole page.
    composer_roots = []
    try:
        root = box.locator(
            "xpath=ancestor::*[.//button or .//*[@role='button']][1]"
        )
        composer_roots.append(root)
        composer_roots.append(
            root.locator(
                "xpath=ancestor::*[.//button or .//*[@role='button']][1]"
            )
        )
    except Exception:
        pass

    submit_re = re.compile(
        r"generate|create\s*(?:image|images|video)?|submit|run|send|"
        r"tạo\s*(?:ảnh|hình)?",
        re.I,
    )
    reject_re = re.compile(
        r"project|new|upload|attach|reference|settings|menu|more|"
        r"delete|remove|cancel|close|back|download|undo|redo",
        re.I,
    )

    for root in composer_roots:
        try:
            controls = root.locator("button,[role='button']")
            for i in range(controls.count() - 1, -1, -1):
                btn = controls.nth(i)
                if not btn.is_visible() or not btn.is_enabled():
                    continue
                lab = _label(btn)
                if submit_re.search(lab) and not reject_re.search(lab):
                    if _safe_click(btn, "composer-labeled"):
                        return True
        except Exception:
            continue

    # 5) Icon-only Generate button: restrict to the composer geometry and
    #    require SVG evidence. This avoids clicking upload/menu/settings.
    try:
        bb = box.bounding_box()
    except Exception:
        bb = None

    if bb:
        bx, by, bw, bh = bb
        box_right = bx + bw
        box_bottom = by + bh
        for root in composer_roots:
            try:
                controls = root.locator("button,[role='button']")
                candidates = []
                for i in range(controls.count() - 1, -1, -1):
                    btn = controls.nth(i)
                    try:
                        if not btn.is_visible() or not btn.is_enabled():
                            continue
                        lab = _label(btn)
                        if reject_re.search(lab):
                            continue
                        cb = btn.bounding_box()
                        if not cb:
                            continue
                        cx, cy, cw, ch = cb
                        near = (
                            cx >= bx - max(80, bw * 0.10)
                            and cx <= box_right + 120
                            and cy >= by - 100
                            and cy <= box_bottom + 140
                        )
                        if not near:
                            continue
                        html_fragment = btn.inner_html(timeout=250) or ""
                        html_low = html_fragment.lower()
                        if "<svg" not in html_low:
                            continue
                        score = 2 + (1 if "<path" in html_low else 0)
                        score += int(max(0, cx - bx) / max(1, bw) * 3)
                        score += int(max(0, cy - by) / max(1, bh) * 2)
                        candidates.append((score, cx, cy, btn))
                    except Exception:
                        continue

                candidates.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
                for score, _cx, _cy, btn in candidates:
                    if _safe_click(btn, f"composer-icon-score-{score}"):
                        deadline = time.monotonic() + 8
                        while time.monotonic() < deadline:
                            if _flow_generation_started(page, before_img_count):
                                return True
                            time.sleep(0.2)
            except Exception:
                continue

    # 6) Submit the actual textbox, but accept it only when Flow enters a
    #    generation state. This mirrors the old tool's Enter fallback safely.
    for key in ("Control+Enter", "Enter"):
        try:
            box.click(timeout=3000)
            box.press(key)
            deadline = time.monotonic() + 8
            while time.monotonic() < deadline:
                if _flow_generation_started(page, before_img_count):
                    workflow_cmd_log(
                        "FLOW", "GENERATE_KEYBOARD_ACCEPTED", key=key
                    )
                    return True
                time.sleep(0.2)
        except Exception:
            continue

    # Diagnostic only: record the visible button labels so the next failure
    # identifies the actual Flow control instead of saying only "not found".
    try:
        visible = []
        buttons = page.locator("button,[role='button']")
        for i in range(buttons.count()):
            b = buttons.nth(i)
            if b.is_visible():
                lab = _label(b)
                if lab:
                    visible.append(lab[:180])
        workflow_cmd_log(
            "FLOW", "GENERATE_CONTROL_NOT_FOUND",
            visible_buttons=visible[-40:],
            url=str(page.url or "")[:500],
            level="ERROR",
        )
    except Exception:
        pass
    return False


def _flow_download_latest_asset(page,before_img_count,timeout=900):
    """Wait for the generated asset and click Flow's visible Download control."""
    deadline=time.monotonic()+max(30,int(timeout))
    while time.monotonic()<deadline:
        if _flow_page_has_captcha(page):
            raise CreativeCaptchaError("Google Flow yêu cầu CAPTCHA.")
        try:
            imgs=page.locator("img"); img_count=imgs.count()
        except Exception:
            imgs=None; img_count=before_img_count

        for sel in ("[aria-label*='download' i]","[title*='download' i]","[data-testid*='download' i]","a[download]","a[href*='download' i]"):
            try:
                loc=page.locator(sel)
                for i in range(loc.count()-1,-1,-1):
                    item=loc.nth(i)
                    if not item.is_visible():
                        continue
                    try:
                        with page.expect_download(timeout=3000) as info:
                            item.click(timeout=2000)
                        return info.value
                    except Exception:
                        continue
            except Exception:
                pass

        if imgs is not None and img_count>before_img_count:
            for i in range(img_count-1,max(-1,img_count-10),-1):
                try:
                    img=imgs.nth(i)
                    if not img.is_visible():
                        continue
                    img.scroll_into_view_if_needed(); img.hover(timeout=2000)
                    parent=img.locator("xpath=ancestor::*[.//img][1]")
                    controls=parent.locator("button,[role='button']")
                    for j in range(controls.count()-1,-1,-1):
                        c=controls.nth(j)
                        if not c.is_visible():
                            continue
                        label=" ".join(filter(None,[c.inner_text(timeout=200),c.get_attribute("aria-label"),c.get_attribute("title")]))
                        if re.search(r"more|options",label,re.I):
                            try:
                                c.click(timeout=2000)
                                menu=page.get_by_role("menuitem",name=re.compile(r"download",re.I)).last
                                if menu.count() and menu.is_visible():
                                    with page.expect_download(timeout=10000) as info:
                                        menu.click(timeout=3000)
                                    return info.value
                            except Exception:
                                pass
                except Exception:
                    pass

        try:
            dl=page.get_by_text(re.compile(r"^Download$",re.I)).last
            if dl.count() and dl.is_visible():
                with page.expect_download(timeout=5000) as info:
                    dl.click(timeout=3000)
                return info.value
        except Exception:
            pass
        time.sleep(CREATIVE_FLOW_POLL_SECONDS)
    raise CreativeFlowTimeout(f"Google Flow đã nhận Generate nhưng chưa có Download sau {timeout}s.")

def _flow_run_real_v115(self,job,worker_id,account_id=None):
    if not account_id:
        raise CreativeAuthError("Flow Job chưa có Account.")
    with _flow_account_serial_lock(account_id):
        pw=None
        try:
            _creative_set_actual_engine(job["job_id"],CREATIVE_ENGINE_FLOW)
            _creative_heartbeat(worker_id,job["job_id"])
            pw,browser,page=_flow_thread_context(self,account_id)
            if _flow_page_has_captcha(page):
                raise CreativeCaptchaError("Google Flow yêu cầu CAPTCHA.")
            meta=_creative_load_json(job.get("metadata_json") or "{}",{})
            prompt=str(meta.get("prompt") or meta.get("image_prompt") or meta.get("visual_prompt") or "").strip()
            if not prompt:
                raise CreativeInvalidRequest(f"{job.get('job_id')} không có scene prompt.")

            workflow_cmd_log("FLOW","PROMPT_IMPORT_START",account_id=account_id,scene=job.get("scene_id"),prompt_chars=len(prompt))
            box=_flow_find_prompt_box(page,30)
            box.click()
            try:
                box.press("Control+A")
            except Exception:
                pass
            # Use real browser text insertion after focus/select, matching the old
            # tool's visible paste operation instead of JS-setting React state.
            try:
                page.keyboard.insert_text(prompt)
            except Exception:
                box.fill(prompt)
            workflow_cmd_log("FLOW","PROMPT_IMPORTED",account_id=account_id,scene=job.get("scene_id"))
            if _flow_page_has_captcha(page):
                raise CreativeCaptchaError("Google Flow yêu cầu CAPTCHA.")
            try: before_img_count=page.locator("img").count()
            except Exception: before_img_count=0
            if not _flow_click_generate_image(page, box, before_img_count):
                raise CreativeFlowUnavailable(
                    "Không tìm thấy nút Generate/Create của Google Flow sau khi đã tạo Project và nhập Prompt."
                )
            workflow_cmd_log("FLOW","GENERATE_CLICKED",account_id=account_id,scene=job.get("scene_id"))
            _creative_heartbeat(worker_id,job["job_id"])
            download=_flow_download_latest_asset(page,before_img_count,CREATIVE_FLOW_RESULT_TIMEOUT)

            out_dir=_project_image_dir(project_dir(job["project_id"]))
            out_dir.mkdir(parents=True,exist_ok=True)
            out=out_dir/f"{job['scene_id']}.png"
            download.save_as(str(out))
            try:
                from PIL import Image
                with Image.open(out) as im:
                    im.load()
                    if im.size!=(CREATIVE_IMAGE_WIDTH,CREATIVE_IMAGE_HEIGHT):
                        im=im.convert("RGB").resize((CREATIVE_IMAGE_WIDTH,CREATIVE_IMAGE_HEIGHT),Image.Resampling.LANCZOS)
                        im.save(out,format="PNG",optimize=True)
            except Exception as exc:
                raise CreativeFlowUnavailable(f"Không normalize được Flow output: {exc}")
            workflow_cmd_log("FLOW","SCENE_OUTPUT_SAVED",account_id=account_id,scene=job.get("scene_id"),output=str(out),width=CREATIVE_IMAGE_WIDTH,height=CREATIVE_IMAGE_HEIGHT)
            return _creative_output_contract(job,out,CREATIVE_ENGINE_FLOW,account_id,worker_id)
        except Exception as exc:
            workflow_cmd_log("FLOW","SCENE_FAILED",account_id=account_id,scene=job.get("scene_id"),error=str(exc)[:1800],level="ERROR")
            raise
        finally:
            # Release only this worker's Playwright connection. Do not close the
            # remote Stable Chrome process/profile.
            try:
                if pw is not None: pw.stop()
            except Exception:
                pass

GoogleFlowAdapter.run=_flow_run_real_v115

_CREATIVE_START_WORKERS_ORIGINAL_V115=globals().get("_creative_start_workers")
def _creative_start_workers(project,count=2,worker_type="FLOW"):
    if str(worker_type).upper()=="FLOW":
        count=1
    return _CREATIVE_START_WORKERS_ORIGINAL_V115(project,count=count,worker_type=worker_type)


# ============================================================
# V11.6.1 — FLOW PROFILE / PLAYWRIGHT RUNTIME HARDENING
# ============================================================
_FLOW_EXTERNAL_DRIVER_PATCHED = False
_FLOW_EXTERNAL_DRIVER_ROOT = ""

def _flow_external_playwright_driver():
    global _FLOW_EXTERNAL_DRIVER_PATCHED, _FLOW_EXTERNAL_DRIVER_ROOT
    if _FLOW_EXTERNAL_DRIVER_PATCHED:
        return _FLOW_EXTERNAL_DRIVER_ROOT
    candidates = []
    env = str(os.getenv("VHUNG_PLAYWRIGHT_RUNTIME") or "").strip()
    if env:
        candidates.append(Path(env).expanduser())
    candidates.extend([
        BASE / "tool" / "playwright",
        BASE / "tool" / "playwright" / "driver",
        BASE / "tool" / "playwright" / "driver" / "package",
    ])
    try:
        candidates.extend([
            Path.home() / "Downloads" / "CloneVoice-1.1.18-win64" / "playwright",
            Path.home() / "Downloads" / "CloneVoice-1.1.18-win64" / "CloneVoice-1.1.18-win64" / "playwright",
        ])
    except Exception:
        pass
    for raw in candidates:
        try:
            p = Path(raw).resolve()
            if p.name.lower() == "package" and p.parent.name.lower() == "driver":
                root = p.parent.parent
            elif p.name.lower() == "driver":
                root = p.parent
            else:
                root = p
            cli = root / "driver" / "package" / "cli.js"
            node = root / "driver" / "node.exe"
            if not cli.is_file():
                continue
            if not node.is_file():
                try:
                    import playwright._impl._driver as _pw_driver
                    current_node = Path(_pw_driver.compute_driver_executable()[0])
                    if current_node.is_file():
                        node = current_node
                except Exception:
                    pass
            if not node.is_file():
                continue
            import playwright._impl._driver as _pw_driver
            import playwright._impl._transport as _pw_transport
            def _compute_external_driver(_node=str(node), _cli=str(cli)):
                return _node, _cli
            _pw_driver.compute_driver_executable = _compute_external_driver
            _pw_transport.compute_driver_executable = _compute_external_driver
            os.environ["VHUNG_PLAYWRIGHT_ACTIVE_ROOT"] = str(root)
            os.environ["VHUNG_PLAYWRIGHT_DRIVER_CLI"] = str(cli)
            os.environ["VHUNG_PLAYWRIGHT_DRIVER_NODE"] = str(node)
            _FLOW_EXTERNAL_DRIVER_ROOT = str(root)
            _FLOW_EXTERNAL_DRIVER_PATCHED = True
            workflow_cmd_log("PLAYWRIGHT", "BUNDLED_DRIVER_SELECTED", root=str(root), cli=str(cli), node=str(node))
            return str(root)
        except Exception as exc:
            workflow_cmd_log("PLAYWRIGHT", "BUNDLED_DRIVER_CANDIDATE_FAILED", path=str(raw), error=str(exc)[:500], level="DEBUG")
    return ""

_PLAYWRIGHT_PREPARE_RUNTIME_ORIGINAL_V1161 = _playwright_prepare_runtime

def _playwright_prepare_runtime_v1161():
    external = _flow_external_playwright_driver()
    if external:
        return external
    try:
        return _PLAYWRIGHT_PREPARE_RUNTIME_ORIGINAL_V1161()
    except Exception:
        return ""
_playwright_prepare_runtime = _playwright_prepare_runtime_v1161

def _playwright_chrome_runtime_check_v1161():
    try:
        runtime_root = _playwright_prepare_runtime()
        from playwright.sync_api import sync_playwright
        pw = sync_playwright().start()
        pw.stop()
        chrome = _flow_chrome_executable()
        if chrome is None:
            return False, "🔴 Không tìm thấy Google Chrome Stable."
        source = f"BUNDLED DRIVER {runtime_root}" if runtime_root else "PYTHON PLAYWRIGHT"
        return True, f"🟢 Playwright OK · {source} · Chrome Stable OK · {chrome}"
    except Exception as exc:
        return False, f"🔴 Playwright runtime lỗi: {str(exc)[:1200]}"
_playwright_chrome_runtime_check = _playwright_chrome_runtime_check_v1161

def _flow_page_has_captcha_v1161(page):
    try:
        url = str(page.url or "").lower()
    except Exception:
        url = ""
    selectors = (
        "iframe[src*='recaptcha' i]",
        "iframe[src*='hcaptcha' i]",
        "[id*='captcha' i]",
        "[class*='captcha' i]",
        "[data-testid*='captcha' i]",
        "[aria-label*='captcha' i]",
    )
    for sel in selectors:
        try:
            loc = page.locator(sel)
            for i in range(loc.count() - 1, -1, -1):
                if loc.nth(i).is_visible():
                    return True
        except Exception:
            pass
    challenge_re = re.compile(
        r"(verify\s+you\s+are\s+human|unusual\s+traffic|security\s+check|"
        r"confirm\s+you(?:'re|\s+are)\s+not\s+a\s+robot)", re.I
    )
    try:
        loc = page.get_by_text(challenge_re)
        for i in range(loc.count() - 1, -1, -1):
            if loc.nth(i).is_visible():
                return True
    except Exception:
        pass
    if "accounts.google.com" in url and any(x in url for x in ("challenge", "signin/v2", "signinoptions")):
        return True
    return False

_flow_page_has_captcha = _flow_page_has_captcha_v1161
GoogleFlowAdapter._captcha_visible = staticmethod(_flow_page_has_captcha_v1161)

def _flow_discover_existing_profiles_v1161():
    _flow_account_schema()
    roots = [FLOW_PROFILE_ROOT]
    extra = str(os.getenv("VHUNG_FLOW_PROFILE_ROOT") or "").strip()
    if extra:
        roots.append(Path(extra).expanduser())
    discovered = []
    for root in roots:
        try:
            root = Path(root).expanduser().resolve()
            if not root.is_dir():
                continue
            for p in sorted(root.iterdir()):
                if not p.is_dir() or p.name.startswith("."):
                    continue
                with _batch_db() as db:
                    row = db.execute(
                        "SELECT account_id FROM creative_accounts WHERE provider='GOOGLE_FLOW' AND session_profile=? LIMIT 1",
                        (str(p),),
                    ).fetchone()
                if row:
                    continue
                aid = p.name if re.fullmatch(r"FLOW-\d{4,}", p.name) else _flow_next_profile_id()
                now = _creative_now()
                with _batch_db() as db:
                    if db.execute("SELECT account_id FROM creative_accounts WHERE account_id=?", (aid,)).fetchone():
                        aid = _flow_next_profile_id()
                    db.execute(
                        "INSERT INTO creative_accounts(account_id,provider,status,session_profile,account_label,cookie_file,debug_port,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?)",
                        (aid, "GOOGLE_FLOW", "DISABLED", str(p), aid, str(_flow_storage_paths(aid)[1]), _flow_debug_port(aid), now, now),
                    )
                discovered.append((aid, str(p)))
                workflow_cmd_log("FLOW_POOL", "PROFILE_AUTO_DISCOVERED", account_id=aid, profile=str(p))
        except Exception as exc:
            workflow_cmd_log("FLOW_POOL", "PROFILE_DISCOVERY_ERROR", root=str(root), error=str(exc)[:700], level="DEBUG")
    return discovered

def _flow_recover_stale_pool_v1161():
    _creative_v3_migrate()
    _flow_discover_existing_profiles_v1161()
    now = _creative_now()
    recovered = 0
    with _batch_db() as db:
        rows = db.execute("SELECT account_id FROM creative_accounts WHERE provider='GOOGLE_FLOW' AND status='BUSY'").fetchall()
        for row in rows:
            aid = str(row["account_id"] or "")
            live = db.execute(
                "SELECT 1 FROM creative_workers WHERE account_id=? AND status='RUNNING' AND job_id IS NOT NULL LIMIT 1",
                (aid,),
            ).fetchone()
            if live:
                continue
            db.execute(
                "UPDATE creative_accounts SET status='ACTIVE',last_error=NULL,cooldown_until=NULL,updated_at=? WHERE account_id=? AND status='BUSY'",
                (now, aid),
            )
            recovered += 1
    if recovered:
        workflow_cmd_log("FLOW_POOL", "STALE_BUSY_RECOVERED", count=recovered)
    return recovered

def _creative_claim_account_checked_v1161(worker_id, job_id, provider=CREATIVE_ENGINE_FLOW):
    _flow_recover_stale_pool_v1161()
    _creative_refresh_cooldowns()
    account_id = _creative_atomic_claim_account(worker_id, job_id, provider)
    if account_id:
        ok, detail = _creative_session_health(account_id, worker_id, force=True)
        if ok:
            return account_id, "OK"
        detail = str(detail or "")
        now = _creative_now()
        is_captcha = any(
            x in detail.upper()
            for x in ("CAPTCHA", "VERIFY YOU ARE HUMAN", "UNUSUAL TRAFFIC", "SECURITY CHECK")
        )
        account_state = "CAPTCHA" if is_captcha else "AUTH_ERROR"
        with _batch_db() as db:
            db.execute(
                "UPDATE creative_accounts SET status=?,last_error=?,updated_at=? WHERE account_id=?",
                (account_state, detail[:1500], now, account_id),
            )
            db.execute(
                "UPDATE creative_jobs SET status='RETRY',error_class=?,error=?,worker_id=NULL,account_id=NULL,ready_at=?,updated_at=? WHERE job_id=? AND status='RUNNING'",
                (account_state, detail[:1500], now, now, job_id),
            )
            db.execute(
                "UPDATE creative_workers SET status='IDLE',job_id=NULL,account_id=NULL,updated_at=? WHERE worker_id=?",
                (now, worker_id),
            )
        return None, detail
    with _batch_db() as db:
        candidates = db.execute(
            "SELECT account_id FROM creative_accounts WHERE provider='GOOGLE_FLOW' AND status='CAPTCHA' AND session_profile IS NOT NULL AND session_profile<>'' ORDER BY updated_at ASC LIMIT 10"
        ).fetchall()
    for row in candidates:
        aid = str(row["account_id"] or "")
        ok, detail = _creative_session_health(aid, worker_id, force=True)
        if ok:
            _creative_set_account_state(aid, "ACTIVE", None, None)
            account_id = _creative_atomic_claim_account(worker_id, job_id, provider)
            if account_id:
                return account_id, "OK"
        else:
            workflow_cmd_log("FLOW_POOL", "PROFILE_HEALTH_STILL_BAD", account_id=aid, detail=str(detail)[:800], level="DEBUG")
    return None, "NO_ACTIVE_ACCOUNT"

_creative_claim_account_checked_v7 = _creative_claim_account_checked_v1161
_creative_claim_account_checked = _creative_claim_account_checked_v1161

_FLOW_ORIGINAL_LAUNCH_UI = launch_ui
def launch_ui_v1161():
    try:
        _flow_recover_stale_pool_v1161()
    except Exception as exc:
        workflow_cmd_log("FLOW_POOL", "STARTUP_RECONCILE_ERROR", error=str(exc)[:700], level="DEBUG")
    return _FLOW_ORIGINAL_LAUNCH_UI()
launch_ui = launch_ui_v1161


if __name__ == '__main__':
    import argparse
    if os.name == 'nt':
        try:
            os.system('title VHƯNG PRO MAX - RUN FAST')
        except Exception:
            pass
    atexit.register(shutdown_runtime_pool)
    atexit.register(cleanup_completed_projects_on_shutdown)
    parser = argparse.ArgumentParser()
    parser.add_argument('--server', choices=list(ENGINE_INFO.keys()), default=None)
    parser.add_argument('--port', type=int, default=None)
    args = parser.parse_args()
    if args.server:
        info = ENGINE_INFO[args.server]
        port = args.port or info['port']
        py = venv_python(info['venv'])
        if not py.exists():
            raise SystemExit(f'Environment chưa có: {py}')
        subprocess.run([str(py), str(info['script']), '--port', str(port)], check=False)
    else:
        app = launch_ui()
        allowed = [str(BASE), str(Path.home()), tempfile.gettempdir()]
        for key in ('voice_output_dir','project_output_dir'):
            chosen = (SETTINGS.get(key) or '').strip()
            if chosen: allowed.append(str(Path(chosen).expanduser()))

        # LOCAL LOOPBACK: keep the Gradio server bound to the local 127.0.0.1 interface.
        # This is a launcher/network setting only; it does not participate in workflow logic.
        app.queue(max_size=None, default_concurrency_limit=None)
        try:
            app.launch(
                server_name='127.0.0.1',
                server_port=7863,
                inbrowser=True,
                share=False,
                show_error=True,
                max_threads=64,
                css=build_css(),
                allowed_paths=allowed,
            )
        except Exception:
            # KHÔNG để CMD tự đóng khi app.py lỗi. Giữ nguyên traceback để
            # nhìn chính xác lỗi startup (ví dụ port 7863 đang bị chiếm).
            print('\n' + '=' * 72, flush=True)
            print('[FAIL] app.py bi loi — CMD SE KHONG TU DONG DONG.', flush=True)
            print('=' * 72, flush=True)
            traceback.print_exc()
            print('\n[INFO] Port Gradio đang cấu hình: 7863', flush=True)
            print('[INFO] Sửa lỗi phía trên rồi nhấn ENTER để đóng CMD.', flush=True)
            if os.name == 'nt':
                try:
                    input('> ')
                except (EOFError, KeyboardInterrupt):
                    pass
            raise
