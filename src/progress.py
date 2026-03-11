"""
progress.py — Thread-safe ingestion progress tracker with async SSE streaming.

Events flow from sync ingestor threads → async SSE handlers via
loop.call_soon_threadsafe into asyncio Queues.

Each job accumulates an ordered event log. SSE subscribers receive
events in real-time; polling clients get the latest snapshot.
"""

import asyncio
import json
import threading
import time
import uuid
from typing import Any

from loguru import logger


class ProgressTracker:
    """Thread-safe progress tracking for ingestion jobs."""

    def __init__(self):
        self._lock = threading.Lock()
        self._jobs: dict[str, dict] = {}
        self._subscribers: dict[str, list[tuple[asyncio.Queue, asyncio.AbstractEventLoop]]] = {}
        self._cancel_events: dict[str, threading.Event] = {}

    def create_job(self, source_file: str, folder_name: str) -> str:
        """Create a new ingestion job. Returns the job_id."""
        job_id = f"j_{uuid.uuid4().hex[:8]}"
        state = {
            "job_id": job_id,
            "source_file": source_file,
            "folder_name": folder_name,
            "status": "pending",
            "progress": 0.0,
            "message": "Queued for ingestion",
            "phase": "",
            "started_at": time.time(),
            "elapsed": 0.0,
            "events": [],
        }
        with self._lock:
            self._jobs[job_id] = state
            self._cancel_events[job_id] = threading.Event()
        self.emit(job_id, "started", message=f"Starting ingestion of {source_file}")
        return job_id

    def emit(self, job_id: str, event_type: str, **kwargs: Any) -> None:
        """
        Emit a progress event from any thread.

        Common event types:
            started, splitting, split_done, part_start, step, step_progress,
            step_done, part_done, done, error
        """
        now = time.time()
        event = {"type": event_type, **kwargs}

        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            event["elapsed"] = round(now - job["started_at"], 1)
            # Update job snapshot from event fields
            for key in ("progress", "message", "phase", "current_part",
                        "total_parts", "part_file"):
                if key in kwargs:
                    job[key] = kwargs[key]
            if event_type == "done":
                job["status"] = "completed"
                job["progress"] = 1.0
            elif event_type == "error":
                job["status"] = "failed"
            elif event_type == "cancelled":
                job["status"] = "cancelled"
            else:
                job["status"] = "running"
            job["elapsed"] = event["elapsed"]
            job["events"].append(event)
            # Keep the subscriber list reference for notification
            subs = list(self._subscribers.get(job_id, []))

        # Push to SSE subscribers (outside the lock to avoid deadlocks)
        for q, loop in subs:
            try:
                loop.call_soon_threadsafe(q.put_nowait, event)
            except Exception:
                pass  # Queue full or loop closed

    def subscribe(self, job_id: str, loop: asyncio.AbstractEventLoop) -> asyncio.Queue:
        """
        Subscribe to events for a job. Returns an asyncio.Queue that receives events.
        Must be called from an async context. Call unsubscribe() when done.
        """
        q: asyncio.Queue = asyncio.Queue(maxsize=200)
        with self._lock:
            if job_id not in self._subscribers:
                self._subscribers[job_id] = []
            self._subscribers[job_id].append((q, loop))
            # Replay accumulated events so the subscriber catches up
            job = self._jobs.get(job_id)
            if job:
                for evt in job["events"]:
                    try:
                        q.put_nowait(evt)
                    except asyncio.QueueFull:
                        break
        return q

    def unsubscribe(self, job_id: str, q: asyncio.Queue) -> None:
        """Remove a subscriber queue."""
        with self._lock:
            subs = self._subscribers.get(job_id, [])
            self._subscribers[job_id] = [(qq, l) for qq, l in subs if qq is not q]

    def cancel_job(self, job_id: str) -> bool:
        """Signal a job to cancel. Returns True if the job exists and was signalled."""
        with self._lock:
            evt = self._cancel_events.get(job_id)
            if not evt:
                return False
            job = self._jobs.get(job_id)
            if job and job["status"] in ("completed", "failed", "cancelled"):
                return False
            evt.set()
        return True

    def is_cancelled(self, job_id: str) -> bool:
        """Check if a job has been signalled for cancellation. Thread-safe."""
        with self._lock:
            evt = self._cancel_events.get(job_id)
        return evt.is_set() if evt else False

    def get_job(self, job_id: str) -> dict | None:
        """Get the current snapshot of a job (without full event log)."""
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None
            snapshot = {k: v for k, v in job.items() if k != "events"}
            snapshot["elapsed"] = round(time.time() - job["started_at"], 1)
            return snapshot

    def list_jobs(self) -> list[dict]:
        """List all jobs (recent first, without full event logs)."""
        with self._lock:
            now = time.time()
            result = []
            for job in self._jobs.values():
                snapshot = {k: v for k, v in job.items() if k != "events"}
                snapshot["elapsed"] = round(now - job["started_at"], 1)
                result.append(snapshot)
        result.sort(key=lambda j: j["started_at"], reverse=True)
        return result


# Global singleton — imported by ingestor, admin, server
tracker = ProgressTracker()
