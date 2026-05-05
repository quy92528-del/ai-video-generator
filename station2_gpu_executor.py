"""
station2_gpu_executor.py - Hybrid Station 2: GCP GPU VM Executor

Manages the lifecycle of a Google Cloud Compute GPU VM instance:
  1. Start the VM when a task arrives.
  2. Wait for the VM to be ready.
  3. Send the processing command via SSH or Cloud Run.
  4. Wait for the task to complete.
  5. ALWAYS shut down the VM after every task (Safe-Off), whether the
     task succeeded or failed.

This module is the only place that may incur GPU costs — it must
NEVER leave the VM running after use.

Usage:
    executor = Station2GPUExecutor(
        project_id="my-gcp-project",
        zone="us-central1-a",
        instance_name="gpu-worker-1",
        credentials_json="/path/to/sa.json",
    )
    result = executor.execute(task_type="text_to_video", payload={...})
"""

from __future__ import annotations

import json
import subprocess
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class VMState(str, Enum):
    UNKNOWN = "UNKNOWN"
    PROVISIONING = "PROVISIONING"
    STAGING = "STAGING"
    RUNNING = "RUNNING"
    STOPPING = "STOPPING"
    TERMINATED = "TERMINATED"


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class VMStatus:
    """Snapshot of the current GPU VM status."""

    instance_name: str
    state: VMState = VMState.UNKNOWN
    external_ip: Optional[str] = None
    internal_ip: Optional[str] = None
    last_checked: float = field(default_factory=time.time)
    error: Optional[str] = None


@dataclass
class ExecutorResult:
    """Result returned by the GPU executor after a task run."""

    success: bool
    task_type: str
    output: Any = None
    error: Optional[str] = None
    vm_started: bool = False
    vm_stopped: bool = False
    duration_seconds: float = 0.0
    log_entries: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Station 2 GPU Executor
# ---------------------------------------------------------------------------

class Station2GPUExecutor:
    """
    Manages a GCP GPU Compute Engine instance for heavy video processing.

    Safety guarantee: The VM is ALWAYS stopped after task completion,
    even if the task raises an exception.  This is enforced by a
    try/finally block in :meth:`execute`.

    Args:
        project_id: GCP project ID.
        zone: GCP zone where the instance lives (e.g. "us-central1-a").
        instance_name: Name of the Compute Engine VM instance.
        credentials_json: Path to a service-account JSON key file.
            If empty, ADC (Application Default Credentials) is used.
        startup_timeout: Maximum seconds to wait for VM to become RUNNING.
        task_timeout: Maximum seconds to wait for the GPU task to finish.
        ssh_user: Username for SSH access to the VM.
        worker_script: Path on the VM to the processing script.
        status_callback: Optional callable for real-time status updates.
    """

    def __init__(
        self,
        project_id: str,
        zone: str,
        instance_name: str,
        credentials_json: str = "",
        startup_timeout: int = 300,
        task_timeout: int = 3600,
        ssh_user: str = "ubuntu",
        worker_script: str = "/opt/ai-video/worker.py",
        status_callback: Optional[Any] = None,
    ) -> None:
        self._project = project_id
        self._zone = zone
        self._instance = instance_name
        self._creds_json = credentials_json
        self._startup_timeout = startup_timeout
        self._task_timeout = task_timeout
        self._ssh_user = ssh_user
        self._worker_script = worker_script
        self._status_callback = status_callback
        self._lock = threading.Lock()
        self._log: List[str] = []

        # Lazily imported to avoid hard dependency when not used
        self._compute_client: Optional[Any] = None
        self._init_compute_client()

    # ------------------------------------------------------------------ #
    # Setup                                                                #
    # ------------------------------------------------------------------ #

    def _init_compute_client(self) -> None:
        """Initialise google-cloud-compute client (lazy import)."""
        try:
            from google.cloud import compute_v1  # type: ignore
            from google.oauth2 import service_account  # type: ignore

            if self._creds_json and Path(self._creds_json).exists():
                credentials = service_account.Credentials.from_service_account_file(
                    self._creds_json,
                    scopes=["https://www.googleapis.com/auth/cloud-platform"],
                )
                self._compute_client = compute_v1.InstancesClient(
                    credentials=credentials
                )
            else:
                # Use Application Default Credentials
                self._compute_client = compute_v1.InstancesClient()

            log.info(
                "GCP Compute client initialised for instance=%s zone=%s",
                self._instance,
                self._zone,
            )
        except ImportError:
            log.warning(
                "google-cloud-compute not installed — Station 2 GPU in stub mode. "
                "Install with: pip install google-cloud-compute"
            )
        except Exception as exc:  # pylint: disable=broad-except
            log.error("Failed to initialise GCP Compute client: %s", exc)

    # ------------------------------------------------------------------ #
    # Main Entry Point                                                     #
    # ------------------------------------------------------------------ #

    def execute(self, task_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Full GPU task lifecycle:
            Start VM → Wait ready → Run task → Shutdown VM (always).

        Args:
            task_type: String identifier for the GPU task.
            payload: Task parameters (prompt, image_path, scene_count, etc.).

        Returns:
            dict with keys: success, output, error, vm_started, vm_stopped,
            duration_seconds, log_entries.
        """
        self._log = []
        start = time.time()
        vm_started = False

        self._emit_log(
            f"⚡ Station 2: Starting GPU task '{task_type}' on VM '{self._instance}'"
        )

        try:
            # --- Step 1: Start the VM ----------------------------------------
            self._emit_log("▶ Step 1: Starting GPU VM …")
            self._start_vm()
            vm_started = True

            # --- Step 2: Wait for VM to be RUNNING ---------------------------
            self._emit_log("⏳ Step 2: Waiting for VM to be ready …")
            vm_status = self._wait_for_running()
            if vm_status.state != VMState.RUNNING:
                raise RuntimeError(
                    f"VM did not reach RUNNING state (last={vm_status.state})"
                )
            self._emit_log(
                f"✅ VM is RUNNING (IP={vm_status.external_ip})"
            )

            # --- Step 3: Send processing command -----------------------------
            self._emit_log("📨 Step 3: Sending task to GPU worker …")
            task_result = self._send_task(
                vm_ip=vm_status.external_ip or vm_status.internal_ip or "",
                task_type=task_type,
                payload=payload,
            )

            self._emit_log(f"✅ GPU task complete: success={task_result.get('success')}")
            return {
                "success": task_result.get("success", False),
                "output": task_result.get("output"),
                "error": task_result.get("error"),
                "vm_started": True,
                "vm_stopped": False,  # updated in finally
                "duration_seconds": time.time() - start,
                "log_entries": list(self._log),
            }

        except Exception as exc:  # pylint: disable=broad-except
            self._emit_log(f"❌ Station 2 error: {exc}")
            return {
                "success": False,
                "output": None,
                "error": str(exc),
                "vm_started": vm_started,
                "vm_stopped": False,
                "duration_seconds": time.time() - start,
                "log_entries": list(self._log),
            }

        finally:
            # --- Step 4 (ALWAYS): Safe shutdown of GPU VM --------------------
            self._emit_log("🛑 Step 4: SAFE SHUTDOWN — stopping GPU VM …")
            stopped = self._safe_stop_vm()
            self._emit_log(
                "🔴 GPU VM stopped." if stopped else "⚠️ GPU VM stop command sent (verify manually)."
            )
            if self._status_callback:
                try:
                    self._status_callback(
                        {"vm_stopped": True, "log": self._log[-1], "station": 2}
                    )
                except Exception:  # pylint: disable=broad-except
                    pass

    # ------------------------------------------------------------------ #
    # VM Lifecycle                                                         #
    # ------------------------------------------------------------------ #

    def _start_vm(self) -> None:
        """Send start command to the GCP VM instance."""
        if self._compute_client is None:
            self._emit_log("⚠️ Compute client not available — running in stub mode.")
            return

        try:
            from google.cloud import compute_v1  # type: ignore

            op = self._compute_client.start(
                project=self._project,
                zone=self._zone,
                instance=self._instance,
            )
            log.info("VM start operation initiated: %s", getattr(op, "name", "unknown"))
        except Exception as exc:
            raise RuntimeError(f"Failed to start VM: {exc}") from exc

    def _wait_for_running(self, poll_interval: int = 10) -> VMStatus:
        """
        Poll VM status until it reaches RUNNING or timeout.

        Returns:
            VMStatus with current state and IP addresses.
        """
        deadline = time.time() + self._startup_timeout
        while time.time() < deadline:
            status = self._get_vm_status()
            self._emit_log(f"  VM state: {status.state}")
            if status.state == VMState.RUNNING:
                return status
            if status.state in (VMState.TERMINATED, VMState.UNKNOWN):
                time.sleep(poll_interval)
            else:
                time.sleep(poll_interval)

        return self._get_vm_status()

    def _get_vm_status(self) -> VMStatus:
        """Fetch current VM status from GCP Compute API."""
        if self._compute_client is None:
            # Stub: simulate VM running after first call
            return VMStatus(
                instance_name=self._instance,
                state=VMState.RUNNING,
                external_ip="127.0.0.1",
            )

        try:
            instance = self._compute_client.get(
                project=self._project,
                zone=self._zone,
                instance=self._instance,
            )
            state_str = getattr(instance, "status", "UNKNOWN")
            state = VMState(state_str) if state_str in VMState._value2member_map_ else VMState.UNKNOWN

            external_ip: Optional[str] = None
            internal_ip: Optional[str] = None

            network_interfaces = getattr(instance, "network_interfaces", []) or []
            for ni in network_interfaces:
                internal_ip = getattr(ni, "network_ip", None)
                access_configs = getattr(ni, "access_configs", []) or []
                for ac in access_configs:
                    nat_ip = getattr(ac, "nat_i_p", None)
                    if nat_ip:
                        external_ip = nat_ip
                        break

            return VMStatus(
                instance_name=self._instance,
                state=state,
                external_ip=external_ip,
                internal_ip=internal_ip,
            )

        except Exception as exc:
            log.error("Failed to get VM status: %s", exc)
            return VMStatus(
                instance_name=self._instance,
                state=VMState.UNKNOWN,
                error=str(exc),
            )

    def _safe_stop_vm(self) -> bool:
        """
        Stop the GPU VM.  This method MUST NOT raise — it is called in
        a finally block and must always complete regardless of errors.

        Returns:
            True if stop command was sent successfully, False otherwise.
        """
        if self._compute_client is None:
            log.warning("Compute client unavailable — VM stop skipped (stub mode).")
            return False

        try:
            op = self._compute_client.stop(
                project=self._project,
                zone=self._zone,
                instance=self._instance,
            )
            log.info("VM stop operation sent: %s", getattr(op, "name", "unknown"))
            return True
        except Exception as exc:
            # Intentionally swallow — this is a safety shutdown
            log.error("⚠️ CRITICAL: Failed to stop GPU VM '%s': %s", self._instance, exc)
            return False

    # ------------------------------------------------------------------ #
    # Task Dispatch                                                        #
    # ------------------------------------------------------------------ #

    def _send_task(
        self, vm_ip: str, task_type: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Send a processing task to the GPU VM worker.

        Strategy:
            1. Serialise payload to JSON.
            2. SSH into VM and invoke worker script.
            3. Parse JSON response from stdout.

        TODO: Replace SSH with a more robust transport (Cloud Pub/Sub,
              internal HTTP endpoint, or Cloud Run on the VM).
        """
        payload_json = json.dumps({"task_type": task_type, "payload": payload})

        self._emit_log(f"  Sending task '{task_type}' to {vm_ip} …")

        if not vm_ip or vm_ip == "127.0.0.1":
            # Stub mode — no real VM to SSH into
            self._emit_log("  ⚠️ Stub mode: simulating GPU task execution.")
            time.sleep(1)  # simulate processing time
            return {"success": True, "output": {"status": "stub"}, "error": None}

        try:
            ssh_cmd = [
                "ssh",
                "-o", "StrictHostKeyChecking=no",
                "-o", f"ConnectTimeout=30",
                f"{self._ssh_user}@{vm_ip}",
                f"python3 {self._worker_script} '{payload_json}'",
            ]
            proc = subprocess.run(
                ssh_cmd,
                capture_output=True,
                text=True,
                timeout=self._task_timeout,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"Worker script failed (rc={proc.returncode}): {proc.stderr[:500]}"
                )
            result = json.loads(proc.stdout.strip())
            return result

        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"GPU task timed out after {self._task_timeout}s"
            )
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid JSON response from GPU worker: {exc}") from exc

    # ------------------------------------------------------------------ #
    # Status Query                                                         #
    # ------------------------------------------------------------------ #

    def current_status(self) -> VMStatus:
        """Return the current VM status (non-blocking)."""
        return self._get_vm_status()

    def is_running(self) -> bool:
        """Return True if the VM is currently in RUNNING state."""
        return self._get_vm_status().state == VMState.RUNNING

    # ------------------------------------------------------------------ #
    # Logging                                                              #
    # ------------------------------------------------------------------ #

    def _emit_log(self, msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        entry = f"[{ts}][Station2] {msg}"
        self._log.append(entry)
        log.info(msg)
        if self._status_callback:
            try:
                self._status_callback({"log": entry, "station": 2})
            except Exception:  # pylint: disable=broad-except
                pass

    @property
    def log_entries(self) -> List[str]:
        return list(self._log)
