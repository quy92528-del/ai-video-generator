"""
parallel_processor.py - Task queue and parallel worker management.

Uses Python's concurrent.futures for in-process parallelism,
with optional Celery / Redis support for distributed workloads.
"""

from __future__ import annotations

import time
import uuid
from concurrent.futures import (
    Future,
    ThreadPoolExecutor,
    as_completed,
)
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TypeVar

from utils.logger import get_logger
from utils.helpers import generate_job_id

log = get_logger(__name__)

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Enums & Data Models
# ---------------------------------------------------------------------------

class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Task:
    """A single unit of work submitted to the processor."""
    task_id: str = field(default_factory=lambda: generate_job_id("task"))
    fn: Optional[Callable[..., Any]] = None
    args: tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None

    @property
    def elapsed(self) -> float:
        if self.started_at is None:
            return 0.0
        end = self.finished_at or time.time()
        return end - self.started_at


@dataclass
class BatchJob:
    """A collection of tasks representing one batch of video generation."""
    job_id: str = field(default_factory=lambda: generate_job_id("batch"))
    tasks: List[Task] = field(default_factory=list)
    total: int = 0
    completed: int = 0
    failed: int = 0
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None

    @property
    def progress(self) -> float:
        """Return completion progress as a value in [0.0, 1.0]."""
        if self.total == 0:
            return 0.0
        return (self.completed + self.failed) / self.total

    @property
    def is_done(self) -> bool:
        return (self.completed + self.failed) >= self.total


# ---------------------------------------------------------------------------
# ParallelProcessor
# ---------------------------------------------------------------------------

class ParallelProcessor:
    """Thread-pool based parallel task processor with progress tracking.

    Args:
        max_workers: Maximum number of parallel worker threads.
        use_celery: Whether to attempt Celery-based distributed execution
                    (falls back to thread pool if Redis is unavailable).
        broker_url: Celery broker URL (Redis).
        result_backend: Celery result backend URL.
    """

    def __init__(
        self,
        max_workers: int = 10,
        use_celery: bool = False,
        broker_url: str = "redis://localhost:6379/0",
        result_backend: str = "redis://localhost:6379/1",
    ) -> None:
        self._max_workers = max_workers
        self._use_celery = use_celery
        self._broker = broker_url
        self._backend = result_backend
        self._celery_app: Any = None
        self._executor: Optional[ThreadPoolExecutor] = None
        self._executor_shutdown: bool = False
        self._jobs: Dict[str, BatchJob] = {}

        if use_celery:
            self._init_celery()

    # ------------------------------------------------------------------ #
    # Setup                                                                #
    # ------------------------------------------------------------------ #

    def _init_celery(self) -> None:
        try:
            from celery import Celery  # type: ignore

            self._celery_app = Celery(
                "ai_video_generator",
                broker=self._broker,
                backend=self._backend,
            )
            self._celery_app.conf.update(
                task_serializer="json",
                result_serializer="json",
                accept_content=["json"],
                worker_prefetch_multiplier=1,
                task_acks_late=True,
                worker_concurrency=self._max_workers,
            )
            log.info("Celery initialised (broker=%s).", self._broker)
        except ImportError:
            log.warning(
                "celery package not installed – using thread pool only."
            )
            self._use_celery = False
        except Exception as exc:  # pylint: disable=broad-except
            log.warning("Celery init failed (%s) – using thread pool.", exc)
            self._use_celery = False

    def _get_executor(self) -> ThreadPoolExecutor:
        if self._executor is None or self._executor_shutdown:
            self._executor = ThreadPoolExecutor(max_workers=self._max_workers)
            self._executor_shutdown = False
        return self._executor

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def submit_batch(
        self,
        fn: Callable[..., T],
        items: List[Any],
        batch_id: Optional[str] = None,
        on_progress: Optional[Callable[[BatchJob], None]] = None,
    ) -> BatchJob:
        """Submit a list of items for parallel processing.

        Each item is passed as the first positional argument to *fn*.

        Args:
            fn: Worker function, called with one item at a time.
            items: List of work items.
            batch_id: Optional identifier (auto-generated if not provided).
            on_progress: Optional callback invoked after each task completes.

        Returns:
            :class:`BatchJob` instance with final task results.
        """
        job = BatchJob(
            job_id=batch_id or generate_job_id("batch"),
            total=len(items),
        )
        self._jobs[job.job_id] = job

        executor = self._get_executor()
        future_to_task: Dict[Future[Any], Task] = {}

        for item in items:
            task = Task(fn=fn, args=(item,))
            job.tasks.append(task)
            future = executor.submit(self._run_task, task)
            future_to_task[future] = task

        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                task.result = future.result()
                task.status = TaskStatus.COMPLETED
                job.completed += 1
                log.debug(
                    "Task %s completed in %.1fs.", task.task_id, task.elapsed
                )
            except Exception as exc:  # pylint: disable=broad-except
                task.status = TaskStatus.FAILED
                task.error = str(exc)
                job.failed += 1
                log.error("Task %s failed: %s", task.task_id, exc)

            if on_progress:
                try:
                    on_progress(job)
                except Exception:  # pylint: disable=broad-except
                    pass

        job.finished_at = time.time()
        elapsed = job.finished_at - job.started_at
        log.info(
            "Batch %s done: %d OK / %d failed in %.1fs.",
            job.job_id,
            job.completed,
            job.failed,
            elapsed,
        )
        return job

    def submit_task(
        self, fn: Callable[..., T], *args: Any, **kwargs: Any
    ) -> Task:
        """Submit a single task for asynchronous execution.

        Args:
            fn: Callable to execute.
            *args: Positional arguments for *fn*.
            **kwargs: Keyword arguments for *fn*.

        Returns:
            :class:`Task` instance.  The task will be updated in-place
            once execution completes.
        """
        task = Task(fn=fn, args=args, kwargs=kwargs)
        executor = self._get_executor()

        def _done_callback(future: Future[Any]) -> None:
            try:
                task.result = future.result()
                task.status = TaskStatus.COMPLETED
            except Exception as exc:  # pylint: disable=broad-except
                task.status = TaskStatus.FAILED
                task.error = str(exc)
            task.finished_at = time.time()

        future = executor.submit(self._run_task, task)
        future.add_done_callback(_done_callback)
        return task

    def get_job(self, job_id: str) -> Optional[BatchJob]:
        """Return a :class:`BatchJob` by ID.

        Args:
            job_id: Job identifier.

        Returns:
            :class:`BatchJob` or ``None`` if not found.
        """
        return self._jobs.get(job_id)

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the thread pool executor.

        Args:
            wait: Whether to wait for pending tasks to finish.
        """
        if self._executor:
            self._executor.shutdown(wait=wait)
            self._executor_shutdown = True
            log.info("Thread pool executor shut down.")

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _run_task(task: Task) -> Any:
        """Execute *task.fn* and record timing."""
        task.status = TaskStatus.RUNNING
        task.started_at = time.time()
        assert task.fn is not None, "Task has no function."
        result = task.fn(*task.args, **task.kwargs)
        task.finished_at = time.time()
        return result

    def __enter__(self) -> "ParallelProcessor":
        return self

    def __exit__(self, *_: Any) -> None:
        self.shutdown()
