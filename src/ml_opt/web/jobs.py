"""One prewarmed solver process, with isolated runs and bounded cancellation."""

from dataclasses import asdict, dataclass, field
import multiprocessing as mp
import queue
import threading
import time
import uuid

from ..config import CantileverConfig
from .worker import worker_loop


class _Mailbox:
    """Ignore late messages from another run when a worker is reused."""

    def __init__(self, messages, run_id):
        self._messages = messages
        self._run_id = run_id
        self.completed = False

    def get(self, block=True, timeout=None):
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            remaining = None if deadline is None else max(0, deadline - time.monotonic())
            run_id, event = self._messages.get(block, remaining)
            if run_id == self._run_id:
                if event["event"] in ("complete", "error"):
                    self.completed = True
                return event


@dataclass
class Job:
    id: str
    process: object
    messages: _Mailbox
    cancel: object
    kind: str = "optimisation"
    cleanup_lock: object = field(default_factory=threading.Lock)
    closed: bool = False


class RunManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._process = self._commands = self._messages = self._cancel = self._ready = None
        self.active: Job | None = None

    def _ensure_worker_locked(self):
        if self._process is not None and self._process.is_alive():
            return
        self._retire_worker_locked()
        context = mp.get_context("spawn")
        self._commands, self._messages = context.Queue(), context.Queue()
        self._cancel, self._ready = context.Event(), context.Event()
        # PERF B1: import the numerical stack once, before the first Run click.
        self._process = context.Process(target=worker_loop,
            args=(self._commands, self._messages, self._cancel, self._ready), daemon=True)
        try:
            self._process.start()
        except BaseException:
            for channel in (self._commands, self._messages):
                channel.close()
                channel.cancel_join_thread()
            self._process = self._commands = self._messages = self._cancel = self._ready = None
            raise

    def warmup(self) -> None:
        with self._lock:
            self._ensure_worker_locked()

    @property
    def worker_ready(self) -> bool:
        with self._lock:
            return (self._process is not None and self._process.is_alive()
                    and self._ready.is_set())

    def start(self, config: CantileverConfig) -> Job:
        return self._start(asdict(config), "optimisation")

    def start_task(self, task: str, payload: dict) -> Job:
        if task not in ("rom_snapshots", "rom_basis"):
            raise ValueError("Unknown training task")
        return self._start({"task": task, **payload}, task)

    def _start(self, configuration: dict, kind: str) -> Job:
        with self._lock:
            if self.active is not None:
                raise RuntimeError("A solver job is already running. Stop it before starting another.")
            self._ensure_worker_locked()
            self._cancel.clear()
            run_id = uuid.uuid4().hex
            job = Job(run_id, self._process, _Mailbox(self._messages, run_id), self._cancel, kind)
            self._commands.put((run_id, configuration))
            self.active = job
            return job

    def stop(self, job_id: str) -> bool:
        with self._lock:
            if self.active is None or self.active.id != job_id:
                return False
            self.active.cancel.set()
            return True

    def finish(self, job: Job) -> None:
        with job.cleanup_lock:
            if job.closed:
                return
            job.cancel.set()
            # Drain a disconnected client's run before clearing its cancel flag.
            deadline = time.monotonic() + 3
            while not job.messages.completed and job.process.is_alive():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    job.messages.get(timeout=min(0.1, remaining))
                except queue.Empty:
                    pass
            with self._lock:
                if self.active is job:
                    if not job.messages.completed or not job.process.is_alive():
                        self._retire_worker_locked()
                    self.active = None
                job.closed = True

    def _retire_worker_locked(self) -> None:
        process = self._process
        if process is None:
            return
        self._cancel.set()
        if process.is_alive():
            self._commands.put(None)
            process.join(timeout=3)
            if process.is_alive():
                process.terminate()
                process.join(timeout=3)
        else:
            process.join()
        if process.is_alive():
            raise RuntimeError("The solver worker could not be stopped")
        for channel in (self._commands, self._messages):
            channel.close()
            channel.cancel_join_thread()
        process.close()
        self._process = self._commands = self._messages = self._cancel = self._ready = None

    def shutdown(self) -> None:
        if self.active is not None:
            self.finish(self.active)
        with self._lock:
            self._retire_worker_locked()
