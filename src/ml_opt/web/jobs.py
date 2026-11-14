"""A single active solver process, with cancellation and owned-process cleanup."""

from dataclasses import asdict, dataclass, field
import multiprocessing as mp
import threading
import uuid

from ..config import CantileverConfig
from .worker import run_optimization


@dataclass
class Job:
    id: str
    process: object
    messages: object
    cancel: object
    cleanup_lock: object = field(default_factory=threading.Lock)
    closed: bool = False


class RunManager:
    def __init__(self):
        self._lock = threading.Lock()
        self.active: Job | None = None

    def start(self, config: CantileverConfig) -> Job:
        with self._lock:
            if self.active is not None:
                raise RuntimeError("An optimisation is already running. Stop it before starting another.")
            context = mp.get_context("spawn")
            messages, cancel = context.Queue(), context.Event()
            process = context.Process(target=run_optimization,
                                       args=(asdict(config), messages, cancel), daemon=True)
            job = Job(uuid.uuid4().hex, process, messages, cancel)
            try:
                process.start()
            except Exception:
                messages.close()
                raise
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
            job.process.join(timeout=3)
            if job.process.is_alive():
                job.process.terminate()
                job.process.join(timeout=3)
            job.messages.close()
            job.messages.cancel_join_thread()
            job.process.close()
            job.closed = True
            with self._lock:
                if self.active is job:
                    self.active = None

    def shutdown(self) -> None:
        if self.active is not None:
            self.finish(self.active)
