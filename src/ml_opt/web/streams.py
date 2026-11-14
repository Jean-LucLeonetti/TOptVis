"""Shared cancellation and lifetime rules for streamed worker jobs."""

import asyncio
import json
import queue

from fastapi.responses import StreamingResponse


def stream_job(manager, job, started):
    async def events():
        try:
            yield json.dumps({"event": "started", "run_id": job.id, **started}) + "\n"
            while True:
                try:
                    event = await asyncio.to_thread(job.messages.get, True, 0.25)
                except queue.Empty:
                    if not job.process.is_alive():
                        yield json.dumps({"event": "error", "message": "The solver process exited unexpectedly."}) + "\n"
                        break
                    continue
                yield json.dumps(event, allow_nan=False) + "\n"
                if event["event"] in ("complete", "error"):
                    break
        finally:
            job.cancel.set()
            await asyncio.shield(asyncio.to_thread(manager.finish, job))

    return StreamingResponse(events(), media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
