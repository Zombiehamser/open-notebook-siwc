"""Bounded command lanes; persistence must not queue behind model inference."""

import asyncio
import os
import signal

from loguru import logger

LIGHT_COMMANDS = ("create_insight", "embed_insight", "embed_source", "embed_note")


async def lane(fetch, execute, stop):
    """Fetch only when this lane can execute; drain active work on shutdown."""
    while not stop.is_set():
        try:
            item = await fetch()
            if item:
                await execute(item)
            else:
                try:
                    await asyncio.wait_for(stop.wait(), timeout=0.5)
                except TimeoutError:
                    pass
        except Exception:
            # Never log command arguments or generated document content.
            logger.exception("Command lane failed; persisted job state retained")
            try:
                await asyncio.wait_for(stop.wait(), timeout=1)
            except TimeoutError:
                pass


async def run():
    from surreal_commands.core.service import command_service

    import commands  # noqa: F401 -- registers the existing command implementations
    from open_notebook.database.repository import ensure_record_id, repo_query

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)

    async def fetch(light):
        operator = "IN" if light else "NOT IN"
        rows = await repo_query(
            "SELECT * FROM command WHERE app = 'open_notebook' "
            f"AND status = 'new' AND name {operator} $names ORDER BY created ASC LIMIT 1",
            {"names": list(LIGHT_COMMANDS)},
        )
        return rows[0] if rows else None

    async def execute(item):
        await command_service.execute_command(
            ensure_record_id(str(item["id"])),
            item["app"] + "." + item["name"],
            item["args"],
            item.get("context"),
        )

    await asyncio.gather(
        lane(lambda: fetch(True), execute, stop),
        lane(lambda: fetch(False), execute, stop),
    )


def main():
    import fcntl

    os.environ["OPEN_NOTEBOOK_PROCESS_ROLE"] = "worker"
    # One owner of both lanes. A second container must not execute the same job.
    with open("/app/data/command-worker.lock", "a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        asyncio.run(run())


if __name__ == "__main__":
    main()
