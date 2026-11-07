"""Regression: a model call cannot delay persisted results or graceful drain."""

import asyncio
import unittest

from open_notebook.queue_worker import lane


class QueueLanesTest(unittest.IsolatedAsyncioTestCase):
    def test_ask_rejects_unbounded_search_fanout(self):
        from pydantic import ValidationError

        from open_notebook.graphs.ask import Strategy

        with self.assertRaises(ValidationError):
            Strategy(
                reasoning="synthetic",
                searches=[{"term": "test", "instructions": "test"}] * 6,
            )

    async def test_persistence_and_shutdown_while_model_is_busy(self):
        stop, started, release, saved = (asyncio.Event() for _ in range(4))
        heavy = ["model", "must-remain-queued"]
        light = ["save"]
        seen = []

        async def fetch(queue):
            return queue.pop(0) if queue else None

        async def execute(item):
            seen.append(item)
            if item == "model":
                started.set()
                await release.wait()
            else:
                saved.set()

        a = asyncio.create_task(lane(lambda: fetch(heavy), execute, stop))
        await asyncio.wait_for(started.wait(), 1)
        b = asyncio.create_task(lane(lambda: fetch(light), execute, stop))
        await asyncio.wait_for(saved.wait(), 1)
        stop.set()
        self.assertFalse(a.done())
        release.set()
        await asyncio.wait_for(asyncio.gather(a, b), 1)
        self.assertEqual(seen, ["model", "save"])
        self.assertEqual(heavy, ["must-remain-queued"])


if __name__ == "__main__":
    unittest.main()
