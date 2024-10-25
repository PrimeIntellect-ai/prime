from collections import deque
from typing import Any
from zeroband.utils.logging import get_logger
import aiohttp
from aiohttp import ClientError
import asyncio


async def _get_external_ip(max_retries=3, retry_delay=5):
    async with aiohttp.ClientSession() as session:
        for attempt in range(max_retries):
            try:
                async with session.get("https://api.ipify.org", timeout=10) as response:
                    response.raise_for_status()
                    return await response.text()
            except ClientError:
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
    return None


class HttpMonitor:
    """
    Logs the status of nodes, and training progress to an API
    """

    def __init__(self, config, *args, **kwargs):
        self.data = []
        self.log_flush_interval = config["monitor"]["log_flush_interval"]
        self.base_url = config["monitor"]["base_url"]
        self.auth_token = config["monitor"]["auth_token"]

        self._logger = get_logger()

        self.run_id = config.get("run_id", None)
        if self.run_id is None:
            raise ValueError("run_id must be set for HttpMonitor")

        self.node_ip_address = None
        self.node_ip_address_fetch_status = None

        # Create event loop only if one doesn't exist
        try:
            self.loop = asyncio.get_event_loop()
        except RuntimeError:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

        self._pending_tasks = deque()
        

    def _remove_duplicates(self):
        seen = set()
        unique_logs = []
        for log in self.data:
            log_tuple = tuple(sorted(log.items()))
            if log_tuple not in seen:
                unique_logs.append(log)
                seen.add(log_tuple)
        self.data = unique_logs

    def set_stage(self, stage: str):
        import time

        # add a new log entry with the stage name
        self.data.append({"stage": stage, "time": time.time()})
        self._handle_send_batch(flush=True)  # it's useful to have the most up-to-date stage broadcasted

    def log(self, data: dict[str, Any]):
        # Lowercase the keys in the data dictionary
        lowercased_data = {k.lower(): v for k, v in data.items()}
        self.data.append(lowercased_data)

        self._handle_send_batch()

    def __del__(self):
        # Ensure all pending tasks are completed before closing
        if hasattr(self, "loop") and self.loop is not None:
            try:
                pending = asyncio.all_tasks(self.loop)
                self.loop.run_until_complete(asyncio.gather(*pending))
            except Exception as e:
                self._logger.error(f"Error cleaning up pending tasks: {str(e)}")
            finally:
                self.loop.close()

    def _cleanup_completed_tasks(self):
        """Remove completed tasks from the pending tasks queue"""
        while self._pending_tasks and self._pending_tasks[0].done():
            task = self._pending_tasks.popleft()
            try:
                task.result()  # This will raise any exceptions that occurred
            except Exception as e:
                self._logger.error(f"Error in completed batch send task: {str(e)}")

    def _handle_send_batch(self, flush: bool = False):
        self._cleanup_completed_tasks()

        if len(self.data) >= self.log_flush_interval or flush:
            batch = self.data[: self.log_flush_interval]
            self.data = self.data[self.log_flush_interval :]
            
            if self.loop.is_running():
                # If we're already in an event loop, create a task
                task = self.loop.create_task(self._send_batch(batch))
                self._pending_tasks.append(task)
            else:
                # If we're not in an event loop, run it directly
                self.loop.run_until_complete(self._send_batch(batch))

    async def _set_node_ip_address(self):
        if self.node_ip_address is None and self.node_ip_address_fetch_status != "failed":
            ip_address = await _get_external_ip()
            if ip_address is None:
                self._logger.error("Failed to get external IP address")
                # set this to "failed" so we keep trying again
                self.node_ip_address_fetch_status = "failed"
            else:
                self.node_ip_address = ip_address
                self.node_ip_address_fetch_status = "success"

    async def _send_batch(self, batch):
        import aiohttp

        self._remove_duplicates()
        await self._set_node_ip_address()

        # set node_ip_address of batch
        batch = [{**log, "node_ip_address": self.node_ip_address} for log in batch]
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.auth_token}"}
        payload = {"logs": batch}
        api = f"{self.base_url}/metrics/{self.run_id}/logs"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(api, json=payload, headers=headers) as response:
                    if response is not None:
                        response.raise_for_status()
                    self._logger.info(f"Sent {len(batch)} logs to server")
        except Exception as e:
            self._logger.error(f"Error sending batch to server: {str(e)}")
            return False

        return True

    async def _finish(self):
        import requests

        # Send any remaining logs
        while self.data:
            batch = self.data
            self.data = []
            await self._send_batch(batch)

        headers = {"Content-Type": "application/json"}
        api = f"{self.base_url}/metrics/{self.run_id}/finish"
        try:
            response = requests.post(api, headers=headers)
            response.raise_for_status()
            return True
        except requests.RequestException as e:
            self._logger.debug(f"Failed to send finish signal to http monitor: {e}")
            return False

    def finish(self):
        self.set_stage("finishing")

        # Clean up any remaining tasks
        pending = asyncio.all_tasks(self.loop)
        self.loop.run_until_complete(asyncio.gather(*pending))
