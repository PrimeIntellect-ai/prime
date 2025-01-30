from typing import Any
from zeroband.utils.logger import get_logger
import aiohttp
from aiohttp import ClientError
import asyncio


async def _get_external_ip(max_retries=3, retry_delay=5):
    async with aiohttp.ClientSession() as session:
        for attempt in range(max_retries):
            try:
                async with session.get('https://api.ipify.org', timeout=10) as response:
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

        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def __del__(self):
        self.loop.close()

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

    def _handle_send_batch(self, flush: bool = False):
        if len(self.data) >= self.log_flush_interval or flush:
            self.loop.run_until_complete(self._send_batch())

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

    async def _send_batch(self):
        import aiohttp

        self._remove_duplicates()
        await self._set_node_ip_address()

        batch = self.data[:self.log_flush_interval]
        # set node_ip_address of batch
        batch = [{**log, "node_ip_address": self.node_ip_address} for log in batch]
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.auth_token}"
        }
        payload = {
            "logs": batch
        }
        api = f"{self.base_url}/metrics/{self.run_id}/logs"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(api, json=payload, headers=headers) as response:
                    if response is not None:
                        response.raise_for_status()
        except Exception as e:
            self._logger.error(f"Error sending batch to server: {str(e)}")
            pass

        self.data = self.data[self.log_flush_interval :]
        return True

    async def _finish(self):
        import requests

        # Send any remaining logs
        while self.data:
            await self._send_batch()

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

        self.loop.run_until_complete(self._finish())
