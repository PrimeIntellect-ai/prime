#!/usr/bin/env python3

import asyncio
import time

import httpx
from prime_cli.api.sandbox import AsyncSandboxClient, CreateSandboxRequest
from tqdm.asyncio import tqdm


async def execute_commands(
    sandbox_id: str, commands: list[str], auth: dict, pbar: tqdm, semaphore: asyncio.Semaphore
) -> tuple[int, list[float]]:
    gateway_url = auth["gateway_url"].rstrip("/")
    url = f"{gateway_url}/{auth['user_ns']}/{auth['job_id']}/exec"
    headers = {"Authorization": f"Bearer {auth['token']}"}

    successful = 0
    execution_times = []

    async with httpx.AsyncClient(timeout=30.0) as http_client:
        tasks = []
        for command in commands:
            task = execute_single(http_client, url, headers, command, sandbox_id, pbar, semaphore)
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, tuple):
                success, exec_time = result
                if success:
                    successful += 1
                    execution_times.append(exec_time)
            elif isinstance(result, Exception):
                print(f"Error in sandbox {sandbox_id}: {result}")

    return successful, execution_times


async def execute_single(
    http_client: httpx.AsyncClient,
    url: str,
    headers: dict,
    command: str,
    sandbox_id: str,
    pbar: tqdm,
    semaphore: asyncio.Semaphore,
) -> tuple[bool, float]:
    async with semaphore:
        payload: dict = {
            "command": command,
            "working_dir": None,
            "env": {},
            "sandbox_id": sandbox_id,
        }

        start = time.time()
        try:
            response = await http_client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            pbar.update(1)
            return True, time.time() - start
        except Exception as e:
            print(f"Error executing command '{command}' on sandbox {sandbox_id}: {e}")
            pbar.update(1)
            return False, 0


async def main() -> None:
    async with AsyncSandboxClient() as client:
        num_sandboxes = 50
        commands_per_sandbox = 1000
        total_commands = num_sandboxes * commands_per_sandbox

        print(f"Sandboxes: {num_sandboxes}")
        print(f"Commands per sandbox: {commands_per_sandbox}")
        print(f"Total commands: {total_commands}")
        print("Target rate: 2000 req/min\n")

        # Create sandboxes
        print("Creating sandboxes...")
        create_tasks = []
        for i in range(num_sandboxes):
            create_tasks.append(
                client.create(
                    CreateSandboxRequest(
                        name=f"sandbox-{i}",
                        docker_image="python:3.11-slim",
                        start_command="tail -f /dev/null",
                        cpu_cores=1,
                        memory_gb=1,
                        timeout_minutes=10,
                    )
                )
            )
        sandboxes = await asyncio.gather(*create_tasks)
        print(f"Created {len(sandboxes)} sandboxes\n")

        # Wait for sandboxes using bulk wait function
        print("Waiting for sandboxes...")
        sandbox_ids = [s.id for s in sandboxes]
        await client.bulk_wait_for_creation(sandbox_ids)
        print("Sandboxes ready\n")

        # Get auth for all sandboxes
        print("Authenticating...")
        auth_tasks = []
        for sandbox in sandboxes:
            auth_tasks.append(client.client.request("POST", f"/sandbox/{sandbox.id}/auth"))
        auth_responses = await asyncio.gather(*auth_tasks)
        auth_map = {sandboxes[i].id: auth_responses[i] for i in range(len(sandboxes))}

        # Generate commands
        sandbox_commands = {}
        for sandbox in sandboxes:
            cmds = []
            for i in range(commands_per_sandbox):
                if i % 2 == 0:
                    cmds.append(f"echo {i}")
                else:
                    cmds.append(f"python -c 'print({i}*2)'")
            sandbox_commands[sandbox.id] = cmds

        # Rate limiting: 33 concurrent requests for ~2000/min
        semaphore = asyncio.Semaphore(33)

        print(f"Executing {total_commands} commands\n")

        start_time = time.time()

        with tqdm(total=total_commands, desc="Progress", unit="cmd") as pbar:
            tasks = []
            for sandbox in sandboxes:
                tasks.append(
                    execute_commands(
                        sandbox.id,
                        sandbox_commands[sandbox.id],
                        auth_map[sandbox.id],
                        pbar,
                        semaphore,
                    )
                )

            results = await asyncio.gather(*tasks)

        duration = time.time() - start_time

        # Statistics
        total_successful = sum(r[0] for r in results)
        all_exec_times = []
        for _, exec_times in results:
            all_exec_times.extend(exec_times)

        avg_exec_time = sum(all_exec_times) / len(all_exec_times) if all_exec_times else 0

        print("\nResults:")
        print(f"  Successful: {total_successful}/{total_commands}")
        print(f"  Duration: {duration:.2f}s")
        rate_per_sec = total_successful / duration
        rate_per_min = rate_per_sec * 60
        print(f"  Rate: {rate_per_sec:.1f} req/s ({rate_per_min:.0f} req/min)")
        print(f"  Avg latency: {avg_exec_time * 1000:.1f}ms")
        print(f"  Success rate: {total_successful / total_commands * 100:.1f}%")

        # Cleanup
        print("\nCleaning up...")
        delete_tasks = [client.delete(s.id) for s in sandboxes]
        await asyncio.gather(*delete_tasks)
        print("Done")


if __name__ == "__main__":
    asyncio.run(main())
