import multiprocessing
import os
import subprocess

import shutil

from zeroband.utils.world_info import get_world_info


def _get_cut_dirs_from_url(url: str) -> int:
    return len(url.rstrip().partition("//")[-1].split("/"))


def _wget(source: str, destination: str) -> None:
    logger = multiprocessing.get_logger()
    cmd = f"wget -P {destination} {source}"

    if shutil.which("wget") is None:
        raise RuntimeError("wget is required but not found. Please install wget and try again.")

    try:
        subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error output: {e.stderr}")
        # print(f"Error output: {e.stderr}")
        raise e


def wget(source: str, destination: str):
    # List of files to download

    os.makedirs(destination, exist_ok=True)

    files = [".metadata"]

    world_info = get_world_info()
    for i in range(world_info.local_world_size):
        files.extend([f"__{i}_0.distcp", f"__{i}_0.pt"])

    processes = []
    for file in files:
        src = f"{source}/{file}"
        process = multiprocessing.Process(target=_wget, args=(src, destination))
        process.start()
        processes.append(process)

    for process in processes:
        process.join()
