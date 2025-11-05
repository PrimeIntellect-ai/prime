"""Value formatting utilities."""

from typing import Any, Dict, List, Optional, Union


def obfuscate_env_vars(env_vars: Dict[str, Any]) -> Dict[str, str]:
    """Obfuscate environment variable values for display."""
    obfuscated: Dict[str, str] = {}
    for key, value in env_vars.items():
        str_value = str(value)  # Ensure value is a string
        if len(str_value) <= 3:
            obfuscated[key] = "*" * len(str_value)
        else:
            obfuscated[key] = str_value[:2] + "*" * (len(str_value) - 4) + str_value[-2:]
    return obfuscated


def obfuscate_secrets(secrets: Dict[str, Any]) -> Dict[str, str]:
    """Obfuscate secret values completely for display (keys only, all values as ***)."""
    obfuscated: Dict[str, str] = {}
    for key in secrets.keys():
        obfuscated[key] = "***"
    return obfuscated


def format_ip_display(ip: Optional[Union[str, List[str]]]) -> str:
    """Format IP address(es) for display, handling both single and list cases."""
    if not ip:
        return "N/A"
    if isinstance(ip, list):
        return ", ".join(ip) if ip else "N/A"
    return str(ip)


def format_price(value: float) -> str:
    """Format price value as currency string."""
    if value == float("inf"):
        return "N/A"
    return f"${value:.2f}"


def format_resources(cpu_cores: int, memory_gb: int, gpu_count: int = 0) -> str:
    """Format resource specifications as compact string."""
    resources = f"{cpu_cores}CPU/{memory_gb}GB"
    if gpu_count > 0:
        resources += f"/{gpu_count}GPU"
    return resources


def format_gpu_spec(gpu_type: str, gpu_count: int) -> str:
    """Format GPU specification as 'Type x Count'."""
    return f"{gpu_type} x{gpu_count}"
