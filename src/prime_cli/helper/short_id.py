import hashlib

from ..api.availability import GPUAvailability


def generate_short_id(gpu_config: GPUAvailability) -> str:
    """Generate a short unique ID for a GPU configuration"""
    location = f"{gpu_config.country or 'N/A'} - {gpu_config.data_center or 'N/A'}"
    config_str = (
        f"{gpu_config.cloud_id}-"
        f"{gpu_config.gpu_type}-"
        f"{gpu_config.socket or 'N/A'}-"
        f"{location}-"
        f"{gpu_config.provider or 'N/A'}-"
        f"{gpu_config.memory.default_count}-"
        f"{gpu_config.vcpu.default_count}"
    )
    return hashlib.md5(config_str.encode()).hexdigest()[:6]
