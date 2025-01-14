from typing import Optional
from pydantic_config import BaseConfig

class ACCOConfig(BaseConfig):
    theta_t_device: Optional[str] = None
