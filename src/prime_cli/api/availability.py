from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class DiskConfig(BaseModel):
    min_count: Optional[int] = Field(None, alias="minCount")
    default_count: Optional[int] = Field(None, alias="defaultCount")
    max_count: Optional[int] = Field(None, alias="maxCount")
    price_per_unit: Optional[float] = Field(None, alias="pricePerUnit")
    step: Optional[int]
    default_included_in_price: Optional[bool] = Field(
        None, alias="defaultIncludedInPrice"
    )
    additional_info: Optional[str] = Field(None, alias="additionalInfo")

    model_config = ConfigDict(populate_by_name=True)


class ResourceConfig(BaseModel):
    min_count: Optional[int] = Field(None, alias="minCount")
    default_count: Optional[int] = Field(None, alias="defaultCount")
    max_count: Optional[int] = Field(None, alias="maxCount")
    price_per_unit: Optional[float] = Field(None, alias="pricePerUnit")
    step: Optional[int]
    default_included_in_price: Optional[bool] = Field(
        None, alias="defaultIncludedInPrice"
    )
    additional_info: Optional[str] = Field(None, alias="additionalInfo")

    model_config = ConfigDict(populate_by_name=True)


class Prices(BaseModel):
    on_demand: Optional[float] = Field(None, alias="onDemand")
    community_price: Optional[float] = Field(None, alias="communityPrice")
    is_variable: Optional[bool] = Field(None, alias="isVariable")
    currency: Optional[str]

    model_config = ConfigDict(populate_by_name=True)

    @property
    def price(self) -> float:
        """Returns the price - either on-demand or community price"""
        if self.community_price is not None:
            return self.community_price
        if self.on_demand is not None:
            return self.on_demand
        return float("inf")


class GPUAvailability(BaseModel):
    cloud_id: str = Field(..., alias="cloudId")
    gpu_type: str = Field(..., alias="gpuType")
    socket: Optional[str]
    provider: Optional[str]
    data_center: Optional[str] = Field(None, alias="dataCenter")
    country: Optional[str]
    gpu_count: int = Field(..., alias="gpuCount")
    gpu_memory: int = Field(..., alias="gpuMemory")
    disk: DiskConfig
    vcpu: ResourceConfig
    memory: ResourceConfig
    internet_speed: Optional[float] = Field(None, alias="internetSpeed")
    interconnect: Optional[int]
    interconnect_type: Optional[str] = Field(None, alias="interconnectType")
    provisioning_time: Optional[int] = Field(None, alias="provisioningTime")
    stock_status: str = Field(..., alias="stockStatus")
    security: Optional[str]
    prices: Prices
    images: Optional[List[str]]
    is_spot: Optional[bool] = Field(None, alias="isSpot")
    prepaid_time: Optional[int] = Field(None, alias="prepaidTime")

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    model_config = ConfigDict(populate_by_name=True)


class AvailabilityClient:
    def __init__(self, client: Any) -> None:
        self.client = client

    def get(
        self,
        regions: Optional[List[str]] = None,
        gpu_count: Optional[int] = None,
        gpu_type: Optional[str] = None,
    ) -> Dict[str, List[GPUAvailability]]:
        """
        Get both single GPU and cluster availability information.

        Args:
            regions: Optional list of regions to filter by
            gpu_count: Optional number of GPUs to filter by
            gpu_type: Optional GPU type to filter by

        Returns:
            Dictionary mapping GPU types to lists of availability information,
            combining both single GPU and cluster availability
        """
        params = {}
        if regions:
            params["regions"] = ",".join(regions)
        if gpu_count:
            params["gpu_count"] = str(gpu_count)
        if gpu_type:
            params["gpu_type"] = gpu_type

        # Get both single GPU and cluster availability
        single_response = self.client.get("/availability", params=params)
        cluster_response = self.client.get("/availability/clusters", params=params)

        # Combine the responses
        combined: Dict[str, List[GPUAvailability]] = {}
        for gpu_type, gpus in single_response.items():
            if gpu_type is not None:  # Ensure gpu_type is not None
                combined[gpu_type] = [GPUAvailability(**gpu) for gpu in gpus]

        for gpu_type, gpus in cluster_response.items():
            if gpu_type is not None:  # Ensure gpu_type is not None
                if gpu_type in combined:
                    combined[gpu_type].extend([GPUAvailability(**gpu) for gpu in gpus])
                else:
                    combined[gpu_type] = [GPUAvailability(**gpu) for gpu in gpus]

        return combined
