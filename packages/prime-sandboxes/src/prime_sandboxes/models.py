"""Pydantic models for Prime Sandboxes SDK."""

import ipaddress
import re
from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, model_validator

from .image_references import is_docker_hub_reference


class SandboxStatus(str, Enum):
    """Sandbox status enum"""

    PENDING = "PENDING"
    PROVISIONING = "PROVISIONING"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    ERROR = "ERROR"
    TERMINATED = "TERMINATED"
    TIMEOUT = "TIMEOUT"


MAX_EGRESS_POLICY_ENTRIES = 1024

# Domain entries are matched by HTTP Host or TLS SNI, read only on these ports.
DOMAIN_RULE_PORTS = (80, 443)

# Header substitution rules contract (mirrors the API implementation):
# SECRET_REFERENCE_PREFIX introduces a "${prime.secrets.<name>}" reference in a
# header template value; the other constants bound the rules per policy.
SECRET_REFERENCE_PREFIX = "${prime.secrets."
MAX_HEADER_RULE_HOSTS = 32
MAX_HEADER_RULE_BYTES = 8192

_SECRET_NAME_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]{0,254}$")
_RESERVED_RULE_HEADERS = frozenset(
    {
        "host",
        ":authority",
        "content-length",
        "transfer-encoding",
        "connection",
        "upgrade",
        "keep-alive",
        "te",
        "trailer",
    }
)
_RESERVED_RULE_HEADER_PREFIXES = ("proxy-",)


class HeaderRule(BaseModel):
    """Headers to set on HTTPS requests to one rule host.

    Each value is a template that may contain "${prime.secrets.<name>}"
    references; a set header replaces any same-name header the sandbox sent.
    """

    headers: Dict[str, str]


class Secret:
    """Reference helper for platform secrets inside header-rule values.

    Distinct from ``CreateSandboxRequest.secrets``, which are per-sandbox
    environment values: a platform secret is stored once per user/team and
    never enters the sandbox; the sandbox's node substitutes its value into
    the HTTPS request headers named by ``network_rules``.
    """

    @classmethod
    def fill(cls, name: str) -> str:
        """Return the "${prime.secrets.<name>}" reference string for ``name``."""
        if not _SECRET_NAME_PATTERN.fullmatch(name):
            raise ValueError(
                f"invalid secret name {name!r}; must be 1-255 uppercase letters, "
                "digits, or underscores, starting with a letter"
            )
        return f"${{prime.secrets.{name}}}"


def _validate_egress_entry(entry: str) -> None:
    """Client-side mirror of the API's egress entry contract.

    Accepts an exact hostname, a leftmost-label ``*.`` wildcard, an IPv4
    address, or an IPv4 CIDR, optionally followed by ``:port`` or
    ``:low-high``; domains accept only ports 80 and 443. Schemes, credentials,
    paths, query strings, bare ``*``, IPv6, and wildcards in other positions
    are rejected. The server canonicalizes entries further (lowercase, IDNA,
    /32) and rejects ports until the platform enables them.
    """
    import ipaddress
    import re

    value = entry.strip()
    if not value:
        raise ValueError("empty entry")

    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        address = None
    if address is not None:
        if address.version != 4:
            raise ValueError(f"'{entry}': IPv6 is not supported")
        return

    host, separator, ports = value.rpartition(":")
    if separator and "://" not in value and not _is_ip_network(value):
        match = re.fullmatch(r"([0-9]{1,5})(?:-([0-9]{1,5}))?", ports)
        low, high = (int(match[1]), int(match[2] or match[1])) if match else (0, 0)
        if not 1 <= low <= high <= 65535:
            raise ValueError(
                f"'{entry}': port must be a number from 1 to 65535 or a low-high range"
            )
        _validate_egress_entry(host)
        if not _is_ip_network(host) and (low != high or low not in DOMAIN_RULE_PORTS):
            raise ValueError(
                f"'{entry}': domain entries support only ports 80 and 443; "
                "use an IP or CIDR entry for other ports"
            )
        return

    if "/" in value:
        try:
            network = ipaddress.ip_network(value, strict=False)
        except ValueError as exc:
            raise ValueError(f"'{entry}' is not a valid IPv4 CIDR") from exc
        if network.version != 4:
            raise ValueError(f"'{entry}': IPv6 is not supported")
        return

    for forbidden, reason in (
        ("://", "schemes are not supported"),
        ("@", "credentials are not supported"),
        ("?", "query strings are not supported"),
    ):
        if forbidden in value:
            raise ValueError(f"'{entry}': {reason}")

    domain = value[2:] if value.startswith("*.") else value
    domain = domain.rstrip(".")
    if not domain:
        raise ValueError(f"'{entry}': domain is empty")
    if "*" in domain:
        raise ValueError(f"'{entry}': wildcard is only supported as the leftmost label")
    for label in domain.split("."):
        if not label:
            raise ValueError(f"'{entry}' contains an empty label")


def _is_ip_network(value: str) -> bool:
    import ipaddress

    try:
        ipaddress.ip_network(value, strict=False)
    except ValueError:
        return False
    return True


def validate_egress_lists(allowlist: Optional[List[str]], denylist: Optional[List[str]]) -> None:
    """Mirror the public API's mutual-exclusion and rule-count contract."""
    if allowlist is not None and denylist is not None:
        raise ValueError(
            "network_allowlist and network_denylist are mutually exclusive; provide at most one"
        )
    for name, entries in (
        ("network_allowlist", allowlist),
        ("network_denylist", denylist),
    ):
        if entries is None:
            continue
        if len(entries) > MAX_EGRESS_POLICY_ENTRIES:
            raise ValueError(f"{name} supports at most {MAX_EGRESS_POLICY_ENTRIES} entries")
        for entry in entries:
            try:
                _validate_egress_entry(entry)
            except ValueError as exc:
                raise ValueError(f"{name}: {exc}") from exc


def _parse_secret_references(value: str) -> List[str]:
    """Parse one template value into its referenced secret names, in order."""
    names: List[str] = []
    while True:
        start = value.find("${")
        if start < 0:
            return names
        if not value.startswith(SECRET_REFERENCE_PREFIX, start):
            raise ValueError(
                f"unsupported interpolation at {value[start : start + 24]!r}; "
                "only ${prime.secrets.<name>} references are allowed"
            )
        rest = value[start + len(SECRET_REFERENCE_PREFIX) :]
        secret_name, terminator, tail = rest.partition("}")
        if not terminator:
            raise ValueError(f"unterminated reference at {value[start:]!r}")
        if not _SECRET_NAME_PATTERN.fullmatch(secret_name):
            raise ValueError(
                f"invalid secret name {secret_name!r} in reference; must be "
                "1-255 uppercase letters, digits, or underscores, "
                "starting with a letter"
            )
        names.append(secret_name)
        value = tail


def validate_header_rules(
    rules: Optional[Dict[str, HeaderRule]]
) -> Optional[Dict[str, HeaderRule]]:
    """Client-side mirror of the API's header substitution rules contract.

    Rule hosts are an exact DNS name or a single leading ``*.`` wildcard
    without ports or paths, canonicalized to lowercase with case-equivalent
    hosts rejected; header names are valid HTTP tokens that are not reserved
    (Host, :authority, framing, hop-by-hop, Proxy-*); values carry no CR/LF
    and only ``${prime.secrets.<name>}`` references; one rule host's headers
    total at most ``MAX_HEADER_RULE_BYTES`` bytes.
    """
    if not rules:
        return rules
    if len(rules) > MAX_HEADER_RULE_HOSTS:
        raise ValueError(
            f"header rules name {len(rules)} hosts; at most {MAX_HEADER_RULE_HOSTS} are allowed"
        )

    canonical: Dict[str, HeaderRule] = {}
    for host, rule in rules.items():
        if not isinstance(rule, HeaderRule):
            rule = HeaderRule.model_validate(rule)
        canonical_host = host.lower()
        if canonical_host in canonical:
            raise ValueError(
                f"header rule hosts {host!r} and an earlier entry differ only by case"
            )
        _validate_header_rule_host(host)
        _validate_header_rule_headers(host, rule.headers)
        canonical[canonical_host] = rule
    return canonical


def _validate_header_rule_host(host: str) -> None:
    """Mirror the API's rule-host contract: DNS names only — IP literals
    (wildcard base included) are rejected because header rules match
    hostnames and the node refuses them; numeric labels stay valid DNS."""
    domain = host[2:] if host.startswith("*.") else host
    if not domain:
        raise ValueError(f"header rule host {host!r} is empty")
    if "*" in domain:
        raise ValueError(
            f"header rule host {host!r}: wildcard is only supported as the leftmost label"
        )
    if len(domain) > 253:
        raise ValueError(f"header rule host {host!r} exceeds 253 characters")
    try:
        ipaddress.ip_address(domain)
    except ValueError:
        pass
    else:
        raise ValueError(
            f"header rule host {host!r}: IP addresses are not supported; use a DNS name"
        )
    for label in domain.split("."):
        if not 1 <= len(label) <= 63:
            raise ValueError(
                f"header rule host {host!r} contains an empty or oversized label"
            )
        if label.startswith("-") or label.endswith("-"):
            raise ValueError(
                f"header rule host {host!r}: labels must not start or end with a hyphen"
            )
        for character in label:
            if not (character.isascii() and (character.isalnum() or character == "-")):
                raise ValueError(
                    f"header rule host {host!r} contains unsupported character {character!r}"
                )


def _validate_header_rule_headers(host: str, headers: Dict[str, str]) -> None:
    total_bytes = 0
    for name, value in headers.items():
        if not _is_valid_header_name(name):
            raise ValueError(f"header rule for {host!r}: {name!r} is not a valid header name")
        lower_name = name.lower()
        if lower_name in _RESERVED_RULE_HEADERS:
            raise ValueError(
                f"header rule for {host!r}: {name!r} is a reserved header and cannot be set"
            )
        if lower_name.startswith(_RESERVED_RULE_HEADER_PREFIXES):
            raise ValueError(
                f"header rule for {host!r}: {name!r} uses the reserved "
                f"{_RESERVED_RULE_HEADER_PREFIXES[0]!r} prefix and cannot be set"
            )
        if "\r" in value or "\n" in value:
            raise ValueError(
                f"header rule for {host!r}: header {name!r} value must not contain CR or LF"
            )
        try:
            _parse_secret_references(value)
        except ValueError as exc:
            raise ValueError(f"header rule for {host!r}, header {name!r}: {exc}") from exc

        total_bytes += len(name.encode("utf-8")) + len(value.encode("utf-8"))
    if total_bytes > MAX_HEADER_RULE_BYTES:
        raise ValueError(
            f"header rule for {host!r} exceeds {MAX_HEADER_RULE_BYTES} bytes of headers"
        )


def _is_valid_header_name(name: str) -> bool:
    """One to 255 HTTP token characters: visible ASCII minus separators."""
    if not 1 <= len(name) <= 255:
        return False
    for character in name:
        if character.isascii() and (
            character.isalnum() or character in "!#$%&'*+-.^_`|~"
        ):
            continue
        return False
    return True


class SandboxEgressPolicy(BaseModel):
    """The desired egress policy: at most one list is non-null."""

    allowlist: Optional[List[str]] = None
    denylist: Optional[List[str]] = None
    # Per-host HTTPS header substitution rules; absent means none.
    rules: Optional[Dict[str, HeaderRule]] = None


class EgressPolicyStatus(BaseModel):
    """Desired vs applied egress policy state for a VM sandbox."""

    policy: SandboxEgressPolicy
    generation: int
    applied_generation: int
    applied: bool

    model_config = ConfigDict(populate_by_name=True)


class LocalhostForward(BaseModel):
    """Forward the sandbox's own loopback ``port`` (127.0.0.1 and [::1]) to the same
    port on peer ``sandbox_id`` (``<sandbox_id>.sandbox.internal``). The peer must
    share a LAN network whose rules allow ``port``, or connections are reset."""

    port: int = Field(ge=1, le=65535)
    sandbox_id: str = Field(alias="peer_sandbox_id")

    model_config = ConfigDict(populate_by_name=True)


class LanRule(BaseModel):
    """A directed allow rule: members matching ``from`` may connect to members
    matching ``to``. Selectors are ``*`` or ``tag:<name>``; ``ports`` limits the
    destination ports (all ports when omitted). A ``from`` of ``internet``
    instead exposes the listed ``ports`` of members matching ``to`` publicly over
    HTTP at ``https://<port>-<sandbox-id>.<gateway domain>``."""

    from_: str = Field(alias="from")
    to: str
    ports: Optional[List[int]] = None

    model_config = ConfigDict(populate_by_name=True)


class LanNetworkMember(BaseModel):
    """A sandbox in a LAN network, reachable at ``<sandbox_id>.sandbox.internal``."""

    sandbox_id: str
    tags: List[str]
    synced: bool


class LanNetwork(BaseModel):
    """A private network between VM sandboxes of one owner."""

    id: str
    name: str
    team_id: Optional[str] = None
    rules: List[LanRule]
    version: int
    synced: bool
    created_at: datetime
    members: Optional[List[LanNetworkMember]] = None


class SSHSession(BaseModel):
    """Ephemeral SSH access to a VM sandbox."""

    session_id: str
    sandbox_id: str
    host: str
    port: int
    expires_at: datetime
    ttl_seconds: int


class AdvancedConfigs(BaseModel):
    """Advanced configuration options for sandbox"""

    # Reserved for future advanced configuration options
    # Allow extra fields for backward compatibility with existing data
    model_config = ConfigDict(extra="allow")


class StartCommand(BaseModel):
    """A process to start without invoking a shell."""

    executable: str = Field(min_length=1)
    args: List[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def validate_no_nul_bytes(self) -> "StartCommand":
        if "\x00" in self.executable:
            raise ValueError("executable must not contain NUL bytes")
        if any("\x00" in arg for arg in self.args):
            raise ValueError("args must not contain NUL bytes")
        return self


class Sandbox(BaseModel):
    """Sandbox model"""

    id: str
    name: str
    docker_image: str = Field(..., alias="dockerImage")
    start_command: Optional[StartCommand] = Field(None, alias="startCommand")
    cpu_cores: float = Field(..., alias="cpuCores")
    memory_gb: float = Field(..., alias="memoryGB")
    disk_size_gb: float = Field(..., alias="diskSizeGB")
    disk_mount_path: str = Field(..., alias="diskMountPath")
    gpu_count: int = Field(..., alias="gpuCount")
    gpu_type: Optional[str] = Field(None, alias="gpuType")
    vm: bool = False
    network_allowlist: Optional[List[str]] = Field(None, alias="networkAllowlist")
    network_denylist: Optional[List[str]] = Field(None, alias="networkDenylist")
    status: str
    timeout_minutes: int = Field(..., alias="timeoutMinutes")
    idle_timeout_minutes: Optional[int] = Field(None, alias="idleTimeoutMinutes")
    termination_reason: Optional[str] = Field(None, alias="terminationReason")
    environment_vars: Optional[Dict[str, Any]] = Field(None, alias="environmentVars")
    secrets: Optional[Dict[str, Any]] = Field(None, alias="secrets")
    advanced_configs: Optional[AdvancedConfigs] = Field(None, alias="advancedConfigs")
    labels: List[str] = Field(default_factory=list)
    created_at: datetime = Field(..., alias="createdAt")
    updated_at: datetime = Field(..., alias="updatedAt")
    started_at: Optional[datetime] = Field(None, alias="startedAt")
    terminated_at: Optional[datetime] = Field(None, alias="terminatedAt")
    exit_code: Optional[int] = Field(None, alias="exitCode")
    error_type: Optional[str] = Field(None, alias="errorType")
    error_message: Optional[str] = Field(None, alias="errorMessage")
    user_id: Optional[str] = Field(None, alias="userId")
    team_id: Optional[str] = Field(None, alias="teamId")
    region: Optional[str] = None
    pending_image_build_id: Optional[str] = Field(default=None, alias="pendingImageBuildId")

    model_config = ConfigDict(populate_by_name=True)


class SandboxListResponse(BaseModel):
    """Sandbox list response model"""

    sandboxes: List[Sandbox]
    total: int
    page: int
    per_page: int = Field(..., alias="perPage")
    has_next: bool = Field(..., alias="hasNext")

    model_config = ConfigDict(populate_by_name=True)


class SandboxCheckpoint(BaseModel):
    """A filesystem checkpoint; only DURABLE checkpoints can be restored."""

    id: str
    sandbox_id: str
    parent_id: Optional[str] = None
    team_id: Optional[str] = None
    state: str
    depth: int
    docker_image: str
    disk_size_bytes: Optional[int] = None
    stored_bytes: Optional[int] = None
    # Quota-size measure: non-hole uncompressed bytes of the packed version.
    # stored_bytes stays the compressed billing metric; do not conflate the two.
    # 0 for grandfathered pre-measurement records; older responses omit it.
    occupied_bytes: int = 0
    error: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class CheckpointLookupError(BaseModel):
    """A missing or inaccessible checkpoint in a batch lookup."""

    checkpoint_id: str
    code: Literal["NOT_FOUND"]
    message: str


class BatchCheckpointResponse(BaseModel):
    """Checkpoints and per-ID errors from a cross-sandbox lookup."""

    checkpoints: List[SandboxCheckpoint]
    errors: List[CheckpointLookupError]


class CheckpointDeleteError(BaseModel):
    """A checkpoint that a sandbox-wide delete could not delete."""

    checkpoint_id: str
    code: Literal["NOT_FOUND", "CONFLICT"]
    message: str


class DeleteSandboxCheckpointsResponse(BaseModel):
    """Deleted checkpoint IDs and per-checkpoint errors for one sandbox."""

    deleted: List[str]
    errors: List[CheckpointDeleteError]


class SandboxStatusSnapshot(BaseModel):
    """Lightweight sandbox lifecycle state returned by a batch status lookup."""

    sandbox_id: str
    status: SandboxStatus
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    pending_image_build_id: Optional[str] = None


class SandboxStatusLookupError(BaseModel):
    """Per-sandbox error returned by a batch status lookup."""

    sandbox_id: str
    code: Literal["NOT_FOUND", "FORBIDDEN", "MANAGED"]
    message: str


class BatchSandboxStatusResponse(BaseModel):
    """Ordered results from a batch sandbox lifecycle status lookup."""

    statuses: List[SandboxStatusSnapshot]
    errors: List[SandboxStatusLookupError]


class VolumeMount(BaseModel):
    """One volume to mount in the sandbox, resolved by name in the
    caller's scope (team scope when team_id is set)."""

    name: str
    mount_path: str
    read_only: bool = False


class Volume(BaseModel):
    """A sandbox volume, owned by exactly one of user/team scope."""

    id: str
    name: str
    user_id: Optional[str] = None
    team_id: Optional[str] = None
    # Newest durable publish, or null for a volume that was never published.
    # Older responses may omit the field entirely.
    head_checkpoint_id: Optional[str] = None
    created_at: datetime
    # Occupied bytes of the latest published version (its checkpoint's
    # occupied_bytes); 0 until the first publish and for grandfathered
    # pre-measurement versions. Older responses omit the field.
    occupied_bytes: int = 0
    # Account-wide storage usage across all owner volumes (sum of the latest
    # published versions) and the whole-GiB limit from the wallet's
    # sandboxVolumeStorageLimitGB.
    # Populated on single-volume GET; always null on list items. Older
    # responses omit both.
    account_storage_used_bytes: Optional[int] = None
    account_storage_limit_gb: Optional[int] = None


class VolumeListResponse(BaseModel):
    """Volume list response model"""

    volumes: List[Volume]
    total: int
    page: int
    per_page: int
    has_next: bool
    # Account-wide storage usage and whole-GiB limit, response-level and
    # present even when ``volumes`` is empty. Older responses omit them.
    storage_used_bytes: Optional[int] = None
    storage_limit_gb: Optional[int] = None


class CreateSandboxRequest(BaseModel):
    """Create sandbox request model"""

    name: str
    docker_image: Optional[str] = None
    checkpoint_id: Optional[str] = Field(None, min_length=1, max_length=64)
    # None lets the server default to amd64; checkpoint restores must omit it.
    arch: Optional[Literal["amd64", "arm64"]] = None
    start_command: Optional[StartCommand] = None
    cpu_cores: float = 1.0
    memory_gb: float = 1.0
    disk_size_gb: Optional[float] = None
    gpu_count: int = 0
    gpu_type: Optional[str] = None
    network_allowlist: Optional[List[str]] = None
    network_denylist: Optional[List[str]] = None
    # Per-host HTTPS header substitution rules; a rule never grants egress.
    network_rules: Optional[Dict[str, HeaderRule]] = None
    timeout_minutes: int = 60
    idle_timeout_minutes: Optional[int] = None
    environment_vars: Optional[Dict[str, str]] = None
    secrets: Optional[Dict[str, str]] = None
    labels: List[str] = Field(default_factory=list)
    team_id: Optional[str] = None
    region: Optional[str] = None
    advanced_configs: Optional[AdvancedConfigs] = None
    idempotency_key: Optional[str] = None
    # Client-side only: when set, the sandbox joins this LAN network with these
    # tags during the creating client's wait_for_creation or bulk_wait_for_creation.
    lan_network_id: Optional[str] = None
    lan_tags: Optional[List[str]] = None
    # Unset volumes are omitted from the create payload, so requests without
    # volume mounts keep their exact previous wire shape.
    volumes: Optional[List[VolumeMount]] = None

    @model_validator(mode="after")
    def validate_restore_source(self) -> "CreateSandboxRequest":
        if self.checkpoint_id:
            if self.docker_image or self.disk_size_gb is not None:
                raise ValueError("omit docker_image and disk_size_gb when restoring a checkpoint")
        else:
            if not self.docker_image:
                raise ValueError("docker_image is required unless checkpoint_id is set")
            if self.disk_size_gb is None:
                self.disk_size_gb = 5.0
        return self

    @model_validator(mode="after")
    def validate_gpu_fields(self) -> "CreateSandboxRequest":
        if self.gpu_count > 0 and not self.gpu_type:
            raise ValueError("gpu_type is required when gpu_count is greater than 0")
        if self.gpu_count == 0 and self.gpu_type is not None:
            raise ValueError("gpu_type requires gpu_count greater than 0")
        return self

    @model_validator(mode="after")
    def validate_network_lists(self) -> "CreateSandboxRequest":
        validate_egress_lists(self.network_allowlist, self.network_denylist)
        self.network_rules = validate_header_rules(self.network_rules)
        if self.lan_tags is not None and self.lan_network_id is None:
            raise ValueError("lan_tags requires lan_network_id")
        return self

    @model_validator(mode="after")
    def validate_idle_timeout(self) -> "CreateSandboxRequest":
        if self.idle_timeout_minutes is None:
            return self
        if self.idle_timeout_minutes < 1:
            raise ValueError("idle_timeout_minutes must be >= 1")
        if self.timeout_minutes > 0 and self.idle_timeout_minutes > self.timeout_minutes:
            raise ValueError(
                "idle_timeout_minutes must be <= timeout_minutes "
                f"(got idle={self.idle_timeout_minutes}, lifetime={self.timeout_minutes})"
            )
        return self


class CommandResponse(BaseModel):
    """Execute command response model"""

    stdout: str
    stderr: str
    exit_code: int


class FileUploadResponse(BaseModel):
    """File upload response model"""

    success: bool
    path: str
    size: int
    timestamp: datetime


class ReadFileResponse(BaseModel):
    """Read file response model"""

    content: str
    size: int
    # VM sandboxes don't support windowed reads yet and omit them.
    total_size: Optional[int] = None
    offset: Optional[int] = None
    truncated: Optional[bool] = None


class SandboxLogsResponse(BaseModel):
    """Sandbox logs response model"""

    logs: str


class BulkDeleteSandboxRequest(BaseModel):
    """Bulk delete sandboxes request model"""

    sandbox_ids: Optional[List[str]] = None
    labels: Optional[List[str]] = None
    team_id: Optional[str] = None
    user_id: Optional[str] = None
    all_users: bool = False


class BulkDeleteSandboxResponse(BaseModel):
    """Bulk delete sandboxes response model"""

    succeeded: List[str]
    failed: List[Dict[str, str]]
    message: str


class ImageVisibility(str, Enum):
    PRIVATE = "PRIVATE"
    PUBLIC = "PUBLIC"


class ImageBuildStatus(str, Enum):
    """Status of an image artifact build."""

    PENDING = "PENDING"
    UPLOADING = "UPLOADING"
    BUILDING = "BUILDING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class ImageArtifactType(str, Enum):
    """Artifact produced for an image."""

    VM_SANDBOX = "VM_SANDBOX"


class ImageOwnerType(str, Enum):
    """Scope that owns an image."""

    PERSONAL = "personal"
    TEAM = "team"
    PLATFORM = "platform"


class ImageListItem(BaseModel):
    """One artifact row returned by the image-list endpoint."""

    id: str
    artifact_type: ImageArtifactType = Field(..., alias="artifactType")
    image_name: str = Field(..., alias="imageName")
    image_tag: str = Field(..., alias="imageTag")
    status: ImageBuildStatus
    full_image_path: Optional[str] = Field(default=None, alias="fullImagePath")
    error_message: Optional[str] = Field(default=None, alias="errorMessage")
    size_bytes: Optional[int] = Field(default=None, alias="sizeBytes")
    visibility: ImageVisibility = ImageVisibility.PRIVATE
    created_at: datetime = Field(..., alias="createdAt")
    started_at: Optional[datetime] = Field(default=None, alias="startedAt")
    completed_at: Optional[datetime] = Field(default=None, alias="completedAt")
    pushed_at: Optional[datetime] = Field(default=None, alias="pushedAt")
    team_id: Optional[str] = Field(default=None, alias="teamId")
    owner_type: ImageOwnerType = Field(default=ImageOwnerType.PERSONAL, alias="ownerType")
    display_ref: Optional[str] = Field(default=None, alias="displayRef")

    model_config = ConfigDict(populate_by_name=True)


class ImageListResponse(BaseModel):
    """A logical-image page and its artifact rows.

    When present, ``total_count`` counts logical images. ``data`` can contain
    more rows than ``limit`` when a logical image has multiple artifact types.
    """

    data: List[ImageListItem]
    total_count: Optional[int] = Field(default=None, alias="totalCount")
    offset: int
    limit: int
    status: Optional[str] = None

    model_config = ConfigDict(populate_by_name=True)


class BuildImageRequest(BaseModel):
    """Request a linux/amd64 or linux/arm64 VM image build.

    Docker Hub sources become public, org-less platform images automatically.
    They cannot use a custom destination, team, or private visibility.
    """

    image_name: Optional[str] = None
    image_tag: Optional[str] = None
    dockerfile_path: str = "Dockerfile"
    source_image: Optional[str] = Field(default=None, alias="sourceImage")
    platform: Literal["linux/amd64", "linux/arm64"] = "linux/amd64"
    team_id: Optional[str] = Field(default=None, alias="teamId")
    visibility: Optional[ImageVisibility] = None
    owner_scope: Optional[Literal["platform"]] = Field(default=None, alias="ownerScope")

    model_config = ConfigDict(populate_by_name=True)

    @model_validator(mode="after")
    def validate_build(self) -> "BuildImageRequest":
        sources = [source.strip() for source in (self.source_image or "").split(",")]
        docker_hub_sources = [
            source for source in sources if source and is_docker_hub_reference(source)
        ]
        if not docker_hub_sources:
            return self

        if self.image_name is not None or self.image_tag is not None:
            raise ValueError("Docker Hub source builds do not accept a custom destination")
        if self.team_id is not None:
            raise ValueError("Docker Hub source builds do not accept team_id")
        if self.visibility == ImageVisibility.PRIVATE:
            raise ValueError("Docker Hub source builds must be public")
        if len(docker_hub_sources) != len([source for source in sources if source]):
            raise ValueError("Docker Hub and non-Docker Hub sources cannot share one request")

        self.visibility = ImageVisibility.PUBLIC
        self.owner_scope = "platform"
        return self


class BuildImageResponse(BaseModel):
    # Server quirk: build_id/upload_url/expires_in stay snake_case on the wire;
    # buildIds/fullImagePath use camelCase.
    build_id: str
    build_ids: List[str] = Field(default_factory=list, alias="buildIds")
    upload_url: Optional[str] = None
    expires_in: Optional[int] = None
    full_image_path: str = Field(..., alias="fullImagePath")
    visibility: Optional[ImageVisibility] = None

    model_config = ConfigDict(populate_by_name=True)

    @model_validator(mode="after")
    def require_dockerfile_upload(self, info: ValidationInfo) -> "BuildImageResponse":
        """Require upload metadata when the caller identifies a Dockerfile build."""
        if info.context and info.context.get("requires_upload"):
            if not self.upload_url or self.expires_in is None:
                raise ValueError("Dockerfile build response requires upload_url and expires_in")
        return self


class SourceImageBuildResult(BaseModel):
    """Build result or error for one requested source image."""

    source_image: str = Field(..., alias="sourceImage")
    build: Optional[BuildImageResponse] = None
    error: Optional[str] = None
    retryable: bool = False

    model_config = ConfigDict(populate_by_name=True)


class BulkBuildImageResponse(BaseModel):
    """Ordered, best-effort results for comma-separated source images."""

    results: List[SourceImageBuildResult]


MAX_IMAGE_UPDATES = 100


class PersonalImageOwner(BaseModel):
    """The authenticated caller's personal image scope."""

    type: Literal["personal"] = "personal"

    model_config = ConfigDict(populate_by_name=True)


class TeamImageOwner(BaseModel):
    """A team image scope."""

    type: Literal["team"] = "team"
    team_id: str = Field(..., alias="teamId")

    model_config = ConfigDict(populate_by_name=True)


class PlatformImageOwner(BaseModel):
    """Org-less platform image scope (admin-managed, always PUBLIC)."""

    type: Literal["platform"] = "platform"

    model_config = ConfigDict(populate_by_name=True)


ImageOwner = Annotated[
    Union[PersonalImageOwner, TeamImageOwner, PlatformImageOwner],
    Field(discriminator="type"),
]


class ImageUpdateSource(BaseModel):
    """Source selector for one logical-image update.

    Either the structured coordinate form (``owner`` + ``name`` + ``tag``) or
    the ``reference`` form (owner-prefixed / plain ``name:tag`` reference).
    The two forms are mutually exclusive.
    """

    owner: Optional[ImageOwner] = None
    name: Optional[str] = None
    tag: Optional[str] = None
    reference: Optional[str] = None

    model_config = ConfigDict(populate_by_name=True)

    @model_validator(mode="after")
    def _require_exactly_one_form(self) -> "ImageUpdateSource":
        coordinate_fields = (self.owner, self.name, self.tag)
        if self.reference is not None:
            if any(value is not None for value in coordinate_fields):
                raise ValueError("source accepts either reference or owner/name/tag, not both")
        elif any(value is None for value in coordinate_fields):
            raise ValueError("source requires owner, name, and tag (or a reference)")
        return self


class ImageUpdatePatch(BaseModel):
    """Partial patch for one logical image; omitted fields keep their value."""

    name: Optional[str] = None
    tag: Optional[str] = None
    owner: Optional[ImageOwner] = None
    visibility: Optional[ImageVisibility] = None

    model_config = ConfigDict(populate_by_name=True)

    @model_validator(mode="after")
    def _require_some_change(self) -> "ImageUpdatePatch":
        if (
            self.name is None
            and self.tag is None
            and self.owner is None
            and self.visibility is None
        ):
            raise ValueError("set must change at least one field")
        if (
            isinstance(self.owner, PlatformImageOwner)
            and self.visibility == ImageVisibility.PRIVATE
        ):
            raise ValueError("platform images are always PUBLIC")
        return self


class ImageUpdateItem(BaseModel):
    """One independent logical-image patch."""

    source: ImageUpdateSource
    set: ImageUpdatePatch

    model_config = ConfigDict(populate_by_name=True)


class UpdateImagesRequest(BaseModel):
    """Explicit list of independent logical-image patches.

    Every update names its source exactly; there is no server-side search
    selection for mutations.
    """

    mode: Literal["explicit"] = "explicit"
    dry_run: bool = Field(default=False, alias="dryRun")
    updates: List[ImageUpdateItem]

    model_config = ConfigDict(populate_by_name=True)


class ImageMutationError(BaseModel):
    """Machine-readable failure for one update item.

    ``code`` is intentionally a plain string so new server-side codes never
    break response parsing.
    """

    code: str
    message: str

    model_config = ConfigDict(populate_by_name=True)


class ImageCoordinateState(BaseModel):
    """Canonical logical-image coordinate plus effective visibility."""

    owner: ImageOwner
    name: str
    tag: str
    visibility: ImageVisibility

    model_config = ConfigDict(populate_by_name=True)


class ImageUpdateResult(BaseModel):
    """Per-item outcome, in deterministic request order."""

    source: ImageUpdateSource
    success: bool
    before: Optional[ImageCoordinateState] = None
    after: Optional[ImageCoordinateState] = None
    error: Optional[ImageMutationError] = None

    model_config = ConfigDict(populate_by_name=True)


class UpdateImagesResponse(BaseModel):
    """Response for both explicit and search update modes."""

    success: bool
    dry_run: bool = Field(default=False, alias="dryRun")
    results: List[ImageUpdateResult] = Field(default_factory=list)

    model_config = ConfigDict(populate_by_name=True)


class BackgroundJob(BaseModel):
    """Background job handle returned when starting a background job"""

    job_id: str
    sandbox_id: str
    stdout_log_file: str
    stderr_log_file: str
    exit_file: str


class BackgroundJobStatus(BaseModel):
    """Status of a background job"""

    job_id: str
    completed: bool
    exit_code: Optional[int] = None
    stdout: Optional[str] = None
    stderr: Optional[str] = None
    stdout_error: Optional[str] = None
    stderr_error: Optional[str] = None
    stdout_truncated: bool = False
    stderr_truncated: bool = False


class BackgroundJobStatusSnapshot(BaseModel):
    """Completion state returned by a platform VM background-job lookup."""

    sandbox_id: str
    job_id: str
    completed: bool
    exit_code: Optional[int] = None


class BackgroundJobStatusLookupError(BaseModel):
    """Per-job error returned by a platform VM background-job lookup."""

    sandbox_id: str
    job_id: str
    code: Literal[
        "NOT_FOUND",
        "FORBIDDEN",
        "MANAGED",
        "NOT_VM",
        "NOT_RUNNING",
        "RUNTIME_ERROR",
    ]
    message: str


class BatchBackgroundJobStatusResponse(BaseModel):
    """Ordered results from a platform VM background-job lookup."""

    statuses: List[BackgroundJobStatusSnapshot]
    errors: List[BackgroundJobStatusLookupError]
