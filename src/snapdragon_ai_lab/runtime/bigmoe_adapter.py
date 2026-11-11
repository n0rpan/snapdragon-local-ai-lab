"""MIT. Pure BigMoe controller adapter prototype; no sockets/processes/filesystem writes.

Transport is injected. Lease ownership is trusted controller evidence, never an HTTP
header/body claim. Connect it to the canonical lease before deploying this module.
"""
from __future__ import annotations
from dataclasses import dataclass
import copy
import json
import math
from pathlib import PurePosixPath
from typing import Any, Protocol

VERSION = "0.40.1"
ENGINE_KIND = "bigmoe-cpu-stream"
SUPPORTS_LLAMA_SLOT_PRIMER = False
PREFIX_CACHE_MODE = "native_resident_only"
# Server defaults differ from CLI. Pin baseline policy in settings and verify live config.
BASELINE = {
    "think": False, "moe-stream": True, "overlap": True, "io-two-wave": False, "row-stream": False,
    "spec": "none", "prefetch": 0, "predict-prefetch": False, "route-ahead": 0,
    "drop-cold-experts": 0, "expert-substitute": 0, "n-expert-used": 0,
    "devices": "cpu", "gpu-layers": 0, "device-cache-mb": 0, "routing-record": 0,
    "cache-type-k": "f16", "cache-type-v": "f16", "temp": 0, "seed": 42,
    "dense-weights": "anon", "dense-mapped": "per_layer_token_embd.weight",
    "threads": 4, "threads-batch": 6, "ubatch": 512, "io-threads": 4,
    "ctx-checkpoints": 2, "checkpoint-min-step": 256,
}
PUBLIC_ROUTES = frozenset({("GET", "/v1/models"), ("POST", "/v1/chat/completions")})
CONTROL_ROUTES = frozenset({("GET", "/api/info"), ("GET", "/api/config"),
                          ("POST", "/api/session/load"), ("POST", "/api/session/unload"),
                          ("POST", "/api/cancel")})

class AdapterError(ValueError):
    """Fixed code only; never include prompt/body/backend error text."""

@dataclass(frozen=True)
class Response:
    status: int
    body: bytes
    content_type: str = "application/json"

class Transport(Protocol):
    def request(self, method: str, path: str, body: dict[str, Any] | None = None) -> Response: ...

def _json(response: Response) -> dict[str, Any]:
    if response.status != 200 or not response.content_type.split(";", 1)[0].strip().lower() == "application/json":
        raise AdapterError("bad_backend_response")
    if len(response.body) > 131072:
        raise AdapterError("backend_body_limit")
    try:
        value = json.loads(response.body)
    except (ValueError, UnicodeError):
        raise AdapterError("invalid_backend_json") from None
    if not isinstance(value, dict):
        raise AdapterError("invalid_backend_schema")
    return value

def _same(actual: Any, expected: Any) -> bool:
    if isinstance(expected, bool):
        return actual is expected
    if isinstance(expected, (int, float)):
        return type(actual) in (int, float) and (type(actual) is int or math.isfinite(actual)) and actual == expected
    return type(actual) is type(expected) and actual == expected

def baseline_policy(microbatch: int = 512) -> dict[str, Any]:
    if type(microbatch) is not int or microbatch not in (512, 4096):
        raise AdapterError("unqualified_microbatch")
    return dict(copy.deepcopy(BASELINE), ubatch=microbatch)

def startup_settings(microbatch: int = 512) -> dict[str, Any]:
    return {"version": 1, "auto_plan": False, "values": baseline_policy(microbatch)}

def validate_settings(value: Any, microbatch: int = 512) -> None:
    if not isinstance(value, dict) or type(value.get("version")) is not int or value["version"] != 1:
        raise AdapterError("invalid_settings")
    if value.get("auto_plan") is not False or not isinstance(value.get("values"), dict):
        raise AdapterError("unsafe_settings")
    if any(not _same(value["values"].get(k), v) for k, v in baseline_policy(microbatch).items()):
        raise AdapterError("baseline_settings_drift")

@dataclass(frozen=True)
class StreamedAdmission:
    # Disk sizes are identity/download facts, not admission inputs.
    model_disk_bytes: int
    metadata_shard_bytes: int
    resident_estimate_bytes: int | None
    measured_peak_bytes: int | None
    admitted_budget_bytes: int | None
    os_headroom_bytes: int

    def require(self, available_bytes: int, *, lease_owned: bool) -> None:
        if lease_owned is not True:
            raise AdapterError("lease_required")
        values = (self.resident_estimate_bytes, self.measured_peak_bytes,
                  self.admitted_budget_bytes, self.os_headroom_bytes, available_bytes)
        if any(type(v) is not int or v <= 0 for v in values):
            raise AdapterError("streamed_budget_unqualified")
        budget = self.admitted_budget_bytes
        assert budget is not None and self.resident_estimate_bytes is not None and self.measured_peak_bytes is not None
        if max(self.resident_estimate_bytes, self.measured_peak_bytes) > budget:
            raise AdapterError("streamed_budget_too_small")
        if available_bytes < budget + self.os_headroom_bytes:
            raise AdapterError("memory_pressure")

@dataclass(frozen=True)
class Spec:
    binary: str
    model_path: str
    models_dir: str
    data_dir: str
    public_model_id: str = "local-qwen-flash-next"
    host: str = "127.0.0.1"
    port: int = 8093
    context: int = 8192
    cache_mib: int = 8192
    api_key_file: str | None = None
    microbatch: int = 512

    @property
    def backend_model_id(self) -> str:
        return PurePosixPath(self.model_path).name

    def validate(self) -> None:
        for path in (self.binary, self.model_path, self.models_dir, self.data_dir):
            if not isinstance(path, str) or "\0" in path or not PurePosixPath(path).is_absolute() or ".." in PurePosixPath(path).parts:
                raise AdapterError("invalid_launch_path")
        if PurePosixPath(self.model_path).parent != PurePosixPath(self.models_dir):
            raise AdapterError("model_outside_root")
        # Candidate controller still defaults to8093; ephemeral loopback ports
        # permit isolated fake-HTTP qualification without touching live ports.
        if self.host != "127.0.0.1" or type(self.port) is not int or not 1 <= self.port <= 65535:
            raise AdapterError("invalid_bind")
        if type(self.context) is not int or self.context not in (8192, 16384):
            raise AdapterError("unqualified_context")
        if type(self.cache_mib) is not int or self.cache_mib not in (8192, 12288):
            raise AdapterError("unqualified_cache")
        baseline_policy(self.microbatch)
        if type(self.public_model_id) is not str or not self.public_model_id or any(c in self.public_model_id for c in "\r\n\0"):
            raise AdapterError("invalid_model_alias")
        if self.api_key_file is not None:
            if type(self.api_key_file) is not str or not PurePosixPath(self.api_key_file).is_absolute() or ".." in PurePosixPath(self.api_key_file).parts or "\0" in self.api_key_file:
                raise AdapterError("invalid_key_path")

    def launch_argv(self, seed: Any) -> list[str]:
        self.validate()
        validate_settings(seed, self.microbatch)
        if self.api_key_file is None:
            raise AdapterError("private_backend_key_required")
        args = [self.binary, "--host", self.host, "--port", str(self.port),
                "--data-dir", self.data_dir, "--models-dir", self.models_dir,
                "--models-only", "--chat-only", "-m", self.model_path,
                "--chatml", "--moe-stream", "--devices", "cpu", "--gpu-layers", "0",
                "--overlap", "--device-cache-mb", "0", "--cache-mb", str(self.cache_mib),
                "--threads", "4", "--threads-batch", "6", "--ctx-size", str(self.context),
                "--ubatch", str(self.microbatch), "--io-threads", "4", "--dense-weights", "anon",
                "--dense-mapped", "per_layer_token_embd.weight", "--drop-cold-experts", "0",
                "--expert-substitute", "0", "--n-expert-used", "0", "--route-ahead", "0",
                "--temp", "0", "--seed", "42", "--cache-type-k", "f16", "--cache-type-v", "f16",
                "--ctx-checkpoints", "2", "--checkpoint-min-step", "256", "--routing-record", "0", "--no-think"]
        if self.api_key_file:
            args += ["--api-key-file", self.api_key_file]
        return args

    def chat_body(self, body: Any) -> dict[str, Any]:
        if not isinstance(body, dict) or not isinstance(body.get("messages"), list) or not body["messages"]:
            raise AdapterError("invalid_chat_request")
        alias = body.get("model", self.public_model_id)
        if type(alias) is not str or alias not in (self.public_model_id, self.backend_model_id):
            raise AdapterError("model_not_found")
        # Native parser ignores body.model entirely. Normalize rather than depend on that.
        result = copy.deepcopy(body)
        result["model"] = self.backend_model_id
        if "think" in result and type(result["think"]) is not bool:
            raise AdapterError("invalid_thinking_schema")
        if result.get("think") is True or result.get("reasoning_effort", "none") != "none":
            raise AdapterError("thinking_not_allowed_by_baseline")
        kwargs = result.get("chat_template_kwargs", {})
        if not isinstance(kwargs, dict) or kwargs.get("enable_thinking", False) is not False:
            raise AdapterError("thinking_not_allowed_by_baseline")
        result["think"] = False
        return result

@dataclass(frozen=True)
class Ready:
    ready: bool
    code: str
    backend_model_id: str = ""

def check_ready(spec: Spec, info_response: Response, config_response: Response) -> Ready:
    try:
        spec.validate()
        info, config = _json(info_response), _json(config_response)
        model = info.get("model")
        if info.get("version") != VERSION:
            raise AdapterError("backend_version_mismatch")
        if info.get("state") != "ready" or info.get("error") != "":
            raise AdapterError("backend_not_ready")
        if not isinstance(model, dict):
            raise AdapterError("invalid_model_schema")
        required = {"path": spec.model_path, "arch": "qwen4exp", "streaming": True,
                    "n_expert_used": 10, "n_ctx": spec.context}
        if any(not _same(model.get(k), v) for k, v in required.items()):
            raise AdapterError("loaded_model_mismatch")
        if config.get("valid") is not True or config.get("reload_required") is not False or config.get("auto_plan") is not False:
            raise AdapterError("unsafe_live_config")
        values = config.get("values")
        expected = dict(baseline_policy(spec.microbatch), **{"model": spec.model_path, "cache-mb": spec.cache_mib, "ctx-size": spec.context})
        if not isinstance(values, dict) or any(not _same(values.get(k), v) for k, v in expected.items()):
            raise AdapterError("live_policy_drift")
        return Ready(True, "ready", spec.backend_model_id)
    except AdapterError as error:
        return Ready(False, str(error))

def allow_route(method: str, path: str, *, controller: bool = False, lease_owned: bool = False) -> bool:
    pair = (method, path)
    if pair in PUBLIC_ROUTES:
        return True
    return controller is True and lease_owned is True and pair in CONTROL_ROUTES

class Adapter:
    def __init__(self, spec: Spec, transport: Transport):
        self.spec, self.transport = spec, transport
        spec.validate()

    def readiness(self, *, lease_owned: bool) -> Ready:
        if not lease_owned:
            return Ready(False, "lease_required")
        try:
            return check_ready(self.spec, self.transport.request("GET", "/api/info"),
                               self.transport.request("GET", "/api/config"))
        except OSError:
            return Ready(False, "backend_unavailable")

    def control(self, action: str, *, lease_owned: bool, admission: StreamedAdmission | None = None, available_bytes: int = 0) -> Response:
        paths = {"load": "/api/session/load", "unload": "/api/session/unload", "cancel": "/api/cancel"}
        path = paths.get(action)
        if path is None or not allow_route("POST", path, controller=True, lease_owned=lease_owned):
            raise AdapterError("lease_required")
        if action == "load":
            if admission is None:
                raise AdapterError("streamed_budget_unqualified")
            admission.require(available_bytes, lease_owned=lease_owned)
        body = {"model": self.spec.model_path} if action == "load" else {}
        return self.transport.request("POST", path, body)

    def chat(self, body: Any, *, lease_owned: bool) -> Response:
        if lease_owned is not True:
            raise AdapterError("lease_required")
        return self.transport.request("POST", "/v1/chat/completions", self.spec.chat_body(body))

    def primer_persistence_actions(self) -> tuple[()]:
        # BigMoe has no llama-server /slots contract. Never call that primer path.
        return ()


def loaded_model_id(spec: Spec, response: Response) -> str:
    value = _json(response)
    data = value.get("data")
    if value.get("object") != "list" or not isinstance(data, list) or len(data) != 1:
        raise AdapterError("invalid_model_listing")
    model = data[0]
    if not isinstance(model, dict) or model.get("object") != "model" or model.get("id") != spec.backend_model_id:
        raise AdapterError("model_listing_mismatch")
    return spec.backend_model_id


def public_model_listing(spec: Spec) -> dict[str, Any]:
    return {"object":"list", "data":[{"id":spec.public_model_id,"object":"model","owned_by":"local"}]}
