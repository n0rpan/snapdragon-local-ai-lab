#!/usr/bin/env python3
"""Launch-scoped OpenCode model selection or read-only planning; no global config writes.

Every selectable model (including Flash-Next) opens the ordinary build agent.
The read-only planning agent remains available as the explicit `plan` action.
MoE launches (`moe`, `ornith`) take the OpenCode context preset from `--mode`
(the controller's stored mode, passed by `local-ai code`), else LOCAL_AI_OPENCODE_MODE:
`fast` (default, O-105), `deep` or `max` (262144, Ornith with an RSS proof; the controller only
stores it when available); any other value fails.
"""
import copy
import json
import os
from pathlib import Path
import sys

MODEL = 'local-qwen/qwen3.8-flash-next-local'
CLASSIC_MODELS = {'moe':'local-qwen/qwen3.6-35b-a3b-local',
                  'ornith':'local-qwen/ornith-35b-a3b-local',
                  'flash':MODEL}
# Retired launch keys (O95: the dense 27B left in v0.1.2) and their OpenCode model ids.
RETIRED_MODELS = {'dense': 'the dense Qwen3.8 27B model was retired in v0.1.2; use moe or ornith'}
RETIRED_MODEL_IDS = ('qwen3.8-27b-local',)
# Qualified request policy: Ornith's production defaults; Flash accepts only
# temperature 0 and no thinking (bigmoe_runtime.shape_chat enforces both).
PINNED_REQUEST_MODELS = frozenset({'ornith', 'flash'})
# MoE-only OpenCode context presets (O-89; O-105: `fast` is the default). `deep` compacts later; `fast` compacts
# earlier. OpenCode 2.2.7 compacts at min(input, context) - max(15%, 16000) when no
# compaction.buffer is set: deep 120000 -> 102000, fast 65536 -> 49536 tokens.
OPENCODE_MODE_ENV = 'LOCAL_AI_OPENCODE_MODE'
DEFAULT_OPENCODE_MODE = 'fast'
# This file is installed alone (~/.local/bin): it mirrors harness_registry.MODE_INPUT / MODE_CAP,
# and tests/test_v013_modes.py keeps the two tables equal. max (V013) = Ornith 262144, compact ~200K.
OPENCODE_MODE_INPUT = {'deep': 120000, 'fast': 65536, 'max': 235000}
OPENCODE_MODE_CONTEXT = {'deep': 131072, 'fast': 131072, 'max': 262144}
MOE_CONTEXT, MOE_OUTPUT, MOE_KEEP_TOKENS = 131072, 16384, 8000
MOE_PRESET_MODELS = ('moe', 'ornith')
# v0.1.3 LONG-APPEND: OpenCode 2.2.7 provider timeouts (ms). Its fetch wrapper aborts a request
# whose response headers take longer than `headerTimeout` (default 300000) and a stream whose next
# SSE *event* takes longer than `chunkTimeout` (default 900000; only `data:` events reset it, SSE
# comments do not), and its session processor retries a transport error that came before any
# output: a prefill longer than 15 min (a 28.5K append at ~83K depth on CPU: ~950-1200 s) would be
# aborted and prefilled again. One hour each; no total timeout (`timeout` unset = none).
MOE_HEADER_TIMEOUT_MS = MOE_CHUNK_TIMEOUT_MS = 3_600_000
# OpenCode 2.2.7 compaction (auto/keep/buffer) is one instance-wide setting: there is
# no per-model compaction, and the legacy provider whitelist/blacklist keys are
# dropped as unsupported. keep=8000 is unsafe for a 16K model (fixed ~5.6K + 8K tail
# exceeds its ~10.4K trigger), so a MoE launch disables every small-context local
# model with the supported per-model `disabled` flag. Disabled models leave the
# instance's model list, and a session pinned to one fails with "Model unavailable"
# instead of running under the MoE keep. Flash launches are unchanged. The retired dense
# 27B id stays in the list: an older OpenCode config may still name it.
SMALL_CONTEXT_MODEL_IDS = tuple(sorted(
    {CLASSIC_MODELS[key].split('/', 1)[1] for key in CLASSIC_MODELS if key not in MOE_PRESET_MODELS}
    | {'qwen-local'} | set(RETIRED_MODEL_IDS)))
PLAN_SYSTEM = ('You are an offline planning and review assistant. Read the task, project instructions '
               'and relevant files. Produce a concrete plan or review with assumptions and checks. '
               'Do not modify files, run shell commands, execute code or start other agents. '
               'For implementation, describe the steps and hand the work to the build agent.')
READ_ONLY_POLICIES = [
    {'action': 'tool.use', 'resource': '*', 'effect': 'deny'},
    {'action': 'tool.use', 'resource': 'read:*', 'effect': 'allow'},
]


def agent_config():
    return {'model': MODEL, 'mode': 'primary', 'system': PLAN_SYSTEM,
            'description': 'Planning (launch via Local AI for read-only access)',
            'request': {'body': {'max_tokens': 2048, 'temperature': 0, 'think': False}}}


def environment_for_named_agent(environment):
    result = dict(environment)
    raw = result.get('OPENCODE_CONFIG_CONTENT', '')
    if not isinstance(raw, str) or len(raw.encode()) > 131072:
        raise ValueError('invalid_launch_overlay')
    overlay = json.loads(raw) if raw else {}
    if not isinstance(overlay, dict):
        raise ValueError('launch_overlay_must_be_object')
    agents = overlay.setdefault('agents', {})
    policies = overlay.setdefault('policies', [])
    if not isinstance(agents, dict) or not isinstance(policies, list):
        raise ValueError('invalid_launch_agents_or_policies')
    agents['local-plan'] = agent_config()
    overlay['default_agent'] = 'local-plan'
    # These rules restrict the whole planning launch even if an agent is changed
    # inside that client. Ordinary OpenCode launches keep their own policies.
    overlay['policies'] = policies + copy.deepcopy(READ_ONLY_POLICIES)
    result['OPENCODE_CONFIG_CONTENT'] = json.dumps(overlay, separators=(',', ':'))
    return result


def opencode_mode(environment):
    mode = environment.get(OPENCODE_MODE_ENV, '') or DEFAULT_OPENCODE_MODE
    if mode not in OPENCODE_MODE_INPUT:
        raise ValueError('unsupported_opencode_mode')
    return mode


def apply_moe_preset(overlay, mode):
    """Launch-scoped MoE limits, keep and small-model isolation for one OpenCode instance."""
    if mode not in OPENCODE_MODE_INPUT:
        raise ValueError('unsupported_opencode_mode')
    providers = overlay.setdefault('providers', {})
    if not isinstance(providers, dict):
        raise ValueError('invalid_launch_providers')
    provider = providers.setdefault('local-qwen', {})
    if not isinstance(provider, dict):
        raise ValueError('invalid_launch_provider')
    models = provider.setdefault('models', {})
    if not isinstance(models, dict):
        raise ValueError('invalid_launch_models')
    limit = {'context': OPENCODE_MODE_CONTEXT[mode], 'input': OPENCODE_MODE_INPUT[mode], 'output': MOE_OUTPUT}
    for key in MOE_PRESET_MODELS:
        entry = models.setdefault(CLASSIC_MODELS[key].split('/', 1)[1], {})
        if not isinstance(entry, dict):
            raise ValueError('invalid_launch_model')
        entry['limit'] = dict(limit)
        entry.pop('disabled', None)
    for model_id in SMALL_CONTEXT_MODEL_IDS:
        entry = models.setdefault(model_id, {})
        if not isinstance(entry, dict):
            raise ValueError('invalid_launch_model')
        entry['disabled'] = True
    # v0.1.3 LONG-APPEND: explicit, generous waits for long prefills (see MOE_CHUNK_TIMEOUT_MS);
    # never shorter than what the incoming overlay already asks for, and never disabled here
    settings = provider.setdefault('settings', {})
    if not isinstance(settings, dict):
        raise ValueError('invalid_launch_settings')
    for key, floor in (('headerTimeout', MOE_HEADER_TIMEOUT_MS), ('chunkTimeout', MOE_CHUNK_TIMEOUT_MS)):
        value = settings.get(key)
        keep = value is False or (isinstance(value, int) and not isinstance(value, bool) and value >= floor)
        if not keep:
            settings[key] = floor
    compaction = overlay.setdefault('compaction', {})
    if not isinstance(compaction, dict):
        raise ValueError('invalid_launch_compaction')
    compaction['auto'] = True
    compaction['keep'] = {'tokens': MOE_KEEP_TOKENS}
    # The trigger must come from OpenCode's default buffer rule (see above).
    compaction.pop('buffer', None)
    compaction.pop('reserved', None)
    return overlay


def environment_for_classic_model(environment, model, mode=None):
    if model not in CLASSIC_MODELS:
        raise ValueError('unsupported_classic_model')
    mode = opencode_mode(environment) if mode is None else mode
    if mode not in OPENCODE_MODE_INPUT:
        raise ValueError('unsupported_opencode_mode')
    result = dict(environment)
    raw = result.get('OPENCODE_CONFIG_CONTENT', '')
    if not isinstance(raw, str) or len(raw.encode()) > 131072:
        raise ValueError('invalid_launch_overlay')
    overlay = json.loads(raw) if raw else {}
    if not isinstance(overlay, dict):
        raise ValueError('launch_overlay_must_be_object')
    overlay['model'] = CLASSIC_MODELS[model]
    if model in PINNED_REQUEST_MODELS:
        # Launch-scoped qualified request policy; ordinary Qwen launches keep
        # their existing provider defaults and agent/tool permissions.
        providers = overlay.setdefault('providers', {})
        if not isinstance(providers, dict):
            raise ValueError('invalid_launch_providers')
        provider = providers.setdefault('local-qwen', {})
        if not isinstance(provider, dict):
            raise ValueError('invalid_launch_provider')
        body = provider.setdefault('body', {})
        if not isinstance(body, dict):
            raise ValueError('invalid_launch_request_body')
        kwargs = body.setdefault('chat_template_kwargs', {})
        if not isinstance(kwargs, dict):
            raise ValueError('invalid_launch_template_kwargs')
        body.setdefault('temperature', 0)
        body.setdefault('cache_prompt', True)
        kwargs.setdefault('enable_thinking', False)
    if model in MOE_PRESET_MODELS:
        apply_moe_preset(overlay, mode)
    result['OPENCODE_CONFIG_CONTENT'] = json.dumps(overlay, separators=(',', ':'))
    return result


# v0.1.3 TOOL-OUTPUT CAP: with LOCAL_AI_TOOL_OUTPUT_CAP=on the launch registers the read-more tool
# (tool_output_cap.py run as a stdio MCP server; codemode false = a direct tool that OpenCode 2.2.7
# exposes as `local_ai_read_tool_output`). The controller caps only requests that offer this tool,
# so a launch without it (flag off, file missing, read-only plan policy) is never capped.
TOOL_CAP_ENV = 'LOCAL_AI_TOOL_OUTPUT_CAP'
TOOL_CAP_SCRIPT_ENV = 'LOCAL_AI_TOOL_OUTPUT_MCP'
TOOL_CAP_SERVER = 'local_ai'


def apply_tool_output_cap(overlay, environment):
    if str(environment.get(TOOL_CAP_ENV, 'off')).strip().lower() not in ('on', '1', 'true'):
        return overlay
    # an explicit path, else next to this file (source tree), else the installed controller runtime
    candidates = ([Path(environment[TOOL_CAP_SCRIPT_ENV])] if environment.get(TOOL_CAP_SCRIPT_ENV) else
                  [Path(__file__).resolve().with_name('tool_output_cap.py'),
                   Path(environment.get('HOME') or Path.home()) / '.local/share/ai-stack/runtime/tool_output_cap.py'])
    script = next((p for p in candidates if p.is_absolute() and p.is_file()), None)
    if script is None:
        return overlay
    mcp = overlay.setdefault('mcp', {})
    if not isinstance(mcp, dict):
        raise ValueError('invalid_launch_mcp')
    servers = mcp.setdefault('servers', {})
    if not isinstance(servers, dict):
        raise ValueError('invalid_launch_mcp_servers')
    entry = {'type': 'local', 'command': [sys.executable or 'python3', str(script), 'mcp'], 'codemode': False}
    passed = {k: environment[k] for k in (TOOL_CAP_ENV + '_STORE', 'XDG_STATE_HOME') if environment.get(k)}
    if passed:
        entry['environment'] = passed
    servers[TOOL_CAP_SERVER] = entry
    return overlay


def with_tool_output_cap(environment):
    if str(environment.get(TOOL_CAP_ENV, 'off')).strip().lower() not in ('on', '1', 'true'):
        return environment
    result = dict(environment)
    raw = result.get('OPENCODE_CONFIG_CONTENT', '')
    overlay = json.loads(raw) if raw else {}
    if not isinstance(overlay, dict):
        raise ValueError('launch_overlay_must_be_object')
    result['OPENCODE_CONFIG_CONTENT'] = json.dumps(apply_tool_output_cap(overlay, environment), separators=(',', ':'))
    return result


def main(arguments=None):
    arguments = sys.argv[1:] if arguments is None else arguments
    model = None
    mode = None
    if len(arguments) == 5 and arguments[0] == '--model' and arguments[1] in MOE_PRESET_MODELS and arguments[2] == '--mode':
        if arguments[3] not in OPENCODE_MODE_INPUT:
            raise ValueError('unsupported_opencode_mode')
        model, mode, arguments = arguments[1], arguments[3], arguments[4:]
    elif len(arguments) == 3 and arguments[0] == '--model' and arguments[1] in RETIRED_MODELS:
        raise ValueError('retired_model: ' + RETIRED_MODELS[arguments[1]])
    elif len(arguments) == 3 and arguments[0] == '--model' and arguments[1] in CLASSIC_MODELS:
        model, arguments = arguments[1], arguments[2:]
    if len(arguments) != 1 or not Path(arguments[0]).is_absolute() or '\0' in arguments[0]:
        raise ValueError('one_absolute_project_directory_required')
    if model is None:
        opencode_mode(os.environ)  # audit S7: a bad LOCAL_AI_OPENCODE_MODE stops a plan launch too
    environment = (environment_for_named_agent(os.environ) if model is None else
                   environment_for_classic_model(os.environ, model, mode))
    environment = with_tool_output_cap(environment)
    # OpenCode 2.2.x's TUI has no --model flag: pass the selected model in its
    # launch-scoped config overlay, not an unsupported CLI argument.
    os.execvpe('opencode', ['opencode', '--standalone', arguments[0]], environment)


if __name__ == '__main__':
    try:
        main()
    except ValueError as error:  # audit S7: an expected refusal is one line, not a traceback
        raise SystemExit(f'local-ai launch: {error}')
