#!/usr/bin/env python3
"""Launch-scoped OpenCode model selection or read-only planning; no global config writes.

Every selectable model (including Flash-Next) opens the ordinary build agent.
The read-only planning agent remains available as the explicit `plan` action.
MoE launches (`moe`, `ornith`) take the OpenCode context preset from
LOCAL_AI_OPENCODE_MODE: `fast` (default, O-105) or `deep`; any other value fails.
"""
import copy
import json
import os
from pathlib import Path
import sys

MODEL = 'local-qwen/qwen3.8-flash-next-local'
CLASSIC_MODELS = {'dense':'local-qwen/qwen3.8-27b-local',
                  'moe':'local-qwen/qwen3.6-35b-a3b-local',
                  'ornith':'local-qwen/ornith-35b-a3b-local',
                  'flash':MODEL}
# Qualified request policy: Ornith's production defaults; Flash accepts only
# temperature 0 and no thinking (bigmoe_runtime.shape_chat enforces both).
PINNED_REQUEST_MODELS = frozenset({'ornith', 'flash'})
# MoE-only OpenCode context presets (O-89). `fast` is the default (O-105: compaction near
# 49.5K inside the 128K window); `deep` compacts later and is slower on heavy tool turns at depth. OpenCode 2.2.7 compacts at min(input, context) - max(15%, 16000) when no
# compaction.buffer is set: deep 120000 -> 102000, fast 65536 -> 49536 tokens.
OPENCODE_MODE_ENV = 'LOCAL_AI_OPENCODE_MODE'
DEFAULT_OPENCODE_MODE = 'fast'
OPENCODE_MODE_INPUT = {'deep': 120000, 'fast': 65536}
MOE_CONTEXT, MOE_OUTPUT, MOE_KEEP_TOKENS = 131072, 16384, 8000
MOE_PRESET_MODELS = ('moe', 'ornith')
# OpenCode 2.2.7 compaction (auto/keep/buffer) is one instance-wide setting: there is
# no per-model compaction, and the legacy provider whitelist/blacklist keys are
# dropped as unsupported. keep=8000 is unsafe for a 16K model (fixed ~5.6K + 8K tail
# exceeds its ~10.4K trigger), so a MoE launch disables every small-context local
# model with the supported per-model `disabled` flag. Disabled models leave the
# instance's model list, and a session pinned to one fails with "Model unavailable"
# instead of running under the MoE keep. Flash/dense launches are unchanged.
SMALL_CONTEXT_MODEL_IDS = tuple(sorted(
    {CLASSIC_MODELS[key].split('/', 1)[1] for key in CLASSIC_MODELS if key not in MOE_PRESET_MODELS}
    | {'qwen-local'}))
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
    limit = {'context': MOE_CONTEXT, 'input': OPENCODE_MODE_INPUT[mode], 'output': MOE_OUTPUT}
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


def main(arguments=None):
    arguments = sys.argv[1:] if arguments is None else arguments
    model = None
    if len(arguments) == 3 and arguments[0] == '--model' and arguments[1] in CLASSIC_MODELS:
        model, arguments = arguments[1], arguments[2:]
    if len(arguments) != 1 or not Path(arguments[0]).is_absolute() or '\0' in arguments[0]:
        raise ValueError('one_absolute_project_directory_required')
    environment = (environment_for_named_agent(os.environ) if model is None else
                   environment_for_classic_model(os.environ, model))
    # OpenCode 2.2.x's TUI has no --model flag: pass the selected model in its
    # launch-scoped config overlay, not an unsupported CLI argument.
    os.execvpe('opencode', ['opencode', '--standalone', arguments[0]], environment)


if __name__ == '__main__':
    main()
