#!/usr/bin/env python3
"""Launch-scoped OpenCode model selection or read-only planning; no global config writes.

Every selectable model (including Flash-Next) opens the ordinary build agent.
The read-only planning agent remains available as the explicit `plan` action.
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


def environment_for_classic_model(environment, model):
    if model not in CLASSIC_MODELS:
        raise ValueError('unsupported_classic_model')
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
    os.execvpe('opencode', ['opencode', '--standalone', arguments[0]], environment)


if __name__ == '__main__':
    main()
