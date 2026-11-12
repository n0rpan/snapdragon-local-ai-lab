#!/usr/bin/env python3
"""Render the example systemd user units from ONE template (S9, audit).

examples/runtime/units/*.service are generated: edit SERVICES/TEMPLATE here and run
`python3 tools/render_units.py --write`. `--check` (the test) fails when an example drifted.
Each unit carries only the environment its own program reads: the controller reads no Studio
variable (SD_*, IMG_OUTDIR, LOCAL_IMAGE_CONFIG_DIR) and Studio does not read LOCAL_LLM_CONFIG.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UNITS = ROOT / 'examples/runtime/units'

# Shared by both services: the one inference lease they coordinate through.
COMMON_ENV = (
    'LOCAL_AI_LEASE=%h/.local/state/local-ai/inference.lock',
)

SERVICES = {
    'local-ai-chat.service': {
        'description': 'Local private AI (controller.py)',
        'program': 'controller.py',
        'env': (
            'LOCAL_LLM_CONFIG=%h/.config/local-ai/llm/runtime.json',
            'PYTHONDONTWRITEBYTECODE=1',
            '# Engine selection (docs/ENGINE-SELECTION.md). The Rust engines are the default',
            '# even without these lines; set one to 0 for the classic llama.cpp / BigMoe fallback.',
            'LOCAL_AI_ENABLE_OXE=1',
            'LOCAL_AI_ENABLE_OXE_FLASH=1',
            '# NPU images need your own Hexagon SDK / QAIRT build; off unless set to 1.',
            '#LOCAL_AI_IMAGE_NPU=1',
        ),
        'cpu_quota': '800%', 'memory_high': '24G', 'memory_max': '25G',
    },
    'prompt-studio-app.service': {
        'description': 'Local private AI (studio)',
        'program': 'studio',
        'env': (
            'LOCAL_IMAGE_CONFIG_DIR=%h/.config/local-ai/images',
            'SD_HOME=%h/.local/share/ai-stack',
            'SD_CLI=%h/.local/share/ai-stack/releases/sd-cpu-repack/sd-cli',
            'IMG_OUTDIR=%h/Pictures/sd',
            'PYTHONDONTWRITEBYTECODE=1',
        ),
        'cpu_quota': '1000%', 'memory_high': '21G', 'memory_max': '22G',
    },
}

TEMPLATE = """[Unit]
Description={description}
[Service]
Type=simple
WorkingDirectory=%h/.local/share/ai-stack/runtime
ExecStart=/usr/bin/python3 %h/.local/share/ai-stack/runtime/{program}
{environment}
Restart=on-failure
RestartSec=5
TimeoutStopSec=25
CPUQuota={cpu_quota}
MemoryHigh={memory_high}
MemoryMax={memory_max}
MemorySwapMax=0
Nice=15
LimitCORE=0
UMask=0077
TasksMax=256
[Install]
WantedBy=default.target
"""


def env_line(item):
    if item.startswith('#') and '=' in item and ' ' not in item.split('=', 1)[0]:
        return '#Environment=' + item[1:]  # a commented-out setting
    return item if item.startswith('#') else 'Environment=' + item


def render(name):
    spec = SERVICES[name]
    environment = '\n'.join(env_line(item) for item in (*COMMON_ENV, *spec['env']))
    return TEMPLATE.format(environment=environment, **{k: v for k, v in spec.items() if k != 'env'})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check', action='store_true', help='exit 1 when an example unit differs')
    mode.add_argument('--write', action='store_true', help='rewrite examples/runtime/units')
    args = parser.parse_args(argv)
    drift = []
    for name in SERVICES:
        text, path = render(name), UNITS / name
        if args.write:
            path.write_text(text)
        elif not path.exists() or path.read_text() != text:
            drift.append(name)
    extra = sorted(p.name for p in UNITS.glob('*.service') if p.name not in SERVICES)
    for name in drift + extra:
        print(f'render_units: {name} is not generated from the template', file=sys.stderr)
    return 1 if (drift or extra) else 0


if __name__ == '__main__':
    sys.exit(main())
