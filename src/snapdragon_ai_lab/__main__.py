import argparse
import json
from pathlib import Path
from .doctor import inspect
from .results import export_screen
from .plugin_install import install

def main(argv=None):
    parser=argparse.ArgumentParser(description='Snapdragon local AI diagnostics and public metrics')
    commands=parser.add_subparsers(dest='command',required=True)
    commands.add_parser('doctor',help='Read-only hardware and power report; starts no models')
    export=commands.add_parser('export-screen',help='Export allowlisted metrics from a synthetic screen artifact')
    export.add_argument('source',type=Path)
    export.add_argument('destination',type=Path)
    plugin=commands.add_parser('install-plugin',help='Install optional Omarchy panel; requires existing local-ai runtime command')
    plugin.add_argument('--dry-run',action='store_true')
    plugin.add_argument('--allow-unsupported',action='store_true',help='Explicit contributor override for other hardware')
    plugin.add_argument('--replace-plugin',metavar='ID',help='Migrate an existing user plugin, preserving its right-bar position and backing up its files')
    plugin.add_argument('--restart-shell',action='store_true',help='After a successful install whose panel code changed, run the Omarchy shell restart (the bar disappears for a moment); off by default')
    args=parser.parse_args(argv)
    if args.command=='doctor':print(json.dumps(inspect(),indent=2))
    elif args.command=='install-plugin':
        # Audit S7: an expected refusal (no Omarchy, unsupported hardware, ...) is one line, not a traceback.
        try:result=install(dry_run=args.dry_run,allow_unsupported=args.allow_unsupported,replace_plugin=args.replace_plugin,restart_shell=args.restart_shell)
        except (RuntimeError,ValueError,OSError) as error:parser.exit(1,'install-plugin: '+(str(error) or type(error).__name__)+'\n')
        print(json.dumps(result,indent=2))
    else:
        result=export_screen(json.loads(args.source.read_text()))
        args.destination.write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
