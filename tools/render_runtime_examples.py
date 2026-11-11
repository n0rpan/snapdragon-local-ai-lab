#!/usr/bin/env python3
"""Render reviewed examples into a NEW staging directory; never install or run."""
import argparse
import json
from pathlib import Path

def render(destination,home=None):
    home=Path.home() if home is None else Path(home)
    if not home.is_absolute() or any(c in str(home) for c in ('\n','\r','\x00')):raise ValueError('absolute single-line home required')
    destination=Path(destination)
    if destination.exists():raise ValueError('destination must not exist')
    source=Path(__file__).resolve().parents[1]/'examples/runtime'
    staged=[]
    for path in source.rglob('*'):
        if not path.is_file():continue
        data=path.read_text().replace('@HOME@',str(home))
        if path.suffix=='.json':json.loads(data)
        staged.append((path.relative_to(source),data))
    destination.mkdir(parents=True,mode=0o700)
    for relative,data in staged:
        target=destination/relative;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(data);target.chmod(0o600)
    return {'staged_files':len(staged),'installed':False,'models_started':False}
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('destination',type=Path)
    print(json.dumps(render(parser.parse_args().destination)))
