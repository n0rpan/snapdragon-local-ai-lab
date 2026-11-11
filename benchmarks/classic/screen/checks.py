"""Synthetic task definitions and isolated checks; no user content is accessed."""
import ast
import json
import os
from pathlib import Path
import re
import resource
import signal
import subprocess
import tempfile

CODE_CASES=[
 ('merge_intervals', 'Write Python code only, without markdown, imports or top-level calls. Define merge_intervals(intervals). Input is a possibly empty list of two-element [start,end] closed intervals in any order. Return a NEW sorted list merging overlapping intervals and touching endpoints. Do not modify any input lists.', '''
cases=[([],[]),([[1,3],[2,6],[8,10],[15,18]],[[1,6],[8,10],[15,18]]),([[3,4],[1,2],[2,3]],[[1,4]]),([[1,10],[2,3]],[[1,10]]),([[-5,-1],[-2,2],[4,4]],[[-5,2],[4,4]]),([[1,1]],[[1,1]])]
for data,expected in cases:
 original=[x[:] for x in data]
 assert merge_intervals(data)==expected
 assert data==original
'''),
 ('dedupe_by_id', 'Write Python code only, without markdown, imports or top-level calls. Define dedupe_by_id(rows). rows is a list of dictionaries, each with a hashable id field. Return a new list retaining the FIRST dictionary for each distinct id, in original order. Do not modify rows or the dictionaries. Additional helper functions are allowed.', '''
for data,expected in [([],[]),([{'id':1,'v':'first'},{'id':2,'v':'two'},{'id':1,'v':'last'}],[{'id':1,'v':'first'},{'id':2,'v':'two'}]),([{'id':None},{'id':0},{'id':None}],[{'id':None},{'id':0}]),([{'id':'a'},{'id':'b'},{'id':'a'}],[{'id':'a'},{'id':'b'}])]:
 original=[dict(x) for x in data]
 out=dedupe_by_id(data)
 assert out==expected and data==original and out is not data
'''),
 ('parse_port', 'Write Python code only, without markdown, imports or top-level calls. Define parse_port(value). Accept ONLY strings. Strip surrounding whitespace, then require one or more ASCII digits (0-9), with no sign or other characters. Return its integer value if between 1 and 65535 inclusive, otherwise raise ValueError. Non-string values must also raise ValueError.', '''
for value,expected in [('1',1),('65535',65535),(' 0080 ',80),('443',443)]:assert parse_port(value)==expected
for value in ['', '0','65536','-1','+80','1.0','80x','１２','٨٠',80,None,True,[],{}]:
 try:parse_port(value)
 except ValueError:pass
 else:raise AssertionError('expected ValueError')
''')]

def code_check(text,name,tests,directory,allowed_imports=()):
    match=re.search(r'```(?:python)?\s*\n(.*?)```',text,re.S)
    code=match.group(1) if match else text.strip()
    try:
        tree=ast.parse(code)
        if not all(isinstance(x,ast.FunctionDef) or allowed_imports and isinstance(x,(ast.Import,ast.ImportFrom)) for x in tree.body):return {'passed':False,'reason':'Unexpected top-level statements'}
        if not any(isinstance(x,ast.FunctionDef) and x.name==name for x in tree.body):return {'passed':False,'reason':'Missing requested function'}
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):
                if any(alias.name not in allowed_imports for alias in node.names):return {'passed':False,'reason':'Imports were prohibited'}
            elif isinstance(node,ast.ImportFrom):
                if node.level or node.module not in allowed_imports:return {'passed':False,'reason':'Imports were prohibited'}
        if any(x.decorator_list or x.args.defaults or any(y is not None for y in x.args.kw_defaults) for x in tree.body if isinstance(x,ast.FunctionDef)):return {'passed':False,'reason':'Unexpected decorators/default expressions'}
    except SyntaxError:return {'passed':False,'reason':'Syntax error or incomplete output'}
    def limit():
        resource.setrlimit(resource.RLIMIT_CPU,(3,3));resource.setrlimit(resource.RLIMIT_AS,(256*1024**2,)*2)
        resource.setrlimit(resource.RLIMIT_FSIZE,(16384,)*2);resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    with tempfile.TemporaryDirectory(prefix='synthetic-check-',dir=directory) as tmp:
        p=Path(tmp)/'test.py';p.write_text(code+'\n'+tests+'\nprint("CHECK_PASSED")\n')
        args=['bwrap','--unshare-all','--die-with-parent','--new-session','--cap-drop','ALL','--clearenv',
              '--ro-bind','/usr','/usr','--symlink','usr/lib','/lib','--proc','/proc','--dev','/dev',
              '--size','16777216','--tmpfs','/tmp','--dir','/work','--ro-bind',str(p),'/work/test.py',
              '--chdir','/work','/usr/bin/python3','-I','/work/test.py']
        output=Path(tmp)/'output.txt'
        with output.open('wb') as stream:
            child=subprocess.Popen(args,stdout=stream,stderr=subprocess.STDOUT,preexec_fn=limit,start_new_session=True)
            try:rc=child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL);child.wait();return {'passed':False,'reason':'Sandbox timeout'}
        report=output.read_text(errors='replace')[:16384]
        if report.startswith('bwrap:'):return {'passed':None,'reason':'Sandbox unavailable: '+report[:500]}
        return {'passed':rc==0 and 'CHECK_PASSED' in report,'exit_code':rc,'failure':None if rc==0 else report[-1500:]}

def json_check(text,expected):
    try:return {'passed':json.loads(text)==expected,'expected':expected}
    except (ValueError,TypeError):return {'passed':False,'reason':'Invalid JSON','expected':expected}

FACTS_PROMPT='''Use only these fictional test observations. CPU cold request: 2.7 seconds. NPU cold request: 4.2 seconds. CPU already-loaded request: 2.0 seconds. NPU already-loaded request: 1.2 seconds. No temperatures or clock speeds were recorded. Return ONLY a JSON object with exactly these keys: cold_winner ("cpu" or "npu"), warm_winner ("cpu" or "npu"), throttling_proven (boolean).'''
FACTS_EXPECTED={'cold_winner':'cpu','warm_winner':'npu','throttling_proven':False}
NORWEGIAN_PROMPT='''Oppsummer følgende fiktive testnotat på naturlig norsk bokmål i nøyaktig tre punkt. Bruk maksimalt 55 ord totalt. Ikke legg til nye fakta. Notat: Q4-modellen trenger 15 GB RAM. Modellen lastes først ved en forespørsel. Etter fire minutter uten bruk lastes den ut igjen. CPU og NPU er testet. GPU er ikke testet. Resultatene sier derfor ikke noe om GPU-ytelsen.'''
LONG_PROMPT='Read the fictional inventory below.\n'+''.join(f'Cabinet {i:03d}: notebook, wooden ruler, blue folder.\n' for i in range(1,101))+'\nThe access code for cabinet 073 is RAVN-731. The code for cabinet 037 is FJORD-317.\n'+''.join(f'Cabinet {i:03d}: pencil, spare label, grey box.\n' for i in range(101,131))+'\nReturn ONLY the access code for cabinet 073.'
