"""Synthetic complete-code workload and independent bounded-LRU checks."""
import ast,hashlib,re,textwrap
PROMPT='''Write Python code only, without markdown or top-level calls. Implement class LRUCache using collections.OrderedDict and threading.RLock; do not use type annotations. Its constructor accepts a positive integer capacity, otherwise raises ValueError. Methods: get(key, default) refreshes recency; if default is omitted a missing key raises KeyError. put(key,value) inserts or updates and refreshes recency, evicting the least recently used item beyond capacity. delete(key) removes and returns its value, raising KeyError if missing. clear(), len(cache), key in cache, and snapshot() are thread-safe too. snapshot returns a NEW list of (key,value) pairs ordered least to most recently used, without changing recency. Stored None is valid. Add concise concurrency and complexity documentation. No tests or example calls in the answer.'''
TESTS='''
C = _make_cache_type()
for cap in [0,-1,False,1.5]:
 try:C(cap)
 except ValueError:pass
 else:raise AssertionError('invalid capacity accepted')
c=C(2);assert len(c)==0 and c.snapshot()==[]
try:c.get('missing')
except KeyError:pass
else:raise AssertionError('missing get did not raise')
sentinel=object();assert c.get('missing',sentinel) is sentinel
c.put('a',None);c.put('b',2);assert c.get('a') is None
assert c.snapshot()==[('b',2),('a',None)]
c.put('c',3);assert 'b' not in c and len(c)==2
c.put('a',4);assert c.snapshot()==[('c',3),('a',4)]
s=c.snapshot();s.clear();assert len(c)==2
assert c.delete('c')==3 and c.snapshot()==[('a',4)]
try:c.delete('missing')
except KeyError:pass
else:raise AssertionError('missing delete did not raise')
c.clear();assert len(c)==0 and c.snapshot()==[]
from concurrent.futures import ThreadPoolExecutor
c=C(32)
def work(t):
 for i in range(400):
  key=(t,i);c.put(key,i);c.get(key,None);key in c;len(c)
  s=c.snapshot();assert isinstance(s,list) and len(s)<=32
with ThreadPoolExecutor(max_workers=6) as pool:list(pool.map(work,range(6)))
s=c.snapshot();assert len(s)==len(c)<=32 and len({k for k,v in s})==len(s)
'''
def check(text,directory,checker):
 # The existing sandbox accepts functions/imports; a factory also scopes the generated class
 # and sentinels without weakening its import, filesystem, network or resource restrictions.
 match=re.search(r'```(?:python)?\s*\n(.*?)```',text,re.S)
 text=match.group(1) if match else text.strip()
 try:ast.parse(text)
 except SyntaxError:return dict(passed=False,reason='incomplete or invalid code')
 wrapped='def _make_cache_type():\n'+textwrap.indent(text,'    ')+'\n    return LRUCache\n'
 return checker(wrapped,'_make_cache_type',TESTS,directory,allowed_imports=('collections','threading'))
