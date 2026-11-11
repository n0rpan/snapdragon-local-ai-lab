"""Collect already-finished M0 receipts. No measurement or payload reads."""
from pathlib import Path
import datetime,json,os,statistics
h=Path.home();opt=Path(os.environ.get('LAB_RESULTS_DIR','results'));repo=Path(__file__).resolve().parents[2]
roots=['flash-m0-tuning-20261110','flash-m0-cache22-admitted-20261110','flash-m0-reclaimed-cache-20261110','flash-m0-layout-screen-20261110','flash-m0-qd-isolated-20261110-v2','flash-m0-contention-isolated-20261110','flash-m0-qd-abba-20261110-v2'];rows=[]
for name in roots:
 for p in sorted((opt/name).glob('*.json')):
  if p.name.endswith('.native.json'):continue
  r=json.loads(p.read_text())
  if not isinstance(r,dict) or 'label' not in r or 'flags' not in r:continue
  row={k:r.get(k) for k in ['label','cache_gib','flags','utc','returncode','failure','wall_seconds','binary_sha256','guard','process_memory_peaks']};row['receipt']=str(p)
  if 'native' in r:
   native=r['native'];row['native']={k:native.get(k) for k in ['ms_per_token','tok_s','breakdown_ms','cache','config','memory','io_max_inflight','barriers_per_token','per_token_ms']}
   proof=Path(r['raw_path']).with_suffix('.affinity.json') if r.get('raw_path') else None
   if proof and proof.exists():
    a=json.loads(proof.read_text());row['crypto_affinity']={k:a[k] for k in ['requested','before','during','after','changed','restored'] if k in a};assert a['restored']
  rows.append(row)
def group(root,label):return next(r for r in rows if root in r['receipt'] and r['label']==label)
def med(root,labels):return statistics.median(group(root,label)['native']['ms_per_token']['median'] for label in labels)
qd='flash-m0-qd-abba-20261110-v2';assert len([r for r in rows if qd in r['receipt'] and not r['failure'] and r['returncode']==0])==4
crypto='flash-m0-contention-isolated-20261110';qm={str(q):med(qd,labels) for q,labels in [(32,['qd32-abba1','qd32-abba4']),(4,['qd4-abba2','qd4-abba3'])]};cm={'default':med(crypto,['crypto-default-r1','crypto-default-r4']),'e00':med(crypto,['crypto-e00-r2','crypto-e00-r3'])}
row=dict(kind='flash-m0-cache-qd-placement-screens',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),timing_only=True,correct_model_inference=False,production_changed=False,qd_abba_median_ms=qm,crypto_abba_median_ms=cm,failed_confirmation=dict(receipt=str(opt/'flash-m0-qd-abba-20261110/failure.json'),excluded=True),notes=['M0 replays routes with dummy activations. No correct-output or completed-task speed claim.','Original f3ff and isolated4f130 builds are separate timing groups, despite identical source commit.','One cache/placement sample is a screen, not a promoted winner.','22GiB cache tripped host1.5GiB floor even after desktop-only reclaim;23/24 skipped.','All temporary workqueue masks restored; no persistent setting or production budget change.'],rows=rows)
p=repo/'benchmarks/results/flash-m0-tuning-20261110.json';p.write_text(json.dumps(row,indent=2)+'\n');print(json.dumps({k:v for k,v in row.items() if k not in ['rows']}))
