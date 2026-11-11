"""Header-only dense-Q4 candidate plan; all expert and other tensor types retained."""
import argparse, datetime, importlib.util, json, re
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
a.output.mkdir(mode=0o700,parents=True,exist_ok=False)
source=Path(__file__).resolve().parents[1]/'flash-memlat-20261110/gguf_header.py'
s=importlib.util.spec_from_file_location('gh',source);gh=importlib.util.module_from_spec(s);s.loader.exec_module(gh);gh.GGML_TYPES[42]=('Q2_0',64,18)
model=Path.home()/'.local/share/ai-stack/models/llm/Qwen3.8-Flash-Next/UD-IQ3_XXS'
rows=[]; seen=set()
for shard in sorted(model.glob('*.gguf')):
 _,ts,_=gh.read_gguf(shard)
 for t in ts:
  assert t.name not in seen; seen.add(t.name)
  expert='_exps.' in t.name
  change=not expert and t.type_name in ('Q6_K','Q8_0')
  target='Q4_K' if change else t.type_name
  incompatible=change and t.shape[0]%256!=0
  if incompatible: target=t.type_name;change=False
  size=t.nelem//256*144 if change else t.nbytes
  rows.append(dict(name=t.name,source_type=t.type_name,target_type=target,shape=t.shape,source_bytes=t.nbytes,target_bytes=size,expert=expert,changed=change,q4_k_row_incompatible=incompatible,source_shard=shard.name))
# Explicit full-name overrides for ALL tensors avoid the quantizer's default mixed-quant policy.
# Single backslashes are literal regex escapes in the whitespace-delimited tensor-type file.
(a.output/'tensor-types.txt').write_text(''.join('^'+re.escape(r['name'])+'$='+r['target_type'].lower()+'\n' for r in rows))
(a.output/'tensor-types-head-q6.txt').write_text(''.join('^'+re.escape(r['name'])+'$='+('q6_k' if r['name']=='output.weight' else r['target_type'].lower())+'\n' for r in rows))
changed=[r for r in rows if r['changed']];experts=[r for r in rows if r['expert']]
# Per-token byte estimate: full matrix reads except embedding gathers, plus other resident
# tensors. This is a storage-layout traffic model, NOT measured DRAM bandwidth.
def traffic(target):
 total=0
 for r in rows:
  if r['expert'] or r['name']=='per_layer_token_embd.weight':continue
  n=r[target+'_bytes']
  if r['name']=='token_embd.weight':n=n//r['shape'][1]
  total+=n
 return total
summary=dict(kind='flash-dense-q4-header-plan',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),header_only=True,tensors=len(rows),changed_tensors=len(changed),q4_k_incompatible=[dict(name=r['name'],shape=r['shape'],source_type=r['source_type']) for r in rows if r['q4_k_row_incompatible']],expert_tensors=len(experts),expert_payload_bytes=sum(r['source_bytes'] for r in experts),changed_source_bytes=sum(r['source_bytes'] for r in changed),changed_target_bytes=sum(r['target_bytes'] for r in changed),nonexpert_bytes_per_token_estimate_before=traffic('source'),nonexpert_bytes_per_token_estimate_after=traffic('target'),notes=['Requantization of existing quantized weights; not a fresh BF16 quant.','All expert and n-gram payloads must be byte-hash verified after conversion.','Traffic estimate counts each dense matrix once and one token embedding row; excludes activations/KV and backend dtype expansion.','Q4_K requires row length divisible by256. Incompatible HC rows stay unchanged in this preliminary manifest; alternate Q4_0 needs an explicit recorded decision.'],tensors_plan=rows)
(a.output/'plan.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k!='tensors_plan'}))
