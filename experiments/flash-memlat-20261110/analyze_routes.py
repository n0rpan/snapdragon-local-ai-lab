"""Expert IDs only: locality and byte-budgeted LRU projections, no tuning."""
from pathlib import Path
from collections import OrderedDict,defaultdict
import argparse,csv,hashlib,json,statistics
ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--trace',type=Path,required=True);ap.add_argument('--header',type=Path,required=True);ap.add_argument('--native-result',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();p=args.trace;audit=json.loads(args.header.read_text());sizes={int(k):sum(t['bytes_per_expert'] for t in v) for k,v in audit['expert_layers'].items()};cells=defaultdict(dict);batches=OrderedDict();rows=0
with p.open() as f:
 for r in csv.DictReader(f):
  t,ph,step,layer,slot,e=(int(r[k]) for k in ['turn','phase','step','layer','slot','expert']);key=(t,ph,step,layer);assert slot not in cells[key];cells[key][slot]=e;rows+=1
  batch=(t,ph,0 if ph==0 else step,layer);batches.setdefault(batch,[]).append(e)
assert all(len(v)==10 and len(set(v.values()))==10 for v in cells.values())
decode=defaultdict(list)
for (turn,phase,step,layer),ids in cells.items():
 if phase==1:decode[layer].append((step,set(ids.values())))
assert len(decode)==48 and all(len(v)==353 for v in decode.values())
locality=[]
for layer,seq in sorted(decode.items()):
 seq.sort();overlaps=[len(a[1]&b[1])/10 for a,b in zip(seq,seq[1:])];seen=set.union(*(v for _,v in seq));locality.append(dict(layer=layer,consecutive_token_recall_mean=statistics.mean(overlaps),consecutive_token_recall_median=statistics.median(overlaps),unique_experts=len(seen)))
curves=[]
for gib in [8,12,13,14,16,20]:
 cache=OrderedDict();used=0;budget=gib*1024**3;tot=defaultdict(int)
 for (_t,ph,_step,layer),ids in batches.items():
  unique=list(dict.fromkeys(ids));protect={(layer,e) for e in unique}
  for e in unique:
   key=(layer,e);tot['lookups']+=1
   if ph==1:tot['decode_lookups']+=1
   if key in cache:
    tot['hits']+=1
    if ph==1:tot['decode_hits']+=1
   else:
    cache[key]=sizes[layer];used+=sizes[layer];tot['read_bytes']+=sizes[layer]
    if ph==1:tot['decode_read_bytes']+=sizes[layer]
  for e in ids:cache.move_to_end((layer,e))
  while used>budget:
   victim,cost=cache.popitem(last=False);assert victim not in protect;used-=cost
 curves.append(dict(cache_gib=gib,overall_hit_percent=100*tot['hits']/tot['lookups'],decode_hit_percent=100*tot['decode_hits']/tot['decode_lookups'],decode_read_mib_per_token=tot['decode_read_bytes']/353/1024**2,**dict(tot)))
answer=json.loads(args.native_result.read_text());assert answer['passed']
report=dict(schema_version=1,scope='Flash P2 fixed353 exact-answer expert-ID trace',passed=True,source_answer_sha256=answer['answer_sha256'],trace_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),trace_records=rows,decode_tokens=353,layers=48,native_top_k=10,prefill_cells=sum(1 for k in cells if k[1]==0),decode_cells=sum(1 for k in cells if k[1]==1),consecutive_token_recall_mean=statistics.mean(x['consecutive_token_recall_mean'] for x in locality),layers_locality=locality,lru_projection=curves,native_metrics=answer['metrics'],production_changed=False,limitations=['Diagnostic tracer adds graph nodes/barriers; timing is not a performance benchmark.','Prefill final layer computes only the last token; missing earlier final-layer cells are normal output cropping.','LRU projection starts empty and ignores startup cache entries; no claim of exact native hit-rate validation.','Consecutive-token route persistence is a baseline prediction; no early-router logits were captured, so actual early-router lookahead accuracy remains unmeasured.','Cache sizes are offline projections only; larger caches have not passed RAM admission.'])
args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k] for k in ['trace_records','decode_cells','prefill_cells','consecutive_token_recall_mean','lru_projection']}))
