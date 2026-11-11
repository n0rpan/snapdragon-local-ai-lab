"""P4: GGUF header-only bytes/types/shapes, with no weight payload reads."""
from pathlib import Path
import argparse,collections,datetime,importlib.util,json,sys
ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--runtime',type=Path,default=Path.home()/'.config/local-ai/llm/runtime.json');ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
s=Path(__file__).parent;sp=importlib.util.spec_from_file_location('budget',s/'gguf_header.py');b=importlib.util.module_from_spec(sp);sp.loader.exec_module(b);b.GGML_TYPES[42]=('Q2_0',64,18)
profile=json.loads(args.runtime.read_text())['profiles']['qwen3.8-flash-next-local'];tensors=[];shards=[];kv={}
for ident in profile['bigmoe']['shards']:
 p=Path(ident['path']);h,ts,size=b.read_gguf(p);assert size==ident['bytes'];tensors+=ts;kv.update(h);shards.append(dict(name=p.name,bytes=size,recorded_sha256=ident['sha256'],header_tensors=len(ts),payload_rehashed=False))
assert len({t.name for t in tensors})==len(tensors)
bytype=collections.defaultdict(lambda:dict(tensors=0,bytes=0,elements=0));groups=collections.defaultdict(lambda:dict(tensors=0,bytes=0,elements=0));layers={};selected=[]
for t in tensors:
 g='routed_experts' if '_exps.' in t.name else 'ngram_table' if 'per_layer_token_embd' in t.name else 'dense_and_other'
 for dest,key in [(bytype,t.type_name),(groups,g)]:
  dest[key]['tensors']+=1;dest[key]['bytes']+=t.nbytes;dest[key]['elements']+=t.nelem
 if g=='routed_experts':
  layer=int(t.name.split('.')[1]);layers.setdefault(layer,[]).append(dict(name=t.name,type=t.type_name,shape=t.shape,bytes=t.nbytes,bytes_per_expert=t.nbytes//512))
 if t.name.startswith('blk.0.') or g=='ngram_table' or t.name in ['output.weight','token_embd.weight']:selected.append(dict(name=t.name,type=t.type_name,shape=t.shape,bytes=t.nbytes))
a='qwen4exp';meta={k:v for k,v in kv.items() if k=='general.architecture' or k.startswith(a+'.') and (not isinstance(v,list) or len(v)<=64)}
assert meta['general.architecture']=='qwen4exp' and meta[a+'.block_count']==48 and meta[a+'.expert_count']==512 and meta[a+'.expert_used_count']==10
assert len(layers)==48 and all(len(v)==3 for v in layers.values())
row=dict(schema_version=1,kind='flash-P4-header-audit',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),model='Qwen3.8-Flash-Next UD-IQ3_XXS',header_only=True,metadata=meta,shards=shards,types=dict(bytype),groups=dict(groups),expert_layers=layers,selected_tensors=selected,routed_expert_bytes_per_native_top10_token=sum(t.nbytes//512*10 for t in tensors if '_exps.' in t.name),all_nonexpert_tensor_storage_bytes=groups['dense_and_other']['bytes'],ngram_storage_bytes=groups['ngram_table']['bytes'],notes=['Model hashes are existing verified-download identities; payload hashes were not recomputed.','Storage bytes are not runtime traffic; selected embedding rows, caches, recurrent state and dtype expansion must be measured.','Q2_0 type42 size64/18 is taken from the pinned BigMoeOnEdge ggml header.'],passed=True)
args.output.write_text(json.dumps(row,indent=2)+'\n');print(json.dumps({k:row[k] for k in ['passed','types','groups','routed_expert_bytes_per_native_top10_token']}))
