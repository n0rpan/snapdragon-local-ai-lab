"""Prompt studio: the single embedded web page (HTML, CSS and script) the Studio serves at "/".
No logic here; the routes it calls are in `studio`.
"""
PAGE = r"""<!doctype html><html><head><meta charset="utf-8">
<title>Prompt studio</title>
<style>
:root{--bg:#14100e;--fg:#eee5dc;--mut:#9a8d80;--acc:#f0a02f;--card:#1e1815;--line:#352a22}
*{box-sizing:border-box}
body{margin:0;font:15px/1.5 ui-sans-serif,system-ui,sans-serif;background:var(--bg);color:var(--fg)}
.wrap{max-width:980px;margin:0 auto;padding:28px 20px 80px}
h1{font-size:20px;font-weight:600;margin:0 0 4px}
.sub{color:var(--mut);font-size:13px;margin-bottom:22px}
.panel{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:16px}
textarea{width:100%;min-height:96px;resize:vertical;background:#0f0c0a;color:var(--fg);
 border:1px solid var(--line);border-radius:10px;padding:12px;font:inherit}
.row{display:flex;gap:14px;flex-wrap:wrap;align-items:center;margin-top:14px}
label{font-size:13px;color:var(--mut);display:flex;align-items:center;gap:7px}
input[type=number]{width:74px;background:#0f0c0a;color:var(--fg);border:1px solid var(--line);
 border-radius:8px;padding:6px 8px}
input[type=range]{accent-color:var(--acc);width:130px}
button{background:var(--acc);color:#1a1206;border:0;border-radius:10px;padding:11px 22px;
 font-weight:650;font-size:15px;cursor:pointer}
button:disabled{opacity:.5;cursor:default}
button.ghost{background:transparent;color:var(--fg);border:1px solid var(--line);font-weight:500;padding:8px 14px;font-size:13px}
#drop,#maskDrop,#eDrop,#eMaskDrop{border:1px dashed var(--line);border-radius:10px;padding:10px;font-size:13px;color:var(--mut);
 cursor:pointer;text-align:center}
#drop.over{border-color:var(--acc);color:var(--acc)}
#thumb,#maskThumb,#eThumb,#eMaskThumb{max-height:90px;border-radius:8px;display:none;margin-top:8px}
#status{margin-top:16px;font-size:13px;color:var(--mut);min-height:20px}
#out{margin-top:20px}
#out img{max-width:100%;border-radius:14px;border:1px solid var(--line);display:block}
#out .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:12px}
#out .grid img{cursor:pointer;border-width:2px}
#out .grid img.chosen{border-color:var(--acc)}
#actions{margin-top:14px}
.hidden{display:none}
kbd{background:#0f0c0a;border:1px solid var(--line);border-radius:5px;padding:1px 5px;font-size:12px}
</style></head><body><div class="wrap">
<h1>Prompt studio</h1>
<div class="sub">Z-Image Turbo · Q4 / Q8 · local image generation · type a prompt, <kbd>Ctrl</kbd>+<kbd>Enter</kbd></div>

<div id="setup" class="sub"></div>
<div class="panel">
  <textarea id="p" placeholder="Describe the image…" autofocus></textarea>
  <div class="row">
    <div id="drop">📎 Drop a photo to edit it (optional)
      <img id="thumb"><input type="file" id="f" accept="image/*" class="hidden">
    </div>
    <label id="ePresetWrap" class="hidden">edit <select id="ePreset">
      <option value="light">Light touch — keep faces and layout</option>
      <option value="restyle">Restyle</option>
      <option value="reimagine">Reimagine</option></select></label>
    <label id="advWrap" class="hidden"><input type="checkbox" id="adv"> advanced</label>
    <label id="sWrap" class="hidden">edit strength <input type="range" id="s" min="0.05" max="0.95" step="0.05" value="0.3">
      <span id="sv">0.30</span></label>
    <div id="maskDrop" class="hidden">Mask (optional): white = change, black = keep
      <img id="maskThumb"><input type="file" id="mf" accept="image/*" class="hidden">
    </div>
  </div>
  <div class="row"><button id="draft" type="button">Fast draft</button><button id="final" type="button">Final image</button><button id="draft512" type="button" class="ghost hidden">Draft 512 (klein)</button><span id="presetHint" class="sub"></span></div>
  <div class="row">
    <label>model <select id="model"><option value="q4">Q4 — smaller</option><option value="q8" disabled>Q8 — downloading</option><option value="klein-q8" disabled>FLUX.2 klein 4B (4 steps)</option><option value="klein-q4" disabled>FLUX.2 klein 4B Q4 (4 steps)</option></select></label>
    <label>size <select id="size"><option value="256">256 × 256</option><option value="512">512 × 512</option><option value="768">768 × 768</option><option value="1024">1024 × 1024</option><option value="custom">Custom</option></select></label>
    <label>width <input type="number" id="w" value="256" min="256" max="1536" step="64"></label>
    <label>height <input type="number" id="h" value="256" min="256" max="1536" step="64"></label>
    <label>steps <input type="number" id="st" value="8" min="1" max="50"></label>
    <label>seed <input type="number" id="seed" placeholder="42 (default)"></label>
    <label>variations <select id="vars"><option>1</option><option>2</option><option>3</option><option>4</option></select></label>
    <label><input type="checkbox" id="cpu" checked> CPU (recommended)</label>
    <label id="engineWrap" class="hidden" title="FLUX.2 klein text-to-image engine">engine <select id="engine"><option value="auto">auto (oxi at 512, else sd.cpp)</option><option value="oxi">oxi (512 only)</option><option value="sdcpp">sd.cpp</option></select></label>
    <button id="go">Generate</button><span id="genEst" class="sub"></span>
  </div>
  <div id="status"></div>
</div>
<div class="panel" id="editPanel" style="margin-top:14px">
  <div class="sub" style="margin-bottom:8px"><b>Edit (instruction)</b> · FLUX.2 klein · upload a photo, say what to change</div>
  <div class="row" style="margin-top:0">
    <div id="eDrop">Photo to edit <img id="eThumb"><input type="file" id="ef" accept="image/*" class="hidden"></div>
    <div id="eMaskDrop">Mask (optional, PNG): white = change, black = keep exactly
      <img id="eMaskThumb"><input type="file" id="emf" accept="image/png" class="hidden"></div>
  </div>
  <textarea id="eInstr" style="min-height:56px;margin-top:12px" placeholder="e.g. change the jacket colour to dark green"></textarea>
  <div class="row">
    <label>size <select id="eSize"><option value="512">512 (default)</option><option value="768">768</option></select></label>
    <label>model <select id="eModel"><option value="q8">Q8</option><option value="q4">Q4 (smaller)</option></select></label>
    <label>seed <input type="number" id="eSeed" placeholder="42 (default)"></label>
    <button id="eGo" type="button">Edit</button><span id="eEst" class="sub"></span>
  </div>
  <div class="sub" id="eNote"></div>
</div>
<div id="out"></div>
<div id="actions" class="panel hidden">
  <div class="sub" id="chosenInfo" style="margin-bottom:6px"></div>
  <div class="row" style="margin-top:0">
    <button id="upBtn" type="button" class="ghost">Quick upscale ×4</button><span id="upEst" class="sub"></span>
  </div>
  <div class="row">
    <button id="refBtn" type="button">Final (refine)</button>
    <label>size <select id="refTarget"></select></label>
    <label>strength <input type="number" id="refStrength" min="0.05" max="0.95" step="0.05" value="0.45"></label>
    <label id="postWrap"><input type="checkbox" id="post"> ESRGAN ×2 after</label>
    <span id="refEst" class="sub"></span>
  </div>
  <div class="sub" id="upNote"></div>
</div>

<script>
const $=id=>document.getElementById(id);
let initImage=null, maskImage=null, editImage=null, editMask=null, editInfo=null, chosen=null, upscaleInfo=null;
let modelInitialized=false;
let presetSettings={draft:{model:'q4',width:256,height:256,steps:8},final:{model:'q4',width:1024,height:1024,steps:8}};
const fmt=s=>s==null?'':(s<90?Math.round(s)+' seconds':Math.round(s/60)+' minutes');
const estNote=()=>upscaleInfo&&!upscaleInfo.estimates_calibrated?' (uncalibrated estimate)':'';
function applyPreset(name){
  const p=presetSettings[name]; if(!p)return;
  $('model').value=p.model==='q8' && $('model').querySelector('[value="q8"]').disabled?'q4':p.model;
  $('w').value=p.width; $('h').value=p.height; $('size').value=p.width===p.height?String(p.width):'custom';
  $('st').value=p.steps; $('cpu').checked=true; syncKlein();
  const seconds=initImage?null:p.expected_seconds;
  const label={draft:'Draft',final:'Final',draft512:'Draft 512 (klein)'}[name]||name;
  $('presetHint').textContent=label+` · ${p.width} × ${p.height}`+(seconds?` · about ${fmt(seconds)}`:'');
  refreshEstimate();
}
$('draft').onclick=()=>applyPreset('draft');
$('final').onclick=()=>applyPreset('final');
$('draft512').onclick=()=>applyPreset('draft512');
['model','w','h','st','cpu'].forEach(id=>$(id).addEventListener('change',()=>{$('presetHint').textContent='';refreshEstimate();}));
// FLUX.2 klein: sizes in multiples of 16 up to 1024, steps fixed at 4 (the field is ignored by the server).
function syncKlein(){
  const k=$('model').value.startsWith('klein');
  ['w','h'].forEach(id=>{$(id).step=k?16:64; $(id).max=k?1024:1536;});
  $('st').disabled=k; $('st').title=k?'FLUX.2 klein always uses 4 steps':'';
  $('engineWrap').classList.toggle('hidden',!k);
}
$('engine').addEventListener('change',async()=>{
  try{await fetch('/engine',{method:'POST',body:JSON.stringify({engine:$('engine').value})});}catch(e){}
  refreshEstimate();
});
$('model').addEventListener('change',syncKlein);
['vars','ePreset','s'].forEach(id=>$(id).addEventListener('change',refreshEstimate));
$('size').onchange=()=>{$('presetHint').textContent='';if($('size').value!=='custom') $('w').value=$('h').value=$('size').value;refreshEstimate();};
function editStrength(){return $('adv').checked?+$('s').value:(window.editPresets||{light:0.3,restyle:0.55,reimagine:0.8})[$('ePreset').value];}
async function refreshEstimate(){
  const q=new URLSearchParams({model:$('model').value,width:$('w').value,height:$('h').value,steps:$('st').value,variations:$('vars').value});
  if(initImage){q.set('edit','1');q.set('strength',editStrength());}
  try{const j=await (await fetch('/estimate?'+q)).json();
    $('genEst').textContent=(j.seconds?`about ${fmt(j.seconds)}${j.calibrated?'':' (uncalibrated estimate)'}`:'')+(j.engine?` · ${j.engine_reason||j.engine}`:'');}
  catch(e){$('genEst').textContent='';}
}
async function refreshModels(){
  try {
    const m=await (await fetch('/models')).json();
    const q8=$('model').querySelector('[value="q8"]');
    q8.disabled=!m.q8.ready;
    const kleinReady=m.klein?.models||{};
    ['klein-q8','klein-q4'].forEach(v=>{const o=$('model').querySelector(`[value="${v}"]`); if(o) o.disabled=!kleinReady[v];});
    if(m.presets)presetSettings=m.presets;
    // The 512 klein draft button is hidden unless its model files are present; nothing else changes.
    const d512=presetSettings.draft512;
    $('draft512').classList.toggle('hidden', !(d512 && kleinReady[d512.model]));
    if(m.edit_presets)window.editPresets=m.edit_presets;
    if(m.engine&&document.activeElement!==$('engine')){$('engine').value=m.engine.engine;
      $('engine').querySelector('[value="oxi"]').disabled=!m.engine.oxi_ready;}
    upscaleInfo=m.upscale||null;
    editInfo=m.edit||null; renderEdit();
    $('cpu').disabled=!m.gpu?.optimized; if($('cpu').disabled)$('cpu').checked=true;
    if(!modelInitialized){
      if(m.q8.ready) $('model').value='q8';
      applyPreset('draft');
      if(upscaleInfo){
        $('refTarget').innerHTML=upscaleInfo.refine_targets.map(t=>`<option value="${t}"${t===upscaleInfo.refine_target?' selected':''}>${t}</option>`).join('');
        $('refStrength').value=upscaleInfo.refine_strength;
      }
      modelInitialized=true;
    }
    $('setup').textContent=(m.q8.ready?'Q8 verified and ready. ':'Q8: '+m.q8.setup_state+'. ')+(m.benchmarks||'');
    if(m.gpu?.optimized){const g=m.gpu.models[$('model').value];if(g)$('setup').textContent+=` Corrected GPU preview: up to ${g.max_width}×${g.max_height}. GPU generation can pause the desktop; CPU is recommended.`; }
    q8.textContent=m.q8.ready?'Q8 — higher precision':`Q8 — ${m.q8.setup_state} (${m.q8.download_mb} / 6273 MB)`;
    renderActions();
  } catch(e) {}
}
refreshModels(); setInterval(refreshModels,10000);
$('drop').onclick=()=>$('f').click();
$('f').onchange=e=>{const f=e.target.files[0]; if(f) upload(f);};
$('maskDrop').onclick=()=>$('mf').click();
$('mf').onchange=e=>{const f=e.target.files[0]; if(f) uploadMask(f);};
$('s').oninput=()=>$('sv').textContent=(+$('s').value).toFixed(2);
$('adv').onchange=()=>{$('sWrap').classList.toggle('hidden',!$('adv').checked);$('ePreset').disabled=$('adv').checked;refreshEstimate();};
['dragover','dragenter'].forEach(ev=>$('drop').addEventListener(ev,e=>{e.preventDefault();$('drop').classList.add('over')}));
['dragleave','drop'].forEach(ev=>$('drop').addEventListener(ev,e=>{e.preventDefault();$('drop').classList.remove('over')}));
$('drop').addEventListener('drop',e=>{const f=e.dataTransfer.files[0]; if(f) upload(f);});

async function sendFile(f){
  const fd=new FormData(); fd.append('file',f,f.name);
  const r=await fetch('/upload',{method:'POST',body:fd}); return r.json();
}
async function upload(f){
  const j=await sendFile(f);
  if(j.error){$('status').textContent='upload failed: '+j.error; return;}
  initImage=j.path; $('presetHint').textContent='';
  $('thumb').src='/file?path='+encodeURIComponent(j.path);
  $('thumb').style.display='block'; $('drop').firstChild.textContent='📎 Photo loaded — click to replace';
  ['ePresetWrap','advWrap','maskDrop'].forEach(id=>$(id).classList.remove('hidden'));
  refreshEstimate();
}
async function uploadMask(f){
  const j=await sendFile(f);
  if(j.error){$('status').textContent='mask upload failed: '+j.error; return;}
  maskImage=j.path;
  $('maskThumb').src='/file?path='+encodeURIComponent(j.path); $('maskThumb').style.display='block';
}

function renderEdit(){
  if(!editInfo)return;
  const ok=editInfo.models[$('eModel').value];
  const est=editInfo.estimates[$('eSize').value];
  $('eEst').textContent=est?`about ${fmt(est)}${editInfo.estimates_calibrated?'':' (uncalibrated estimate)'}${editMask?' + mask composite':''}`:'';
  $('eNote').textContent=ok?'':'Edit model files are not installed for this choice; see docs/STUDIO-EDIT.md.';
  if(!$('go').disabled)$('eGo').disabled=!ok;
}
['eSize','eModel'].forEach(id=>$(id).addEventListener('change',renderEdit));
$('eDrop').onclick=()=>$('ef').click();
$('ef').onchange=async e=>{const f=e.target.files[0]; if(!f)return; const j=await sendFile(f);
  if(j.error){$('status').textContent='upload failed: '+j.error; return;}
  editImage=j.path; $('eThumb').src='/file?path='+encodeURIComponent(j.path); $('eThumb').style.display='block';};
$('eMaskDrop').onclick=()=>$('emf').click();
$('emf').onchange=async e=>{const f=e.target.files[0]; if(!f)return; const j=await sendFile(f);
  if(j.error){$('status').textContent='mask upload failed: '+j.error; return;}
  editMask=j.path; $('eMaskThumb').src='/file?path='+encodeURIComponent(j.path); $('eMaskThumb').style.display='block'; renderEdit();};
$('eGo').onclick=()=>{
  const instruction=$('eInstr').value.trim();
  if(!editImage){$('status').textContent='Upload the photo to edit first.'; return;}
  if(!instruction){$('status').textContent='Type what to change first.'; return;}
  const body={instruction, init:editImage, size:+$('eSize').value, model:$('eModel').value,
    seed:$('eSeed').value?+$('eSeed').value:null};
  if(editMask) body.mask=editMask;
  submit('/edit',body);
};
$('go').onclick=run;
$('p').addEventListener('keydown',e=>{if((e.ctrlKey||e.metaKey)&&e.key==='Enter') run();});

async function submit(url,body){
  if($('go').disabled) return;
  setBusy(true); $('status').textContent='starting…';
  try {
    const r=await fetch(url,{method:'POST',body:JSON.stringify(body)});
    const {id,error}=await r.json();
    if(error){$('status').textContent='error: '+error; setBusy(false); return;}
    poll(id);
  } catch(e){$('status').textContent='Connection lost; reload to reconnect.'; setBusy(false);}
}
function setBusy(b){['go','upBtn','refBtn','eGo'].forEach(id=>$(id).disabled=b); if(!b){renderActions();renderEdit();}}

async function run(){
  const prompt=$('p').value.trim();
  if(!prompt){$('status').textContent='Type a prompt first.'; return;}
  const body={prompt, model:$('model').value, width:+$('w').value, height:+$('h').value, steps:+$('st').value,
    seed:$('seed').value?+$('seed').value:null, cpu:$('cpu').checked, init:initImage, variations:+$('vars').value};
  if(initImage){
    if($('adv').checked){body.edit_preset='custom'; body.strength=+$('s').value;} else body.edit_preset=$('ePreset').value;
    if(maskImage) body.mask=maskImage;
  }
  submit('/generate',body);
}

function showImages(images){
  if(images.length===1){
    $('out').innerHTML=`<img src="/file?path=${encodeURIComponent(images[0].path)}&t=${Date.now()}">`;
    choose(images[0], null);
  } else {
    $('out').innerHTML='<div class="sub">Pick one to upscale or refine.</div><div class="grid">'+images.map((im,i)=>
      `<img data-i="${i}" title="seed ${im.seed}" src="/file?path=${encodeURIComponent(im.path)}&t=${Date.now()}">`).join('')+'</div>';
    $('out').querySelectorAll('.grid img').forEach(el=>el.onclick=()=>choose(images[+el.dataset.i],el));
    chosen=null; $('actions').classList.add('hidden');
  }
}
function choose(image,el){
  chosen=image;
  $('out').querySelectorAll('.grid img').forEach(x=>x.classList.toggle('chosen',x===el));
  renderActions();
}
function renderActions(){
  if(!chosen){$('actions').classList.add('hidden'); return;}
  $('actions').classList.remove('hidden');
  const a=chosen.actions||{}, busy=$('go').disabled;
  $('chosenInfo').textContent=`Chosen: ${chosen.width} × ${chosen.height} · seed ${chosen.seed}`;
  const canUp=a.upscale!=null;
  $('upBtn').classList.toggle('hidden',!canUp); $('upBtn').disabled=busy||!canUp;
  $('upEst').textContent=canUp?`ESRGAN only, ${chosen.width*(upscaleInfo?.esrgan_scale||4)} px · keeps the image exactly, a little soft · about ${fmt(a.upscale)}${estNote()}`:'';
  $('upNote').textContent=canUp?'':'Quick upscale is unavailable: no ESRGAN model is installed. Refine will enlarge with a plain resize first.';
  $('postWrap').classList.toggle('hidden',!a.post_upscale);
  const t=$('refTarget').value, s=a.refine?.[t];
  $('refBtn').disabled=busy||s==null;
  $('refEst').textContent=s!=null?`same prompt and seed, keeps the layout and adds detail · about ${fmt(s)} at strength ${upscaleInfo?.refine_strength??0.45}${estNote()}`:'not available for this image size';
}
$('refTarget').onchange=renderActions;
$('upBtn').onclick=()=>{ if(chosen) submit('/upscale',{image:chosen.id}); };
$('refBtn').onclick=()=>{ if(chosen) submit('/refine',{image:chosen.id, target:+$('refTarget').value,
  strength:+$('refStrength').value, post_upscale:$('post').checked && !$('postWrap').classList.contains('hidden')}); };

async function poll(id){
  const t0=Date.now();
  while(true){
    let j;
    try {j=await (await fetch('/job?id='+id)).json();}
    catch(e){$('status').textContent='Reconnecting…'; await new Promise(r=>setTimeout(r,2000)); continue;}
    if(j.state==='done'){
      $('status').textContent=`done in ${((Date.now()-t0)/1000).toFixed(1)}s${j.engine?' · '+j.engine:''}${j.engine_note?' ('+j.engine_note+')':''} · ${j.path}`;
      setBusy(false);
      if(j.images&&j.images.length) showImages(j.images);
      else $('out').innerHTML=`<img src="/file?path=${encodeURIComponent(j.path)}&t=${Date.now()}">`;
      return;
    }
    if(j.state==='error'){$('status').textContent='failed: '+j.error; setBusy(false); return;}
    $('status').textContent=`${j.step?j.step+' · ':''}${j.state}… ${((Date.now()-t0)/1000).toFixed(0)}s`;
    await new Promise(r=>setTimeout(r,900));
  }
}
</script></body></html>"""
