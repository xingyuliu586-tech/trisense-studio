'use strict';
const $ = (id) => document.getElementById(id);
const state = {research:null, samples:[], result:null, file:null, heatmap:'audio', busy:false, selection:0};
const colors = ['#a4b3bf','#7c98b0','#c2a47c','#167d8d'];
const labels = ['文本','语音','视觉'];
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const num = (x, n=3) => Number.isFinite(x) ? x.toFixed(n) : '—';
const signed = (x,n=3) => Number.isFinite(x) ? `${x>=0?'+':''}${x.toFixed(n)}` : '—';
const pct = (x) => Number.isFinite(x) ? `${(100*x).toFixed(1)}%` : '—';
function error(message) { $('error').hidden = !message; $('error').textContent = message || ''; }
async function api(path, options={}) {
  if (window.trisenseStaticAPI) return window.trisenseStaticAPI(path, options);
  const response = await fetch(path, options);
  const data = await response.json();
  if (!response.ok) throw new Error(typeof data.detail==='string' ? data.detail : '输入内容未通过校验，请检查文件格式和参数。');
  return data;
}
function route() {
  const names={overview:'项目概览',analyze:'分析工作台',experiments:'实验对比',method:'方法与使用'};
  const page=location.hash.slice(1) in names ? location.hash.slice(1) : 'overview';
  document.querySelectorAll('.page').forEach(el=>el.hidden=el.id!==`page-${page}`);
  document.querySelectorAll('[data-page]').forEach(el=>{el.classList.toggle('active',el.dataset.page===page); if(el.dataset.page===page) el.setAttribute('aria-current','page'); else el.removeAttribute('aria-current');});
  $('breadcrumb').textContent=names[page];
  window.scrollTo(0,0);
  if(page==='analyze') requestAnimationFrame(drawFeatures);
}
const svgText=(x,y,t,extra='')=>`<text x="${x}" y="${y}" fill="#6f7e88" font-size="14" ${extra}>${esc(t)}</text>`;
function bars(rows,metric,compact=false) {
  const W=570,H=compact?225:250,left=128,right=85,top=20,step=45,plot=W-left-right;
  const max=Math.max(...rows.map(r=>r[metric+'_mean']+r[metric+'_sd']))*1.12;
  let s=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="四种模型 ${esc(metric)} 均值与标准差">`;
  for(let i=0;i<=4;i++){let x=left+plot*i/4;s+=`<path d="M${x},10V${top+step*4-10}" stroke="#edf1f3"/>`+svgText(x,top+step*4+14,num(max*i/4,2),'text-anchor="middle"');}
  rows.forEach((r,i)=>{let y=top+step*i,w=plot*r[metric+'_mean']/max,sd=plot*r[metric+'_sd']/max;
    s+=svgText(left-12,y+19,r.model,'text-anchor="end"')+`<rect x="${left}" y="${y}" width="${w}" height="28" rx="3" fill="${colors[i]}"/><path d="M${left+w-sd},${y+14}h${2*sd} M${left+w-sd},${y+9}v10 M${left+w+sd},${y+9}v10" stroke="#3a5965" fill="none"/>`+svgText(left+w+sd+9,y+19,num(r[metric+'_mean']));});
  return s+'</svg>';
}
function lines(series,{ticks,yLabel='MAE',min,max}={}) {
  const W=570,H=265,L=47,R=18,T=22,B=45,values=series.flatMap(s=>s.values);
  min=min ?? Math.floor((Math.min(...values)-.02)*20)/20;max=max ?? Math.ceil((Math.max(...values)+.02)*20)/20;
  if(max<=min) max=min+.1;
  const x=i=>L+i*(W-L-R)/Math.max(1,ticks.length-1),y=v=>T+(max-v)/(max-min)*(H-T-B);
  let s=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(yLabel)} 曲线">`+svgText(L,12,yLabel);
  for(let i=0;i<=4;i++){const v=min+(max-min)*i/4;s+=`<path d="M${L},${y(v)}H${W-R}" stroke="#e9eef1"/>`+svgText(L-8,y(v)+4,num(v,2),'text-anchor="end"');}
  ticks.forEach((t,i)=>s+=svgText(x(i),H-B+24,t,'text-anchor="middle"'));
  series.forEach((r,i)=>{const color=r.color||colors[i];s+=`<polyline points="${r.values.map((v,j)=>`${x(j)},${y(v)}`).join(' ')}" stroke="${color}" stroke-width="2.2" fill="none"/>`;r.values.forEach((v,j)=>s+=`<circle cx="${x(j)}" cy="${y(v)}" r="3" fill="${color}"><title>${esc(r.name)} · ${esc(ticks[j])}: ${num(v,4)}</title></circle>`);});
  return s+'</svg><div class="legend">'+series.map((r,i)=>`<span><i style="background:${r.color||colors[i]}"></i>${esc(r.name)}</span>`).join('')+'</div>';
}
function renderResearch() {
  const r=state.research,q=r.q2;
  $('metric-cards').innerHTML=[['Macro-F1',num(q.f1_macro),'三类极性 · 宏平均'],['MAE',num(q.mae),'连续情感强度 · 越低越好'],['Accuracy',pct(q.accuracy),'内部验证 · 728 条样本'],['Pearson',num(q.pearson),'预测与标注的相关系数']].map(([name,value,foot])=>`<article class="metric"><div class="metric-name">${name}</div><div class="metric-value">${value}</div><div class="metric-foot">${foot}</div></article>`).join('');
  $('home-benchmark').innerHTML=bars(r.comparison,'mae',true);
  $('comparison-chart').innerHTML=bars(r.comparison,$('metric-select').value);
  const scenarios=[...new Set(r.stress.map(x=>x.scenario))],names={clean:'完整',text30:'文本 30%',audio30:'语音 30%',vision30:'视觉 30%',joint30:'同步 30%'};
  $('stress-chart').innerHTML=lines(r.comparison.map(m=>({name:m.model,values:scenarios.map(s=>r.stress.find(v=>v.model===m.model&&v.scenario===s).mae_mean)})),{ticks:scenarios.map(s=>names[s]||s),yLabel:'MAE'});
  $('comparison-table').innerHTML='<table><thead><tr><th>模型</th><th>下游参数</th><th>Accuracy</th><th>Macro-F1</th><th>MAE ↓</th><th>Pearson</th></tr></thead><tbody>'+r.comparison.map(m=>`<tr class="${m.model==='MaskedFusionNet'?'highlight':''}"><td>${esc(m.model)}</td><td>${m.parameters.toLocaleString()}</td>${['accuracy','f1_macro','mae','pearson'].map(k=>`<td>${num(m[k+'_mean'])} ± ${num(m[k+'_sd'])}</td>`).join('')}</tr>`).join('')+'</tbody></table>';
  const training=Object.entries(r.training),n=Math.max(...training.map(([,v])=>v.length));
  $('learning-chart').innerHTML=lines(training.map(([k,v],i)=>({name:k==='baseline'?'基线':'缺失增强',values:v.map(x=>x.f1_macro),color:i?'#b69b76':'#167d8d'})),{ticks:Array.from({length:n},(_,i)=>i+1),yLabel:'Macro-F1'});
  $('protocol').innerHTML=`<div class="protocol-grid"><div><span>训练 / 验证</span>${r.train_n.toLocaleString()} / ${r.valid_n}</div><div><span>随机种子</span>${r.protocol.seeds.join(' / ')}</div><div><span>最长训练</span>${r.protocol.epochs_max} 个 epoch</div><div><span>任务目标</span>三类极性 + 强度回归</div></div><ul class="protocol-list"><li>四模型在相同数据划分下比较，未使用缺失增强。误差条为 3 个种子的样本标准差，不是置信区间。</li><li>验证集参与早停与选模。正式 Q2 集成采用基线和增强模型，不能直接与架构实验混为同一协议。</li><li>原文本压力实验将词元替换为 UNK 后重新编码 BERT；历史结果保留此协议。实时工作台只开放音视频局部缺失。</li><li>附件三、四无真实标签。类别概率与预测强度仅作为模型输出展示。</li></ul>`;
}
function setBusy(b) { state.busy=b; $('controls-fieldset').disabled=b; $('analyze').textContent=b?'正在计算…':'运行分析'; updateExport(); }
function updateExport() { const off=state.busy||!state.result?.prediction; $('export-json').disabled=off;$('export-csv').disabled=off; }
function parameters() {return {predictor:$('predictor').value,mode:$('missing-mode').value,rate:$('missing-mode').value==='none'?0:Number($('rate').value)/100,location:$('location').value};}
function inputStatus() {
  const row=state.samples.find(s=>s.id===$('sample').value);
  $('sample').disabled=!!state.file;
  $('clear-file').hidden=!state.file;
  $('upload-title').textContent=state.file ? state.file.name : '导入自己的特征';
  $('sample-hint').textContent=state.file?'已选择文件；点击运行后校验并计算。':row?.synthetic?'随机数值特征，不代表真实视频。':row?.live_available?'本机已导入特征，可以重新推理。':'仅保存历史结果；未导入原始特征。';
  $('analyze').disabled=state.busy||(!state.file&&!row?.live_available);
  const off=$('missing-mode').value==='none';$('rate').disabled=off;$('location').disabled=off;
}
async function selectSample() {
  const seq=++state.selection;
  error(null);inputStatus();
  try {const data=await api('/api/samples/'+encodeURIComponent($('sample').value));if(seq!==state.selection)return;state.result=data;renderResult();} catch(e){error(e.message);}
}
async function run() {
  if(state.busy)return;
  error(null);setBusy(true);
  try {
    let data;
    if(state.file){const form=new FormData();form.append('file',state.file);Object.entries(parameters()).forEach(([k,v])=>form.append(k,String(v)));data=await api('/api/upload',{method:'POST',body:form});}
    else data=await api('/api/analyze',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({sample_id:$('sample').value,...parameters()})});
    state.result=data;renderResult();
  } catch(e){error(e.message);} finally {setBusy(false);inputStatus();}
}
function renderResult() {
  const r=state.result,p=r.prediction,live=r.source==='live';
  const banner=r.synthetic?'合成接口样例 · 随机特征，仅验证软件操作与计算流程，不用于情感识别性能展示。':live?'实时推理 · 使用已训练检查点在本机计算，上传文件不会保存到磁盘。':'历史实验记录 · '+r.id+' · 以下为原 Q3 实验结果，本次尚未重新计算。';
  $('result-banner').classList.toggle('synthetic',r.synthetic);$('result-banner').textContent=banner;
  $('result-source').textContent=live?(r.predictor==='ensemble'?'Q2 加权集成':'Q3 单模型'):'原实验快照';
  if(!p){$('prediction').innerHTML='<div class="empty-note">样例已载入。点击“运行分析”，生成本次预测及解释。</div>';}
  else {
    const probs=p.probabilities?'<div class="prob-title">三类概率 · 未经校准</div>'+p.probabilities.map((v,i)=>`<div class="prob-row"><span>${['负向','中性','正向'][i]}</span><div class="bar-track"><div class="bar-fill ${i===p.class_id?'highlight':''}" style="width:${v*100}%"></div></div><strong>${pct(v)}</strong></div>`).join(''):`<div class="prob-title">预测类别概率</div><div class="big-confidence">${pct(p.confidence)}</div><p class="caption">历史快照未存储完整概率分布，不补造其他类别数值。</p>`;
    $('prediction').innerHTML=`<div><div class="prediction-label tone-${['negative','neutral','positive'][p.class_id]}">${esc(p.label)}</div><div class="strength-line">情感强度 <strong>${signed(p.strength)}</strong></div><div class="strength-track"><i class="strength-point" style="left:${Math.max(0,Math.min(100,(p.strength+3)/6*100))}%"></i></div><div class="strength-scale"><span>−3 负向</span><span>0</span><span>+3 正向</span></div></div><div>${probs}</div>`;
    if(live)$('prediction').innerHTML+=`<div class="prediction-meta">本次计算 ${num(r.elapsed_ms,0)} ms · CPU · 包含归因与局部遮挡</div>`;
  }
  const a=r.attribution;
  if(a){const max=Math.max(.001,...a.signed.map(Math.abs));$('attribution').innerHTML=a.signed.map((v,i)=>`<div class="contribution-row"><span>${labels[i]}</span><div class="signed-track"><i class="signed-bar" style="left:${v>=0?50:50-Math.abs(v)/max*47}%;width:${Math.abs(v)/max*47}%;background:${v>=0?'#4d9b9c':'#ba7c7c'}"></i></div><span class="signed-number">${signed(v)}</span></div>`).join('')+`<div class="relative-row">${a.relative.map((v,i)=>`<span>${labels[i]}相对作用<strong>${pct(v)}</strong></span>`).join('')}</div><div class="contribution-foot">主要作用：${esc(a.primary)}${Number.isFinite(a.efficiency_error)?` · 守恒残差 ${a.efficiency_error.toExponential(1)}`:''}</div>`;}else $('attribution').innerHTML='<div class="empty-note">运行后展示精确 Shapley 模态贡献。</div>';
  if(r.baseline){$('intervention').innerHTML=`<div class="comparison-mini"><div><small>原始强度 · ${esc(r.baseline.label)}</small><strong>${signed(r.baseline.strength)}</strong></div><div><small>当前强度 · ${esc(p.label)}</small><strong>${signed(p.strength)}</strong></div></div><div class="delta-line">强度变化 ${signed(p.strength-r.baseline.strength)}</div><p class="caption">叠加缺失：${esc(({none:'无',audio:'语音',vision:'视觉','audio+vision':'语音 + 视觉'})[r.intervention.mode])}；覆盖 ${r.intervention.positions.length} 个内容位置。原来已缺失的位置不会恢复。</p>`;}
  else if(r.evidence){const e=r.evidence;$('intervention').innerHTML=`<div class="evidence-list"><div>历史文本窗口<span>词元位置 ${esc(e.text_positions)}</span></div><div>历史语音区间<span>${esc(e.audio_seconds)} s</span></div><div>历史视觉证据<span>${e.vision_status==='可用'?esc(e.vision_seconds)+' s · 帧 '+esc(e.vision_frame):esc(e.vision_status)}</span></div></div><p class="caption">导入同一样本的兼容特征后才能重新计算干预。</p>`;}
  else $('intervention').innerHTML='<div class="empty-note">选择语音或视觉缺失后运行，可比较干预前后的强度。</div>';
  renderLocal();drawFeatures();updateExport();
}
function drawFeatures() {
  const f=state.result?.features,visible=!$('page-analyze').hidden;
  $('mask-canvas').hidden=!f;$('heatmap-canvas').hidden=!f;
  if(!f){$('feature-status').innerHTML='<div class="empty-note">这条历史记录未附带原始特征。导入对应 NPZ 后，显示实际可用性与热图。</div>';$('heatmap-note').textContent='';return;}
  $('feature-status').innerHTML=`<div class="coverage"><span>有效序列长度 <strong>${f.length} / 50</strong></span>${f.coverage.map((v,i)=>`<span>${labels[i]}内容覆盖 <strong>${pct(v)}</strong></span>`).join('')}</div>`;
  if(!visible)return;
  function surface(id,height){const c=$(id),w=Math.max(250,c.clientWidth),dpr=window.devicePixelRatio||1;c.width=Math.round(w*dpr);c.height=Math.round(height*dpr);c.style.height=height+'px';const g=c.getContext('2d');g.scale(dpr,dpr);g.font='10px "Segoe UI", "Microsoft YaHei", sans-serif';g.fillStyle='#70808b';return {g,w};}
  const {g,w}=surface('mask-canvas',104),left=42,right=12,cw=(w-left-right)/f.length;
  f.available.forEach((row,i)=>{g.fillStyle='#70808b';g.fillText(labels[i],0,21+i*23);row.forEach((v,j)=>{g.fillStyle=v?['#6a97b5','#ba9775','#6f9e91'][i]:'#edf1f3';g.fillRect(left+j*cw,10+i*23,Math.max(1,cw-1),16);});});
  g.fillStyle='#83929c';for(let i=0;i<f.length;i+=Math.max(1,Math.ceil(f.length/10)))g.fillText(String(i),left+i*cw,94);
  const heat=f.heatmaps[state.heatmap],surf=surface('heatmap-canvas',259),h=surf.g,ch=8,cellW=(surf.w-left-right)/f.length,m=['text','audio','vision'].indexOf(state.heatmap);
  const color=z=>{const t=Math.min(1,Math.abs(z)/3),a=[245,246,245],b=z<0?[63,117,158]:[183,85,80];return `rgb(${a.map((v,i)=>Math.round(v+(b[i]-v)*t)).join(',')})`;};
  heat.values.forEach((row,i)=>{if(i%4===0){h.fillStyle='#83929c';h.fillText(String(heat.channels[i]),0,13+i*ch);}row.forEach((v,j)=>{h.fillStyle=f.available[m][j]?color(v):'#e5e9ec';h.fillRect(left+j*cellW,5+i*ch,Math.ceil(cellW),ch);});});
  h.fillStyle='#83929c';h.fillText('通道',0,218);for(let i=0;i<f.length;i+=Math.max(1,Math.ceil(f.length/10)))h.fillText(String(i),left+i*cellW,218);
  const cx=left,cy=235;for(let i=0;i<100;i++){h.fillStyle=color(-3+6*i/99);h.fillRect(cx+i*1.1,cy,1.2,8);}h.fillStyle='#83929c';h.fillText('−3',cx,255);h.fillText('0',cx+53,255);h.fillText('+3',cx+100,255);h.fillText('样本内通道 z-score',cx+130,243);
  $('heatmap-note').textContent=`均匀选取 ${heat.channels.length} 个通道；仅为显示按本样本各通道的可用位置标准化，截断至 ±3。灰色表示不可用，横轴为词元索引（从 0 开始），非秒数；颜色不是情感极性。`;
}
function renderLocal(){const local=state.result?.local_effects||{},a=local.audio||[],v=local.vision||[];
  if(!a.length&&!v.length){$('local-chart').innerHTML='<div class="empty-note">实时运行后展示音视频局部窗口的概率影响。已全部缺失的模态不生成虚构窗口。</div>';return;}
  const all=a.length?a:v,ticks=all.map(x=>x.start),series=[];
  if(a.length)series.push({name:'语音',values:a.map(x=>x.delta),color:'#b79a77'});
  if(v.length)series.push({name:'视觉',values:v.map(x=>x.delta),color:'#479088'});
  $('local-chart').innerHTML=lines(series,{ticks,yLabel:'概率差 Δp'});
}
function download(format){const r=state.result;if(!r?.prediction)return;
  const meta={exported_at:new Date().toISOString(),app:'TriSense Studio 1.0.0',...r};let text,type;
  if(format==='json'){text=JSON.stringify(meta,null,2);type='application/json';}
  else {const rows=[['field','value'],['source',r.source],['synthetic',r.synthetic],['sample_id',r.id],['predictor',r.predictor],['class',r.prediction.label],['strength',r.prediction.strength],['class_probability',r.prediction.confidence],['primary_modality',r.attribution?.primary]];r.attribution?.signed.forEach((x,i)=>rows.push([`shapley_${['text','audio','vision'][i]}`,x]));rows.push(['note','Internal validation only; probabilities are not calibrated; synthetic data is not a real video.']);const quote=x=>`"${String(x??'').replace(/"/g,'""')}"`;text='\ufeff'+rows.map(row=>row.map(quote).join(',')).join('\r\n');type='text/csv;charset=utf-8';}
  const url=URL.createObjectURL(new Blob([text],{type})),a=document.createElement('a');a.href=url;a.download=`trisense-${r.id}-${r.source}.${format}`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
}
window.addEventListener('hashchange',route);window.addEventListener('resize',()=>requestAnimationFrame(drawFeatures));
$('metric-select').addEventListener('change',()=>{$('comparison-chart').innerHTML=bars(state.research.comparison,$('metric-select').value);});
$('sample').addEventListener('change',selectSample);$('missing-mode').addEventListener('change',inputStatus);
$('rate').addEventListener('input',()=>{$('rate-value').textContent=$('rate').value+'%';});
$('file').addEventListener('change',()=>{const file=$('file').files[0];if(file&&(!file.name.toLowerCase().endsWith('.npz')||file.size>8*1024*1024)){error('请选择小于 8 MB 的 .npz 文件。');$('file').value='';return;}state.file=file||null;error(null);inputStatus();if(file)$('result-banner').textContent='新文件已选择，尚未运行。下方仍为上次结果，请点击“运行分析”更新。';});
$('clear-file').addEventListener('click',()=>{state.file=null;$('file').value='';selectSample();});
$('analyze').addEventListener('click',run);
$('try-synthetic').addEventListener('click',async()=>{state.file=null;$('file').value='';$('sample').value='synthetic';await selectSample();await run();});
document.querySelectorAll('[data-heatmap]').forEach(btn=>btn.addEventListener('click',()=>{state.heatmap=btn.dataset.heatmap;document.querySelectorAll('[data-heatmap]').forEach(b=>b.classList.toggle('selected',b===btn));drawFeatures();}));
$('export-json').addEventListener('click',()=>download('json'));$('export-csv').addEventListener('click',()=>download('csv'));
async function boot(){try{const [health,research,samples]=await Promise.all([api('/api/health'),api('/api/research'),api('/api/samples')]);state.research=research;state.samples=samples;$('connection').textContent=health.models_present?(window.trisenseStaticAPI?'浏览器分析就绪':'本地服务已就绪'):'服务已启动 · 缺少模型';$('sample').innerHTML=samples.map(s=>`<option value="${esc(s.id)}">${s.synthetic?'合成接口样例':`${s.id} · ${s.label} · ${s.live_available?'可运行':'历史记录'}`}</option>`).join('');renderResearch();route();await selectSample();}catch(e){error('服务加载失败：'+e.message);}finally{$('loading').hidden=true;}}
boot();
