/* Static Pages adapter: local files are processed in this browser tab. */
'use strict';
window.trisenseStaticAPI=(()=>{
  const cache=new Map(),models={};
  async function asset(name){if(!cache.has(name))cache.set(name,fetch('./'+name).then(r=>{if(!r.ok)throw Error(`网页资源 ${name} 加载失败。`);return r.json();}));return cache.get(name);}
  async function model(name){if(!models[name])models[name]=asset(`browser-models/${name}.json`).then(window.TriSenseBrowser.decodeLayers);return models[name];}
  async function requiredModels(kind){const named=kind==='q3'?['q3_predictor']:['q2_baseline','q2_robust'];const entries=await Promise.all(named.map(async name=>[name,await model(name)]));return Object.fromEntries(entries);}
  async function analyze(raw,predictor,mode,rate,location,id){
    if(!['q3','ensemble'].includes(predictor)||!['none','audio','vision','audio+vision'].includes(mode)||!['early','middle','late'].includes(location)||!Number.isFinite(rate)||rate<0||rate>1)throw Error('分析参数无效。');
    const sample=window.TriSenseBrowser.sampleFromArrays(raw),[net,norm]=await Promise.all([requiredModels(predictor),asset('normalization.json')]);
    const result=await window.TriSenseBrowser.analyze(sample,predictor,mode,rate,location,net,norm);result.id=id;return result;
  }
  return async(path,options={})=>{
    if(path==='/api/health')return {status:'ok',version:'1.0.0',models_present:true,device:'当前浏览器',network_required:true};
    if(path==='/api/research')return asset('research.json');
    if(path==='/api/samples'){const rows=await asset('cases.json');return rows.map(r=>({id:r.id,label:r.prediction.label,strength:r.prediction.strength,live_available:false,synthetic:false})).concat([{id:'synthetic',label:'合成接口样例',live_available:true,synthetic:true}]);}
    if(path.startsWith('/api/samples/')){const id=decodeURIComponent(path.slice('/api/samples/'.length));
      if(id==='synthetic'){const sample=window.TriSenseBrowser.sampleFromArrays(await asset('synthetic_sample.json'));return {id,source:'synthetic',synthetic:true,predictor:'q3',prediction:null,attribution:null,features:window.TriSenseBrowser.featureView(sample),live_available:true,notes:['随机生成的特征，仅验证软件流程；不代表真实视频情感。']};}
      const row=(await asset('cases.json')).find(x=>x.id===id);if(!row)throw Error('未找到该样本。');return {...row,live_available:false};
    }
    if(path==='/api/analyze'){const input=JSON.parse(options.body||'{}');if(input.sample_id!=='synthetic')throw Error('此历史样本没有公开特征。请选择合成样例，或导入自己的兼容 NPZ。');return analyze(await asset('synthetic_sample.json'),input.predictor||'q3',input.mode||'none',input.rate??0,input.location||'middle','synthetic');}
    if(path==='/api/upload'){const form=options.body,file=form?.get('file');if(!(file instanceof File))throw Error('请选择 NPZ 文件。');if(!file.name.toLowerCase().endsWith('.npz')||file.size>8*1024*1024)throw Error('仅接受不超过 8 MB 的 NPZ 文件。');const raw=await window.TriSenseBrowser.unzipNPZ(await file.arrayBuffer());return analyze(raw,String(form.get('predictor')||'q3'),String(form.get('mode')||'none'),Number(form.get('rate')||0),String(form.get('location')||'middle'),'uploaded');}
    throw Error('未知请求。');
  };
})();
