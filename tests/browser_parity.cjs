/* Run with node tests/browser_parity.cjs after scripts/browser_reference.py. */
'use strict';
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict'),vm=require('node:vm');
const root=path.resolve(__dirname,'..');global.window=global;
vm.runInThisContext(fs.readFileSync(path.join(root,'docs/browser-engine.js'),'utf8'),{filename:'browser-engine.js'});
const browser=window.TriSenseBrowser;
const sample=browser.sampleFromArrays(JSON.parse(fs.readFileSync(path.join(root,'docs/synthetic_sample.json'))));
const norm=JSON.parse(fs.readFileSync(path.join(root,'docs/normalization.json')));
const python=JSON.parse(fs.readFileSync(path.join(__dirname,'browser_reference.json')));
const models={};for(const name of ['q3_predictor','q2_baseline','q2_robust'])models[name]=browser.decodeLayers(JSON.parse(fs.readFileSync(path.join(root,'docs/browser-models',name+'.json'))));
const close=(a,b,margin=.002)=>assert.ok(Math.abs(a-b)<margin,`Python ${b}, browser ${a}, difference ${Math.abs(a-b)}`);
(async()=>{
  for(const [key,expected] of Object.entries(python)){
    const [predictor,mode,rate]=key.split(':'),actual=await browser.analyze(sample,predictor,mode,Number(rate),'middle',models,norm);
    assert.equal(actual.prediction.class_id,expected.prediction.class_id,key);
    close(actual.prediction.strength,expected.prediction.strength);
    for(let i=0;i<3;i++){
      close(actual.prediction.probabilities[i],expected.prediction.probabilities[i]);
      close(actual.attribution.signed[i],expected.attribution.signed[i]);
    }
    assert.deepEqual(actual.intervention.positions,expected.intervention.positions);
    close(actual.attribution.signed.reduce((a,b)=>a+b,0),actual.attribution.full_probability-actual.attribution.empty_probability,1e-5);
    console.log(key,'class',actual.prediction.label,'strength',actual.prediction.strength.toFixed(3));
  }
  const raw=fs.readFileSync(path.join(root,'docs/synthetic_sample.npz'));
  const parsed=await browser.unzipNPZ(raw.buffer.slice(raw.byteOffset,raw.byteOffset+raw.byteLength));
  const uploaded=browser.sampleFromArrays(parsed);
  assert.equal(uploaded.synthetic,true);
  assert.equal(uploaded.length,sample.length);
  assert.equal(uploaded.text[0],sample.text[0]);
  console.log('Browser inference and NPZ parsing match the Python reference.');
})().catch(e=>{console.error(e);process.exitCode=1;});
