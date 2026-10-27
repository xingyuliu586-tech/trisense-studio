/* Browser-only port of the original PyTorch inference graph. No external calls. */
'use strict';
window.TriSenseBrowser = (()=>{
  const DIMS={text:768,audio:74,vision:35},KEYS=['text','audio','vision'];
  const LABELS=['负向','中性','正向'];
  const clip=(x,a,b)=>Math.max(a,Math.min(b,x));
  function erf(x){const sign=x<0?-1:1; x=Math.abs(x);const t=1/(1+.3275911*x);return sign*(1-(((((1.061405429*t-1.453152027)*t)+1.421413741)*t-.284496736)*t+.254829592)*t*Math.exp(-x*x));}
  const gelu=x=>.5*x*(1+erf(x/Math.SQRT2));
  function softmax(v){const high=Math.max(...v),e=v.map(x=>Math.exp(x-high)),sum=e.reduce((a,b)=>a+b,0);return e.map(x=>x/sum);}
  function decodeLayers(raw){const result={};for(const [name,value] of Object.entries(raw)){const binary=atob(value.data),bytes=new Uint8Array(binary.length);for(let i=0;i<binary.length;i++)bytes[i]=binary.charCodeAt(i);result[name]=new Float32Array(bytes.buffer);}return result;}
  function flatten(value,shape,name){let a=value;for(let i=0;i<shape.length;i++){if(!Array.isArray(a)||a.length!==shape[i])throw Error(`${name} 形状不符合接口。`);a=a[0];}const out=new Float32Array(shape.reduce((x,y)=>x*y,1));let n=0;function walk(v,depth){if(depth===shape.length){if(!Number.isFinite(v)||Math.abs(v)>1e8)throw Error(`${name} 包含异常数值。`);out[n++]=v;return;}for(const x of v)walk(x,depth+1);}walk(value,0);return out;}
  function sampleFromArrays(raw){
    if(raw.schema!=='mosei-aligned-50-v1')throw Error('需要兼容的 mosei-aligned-50-v1 特征；第一问自提取特征不能直接套用。');
    const arrays={};for(const [key,d] of Object.entries(DIMS))arrays[key]=flatten(raw[key],[50,d],key);
    const tokens=flatten(raw.text_bert,[3,50],'text_bert');let length=0;
    for(let i=0;i<50;i++){const mask=tokens[50+i],type=tokens[100+i],id=tokens[i];if(!Number.isInteger(mask)||![0,1].includes(mask)||![0,1].includes(type)||!Number.isInteger(id)||id<0||id>=30522)throw Error('text_bert 含无效 token / mask / segment。');length+=mask;}
    if(length<3||length>50)throw Error('有效词元长度必须为 3–50。');
    for(let i=0;i<50;i++)if(tokens[50+i]!==Number(i<length))throw Error('有效词元必须为连续前缀。');
    for(const key of ['audio','vision'])for(let i=length*DIMS[key];i<50*DIMS[key];i++)if(arrays[key][i]!==0)throw Error(`${key} 填充行必须为零。`);
    return {...arrays,tokens,length,synthetic:raw.synthetic===true||raw.synthetic===1};
  }
  function availability(sample){const out=new Uint8Array(150);for(let t=0;t<50;t++){
      out[t*3]=Number(sample.tokens[50+t]!==0&&sample.tokens[t]!==100);
      for(let j=1;j<3;j++){const key=KEYS[j],d=DIMS[key];let found=0;for(let k=0;k<d;k++)if(sample[key][t*d+k]!==0){found=1;break;}out[t*3+j]=found;}
    }return out;}
  function normalized(sample,norm){const x={text:new Float32Array(sample.text),audio:new Float32Array(50*74),vision:new Float32Array(50*35),available:availability(sample)};
    for(const key of ['audio','vision']){const d=DIMS[key];for(let t=0;t<50;t++)if(x.available[t*3+KEYS.indexOf(key)])for(let k=0;k<d;k++)x[key][t*d+k]=clip((sample[key][t*d+k]-norm[key].mean[k])/norm[key].std[k],-10,10);}
    return x;
  }
  function without(x,subset){const v={text:new Float32Array(x.text),audio:new Float32Array(x.audio),vision:new Float32Array(x.vision),available:new Uint8Array(x.available)};
    KEYS.forEach((key,j)=>{if(subset&(1<<j))return;v[key].fill(0);for(let t=0;t<50;t++)v.available[t*3+j]=0;});return v;}
  function maskPosition(x,mode,rate,location,length){const v={text:x.text,audio:new Float32Array(x.audio),vision:new Float32Array(x.vision),available:new Uint8Array(x.available)};
    const n=length-2,count=rate>0?Math.max(1,Math.min(n,Math.round(n*rate))):0;
    const start=location==='early'?1:location==='late'?1+n-count:1+Math.floor((n-count)/2),positions=[];
    if(mode!=='none')for(let t=start;t<start+count;t++){positions.push(t);for(const [key,j] of [['audio',1],['vision',2]])if(mode===key||mode==='audio+vision'){v[key].fill(0,t*DIMS[key],(t+1)*DIMS[key]);v.available[t*3+j]=0;}}
    return {value:v,positions};
  }
  function matVec(out,weight,bias,vector,inputSize){for(let o=0;o<out.length;o++){let s=bias?bias[o]:0,base=o*inputSize;for(let k=0;k<inputSize;k++)s+=weight[base+k]*vector[k];out[o]=s;}return out;}
  function normVector(v,gamma,beta){let mean=0,variance=0;for(const x of v)mean+=x;mean/=v.length;for(const x of v)variance+=(x-mean)**2;variance/=v.length;const denominator=Math.sqrt(variance+1e-5);for(let i=0;i<v.length;i++)v[i]=(v[i]-mean)/denominator*gamma[i]+beta[i];return v;}
  function convolve(input,weight,bias){const out=new Float32Array(50*96);for(let t=0;t<50;t++)for(let o=0;o<96;o++){
      let sum=bias[o],base=o*96*3;
      for(let i=0;i<96;i++){const b=base+i*3;if(t>0)sum+=input[(t-1)*96+i]*weight[b];sum+=input[t*96+i]*weight[b+1];if(t<49)sum+=input[(t+1)*96+i]*weight[b+2];}out[t*96+o]=sum;
    }return out;}
  function forward(model,x){const w=n=>model[n],fused=new Float32Array(50*96),active=new Uint8Array(50),temp=new Float32Array(96);
    for(let t=0;t<50;t++){
      let seen=0;const projected=[],scores=[];
      for(let j=0;j<3;j++){
        if(!x.available[t*3+j]){projected.push(null);scores.push(-1e4);continue;}
        seen=1;const key=KEYS[j],d=DIMS[key],input=x[key].subarray(t*d,(t+1)*d),vec=new Float32Array(96),prefix=`proj.${j}.`;
        matVec(vec,w(prefix+'0.weight'),w(prefix+'0.bias'),input,d);
        normVector(vec,w(prefix+'1.weight'),w(prefix+'1.bias'));
        for(let k=0;k<96;k++)vec[k]=gelu(vec[k]);
        let score=w(`gate.${j}.bias`)[0],g=w(`gate.${j}.weight`);for(let k=0;k<96;k++)score+=g[k]*vec[k];projected.push(vec);scores.push(score);
      }
      if(!seen)continue;active[t]=1;const weights=softmax(scores),a=w('availability.weight'),pos=w('position.weight');
      for(let k=0;k<96;k++){let sum=pos[t*96+k];for(let j=0;j<3;j++)if(projected[j])sum+=weights[j]*projected[j][k]+a[k*3+j];fused[t*96+k]=sum;}
    }
    const convA=convolve(fused,w('conv1.weight'),w('conv1.bias')),stage=new Float32Array(50*96);
    for(let i=0;i<stage.length;i++)stage[i]=fused[i]+gelu(convA[i]);
    const convB=convolve(stage,w('conv2.weight'),w('conv2.bias')),out=new Float32Array(50*96),scores=[];
    const lnw=w('layer_norm.weight'),lnb=w('layer_norm.bias'),aw=w('attention.weight'),ab=w('attention.bias')[0];
    for(let t=0;t<50;t++){
      if(!active[t]){scores.push(-1e4);continue;}
      for(let k=0;k<96;k++)temp[k]=stage[t*96+k]+gelu(convB[t*96+k]);
      normVector(temp,lnw,lnb);
      let score=ab;for(let k=0;k<96;k++){out[t*96+k]=temp[k];score+=aw[k]*temp[k];}scores.push(score);
    }
    const weights=softmax(scores),pooled=new Float32Array(192);let count=0;
    for(let t=0;t<50;t++)if(active[t]){count++;for(let k=0;k<96;k++){const v=out[t*96+k];pooled[k]+=weights[t]*v;pooled[96+k]+=v;}}
    if(count)for(let k=0;k<96;k++)pooled[96+k]/=count;
    const shared=matVec(new Float32Array(96),w('shared.0.weight'),w('shared.0.bias'),pooled,192);
    for(let k=0;k<96;k++)shared[k]=gelu(shared[k]);
    const logits=matVec(new Float32Array(3),w('classifier.weight'),w('classifier.bias'),shared,96);
    const reg=matVec(new Float32Array(1),w('regressor.weight'),w('regressor.bias'),shared,96)[0];
    return {prob:softmax(Array.from(logits)),strength:clip(reg,-3,3)};
  }
  function predict(models,x,kind){if(kind==='q3')return forward(models.q3_predictor,x);const b=forward(models.q2_baseline,x),r=forward(models.q2_robust,x);return {prob:b.prob.map((v,i)=>.75*v+.25*r.prob[i]),strength:clip(.75*b.strength+.25*r.strength,-3,3)};}
  function describe(p){const cls=p.prob.indexOf(Math.max(...p.prob));return {class_id:cls,label:LABELS[cls],strength:p.strength,probabilities:p.prob,confidence:p.prob[cls]};}
  const tick=()=>new Promise(resolve=>setTimeout(resolve,0));
  async function analyze(sample,kind,mode,rate,location,models,norm){const start=performance.now(),base=normalized(sample,norm),original=describe(predict(models,base,kind)),{value,positions}=maskPosition(base,mode,rate,location,sample.length),pred=describe(predict(models,value,kind));
    const values=[];for(let subset=0;subset<8;subset++){values.push(predict(models,without(value,subset),kind).prob[pred.class_id]);if(subset%2===1)await tick();}
    const phi=[0,0,0];for(let j=0;j<3;j++)for(let sub=0;sub<8;sub++)if(!(sub&(1<<j))){const k=((sub&1)?1:0)+((sub&2)?1:0)+((sub&4)?1:0);phi[j]+=(k===1?1/6:1/3)*(values[sub|(1<<j)]-values[sub]);}
    const amount=phi.map(Math.abs),total=amount.reduce((a,b)=>a+b,0),relative=amount.map(v=>total>1e-10?v/total:0),error=Math.abs(phi.reduce((a,b)=>a+b,0)-(values[7]-values[0]));
    if(error>1e-5)throw Error('模态贡献守恒校验失败。');
    const local={audio:[],vision:[]},width=Math.min(3,sample.length-2),end=sample.length-1;
    let starts=[];for(let s=1;s<=end-width;s+=2)starts.push(s);if(!starts.length||starts.at(-1)!==end-width)starts.push(end-width);
    for(const [key,j] of [['audio',1],['vision',2]]){
      let exists=false;for(let t=1;t<sample.length-1;t++)if(value.available[t*3+j]){exists=true;break;}if(!exists)continue;
      for(const s of starts){const v={text:value.text,audio:new Float32Array(value.audio),vision:new Float32Array(value.vision),available:new Uint8Array(value.available)};v[key].fill(0,s*DIMS[key],(s+width)*DIMS[key]);for(let t=s;t<s+width;t++)v.available[t*3+j]=0;
        local[key].push({start:s,stop:s+width,delta:pred.confidence-predict(models,v,kind).prob[pred.class_id]});
      }await tick();
    }
    const changed={...sample,audio:new Float32Array(sample.audio),vision:new Float32Array(sample.vision)};
    for(const t of positions)for(const key of ['audio','vision'])if(mode===key||mode==='audio+vision')changed[key].fill(0,t*DIMS[key],(t+1)*DIMS[key]);
    return {source:'live',predictor:kind,synthetic:sample.synthetic,prediction:pred,baseline:original,
      intervention:{mode,rate,location,positions},attribution:{signed:phi,relative,primary:total?['文本','语音','视觉'][relative.indexOf(Math.max(...relative))]:'无可归因差异',coalitions:values,empty_probability:values[0],full_probability:values[7],efficiency_error:error,target:pred.label},
      local_effects:local,features:featureView(changed),elapsed_ms:Math.round((performance.now()-start)*10)/10,
      notes:['类别概率未经校准；不是预测正确率。','归因显示模型在零基线下的响应，不表示因果关系。']};
  }
  function featureView(sample){const mask=availability(sample),L=sample.length,available=[[],[],[]],coverage=[0,0,0],heatmaps={};for(let t=0;t<L;t++)for(let j=0;j<3;j++){const v=mask[t*3+j];available[j].push(v);if(t>0&&t<L-1)coverage[j]+=v;}
    for(let j=0;j<3;j++)coverage[j]=Math.round(coverage[j]/(L-2)*1e4)/1e4;
    KEYS.forEach((key,j)=>{const d=DIMS[key],channels=Array.from({length:24},(_,i)=>Math.floor(i*(d-1)/23)),values=channels.map(c=>{
        const entries=[];for(let t=0;t<L;t++)if(available[j][t])entries.push(sample[key][t*d+c]);
        const mean=entries.length?entries.reduce((a,b)=>a+b,0)/entries.length:0,sd=entries.length?Math.sqrt(entries.reduce((a,b)=>a+(b-mean)**2,0)/entries.length):0;
        return Array.from({length:L},(_,t)=>available[j][t]?Math.round(clip((sample[key][t*d+c]-mean)/Math.max(sd,1e-6),-3,3)*1e3)/1e3:0);
      });heatmaps[key]={channels:channels.map(x=>x+1),values};});
    return {length:L,content_positions:L-2,available,coverage,heatmaps};
  }
  async function unzipNPZ(buffer){const view=new DataView(buffer),bytes=new Uint8Array(buffer),limit=8*1024*1024;if(buffer.byteLength>limit)throw Error('单样本 NPZ 文件超过 8 MB。');
    let e=-1;for(let i=buffer.byteLength-22;i>=Math.max(0,buffer.byteLength-65557);i--)if(view.getUint32(i,true)===0x06054b50){e=i;break;}if(e<0)throw Error('无法读取 NPZ 压缩包。');
    const count=view.getUint16(e+10,true),offset=view.getUint32(e+16,true),decoder=new TextDecoder(),out={};let pos=offset,total=0;
    if(count>12)throw Error('NPZ 包含过多字段。');
    for(let i=0;i<count;i++){
      if(view.getUint32(pos,true)!==0x02014b50)throw Error('NPZ 目录损坏。');
      const method=view.getUint16(pos+10,true),size=view.getUint32(pos+20,true),rawSize=view.getUint32(pos+24,true),nlen=view.getUint16(pos+28,true),extra=view.getUint16(pos+30,true),comment=view.getUint16(pos+32,true),local=view.getUint32(pos+42,true),name=decoder.decode(bytes.subarray(pos+46,pos+46+nlen));
      pos+=46+nlen+extra+comment;total+=rawSize;if(total>limit||size>limit)throw Error('NPZ 解压后过大。');
      if(!name.endsWith('.npy')||name.includes('/')||name.includes('\\')||name.slice(0,-4) in out)throw Error('NPZ 字段名异常。');
      if(view.getUint32(local,true)!==0x04034b50)throw Error('NPZ 数据区损坏。');
      const dataStart=local+30+view.getUint16(local+26,true)+view.getUint16(local+28,true),slice=bytes.slice(dataStart,dataStart+size);let decoded;
      if(method===0)decoded=slice;
      else if(method===8){const reader=new Blob([slice]).stream().pipeThrough(new DecompressionStream('deflate-raw')).getReader(),chunks=[];let used=0;while(true){const part=await reader.read();if(part.done)break;used+=part.value.length;if(used>limit)throw Error('NPZ 解压后过大。');chunks.push(part.value);}decoded=new Uint8Array(used);let cursor=0;for(const c of chunks){decoded.set(c,cursor);cursor+=c.length;}}
      else throw Error('不支持的 NPZ 压缩方式。');
      if(decoded.length!==rawSize)throw Error('NPZ 数据长度错误。');out[name.slice(0,-4)]=readNpy(decoded);
    }return out;
  }
  function readNpy(bytes){const view=new DataView(bytes.buffer,bytes.byteOffset,bytes.byteLength),decoder=new TextDecoder();if(bytes.length<10||bytes[0]!==0x93||decoder.decode(bytes.subarray(1,6))!=='NUMPY')throw Error('NPY 文件头无效。');
    const major=bytes[6],headerLength=major===1?view.getUint16(8,true):view.getUint32(8,true),start=major===1?10:12,header=decoder.decode(bytes.subarray(start,start+headerLength)),descr=header.match(/'descr':\s*'([^']+)'/)?.[1],shapeText=header.match(/'shape':\s*\(([^)]*)\)/)?.[1],fortran=/fortran_order':\s*True/.test(header);
    if(!descr||shapeText===undefined||fortran)throw Error('NPY 格式或数组顺序不支持。');
    const shape=shapeText.split(',').map(x=>x.trim()).filter(Boolean).map(Number),count=shape.reduce((a,b)=>a*b,1),dataStart=start+headerLength,type=descr.slice(-2),little=descr[0]!=='>';
    if(count>50000||dataStart>bytes.length)throw Error('数组长度错误。');
    if(descr.includes('U')){const n=Number(descr.match(/U(\d+)/)?.[1]);if(!Number.isInteger(n)||n>100||bytes.length<dataStart+4*n)throw Error('schema 字符串无效。');let s='';for(let i=0;i<n;i++){const char=view.getUint32(dataStart+i*4,little);if(char)s+=String.fromCodePoint(char);}return s;}
    const sizes={f4:4,f8:8,i8:8,i4:4,u1:1,b1:1,i2:2,u2:2};if(!(type in sizes)||bytes.length<dataStart+count*sizes[type])throw Error('仅支持数值 NPY 数组。');
    const arr=new Array(count);for(let i=0;i<count;i++){const p=dataStart+i*sizes[type];arr[i]=type==='f4'?view.getFloat32(p,little):type==='f8'?view.getFloat64(p,little):type==='i8'?Number(view.getBigInt64(p,little)):type==='i4'?view.getInt32(p,little):type==='i2'?view.getInt16(p,little):type==='u2'?view.getUint16(p,little):view.getUint8(p);}
    if(!shape.length)return arr[0];function nest(values,depth){if(depth===shape.length-1)return values.splice(0,shape[depth]);return Array.from({length:shape[depth]},()=>nest(values,depth+1));}return nest(arr,0);
  }
  return {decodeLayers,sampleFromArrays,analyze,featureView,unzipNPZ,forward,normalized,predict};
})();
