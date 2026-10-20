"""Self-contained, portable experiment report and GIF from actual held-out outputs."""
from __future__ import annotations
import argparse
import base64
import html
import io
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from drive_gaussian_world.common import cpu_runtime,fresh_dir,write_json,read_json

def png_data(rgb):
    image=Image.fromarray(np.uint8(np.clip(rgb.transpose(1,2,0),0,1)*255))
    b=io.BytesIO();image.save(b,format='PNG')
    return 'data:image/png;base64,'+base64.b64encode(b.getvalue()).decode()

def make_demo(dataset,checkpoint,output,split='test',window=0):
    cpu_runtime()
    import torch
    from torch.utils.data import default_collate
    from drive_gaussian_world.data.windows import DrivingWindows
    from drive_gaussian_world.train import load_model
    from drive_gaussian_world.evaluate import baseline,trajectory,evaluate
    from drive_gaussian_world.planning.counterfactual import rank_maneuvers
    out=fresh_dir(output);model,ckpt=load_model(checkpoint);cfg=ckpt['config']
    ds=DrivingWindows(dataset,split,cfg['history'],cfg['horizon'])
    batch=default_collate([ds[window]])
    with torch.inference_mode():
        action=batch['condition']['actions']
        if cfg['no_actions']:action=torch.zeros_like(action)
        pred=model(batch['observation'],action,batch['condition']['dt'])
        ref=baseline(batch)
    counterfactual=rank_maneuvers(model,batch['observation'],cfg['horizon'],float(batch['condition']['dt'][0,0]))
    write_json(out/'counterfactuals.json',counterfactual)
    metrics=evaluate(dataset,checkpoint,out/'evaluation.json',split)
    target=batch['target'];frames=[];payload=[]
    for i in range(cfg['horizon']):
        row=Image.new('RGB',(960,145),(15,23,38));draw=ImageDraw.Draw(row)
        for j,(title,rgb) in enumerate([('Ground truth future (evaluation only)',target['rgb'][0,i].numpy()),
                                      ('World model / logged motion condition',pred['rgb'][0,i].numpy()),
                                      ('Last-frame persistence',ref['rgb'][0,i].numpy())]):
            draw.text((j*320+8,8),title,fill='white')
            im=Image.fromarray((rgb.transpose(1,2,0).clip(0,1)*255).astype('uint8')).resize((312,104))
            row.paste(im,(j*320+4,30))
        frames.append(row)
        p=pred['tokens'][0,i].numpy();g=target['tokens'][0,i].numpy()
        pm=pred['mask_logits'][0,i].sigmoid().numpy()>.5;gm=target['mask'][0,i].numpy()
        payload.append({'gt':png_data(target['rgb'][0,i].numpy()),'prediction':png_data(pred['rgb'][0,i].numpy()),
            'persistence':png_data(ref['rgb'][0,i].numpy()),
            'surfaces_predicted':(p[pm,:3]*40).tolist(),'colors_predicted':p[pm,3:6].tolist(),
            'surfaces_reference':(g[gm,:3]*40).tolist()})
    frames[0].save(out/'rollout.gif',save_all=True,append_images=frames[1:],duration=450,loop=0)
    frames[-1].save(out/'comparison.png')
    history=[png_data(x) for x in batch['observation']['rgb'][0].numpy()]
    content={'frames':payload,'history':history,'metrics':metrics,'counterfactuals':counterfactual,
             'scene_id':batch['scene_id'][0],'anchor':int(batch['anchor'][0]),
             'predicted_path':trajectory(pred['ego'])[0].tolist(),'reference_path':trajectory(target['ego'])[0].tolist()}
    write_json(out/'demo_data.json',content)
    template='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>DriveGaussianWorld · CPU experiment</title><style>
body{font:15px system-ui;background:#0c1423;color:#dbe5f5;max-width:1250px;margin:36px auto;padding:0 20px}h1{font-size:32px}h2{font-size:21px;color:#7dd3fc}p{line-height:1.7}.badge{padding:6px 12px;background:#123a46;border-radius:16px;color:#6ee7b7}section{background:#152137;border:1px solid #27364e;border-radius:14px;padding:20px;margin:20px 0}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}img{width:100%;image-rendering:auto;border-radius:8px}canvas{max-width:100%;background:#0c1423;border-radius:8px}.history{display:flex;gap:12px}.history img{width:30%}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:9px;text-align:left;border-bottom:1px solid #30405b}.warn{color:#fcd34d}input{width:60%}@media(max-width:700px){.grid{grid-template-columns:1fr}body{margin-top:16px}table{font-size:10px}}
</style><h1>DriveGaussianWorld</h1><span class="badge">DGGT → Gaussian state → action-conditioned dynamics</span>
<p>面向行车场景的 CPU 工程实验。这里展示真实留出日志的实际预测；图像分辨率为 96×32，DGGT 使用降分辨率 CPU 推理。小样本结果用于验证系统，不能代表标准基准表现。</p>
<section><h2>历史观测</h2><p id="scene"></p><div class="history" id="history"></div></section>
<section><h2>开环未来预测</h2><p>这里的模型预测使用日志中的未来运动条件；真值未来仅用于评估。另有不提供未来运动的指标。</p><input id="step" type="range" min="0" value="0"><span id="label"></span>
<div class="grid"><div><p>真值未来图像</p><img id="gt"></div><div><p>世界模型预测</p><img id="prediction"></div><div><p>保持末帧基线</p><img id="persistence"></div></div></section>
<section><h2>Gaussian 表面与轨迹</h2><div class="grid"><div><p>预测 Gaussian：相机 x/z 平面</p><canvas id="surfaces" width="380" height="300"></canvas></div><div><p>多步自车轨迹：米，前向/左向</p><canvas id="path" width="380" height="300"></canvas></div><div><p class="warn">稀疏可见表面不是占据真值。轨迹与动作均是离线研究输出，不是车辆控制接口。</p><p id="candidate"></p></div></div></section>
<section><h2>留出日志评估</h2><div id="metrics"></div><p class="warn">日志级划分，窗口有重叠，路线可能接近；伪几何指标不能证明真实重建精度。无闭环驾驶或安全能力结论。</p></section>
<script>const D=__DATA__;document.getElementById('scene').textContent=D.scene_id+' / anchor '+D.anchor;document.getElementById('history').innerHTML=D.history.map(x=>'<img src="'+x+'">').join('');const slider=document.getElementById('step');slider.max=D.frames.length-1;
function line(canvas,series,colors){let c=canvas.getContext('2d');c.clearRect(0,0,380,300);series.forEach((pts,i)=>{c.beginPath();c.strokeStyle=colors[i];c.lineWidth=3;c.moveTo(40,260);pts.forEach(p=>c.lineTo(40+p[0]*12,260-p[1]*18));c.stroke()});c.fillStyle='#cbd5e1';c.fillText('cyan: predicted; yellow: reference',12,18)}
function render(){let t=+slider.value,f=D.frames[t];['gt','prediction','persistence'].forEach(k=>document.getElementById(k).src=f[k]);document.getElementById('label').textContent=' future step '+(t+1);let c=document.getElementById('surfaces').getContext('2d');c.clearRect(0,0,380,300);f.surfaces_predicted.forEach((p,i)=>{let col=f.colors_predicted[i];c.fillStyle='rgb('+col.map(x=>Math.round(x*255)).join(',')+')';c.beginPath();c.arc(190+p[0]*6,290-p[2]*3,3,0,7);c.fill()});c.fillStyle='#cbd5e1';c.fillText('x: right (+), z: forward (+)',12,18);line(document.getElementById('path'),[D.predicted_path.slice(0,t+1),D.reference_path.slice(0,t+1)],['#67e8f9','#fde047'])}slider.oninput=render;render();
let cols=['rgb_l1','psnr_db','ego_ade_m','ego_fde_m','gaussian_chamfer_m'];document.getElementById('metrics').innerHTML='<table><tr><th>method</th>'+cols.map(k=>'<th>'+k+'</th>').join('')+'</tr>'+Object.entries(D.metrics.metrics).map(([name,m])=>'<tr><td>'+name+'</td>'+cols.map(k=>'<td>'+m[k].toFixed(3)+'</td>').join('')+'</tr>').join('')+'</table>';document.getElementById('candidate').textContent='候选：'+D.counterfactuals.ranked.map(x=>x.maneuver+' ('+x.score.toFixed(2)+')').join('，')+'。仅展示开环排序。';</script></html>'''
    (out/'report.html').write_text(template.replace('__DATA__',json.dumps(content,ensure_ascii=False).replace('</','<\\/')))
    return str(out/'report.html')

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--output',required=True);p.add_argument('--split',default='test');p.add_argument('--window',type=int,default=0)
    print(make_demo(**vars(p.parse_args())))
if __name__=='__main__':main()
