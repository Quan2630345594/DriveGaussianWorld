import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
import urllib.error
from drive_gaussian_world.train import train

def test_cpu_api_contract_and_request_limits(dataset,tmp_path):
    checkpoint=train(dataset,tmp_path/'api_train',steps=1,batch_size=2,hidden=32)
    with socket.socket() as s:
        s.bind(('127.0.0.1',0));port=s.getsockname()[1]
    log=(tmp_path/'server.log').open('w')
    process=subprocess.Popen([sys.executable,'-m','drive_gaussian_world','serve','--dataset',str(dataset),
        '--checkpoint',checkpoint,'--port',str(port)],stdout=log,stderr=log,env=os.environ.copy())
    url=f'http://127.0.0.1:{port}'
    try:
        for _ in range(60):
            if process.poll() is not None:raise RuntimeError('server failed to start')
            try:
                health=json.load(urllib.request.urlopen(url+'/health',timeout=1));break
            except (urllib.error.URLError,TimeoutError):time.sleep(.1)
        else:raise RuntimeError('server startup timed out')
        assert health['device']=='cpu'
        body=json.dumps({'window':0,'dt':.4,'actions':[[0,0]]*4}).encode()
        request=urllib.request.Request(url+'/api/rollout',data=body,headers={'Content-Type':'application/json'})
        response=json.load(urllib.request.urlopen(request,timeout=10))
        assert len(response['rgb'])==4
        assert len(response['gaussians'][0])==64
        assert response['deployment_eligible'] is False
        assert not {'target','future_rgb','ground_truth'}&response.keys()
        bad=json.dumps({'window':0,'actions':[[100,0]]*4}).encode()
        with __import__('pytest').raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(urllib.request.Request(url+'/api/rollout',data=bad),timeout=5)
        assert e.value.code==400
    finally:
        process.terminate();process.wait(timeout=10);log.close()
