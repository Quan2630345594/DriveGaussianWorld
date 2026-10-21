import io
from pathlib import Path
import zipfile
import pytest
import drive_gaussian_world.data.range_collect as rc

def test_range_zip_member_integrity(tmp_path,monkeypatch):
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr('safe/oxts/data/0000000000.txt','hello-driving')
    blob=stream.getvalue()
    def fake(url,start,end):return blob[start:end+1],{'Content-Range':f'bytes {start}-{end}/{len(blob)}'}
    monkeypatch.setattr(rc,'get_range',fake)
    remote=rc.RemoteZip('https://official.example/data.zip')
    with zipfile.ZipFile(remote) as z:info=z.infolist()[0]
    report=rc.member(remote.url,info,tmp_path)
    assert report['bytes']==13
    assert (tmp_path/'safe/oxts/data/0000000000.txt').read_text()=='hello-driving'
    (tmp_path/'safe/oxts/data/0000000000.txt').write_text('bad')
    with pytest.raises(ValueError,match='changed cached'):rc.member(remote.url,info,tmp_path)

def test_unsafe_member_rejected(tmp_path):
    info=zipfile.ZipInfo('../outside')
    with pytest.raises(ValueError,match='unsafe member'):rc.member('https://example',info,tmp_path)


def test_explicit_project_proxy(monkeypatch):
    from drive_gaussian_world.data.collect import download_session
    monkeypatch.delenv('DGW_HTTP_PROXY',raising=False)
    direct=download_session();assert not direct.trust_env and not direct.proxies
    monkeypatch.setenv('DGW_HTTP_PROXY','http://127.0.0.1:17892')
    proxied=download_session()
    assert not proxied.trust_env and proxied.proxies['https']=='http://127.0.0.1:17892'


def test_driving_plan_budget_and_duplicate_rejection(tmp_path):
    from drive_gaussian_world.data.download_drives import plan_download
    with pytest.raises(ValueError,match='below 50'):plan_download([],tmp_path,50_000_000_000)
    repeated=[{'date':'2011_09_26','drive':'0005','split':s} for s in ('train','val','test')]
    with pytest.raises(ValueError,match='duplicate'):plan_download(repeated,tmp_path)


def test_drive_extract_preserves_only_requested_sensors(tmp_path):
    from drive_gaussian_world.data.download_drives import extract_checked
    archive=tmp_path/'test.zip';root=tmp_path/'raw'
    with zipfile.ZipFile(archive,'w') as z:
        z.writestr('date/log/image_02/data/0000000000.png',b'left')
        z.writestr('date/log/image_03/data/0000000000.png',b'right')
        z.writestr('date/log/image_00/data/0000000000.png',b'gray')
        z.writestr('date/log/oxts/data/0000000000.txt',b'pose')
        z.writestr('date/log/velodyne_points/data/0000000000.bin',b'lidar')
    records=extract_checked(archive,root)
    assert len(records)==4
    assert not (root/'date/log/image_00/data/0000000000.png').exists()
    assert extract_checked(archive,root)==records
    (root/records[0]['path']).write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed raw'):extract_checked(archive,root)


def test_multi_drive_budget_rejects_before_large_download(tmp_path,monkeypatch):
    import drive_gaussian_world.data.download_drives as dd
    class Remote:
        size=20_000_000_000;etag='public-test'
        def __init__(self,url):pass
    class Zip:
        def __init__(self,remote):pass
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def infolist(self):
            a=zipfile.ZipInfo('date/log/image_02/data/0000000000.png');a.file_size=1
            b=zipfile.ZipInfo('date/log/oxts/data/0000000000.txt');b.file_size=20
            return [a,b]
    def motion(url,info,root):
        p=Path(root)/info.filename;p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text('0 0 0 0 0 0 0 0 5 0')
        return {'path':info.filename}
    monkeypatch.setattr(dd,'RemoteZip',Remote);monkeypatch.setattr(dd.zipfile,'ZipFile',Zip)
    monkeypatch.setattr(dd,'member',motion)
    plan=[{'date':'2011_09_26','drive':f'{i:04d}','split':s} for i,s in enumerate(('train','val','test'))]
    with pytest.raises(ValueError,match='exceeds budget'):dd.plan_download(plan,tmp_path)


def test_archive_resume_checks_range(tmp_path,monkeypatch):
    import drive_gaussian_world.data.download_drives as dd
    path=tmp_path/'drive.zip';path.with_suffix('.part').write_bytes(b'ab')
    class Response:
        status_code=206;headers={'Content-Range':'bytes 2-5/6'}
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def raise_for_status(self):pass
        def iter_content(self,size):yield b'cdef'
    class Session:
        def get(self,url,**kwargs):
            assert kwargs['headers']=={'Range':'bytes=2-'}
            return Response()
    monkeypatch.setattr(dd,'download_session',Session)
    assert dd.fetch_resume('https://official.example/drive.zip',path,6).read_bytes()==b'abcdef'
    assert not path.with_suffix('.part').exists()


def test_multi_drive_collection_completes_and_resumes(tmp_path,monkeypatch):
    import json
    import drive_gaussian_world.data.download_drives as dd
    import drive_gaussian_world.data.collect as source
    plan=[{'date':'2011_09_26','drive':f'{i:04d}','split':s} for i,s in enumerate(('train','val','test'))]
    plan_path=tmp_path/'plan.json';plan_path.write_text(json.dumps(plan))
    jobs=[{**e,'url':e['drive'],'archive_bytes':100,'frame_ids':[0]} for e in plan]
    monkeypatch.setattr(dd,'plan_download',lambda spec,root,limit:{'max_bytes':limit,'archive_bytes':300,'episodes':jobs})
    def calibration(url,path):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'calibration');return path
    monkeypatch.setattr(source,'fetch',calibration);monkeypatch.setattr(source,'extract',lambda *args:[])
    def archive(url,path,expected):
        path.parent.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(path,'w') as z:z.writestr(f'date/drive_{url}/image_02/data/0000000000.png',b'image')
        return path
    monkeypatch.setattr(dd,'fetch_resume',archive)
    root=tmp_path/'collection';result=dd.collect(root,plan_path,max_bytes=1_000_000)
    assert len(result['archives'])==3
    assert json.loads((root/'status.json').read_text())['status']=='COMPLETED'
    assert not list((root/'archives').glob('*_sync.zip'))
    def forbidden(*args):raise AssertionError('completed archives should not be downloaded again')
    monkeypatch.setattr(dd,'fetch_resume',forbidden)
    assert len(dd.collect(root,plan_path,max_bytes=1_000_000)['archives'])==3


def test_parallel_ranges_reuse_prefix_and_assemble(tmp_path,monkeypatch):
    import drive_gaussian_world.data.download_drives as dd
    payload=b'abcdefghijkl';path=tmp_path/'drive.zip';path.with_suffix('.part').write_bytes(payload[:4])
    calls=[]
    class Response:
        status_code=206
        def __init__(self,start,end):self.start=start;self.end=end;self.headers={'Content-Range':f'bytes {start}-{end}/{len(payload)}'}
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def raise_for_status(self):pass
        def iter_content(self,size):yield payload[self.start:self.end+1]
    class Session:
        def get(self,url,headers,**kwargs):
            start,end=map(int,headers['Range'].split('=')[1].split('-'));calls.append((start,end));return Response(start,end)
    monkeypatch.setattr(dd,'download_session',Session)
    assert dd.fetch_parallel('https://official.example/drive.zip',path,len(payload),block_bytes=4,workers=2).read_bytes()==payload
    assert sorted(calls)==[(4,7),(8,11)]
    assert not path.with_suffix('.chunks').exists()


def test_range_resume_after_interrupted_assembly(tmp_path,monkeypatch):
    import drive_gaussian_world.data.download_drives as dd
    from drive_gaussian_world.common import write_json
    payload=b'abcdefghijkl';path=tmp_path/'drive.zip';path.with_suffix('.part').write_bytes(payload[:6])
    cache=path.with_suffix('.chunks');cache.mkdir()
    write_json(cache/'source.json',{'url':'https://official.example/drive.zip','expected_bytes':12,'block_bytes':4})
    (cache/'000000000004-000000000007.chunk').write_bytes(payload[4:8])
    (cache/'000000000008-000000000011.chunk').write_bytes(payload[8:12])
    class Response:
        status_code=206
        def __init__(self,start,end):self.start=start;self.end=end;self.headers={'Content-Range':f'bytes {start}-{end}/12'}
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def raise_for_status(self):pass
        def iter_content(self,size):yield payload[self.start:self.end+1]
    class Session:
        def get(self,url,headers,**kwargs):return Response(*map(int,headers['Range'].split('=')[1].split('-')))
    monkeypatch.setattr(dd,'download_session',Session)
    assert dd.fetch_parallel('https://official.example/drive.zip',path,12,block_bytes=4,workers=2).read_bytes()==payload
    assert not cache.exists()


def test_transfer_worker_configuration(tmp_path,monkeypatch):
    import drive_gaussian_world.data.download_drives as dd
    captured=[]
    def fake(url,path,expected_bytes,workers):captured.append(workers);return path
    monkeypatch.setattr(dd,'fetch_parallel',fake)
    monkeypatch.setenv('DGW_TRANSFER_WORKERS','16')
    path=tmp_path/'drive.zip';assert dd.fetch_resume('https://official.example/x',path,100_000_000)==path
    assert captured==[16]
    monkeypatch.setenv('DGW_TRANSFER_WORKERS','33')
    with pytest.raises(ValueError,match='within'):dd.fetch_resume('https://official.example/x',path,100_000_000)
