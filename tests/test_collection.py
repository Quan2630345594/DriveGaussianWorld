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
