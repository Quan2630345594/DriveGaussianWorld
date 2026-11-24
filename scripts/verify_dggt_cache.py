"""Verify full causal cache coverage and unchanged labels without loading a model."""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from drive_gaussian_world.common import read_json, sha256, write_json
from drive_gaussian_world.data.schema import validate_dataset


def verify(source, cache):
    source,cache=Path(source),Path(cache)
    validation=validate_dataset(cache)
    original=read_json(source/'dataset.json')
    index=read_json(cache/'dataset.json')
    protocol=read_json(cache/'protocol.json')
    assert protocol['input_dataset_sha256']==sha256(source/'dataset.json')
    assert len(index['episodes'])==len(original['episodes'])
    results=[];splits=Counter();counts=Counter()
    for expected,episode in zip(original['episodes'],index['episodes']):
        assert (expected['scene_id'],expected['split'])==(episode['scene_id'],episode['split'])
        original_sources=read_json(source/expected['sources'])['frames']
        cached_sources=read_json(cache/episode['sources'])['frames']
        assert original_sources==cached_sources
        metadata=read_json(cache/episode['geometry_metadata'])
        assert metadata['checkpoint_sha256']==protocol['checkpoint_sha256']
        assert metadata['output_sha256']==sha256(cache/episode['path'])
        assert metadata['input_width']==protocol['width']
        assert metadata['max_context_frames']==protocol['context']
        assert metadata['precision']==protocol['precision']
        with np.load(source/expected['path'],allow_pickle=False) as a, np.load(cache/episode['path'],allow_pickle=False) as b:
            for key in a.files:
                if key not in ('tokens','token_mask'):np.testing.assert_array_equal(a[key],b[key])
            frame_ids=b['frame_ids'];assert len(metadata['frames'])==len(frame_ids)
            assert [f['frame_id'] for f in original_sources]==frame_ids.tolist()
            for i,record in enumerate(metadata['frames']):
                assert record['target_frame_id']==int(frame_ids[i])
                assert record['source_frame_ids']==frame_ids[max(0,i-protocol['context']+1):i+1].tolist()
                assert record['valid_tokens']==int(b['token_mask'][i].sum())
            invalid=int((~b['token_mask'].any(axis=1)).sum())
            results.append(dict(scene_id=episode['scene_id'],split=episode['split'],
                frames=len(frame_ids),invalid_geometry_frames=invalid,
                valid_token_fraction=float(b['token_mask'].mean()),sha256=episode['sha256']))
            counts[episode['split']]+=len(frame_ids);splits[episode['split']]+=1
    assert not (cache/'.progress').exists()
    return dict(status='PASS',episodes=results,logs=sum(splits.values()),frames=sum(counts.values()),
        split_logs=dict(splits),split_frames=dict(counts),protocol=protocol,
        invalid_geometry_frames=sum(e['invalid_geometry_frames'] for e in results),
        dataset_sha256=sha256(cache/'dataset.json'),schema_validation=validation)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True)
    p.add_argument('--cache',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();report=verify(args.source,args.cache)
    write_json(args.output,report)
    print({k:v for k,v in report.items() if k not in ('episodes','schema_validation')})
