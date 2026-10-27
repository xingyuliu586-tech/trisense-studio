from io import BytesIO
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from trisense.api import app
from trisense.engine import Engine, ROOT
from trisense.features import FeatureError, SCHEMA, read_features, mask_segment


@pytest.fixture(scope='module')
def client():
    with TestClient(app) as value:
        yield value


@pytest.fixture(scope='module')
def engine():
    return Engine()


def sample():
    return read_features(ROOT / 'examples/synthetic_sample.npz')


def payload(arrays=None, **changes):
    values = sample().arrays if arrays is None else arrays
    values = dict(values, schema=np.array(SCHEMA), synthetic=np.array(True))
    values.update(changes)
    buf = BytesIO()
    np.savez_compressed(buf, **values)
    return buf.getvalue()


def test_home_and_local_docs(client):
    assert client.get('/').status_code == 200
    assert client.get('/static/app.js').status_code == 200
    assert 'unpkg.com' not in client.get('/docs').text
    assert client.get('/openapi.json').status_code == 200
    assert client.get('/api/health').json()['models_present']


def test_recorded_results_do_not_masquerade_as_live(client):
    rows = client.get('/api/samples').json()
    assert len(rows) == 21
    record = client.get('/api/samples/S01').json()
    assert record['source'] == 'recorded'
    assert record['prediction']['probabilities'] is None
    if not record['live_available']:
        assert client.post('/api/analyze', json={'sample_id':'S01'}).status_code == 409


@pytest.mark.parametrize('predictor', ['q3','ensemble'])
def test_real_checkpoint_inference_and_exact_shapley(client, predictor):
    result = client.post('/api/analyze',json={'predictor':predictor}).json()
    assert result['source'] == 'live' and result['synthetic']
    assert -3 <= result['prediction']['strength'] <= 3
    assert sum(result['prediction']['probabilities']) == pytest.approx(1, abs=1e-6)
    a = result['attribution']
    assert len(a['coalitions']) == 8
    assert sum(a['signed']) == pytest.approx(a['full_probability']-a['empty_probability'],abs=1e-6)
    assert sum(a['relative']) == pytest.approx(1,abs=1e-6)
    assert a['full_probability'] == pytest.approx(result['prediction']['confidence'],abs=1e-6)


def test_fully_removed_visual_has_no_contribution_or_local_evidence(client):
    r=client.post('/api/analyze',json={'mode':'vision','rate':1}).json()
    assert r['features']['coverage'][2] == 0
    assert r['attribution']['signed'][2] == pytest.approx(0,abs=1e-7)
    assert r['local_effects']['vision'] == []
    assert r['local_effects']['audio']


def test_zero_rate_and_non_mutating_intervention(engine):
    x=sample();before=x.arrays['audio'].copy()
    y,positions=mask_segment(x,'audio',1,'late')
    assert positions == list(range(1,x.length-1))
    assert np.array_equal(x.arrays['audio'],before)
    assert not y.arrays['audio'].any()
    r=engine.analyze(x, mode='audio',rate=0)
    assert r['prediction'] == r['baseline']


def test_safe_upload_matches_builtin_result(client):
    raw=(ROOT/'examples/synthetic_sample.npz').read_bytes()
    r=client.post('/api/upload',files={'file':('sample.npz',raw,'application/octet-stream')})
    assert r.status_code == 200
    expected=client.post('/api/analyze',json={}).json()
    assert r.json()['prediction'] == expected['prediction']
    assert r.json()['synthetic']


@pytest.mark.parametrize('changes', [
    {'text':np.zeros((49,768))},
    {'text':np.full((50,768),np.nan)},
    {'vision':np.full((50,35),np.inf)},
    {'text':np.empty((50,768),dtype=object)},
    {'schema':np.array('q1-self-extracted')},
    {'text_bert':np.zeros((3,50))},
])
def test_reject_invalid_arrays(changes):
    with pytest.raises(FeatureError):
        read_features(payload(**changes))


def test_reject_noncontiguous_mask_and_nonzero_padding():
    data=sample().arrays
    data['text_bert'][1,2]=0
    with pytest.raises(FeatureError):read_features(payload(data))
    data=sample().arrays;data['audio'][49,0]=1
    with pytest.raises(FeatureError):read_features(payload(data))


def test_request_boundaries(client):
    assert client.post('/api/upload',files={'file':('bad.pkl',b'pickle')}).status_code == 415
    assert client.post('/api/upload',files={'file':('bad.npz',b'not a zip')}).status_code == 422
    assert client.post('/api/analyze',json={'rate':1.1}).status_code == 422
    assert client.post('/api/analyze',json={'sample_id':'../private'}).status_code == 422
    assert client.get('/api/samples/no-such-sample').status_code == 404
    assert client.post('/api/analyze',headers={'origin':'https://example.com'},json={}).status_code == 403
    assert client.get('/api/health',headers={'host':'evil.example'}).status_code == 400
    assert client.post('/api/upload',content=b'x'*(9*1024*1024+1)).status_code == 413


def test_result_artifacts_have_no_original_transcript_or_machine_paths():
    cases=(ROOT/'assets/cases.json').read_text(encoding='utf-8')
    assert len(json.loads(cases)) == 20
    assert 'raw_text' not in cases and 'C:\\Users' not in cases


def test_minimum_length_has_valid_local_window(engine):
    s=sample();s.arrays['text_bert'][1,3:]=0;s.arrays['audio'][3:]=0;s.arrays['vision'][3:]=0
    result=engine.analyze(s)
    for records in result['local_effects'].values():
        assert records[0]['start'] == 1 and records[0]['stop'] == 2


def test_special_token_rows_do_not_create_false_local_evidence(engine):
    s=sample();s.arrays['vision'][0,0]=1.0;s.arrays['vision'][s.length-1,0]=1.0
    r=engine.analyze(s,mode='vision',rate=1)
    assert r['local_effects']['vision']==[]
