"""Extract the release into a checked temporary folder and test it as a clone."""
from pathlib import Path
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'dist'
ARCHIVE=OUTPUT/'TriSense-Studio-v1.0.0.zip'
REQUIRED={'TriSense-Studio/README.md','TriSense-Studio/run.py','TriSense-Studio/models/q3_predictor.pt',
          'TriSense-Studio/examples/synthetic_sample.npz','TriSense-Studio/web/app.js'}
FORBIDDEN={'local_data','.runtime','.venv','uploads','exports','__pycache__','.git'}


def main():
    if not ARCHIVE.is_file():
        raise SystemExit('Run python scripts/package_release.py first.')
    OUTPUT.mkdir(exist_ok=True)
    temporary=Path(tempfile.mkdtemp(prefix='_smoke_',dir=OUTPUT)).resolve()
    if not temporary.is_relative_to(OUTPUT.resolve()):
        raise RuntimeError('Temporary extraction escaped dist/.')
    try:
        with zipfile.ZipFile(ARCHIVE) as archive:
            names=set(archive.namelist())
            assert REQUIRED <= names
            assert all(n.startswith('TriSense-Studio/') and not (set(Path(n).parts)&FORBIDDEN)
                       and '..' not in Path(n).parts for n in names)
            archive.extractall(temporary)
        clone=temporary/'TriSense-Studio'
        env=os.environ.copy()
        env['PYTHONPATH']=os.pathsep.join([str(ROOT/'.runtime'),str(clone)])
        subprocess.run([sys.executable,'-m','pytest','tests','-q'],cwd=clone,env=env,check=True)
        subprocess.run([sys.executable,'-c',
                        "from trisense.api import app;from fastapi.testclient import TestClient;"
                        "r=TestClient(app).post('/api/analyze',json={});"
                        "assert r.status_code==200 and r.json()['source']=='live'"],
                       cwd=clone,env=env,check=True)
        print(f'Isolated release check passed ({len(names)} published files).')
    finally:
        assert temporary.resolve().is_relative_to(OUTPUT.resolve())
        shutil.rmtree(temporary)


if __name__=='__main__':main()
