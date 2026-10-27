"""Create an allow-listed release; never include the surrounding research folder."""
from pathlib import Path
import hashlib
import zipfile

ROOT=Path(__file__).resolve().parents[1]
FILES=('README.md','LICENSE','requirements.txt','requirements-dev.txt','run.py','start.cmd','setup.ps1','.gitignore')
TREES={'.github':{'.yml'},'trisense':{'.py'},'web':{'.html','.css','.js','.svg'},
       'models':{'.pt'},'assets':{'.json'},'examples':{'.npz'},'tests':{'.py'},
       'scripts':{'.py'},'docs':{'.md','.json','.png','.html','.css','.js','.svg','.npz'}}


def release_files():
    paths=[ROOT/name for name in FILES]
    for folder,extensions in TREES.items():
        paths.extend(p for p in (ROOT/folder).rglob('*') if p.is_file() and p.suffix in extensions and '__pycache__' not in p.parts)
    paths.append(ROOT/'docs/.nojekyll')
    for p in paths:
        if p.is_symlink() or not p.resolve().is_relative_to(ROOT):
            raise RuntimeError(f'Unexpected file location: {p.name}')
    return sorted(paths)


def main():
    target=ROOT/'dist';target.mkdir(exist_ok=True)
    output=target/'TriSense-Studio-v1.0.0.zip'
    paths=release_files()
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path,'TriSense-Studio/'+path.relative_to(ROOT).as_posix())
    digest=hashlib.sha256(output.read_bytes()).hexdigest()
    (target/'SHA256SUMS.txt').write_text(f'{digest}  {output.name}\n',encoding='utf-8')
    print(f'{len(paths)} files, {output.stat().st_size:,} bytes: {output}')
    print(f'SHA256: {digest}')


if __name__=='__main__':main()
