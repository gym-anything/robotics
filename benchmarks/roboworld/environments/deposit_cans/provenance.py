"""Exact source/config receipts for generated evidence, without credentials."""
from dataclasses import asdict
import hashlib,importlib.metadata,json,subprocess,sys,tarfile
from pathlib import Path
from datetime import datetime,timezone

def record(directory,simulation,arguments):
    root=Path(__file__).resolve().parents[1];directory=Path(directory)
    paths=set((root/'deposit_cans').glob('*.py'))
    for module in tuple(sys.modules.values()):
        raw=getattr(module,'__file__',None)
        if not raw:continue
        p=Path(raw).resolve()
        if p.is_relative_to(root) and p.suffix in ('.py','.mdl'):paths.add(p)
    paths.update((root/'deposit_cans').glob('*.mdl'))
    sources={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths) if p.is_file()}
    with tarfile.open(directory/'source_snapshot.tar.gz','w:gz') as archive:
        for path in sources:archive.add(root/path,arcname=path,recursive=False)
    def git(*args):return subprocess.check_output(['git',*args],cwd=root,text=True).strip()
    receipt=dict(created_utc=datetime.now(timezone.utc).isoformat(),git_commit=git('rev-parse','HEAD'),
        package_changes=git('status','--short','--','deposit_cans'),sources=sources,
        config=asdict(simulation.config),arguments=arguments,robot_sources=simulation.assembly.sources,
        packages={p:importlib.metadata.version(p) for p in ('newton','warp-lang','numpy')},
        device=str(simulation.model.device),geometry_checks=simulation.assembly.geometry_checks,
        note='Source snapshot preserves the implementation used; a dirty checkpoint is not a released build.')
    (directory/'provenance.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt
