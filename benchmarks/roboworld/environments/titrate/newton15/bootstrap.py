"""Provision hash-pinned assets and locate the separate render-only runtime.

Run with the dedicated Python environment after installing requirements.txt.
No simulator is started, existing sessions are untouched, and no global Python
installation, service, firewall, or machine environment variable is modified.
"""

import argparse
from importlib.metadata import version, PackageNotFoundError
import json
from pathlib import Path

from .runtime import configure, require_disk


def dependency_check():
    found, errors = {}, []
    for line in Path(__file__).with_name('requirements.txt').read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        name, expected = line.split('==')
        try:
            actual = version(name)
        except PackageNotFoundError:
            actual = None
        found[name] = actual
        if actual != expected:
            errors.append(f'{name}: expected {expected}, found {actual}')
    if errors:
        raise RuntimeError('Install the pinned requirements in this interpreter: '+ '; '.join(errors))
    return found


def setup(isaac_root):
    cache = configure()
    require_disk(cache)
    versions = dependency_check()
    root = Path(isaac_root).expanduser().resolve()
    interpreter = next((root/name for name in ('python.bat', 'python.sh') if (root/name).is_file()), None)
    if interpreter is None:
        raise FileNotFoundError('Expected the installed Isaac Sim 5.1 root containing python.bat or python.sh')
    from .assets import install
    urdf = install(root)  # verifies every installed URDF/mesh against the manifest
    from whiteboard.visual_assets import asset
    appearance = [asset('beige_wall_001', kind) for kind in ('diff', 'rough', 'nor_gl')]
    appearance.append(asset('kiara_5_noon', 'hdri'))
    from whiteboard.web_assets import provision
    provision()
    result = {'render_interpreter': str(interpreter), 'panda_urdf': str(urdf),
        'versions': versions, 'appearance_assets': [str(p) for p in appearance],
        'qualification': 'Dependency and pinned-asset preflight only; no physics or rendering executed'}
    temporary = cache/'runtime.json.tmp'
    temporary.write_text(json.dumps(result, indent=2), encoding='utf-8')
    temporary.replace(cache/'runtime.json')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--isaac-root', type=Path, required=True)
    print(json.dumps(setup(parser.parse_args().isaac_root), indent=2))
