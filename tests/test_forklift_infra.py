"""Forklift infrastructure: the installed wheel builds a world, the doctor checks what a launch needs, invalid values
are refused rather than substituted, the declared client timeout and wall-time note, the render worker's release
interval, and the per-action physics trace.

No GPU and no renderer. Fast checks need no physics. Two checks build a world on CPU MuJoCo: the served backend at
warehouse/start-and-park seed 0 (about 45-90 s on a shared host, once per module) and, in a child process, the same
world from a wheel of this tree installed alone into a scratch folder (wheel build and install about 1 minute, world
about 1-1.5 minutes). The wheel check needs the forklift and H1-2 assets installed (gar doctor forklift); it is skipped,
saying so, only when they are not.
"""

import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

from gym_anything_robotics.errors import UnsupportedConfigurationError
from tests.packages import PACKAGES, REPO

ROOT = PACKAGES


def _modified_rig():
    """The catalogue's forklift-h1-2 with its instance's mount renamed and its own id."""
    from robots.catalog import resolve
    d = resolve("forklift-h1-2").to_dict()
    d["id"] = "auditor-h1-2-roof"
    for inst in d["instances"].values():
        inst["mount"] = "roof"
    return d


class _Reached(Exception):
    """Raised by a stand-in world: the backend accepted the configuration and went on to build."""


@pytest.fixture
def no_world(monkeypatch):
    import forklift.env

    class NoWorld:
        def __init__(self, *a, **k):
            raise _Reached()
    monkeypatch.setattr(forklift.env, "ForkliftEnv", NoWorld)


# ------------------------------------------------------------------------------------------------ refusals
def test_task_resolver_refuses_a_non_integer_seed():
    from gym_anything_robotics import tasks
    for bad in (1.5, 1.0, "1", True, None, 1000, -1):
        with pytest.raises(ValueError, match="seed"):
            tasks.resolve("forklift", {"task": "start-and-park", "seed": bad})
    spec = tasks.resolve("forklift", {"task": "start-and-park", "seed": 1})
    assert spec.launch.backend_options == {"task": "start-and-park", "seed": 1}


def test_cli_select_seed_1_5_is_refused_not_run_as_1():
    """`gar tasks resolve forklift --select seed=1.5` (the CLI parses the value as JSON: 1.5) is refused."""
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    p = subprocess.run([sys.executable, "-m", "gym_anything_robotics.cli", "tasks", "resolve", "forklift", "--select",
                        "task=start-and-park", "--select", "seed=1.5", "--json"], capture_output=True, text=True,
                       cwd=ROOT, env=env, timeout=300)
    assert p.returncode != 0 and "seed" in (p.stdout + p.stderr) and "seed-1" not in p.stdout, (p.stdout, p.stderr)


def test_backend_refuses_a_setup_the_operator_would_replace(tmp_path, no_world, monkeypatch):
    from robots.catalog import resolve
    from gym_anything_robotics.backends.newton_forklift import NewtonForkliftBackend
    with pytest.raises(UnsupportedConfigurationError, match="refused"):
        NewtonForkliftBackend(robot=_modified_rig(), output_dir=tmp_path / "a", cameras=())
    path = tmp_path / "roof.json"
    path.write_text(json.dumps(_modified_rig()))
    with pytest.raises(UnsupportedConfigurationError, match="refused"):
        NewtonForkliftBackend(robot=str(path), output_dir=tmp_path / "b", cameras=())
    assert not (tmp_path / "a").exists() and not (tmp_path / "b").exists()
    # the backend's own check, independent of the validator: a validator that maps any H1-2 setup to the alias must
    # not get a modified setup built as the stock rig
    import forklift.robot_setup
    monkeypatch.setattr(forklift.robot_setup, "validate_setup", lambda selected: "h1-2")
    with pytest.raises(UnsupportedConfigurationError, match="would be replaced by it"):
        NewtonForkliftBackend(robot=_modified_rig(), output_dir=tmp_path / "c", cameras=())
    assert not (tmp_path / "c").exists()
    # the catalogue rig itself, as an object, its dict, its id or the alias, is what the operator builds: accepted
    rig = resolve("forklift-h1-2")
    for ok in ("h1-2", "forklift-h1-2", rig, rig.to_dict()):
        with pytest.raises(_Reached):
            NewtonForkliftBackend(robot=ok, output_dir=tmp_path / f"ok{id(ok)}", cameras=())


def test_registry_refuses_a_setup_the_operator_would_replace(no_world):
    """Through GAR's selection (registry.create -> robots.catalog.select_for_backend -> forklift.robot_setup
    validate_setup) the backend receives an alias, so the refusal of a modified setup is the validator's; the launch
    configuration (gar run --robot FILE) checks it the same way before a session starts."""
    from gym_anything_robotics import launch, registry
    with pytest.raises(ValueError, match="refused"):
        registry.create("forklift", robot=_modified_rig(), task="start-and-park", seed=0, cameras=())
    with pytest.raises(ValueError, match="refused"):
        launch.LaunchSpec("forklift", robot=_modified_rig())
    renamed = dict(_modified_rig(), id="my-h1-2")
    from robots.catalog import resolve
    renamed["instances"] = resolve("forklift-h1-2").to_dict()["instances"]
    with pytest.raises(_Reached):                      # the rig under another name builds the same robot: accepted
        registry.create("forklift", robot=renamed, task="start-and-park", seed=0, cameras=())


def test_backend_checks_renderer_options_before_building(tmp_path, no_world):
    from gym_anything_robotics.backends.newton_forklift import NewtonForkliftBackend
    for bad in ({"idle_release_s": -1}, {"idle_release_s": float("nan")}, {"idle_release_s": True}, {"samples": 0},
                {"samples": 2.5}, {"width": 640}, {"trace": "yes"}, {"trace_bytes": 10}, []):
        with pytest.raises(UnsupportedConfigurationError):
            NewtonForkliftBackend(output_dir=tmp_path / "x", cameras=(), renderer_options=bad)
        assert not (tmp_path / "x").exists(), bad
    with pytest.raises(_Reached):
        NewtonForkliftBackend(output_dir=tmp_path / "y", cameras=(), renderer_options={"idle_release_s": 0, "samples": 4})


# ------------------------------------------------------------------------------------------------ doctor and errors
def test_doctor_fails_with_the_fixing_commands_on_empty_asset_folders(tmp_path, monkeypatch):
    from gym_anything_robotics import launch
    (tmp_path / "fa").mkdir()
    (tmp_path / "ra").mkdir()
    monkeypatch.setenv("GAR_FORKLIFT_ASSETS", str(tmp_path / "fa"))
    monkeypatch.setenv("GAR_ROBOT_ASSETS", str(tmp_path / "ra"))
    import robots.assets
    monkeypatch.setattr(robots.assets, "_verified", {})
    r = launch.doctor(launch.LaunchSpec("forklift"))
    assert r["ok"] is False
    rows = {c["check"]: c for c in r["environment_checks"]["checks"]}
    fa = rows["forklift assets (assets.lock.json)"]
    h1 = next(c for k, c in rows.items() if k.startswith("robot assets: h1_2"))
    assert not fa["ok"] and "python -m forklift.assets.install" in fa["detail"] and fa["fix"]
    assert not h1["ok"] and "robots.models.h1_2.install" in h1["detail"] and "install" in h1["fix"]
    assert "python -m forklift.assets.install" in r["environment_checks"]["hint"]
    assert "h1_2" in r["environment_checks"]["hint"]


def test_robot_asset_errors_say_what_to_run_for_every_model(tmp_path, monkeypatch):
    from robots import assets
    monkeypatch.delenv("GAR_ROBOT_ASSETS", raising=False)
    want = {"h1_2": "robots.models.h1_2.install", "ur10e": "robots.models.ur10e.install",
            "g1": "whiteboard.assets", "panda": "titrate.newton15.assets", "fr3": "Newton downloads"}
    for name, text in want.items():
        with pytest.raises(ValueError) as exc:
            assets.verify(name, tmp_path)
        assert "asset missing or outside package" in str(exc.value) and text in str(exc.value), (name, exc.value)


def test_env_default_output_is_outside_the_package(tmp_path, monkeypatch):
    from forklift import env as E
    monkeypatch.setenv("GAR_WORKSPACE", str(tmp_path))
    out = E._default_output()
    assert out.is_dir() and out.is_relative_to(tmp_path) and not out.is_relative_to(Path(E.__file__).parent)
    monkeypatch.delenv("GAR_WORKSPACE")
    out = E._default_output()
    assert out.is_dir() and not out.is_relative_to(Path(E.__file__).parent)
    out.rmdir()


def test_each_robot_gets_its_own_runtime_note():
    """A drive-kit session is not told the H1-2's figures."""
    from gym_anything_robotics.backends import newton_forklift as NF
    notes = {r: NF.wall_time_note(r) for r in NF.RUNTIME}
    assert set(notes) == {"h1-2", "drive-kit"} and notes["h1-2"] != notes["drive-kit"]
    for robot, note in notes.items():
        assert NF.RUNTIME[robot], robot
        d = NF.RUNTIME[robot]["shortest_command_within_200x_s"]
        assert f"shortest_command_within_200x_s ({d * 1e3:g} ms)" in note or f"shortest_command_within_200x_s ({d:g} s)" in note
        # its own robot and its own per-request figures, not the other robot's
        other = [r for r in notes if r != robot][0]
        assert NF.ROBOT_NAMES[robot] in note and NF.ROBOT_NAMES[other] not in note
        figures = lambda r: [f"median {NF._n(c['median'])}, p99 {NF._n(c['p99'])}" for c in NF.RUNTIME[r]["per_step"].values()]
        assert all(x in note for x in figures(robot)) and not all(x in note for x in figures(other))
    assert "No served-runtime measurement" in NF.wall_time_note("g1")


# ------------------------------------------------------------------------------------------------ served backend (physics)
@pytest.fixture(scope="module")
def backend(tmp_path_factory):
    from gym_anything_robotics.backends.newton_forklift import NewtonForkliftBackend
    b = NewtonForkliftBackend(task="start-and-park", seed=0, output_dir=tmp_path_factory.mktemp("forklift-infra") / "s")
    yield b
    b.close()


# ------------------------------------------------------------------------------------------------ installed wheel
CHILD = textwrap.dedent('''
    import json, sys
    from pathlib import Path
    target, repo = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    # the runtime's editable install of the checkout must not serve anything: only the scratch target may
    sys.meta_path[:] = [f for f in sys.meta_path if "editable" not in type(f).__module__.lower()
                        and "editable" not in repr(f).lower()]
    sys.path[:] = [p for p in sys.path if p and not Path(p).resolve().is_relative_to(repo)]
    import forklift, gym_anything_robotics
    from gym_anything_robotics import registry
    from gym_anything_robotics.actions import NamedJointTarget
    from forklift.robot_setup import ALIASES
    robots_built = {}
    for alias in [a for a in ALIASES if a != "h1-2"]:     # every other offered robot
        try:
            e2 = registry.create("forklift", task="start-and-park", seed=0, robot=alias, cameras=())
            st = e2._backend.act(NamedJointTarget({}, ticks=40))
            rr = e2._backend.evaluate()
            robots_built[alias] = dict(executed=st.info["executed_ticks"], evaluation=rr["schema"], fault=rr["fault"],
                                       setup=e2._backend.robot_configuration().get("setup_id"))
            e2._backend.close()
        except Exception as exc:
            robots_built[alias] = dict(error=f"{type(exc).__name__}: {exc}")
    env = registry.create("forklift", task="start-and-park", seed=0, cameras=())
    b = env._backend
    step = b.act(NamedJointTarget({}, ticks=40))
    try:
        r = b.evaluate()
    except Exception as exc:                 # reported, and failed below, apart from the build
        r = dict(schema=None, fault=None, error=f"{type(exc).__name__}: {exc}")
    from forklift import doctor
    d = doctor.checks(None)
    out = dict(forklift=forklift.__file__, gar=gym_anything_robotics.__file__, executed=step.info["executed_ticks"],
               robots_built=robots_built,
               evaluation=r["schema"], evaluation_error=r.get("error"), fault=r["fault"], env_output=str(b.env.output),
               visuals=sys.modules["forklift.visuals"].__file__,
               h1_2_installer=(target / "robots/models/h1_2/install.py").is_file(),
               extra_sources=sorted(p.name for p in (target / "forklift/assets").glob("extra_sources*.json")),
               doctor_ok=d["ok"], doctor_failed=[c["check"] for c in d["checks"] if not c["ok"]],
               from_checkout=sorted(n for n, m in sys.modules.items() if getattr(m, "__file__", None)
                                    and Path(m.__file__).resolve().is_relative_to(repo)))
    b.close()
    print("RESULT " + json.dumps(out))
''')


def _copy_sources(dest):
    """The tree's Python files and package data (the packages pyproject.toml ships), without caches or recordings."""
    import shutil
    import tomllib
    t = tomllib.loads((REPO / "pyproject.toml").read_text())
    st = t["tool"]["setuptools"]
    tops = sorted({p.split(".")[0].rstrip("*") for p in st["packages"]["find"]["include"]})
    files = set()
    for top in tops:
        for d, dirs, names in os.walk(ROOT / top):
            dirs[:] = [x for x in dirs if not x.startswith(".") and x != "__pycache__"]
            files.update(Path(d) / n for n in names if n.endswith(".py"))
    for pkg, patterns in st["package-data"].items():
        base = ROOT / pkg.replace(".", "/")
        for pat in patterns:
            files.update(p for p in base.glob(pat) if p.is_file())
    for f in files:
        out = dest / f.relative_to(ROOT)
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, out)
    shutil.copy2(REPO / "pyproject.toml", dest / "pyproject.toml")


def test_wheel_installed_alone_builds_a_forklift_world(tmp_path, monkeypatch):
    """The wheel, installed alone, builds start-and-park with the default H1-2 and with every other offered robot."""
    from forklift import doctor
    assets = os.environ.get("GAR_FORKLIFT_ASSETS") or str(ROOT / "forklift/.cache/assets")
    monkeypatch.setenv("GAR_FORKLIFT_ASSETS", assets)     # the installed copy's own default cache would be empty
    if not (doctor.forklift_assets()[0] and doctor.h1_2_assets()[0]):
        pytest.skip("forklift or H1-2 assets are not installed here (gar doctor forklift); a world cannot be built")
    src, wheels, target = tmp_path / "src", tmp_path / "wheels", tmp_path / "site"
    _copy_sources(src)
    p = subprocess.run([sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation", "-q", "-w",
                        str(wheels), str(src)], capture_output=True, text=True, timeout=1200)
    assert p.returncode == 0, p.stderr[-3000:]
    wheel = next(wheels.glob("gym_anything_robotics-*.whl"))
    p = subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--no-compile", "-q", "--target",
                        str(target), str(wheel)], capture_output=True, text=True, timeout=1200)
    assert p.returncode == 0, p.stderr[-3000:]
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.update(PYTHONPATH=str(target), GAR_FORKLIFT_ASSETS=assets, GAR_WORKSPACE=str(tmp_path / "ws"))
    p = subprocess.run([sys.executable, "-c", CHILD, str(target), str(ROOT)], capture_output=True, text=True,
                       cwd=tmp_path, env=env, timeout=1800)
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith("RESULT ")]
    assert p.returncode == 0 and lines, (p.stdout[-2000:], p.stderr[-4000:])
    r = json.loads(lines[-1][len("RESULT "):])
    print("installed wheel:", wheel.name, wheel.stat().st_size, "bytes;", json.dumps(r))
    for k in ("forklift", "gar", "visuals"):
        assert Path(r[k]).resolve().is_relative_to(target.resolve()), r
    assert r["from_checkout"] == [], r["from_checkout"]
    assert r["executed"] == 40 and r["fault"] is None
    assert r["evaluation"] == "gar.evaluation.v1", r["evaluation_error"]
    from forklift.robot_setup import ALIASES
    assert set(r["robots_built"]) == set(ALIASES) - {"h1-2"}, r["robots_built"]
    for alias, rb in r["robots_built"].items():          # each offered robot builds from the wheel and runs
        assert rb.get("executed") == 40 and rb.get("evaluation") == "gar.evaluation.v1" and rb.get("fault") is None, (alias, rb)
        assert rb["setup"] == ALIASES[alias], (alias, rb)
    assert not Path(r["env_output"]).resolve().is_relative_to(target.resolve())
    assert r["h1_2_installer"] and r["extra_sources"] == ["extra_sources.json", "extra_sources_dockyard.json",
                                                          "extra_sources_interior.json"]
    assert r["doctor_ok"], r["doctor_failed"]
