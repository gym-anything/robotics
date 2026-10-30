"""Opt-in pure-Python harness (GAR_FORKLIFT_PURE_TESTS=1).

It refuses simulator imports and stubs robots.catalog when robots is not installed. Ordinary test runs do
neither.
"""
import importlib.abc
import importlib.util
import os
import sys
import types


class NoSimulator(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'newton', 'warp', 'mujoco', 'pxr'} or fullname in {
            'forklift.world', 'forklift.env', 'forklift.render', 'forklift.render_worker',
        }:
            raise AssertionError(f'Simulator/runtime import forbidden by pure-Python test suite: {fullname}')


if os.environ.get('GAR_FORKLIFT_PURE_TESTS') == '1':
    sys.meta_path.insert(0, NoSimulator())
    if importlib.util.find_spec('robots') is None:
        robots = types.ModuleType('robots')
        robots.__path__ = []
        catalog = types.ModuleType('robots.catalog')
        catalog.ENVIRONMENTS = {'forklift': {'aliases': {'h1-2': 'forklift-h1-2', 'drive-kit': 'forklift-drive-kit'}}}
        robots.catalog = catalog
        sys.modules['robots'] = robots
        sys.modules['robots.catalog'] = catalog
