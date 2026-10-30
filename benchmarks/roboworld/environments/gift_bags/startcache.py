"""Where start states are found and where they are written.

A start state (the tote's shipping state, its handle assembly, each wrapped present, the sealed carton) is the end of a
simulation that takes from 23 s (the carton) to half an hour (the tote's shipping state). Each is a file named by a key: a hash of its generator's code (code()), of the
contact world's physics version and of its physical inputs, so a file can only be read by the code that would make it.
Two places hold them:
  gift_bags/start_states/           shipped with the code (committed). A fresh checkout launches every preset and every
                                    wrapped-present variant without building anything.
  .cache/gift_bags/start_states/    the local cache (git-ignored). A build writes here; `python -m gift_bags.prebuild
                                    --ship` copies the current code's files into the shipped folder.
A generator edit changes the keys: the shipped files then no longer match, the first launch builds again, and a contract
test (test_start_states_for_the_current_code_are_shipped) fails until the new files are shipped."""
import ast
import inspect
import textwrap
from pathlib import Path

SHIPPED = Path(__file__).resolve().parent/"start_states"
CACHE = Path(__file__).resolve().parents[1]/".cache"/"gift_bags"/"start_states"


def code(obj):
    """A module or function as the code it runs, for the start-state keys: its syntax tree with the docstrings, the
    comments and the formatting gone, and every raised exception's message removed (the exception's type stays),
    serialised the same way on every Python version (fields that are empty or None are left out). Rewording a
    comment, a docstring or an error message does not change a key."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(obj)))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            b = node.body
            if b and isinstance(b[0], ast.Expr) and isinstance(b[0].value, ast.Constant) and isinstance(b[0].value.value, str):
                node.body = b[1:] or [ast.Pass()]
        if isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call):
            node.exc.args, node.exc.keywords = [], []

    def ser(n):
        if isinstance(n, ast.AST):
            parts = []
            for f in n._fields:
                v = getattr(n, f, None)
                if v is None or v == [] or f in ("type_comment", "type_params", "kind"):
                    continue
                parts.append(f"{f}={ser(v)}")
            return f"{type(n).__name__}({','.join(parts)})"
        if isinstance(n, list):
            return "[" + ",".join(ser(x) for x in n) + "]"
        return repr(n)
    return ser(tree)


def locate(name, cache=None):
    """Path of the start-state file `name`: the shipped file when it exists and no explicit cache folder was asked
    for, else the file's place in the cache (it may not exist yet: the caller builds it and writes it there)."""
    if cache is None and (SHIPPED/name).is_file():
        return SHIPPED/name
    cache = Path(cache) if cache else CACHE
    cache.mkdir(parents=True, exist_ok=True)
    return cache/name


def where(path):
    """'shipped', 'cache' or 'missing' for a start-state path."""
    path = Path(path)
    if not path.is_file():
        return "missing"
    return "shipped" if path.parent == SHIPPED else "cache"
