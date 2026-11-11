"""Contract `titration-tasks-v2`: the render-checked eight.

A selection opts in with "contract": "titration-tasks-v2"; without it every selection keeps
the released behaviour. Under this contract the seven tasks other than titration run the
released catalogue unchanged (presets in task_presets, instructions in task_catalogue,
evaluators in task_evaluators, sessions in task_session) and differ in one thing, the fix:
they report the single contract string `titration-tasks-v2` (the task is the evaluation's
`task` field) instead of `titration-tasks-v1/<task>`. Titration keeps its released selection
{"variation_id": 0} and contract `titration-progress-v1`.

Importing or resolving never constructs a simulator. Nothing here is in a physics-hashed
source list (liquid_snapshot.source_identity, mechanical_snapshot.identity).
"""

import copy

from .task import REWARD_CONTRACT
from .task_catalogue import contract as released_contract, task_name

CONTRACT = 'titration-tasks-v2'
TASK_SET = tuple({'task': name, 'variation_id': 0, 'contract': CONTRACT} for name in (
    'start-stirring', 'centre-flask', 'prime-burette', 'run-in',
    'drain-burette', 'decant-acid', 'finish-titration')) + ({'variation_id': 0},)


def contract(name):
    return REWARD_CONTRACT if task_name(name) == 'titration' else CONTRACT


def check_contract(value):
    if value != CONTRACT:
        raise ValueError(f'Unknown titration contract {value!r}; choose {CONTRACT!r} or omit contract')
    return value


def relabel_selection(item):
    """A resolved catalogue selection (variations.select) under this contract."""
    item = copy.deepcopy(item)
    if 'task_contract' in item:
        item['task_contract'] = contract(item['task_name'])
    return item


def resolve(selection):
    """GAR resolver for selections naming this contract (dispatched from benchmark.resolve)."""
    from .benchmark import resolve as resolve_released
    check_contract(selection.get('contract'))
    rest = {key: value for key, value in selection.items() if key != 'contract'}
    if task_name(rest.get('task', 'titration')) == 'titration':
        raise ValueError(f'Contract {CONTRACT!r} covers the seven tasks other than titration; titration keeps '
                         'its released selection without a contract, e.g. {"variation_id": 0}')
    resolved = resolve_released(rest)
    # The v2 task id carries the contract before the variation, so v2 results never share a v1 id.
    head, variation = resolved['task_id'].rsplit('/', 1)
    resolved['task_id'] = f'{head}/{CONTRACT}/{variation}'
    resolved['selection'] = relabel_selection(resolved['selection'])
    resolved['backend_options'] = {**resolved['backend_options'], 'contract': CONTRACT}
    return resolved


def _relabel(result, label):
    if not isinstance(result, dict):
        return copy.deepcopy(result)
    result = copy.deepcopy(result)
    if 'contract' in result:
        result['contract'] = label
    if isinstance(result.get('reward'), dict) and 'contract' in result['reward']:
        result['reward']['contract'] = label
    return result


class ContractEvaluator:
    """The released CatalogueEvaluator, unchanged, reporting `titration-tasks-v2`.

    Results, checkpoints and restores carry this contract; the inner evaluator and its own
    checkpoint checks see the released labels, so every predicate, grade and score is the
    released one.
    """

    def __init__(self, inner):
        self.inner = inner
        self.contract = contract(inner.name)
        self.released_contract = released_contract(inner.name)

    def __getattr__(self, name):
        if name == 'inner':
            raise AttributeError(name)
        return getattr(self.inner, name)

    @property
    def result(self):
        return _relabel(self.inner.result, self.contract)

    def update(self, tick, **measurements):
        self.inner.update(tick, **measurements)
        return self.result

    def snapshot(self):
        state = self.inner.snapshot()
        state['contract'] = self.contract
        state['result'] = _relabel(state['result'], self.contract)
        return state

    def _released(self, state):
        if not isinstance(state, dict) or state.get('contract') != self.contract:
            raise ValueError('Task checkpoint identity mismatch')
        state = {**state, 'contract': self.released_contract}
        if 'result' in state:
            state['result'] = _relabel(state['result'], self.released_contract)
        return state

    def validate(self, state):
        self.inner.validate(self._released(state))

    def restore(self, state):
        self.inner.restore(self._released(state))


def make_session(config=None, *, task='titration', **kwargs):
    """task_session.make_session under this contract; titration's session is the released one."""
    from .task_session import make_session as make_released
    session = make_released(config, task=task, **kwargs)
    if task_name(task) == 'titration':
        return session
    try:
        session.evaluator = ContractEvaluator(session.evaluator)
        # The episode-start checkpoint that reset restores carries this contract too. Nothing
        # has advanced since make_released took it, so this is the same start state.
        session._initial = session.snapshot()
    except BaseException:
        session.close()
        raise
    return session
