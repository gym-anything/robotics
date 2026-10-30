"""Eight jobs; importing/resolving this catalogue never constructs a simulator."""

from .task import REWARD_CONTRACT, description as titration_description

CONTRACT = 'titration-tasks-v1'
TASKS = {
    'titration': 'hardest',
    'start-stirring': 'entry',
    'centre-flask': 'entry',
    'prime-burette': 'medium',
    'run-in': 'medium',
    'drain-burette': 'medium',
    'decant-acid': 'hard',
    'finish-titration': 'hard',
}
TASK_SET = tuple({'task': name, 'variation_id': 0} for name in (
    'start-stirring', 'centre-flask', 'prime-burette', 'run-in',
    'drain-burette', 'decant-acid', 'finish-titration', 'titration'))


def task_name(value):
    if not isinstance(value, str) or value not in TASKS:
        raise ValueError(f'Unknown titration task {value!r}; choose {tuple(TASKS)}')
    return value


def contract(name):
    return REWARD_CONTRACT if task_name(name) == 'titration' else CONTRACT + '/' + name


def description(name, reagents):
    task_name(name)
    if name == 'titration':
        return titration_description(reagents)
    if name == 'run-in':
        v = reagents.aliquot_ml * reagents.acid_molar / reagents.base_molar
        return (f'A rough titration of this sample took {v:g} mL of base. Run base from the burette '
                f'into the flask until about {v-1:g} mL has gone in, then close the stopcock. '
                'The solution should still be colourless.')
    return {
        'start-stirring': 'Start the magnetic stirrer under the flask and set it to a steady stir that mixes the solution without splashing it.',
        'centre-flask': 'The flask has been nudged off-centre on the stirrer: the burette tip no longer points into its mouth. Move the flask back under the burette tip, without spilling any of it.',
        'prime-burette': 'Before titrating, prime the burette: run about 1 mL of base through the tip into the waste beaker, close the stopcock, and put the flask back under the burette. None of the priming base goes into the flask.',
        'drain-burette': 'The session is over. Drain all the base left in the burette into the waste beaker, keeping it out of the flask, and leave the burette empty with its stopcock open.',
        'decant-acid': 'Pour about half of the hydrochloric acid, roughly 10 mL, from its stock bottle into the clean beaker, so it can be pipetted from there, and stand the bottle back upright on the bench.',
        'finish-titration': 'This titration was paused just before its endpoint: most of the base is already in and the stirred solution is still colourless. Finish it: add base until the first faint pink persists throughout the solution for 30 seconds.',
    }[name]
