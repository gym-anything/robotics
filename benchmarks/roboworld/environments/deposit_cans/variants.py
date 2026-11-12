"""Authored inventory variants; selection never moves an existing episode."""
from gym_anything_robotics import task_folders

PACKAGE='deposit_cans'
# The presets in order; each one's inventory is INVENTORY in its task folder's setup.py (tasks/<variant>/).
PRESET_NAMES=('reference','high_fill',
         # Same scene, bag and 48 pre-existing cans as reference; only which cans are targets, where
         # they start, and (sort_cups) where the three reusable cups start.
         'single_can','sort_cups','empty_tub',
         # lid_cans, sort_waste and tub_cups: the same scene and 48 pre-existing cans. Each inventory tuple is unique
         # (preset() relies on it).
         'lid_cans','sort_waste','tub_cups')
_PRESETS={}

def task_setup(variant):
    """The variant's setup.py (tasks/<variant>/), or None for an inventory outside the catalogue."""
    if variant is None or not task_folders.exists(PACKAGE,variant):return None
    return task_folders.setup(PACKAGE,variant)

def task_verifier(variant):
    """The variant's verifier.py (tasks/<variant>/), or None for an inventory outside the catalogue."""
    if variant is None or not task_folders.exists(PACKAGE,variant):return None
    return task_folders.verifier(PACKAGE,variant)

def presets():
    """variant -> its inventory (target_count, tub_count, preload_count), built once: the task folders' modules are
    loaded on first use."""
    if not _PRESETS:_PRESETS.update((k,task_setup(k).INVENTORY) for k in PRESET_NAMES)
    return _PRESETS

def __getattr__(name):
    if name=='PRESETS':return presets()
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')

# Catalogue order, easiest first.
CATALOGUE=('single_can','lid_cans','sort_cups','sort_waste','empty_tub','tub_cups','reference','high_fill')

# The eight-task catalogue's contract (the high_fill evaluator, task.py Progress._high_fill, with the
# task-specific items below). `reference` keeps its default meaning: a bare selection is v1 (the reference
# instruction, the count-score evaluator and the plain task id); the 'contract' selector picks TASKS_V2_CONTRACT
# explicitly, as the eight-task catalogue's reference entry does. The first choice is the default. No other variant
# has a choice.
TASKS_V2_CONTRACT='deposit-tasks-v2'
CONTRACT_CHOICES={'reference':('v1',TASKS_V2_CONTRACT)}
# The eight-task catalogue, easiest first: each task's exact
# selection, as listed in benchmark.json "task_set".
TASK_SET=tuple(dict(variant=k,**({'contract':TASKS_V2_CONTRACT} if k=='reference' else {}),seed=0) for k in CATALOGUE)

def preset(config):
    """Catalogue preset whose exact inventory a configuration builds; None for fixtures and probes."""
    inventory=(config.target_count,config.tub_count,config.preload_count)
    PRESETS=presets()
    return next((k for k,v in PRESETS.items() if inventory==(v['target_count'],v['tub_count'],v['preload_count'])),None)

def resolve(variant='reference',seed=0,contract=None):
    PRESETS=presets()
    if variant not in PRESETS:raise ValueError('Unknown deposit-cans variation: '+str(variant))
    if type(seed)is not int or not 0<=seed<2**32:raise ValueError('Seed must be an integer in [0,2**32)')
    choices=CONTRACT_CHOICES.get(variant)
    if contract is not None and (not choices or contract not in choices):
        raise ValueError(f'Unknown task contract {contract!r} for {variant}; selectable: '
            +', '.join(f'{k}: {list(v)}' for k,v in CONTRACT_CHOICES.items()))
    config=dict(PRESETS[variant],seed=seed)
    # The default (first) choice resolves with no contract in the configuration (Config.task_contract '' keeps the
    # standing contract). Only another choice is carried explicitly.
    if choices and contract not in (None,choices[0]):config['task_contract']=contract
    return dict(id=variant,seed=seed,config=config)

def contract():
    from .task import TASK_CONTRACTS
    PRESETS=presets()
    return dict(supported=True,selectors=['variant','seed','contract'],variants=[
        dict(id=k,**PRESETS[k],total_cans=PRESETS[k]['target_count']+PRESETS[k]['preload_count'],task_contract=TASK_CONTRACTS[k],
             **({'contract_choices':list(CONTRACT_CHOICES[k])} if k in CONTRACT_CHOICES else {}))for k in CATALOGUE],
        seed_range=[0,2**32-1],
        seed_effect='Initial can yaw only (high_fill: spin of each pre-existing can about its own axis in the authored resting pile); no claim of arbitrary layout generation.',
        contract_selection=f"reference only: 'contract' v1 (default: the count-score evaluator, the reference instruction and the plain task id) or {TASKS_V2_CONTRACT} (the eight-task catalogue's entry).",
        task_set=[dict(s) for s in TASK_SET],
        qualification='Component-qualified v6/scene7 seed0 reference/high_fill inventories; continued loaded-pile packing is documented. '
        'single_can, sort_cups, empty_tub, lid_cans, sort_waste and tub_cups reuse the reference '
        'scene and component evidence; they are not separately qualified. Other seeds are not independently qualified.')

def task_id(variant,seed,contract=None):
    """reference keeps its plain id under v1 (its default); under another contract the id names it."""
    if variant in CONTRACT_CHOICES:
        contract=contract or CONTRACT_CHOICES[variant][0]
        if contract!='v1':return f'deposit-cans/v1/{variant}/{contract}/seed-{seed}'
    return f'deposit-cans/v1/{variant}/seed-{seed}'

def task_resolve(selection):
    from .task import description
    v=resolve(selection.get('variant','reference'),selection.get('seed',0),selection.get('contract'))
    chosen=v['config'].get('task_contract')
    picked=dict(variant=v['id'],seed=v['seed'],**({'contract':chosen} if chosen else {}))
    return dict(task_id=task_id(v['id'],v['seed'],chosen),description=description(v['id'],chosen),
        backend_options=dict(variant=v['id'],**({'contract':chosen} if chosen else {}),config=dict(seed=v['seed'])),
        selection=picked,split=None)
