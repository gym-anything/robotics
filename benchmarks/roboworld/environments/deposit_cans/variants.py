"""Authored inventory variants; selection never moves an existing episode."""
PRESETS={'reference':dict(target_count=24,tub_count=16,preload_count=48),
         'high_fill':dict(target_count=24,tub_count=16,preload_count=120),
         # Same scene, bag and 48 pre-existing cans as reference; only which cans are targets, where
         # they start, and (sort_cups) where the three reusable cups start.
         'single_can':dict(target_count=1,tub_count=0,preload_count=48),
         'sort_cups':dict(target_count=8,tub_count=0,preload_count=48),
         'empty_tub':dict(target_count=16,tub_count=16,preload_count=48),
         # lid_cans, sort_waste and tub_cups: the same scene and 48 pre-existing cans. Each inventory tuple is unique
         # (preset() relies on it).
         'lid_cans':dict(target_count=3,tub_count=0,preload_count=48),
         'sort_waste':dict(target_count=6,tub_count=0,preload_count=48),
         'tub_cups':dict(target_count=8,tub_count=8,preload_count=48)}
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

# Task-specific start placements, world metres (z of a cup is its base). Counter cans stand at
# x = -0.60 + 0.15 (j % 4), y = -0.53 (j < 4) or +0.53 (assembly.can_inventory); arm bases at (0.12, +-0.65, 1.02).
# sort_cups: the three reusable cups stand midway between the counter cans of the right-hand row (4.5 mm from a
# can at the cup rim), all on the right arm's side, which also reaches the tray at (-0.43, 0.375): no hand-off.
SORT_CUPS_ON_COUNTER=((-.525,.53,.903),(-.375,.53,.903),(-.225,.53,.903))
# tub_cups: the three reusable cups stand upright on the tub floor's flat +y end (tub-frame x -0.10, 0, 0.10;
# y 0.15, beyond the floor ribs at |y| < 0.095), 4 mm above it; the tub's eight cans are its bottom rows 0-3.
TUB_CUPS=((-.47,.15,.909),(-.37,.15,.909),(-.27,.15,.909))
# sort_waste: two single-use paper cups on the counter's right-hand row, in the two slots the six cans leave free.
PAPER_CUPS=((-.30,.53,.903),(-.15,.53,.903))
# lid_cans: the three target cans stand upright on the station lid's flat roof (station-frame xy; the roof is
# flat at z 1.016 between the slot edge |y| 0.05715 and the roof edge |y| 0.2276, station.lid_mesh revision 3).
STATION_XY=(.49,0.)
LID_ROOF_Z=1.016
LID_CANS_XY=((-.12,-.14),(.12,-.14),(0.,.14))

def preset(config):
    """Catalogue preset whose exact inventory a configuration builds; None for fixtures and probes."""
    inventory=(config.target_count,config.tub_count,config.preload_count)
    return next((k for k,v in PRESETS.items() if inventory==(v['target_count'],v['tub_count'],v['preload_count'])),None)

def resolve(variant='reference',seed=0,contract=None):
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
