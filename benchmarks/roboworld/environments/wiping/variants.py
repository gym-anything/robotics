"""Bounded seeded kitchen variations; no implicit physics-parameter changes."""
PRESETS=('reference','oil-heavy','sticky-food')

def resolve(name='reference',seed=0):
    if name not in PRESETS:raise ValueError('Unknown variant; choose '+', '.join(PRESETS))
    if type(seed) is not int or not 0<=seed<10000:raise ValueError('Seed must be integer 0..9999')
    return dict(id=name,seed=seed,config=dict(seed=seed,soil_preset=name))

def contract():
    return dict(supported=True,selectors=['variant','seed'],variants=list(PRESETS),seed_range=[0,9999],
                qualification='Named soil compositions and seeded patch/crumb positions; not 30000 independently qualified episodes')

def task_resolve(selection):
    from .task import description
    v=resolve(selection.get('variant','reference'),selection.get('seed',0))
    return dict(task_id=f"wipe-work-surfaces/v1/{v['id']}/seed-{v['seed']}",description=description(v['id']),
        backend_options=dict(variant=v['id'],config=dict(seed=v['seed'])),
        selection=dict(variant=v['id'],seed=v['seed']),split=None)
