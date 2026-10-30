"""Explicit liquid reference plus native layout/charge selection."""
import json
from pathlib import Path


def resolve(selection):
    from .variations import select
    reference = json.loads(Path(__file__).with_name("reference.json").read_text())
    item = select(selection if selection else {"variation_id": 0},
                  spacing_m=reference["liquid_numerics"]["spacing_m"])
    # Native constructor rejects variation_id together with config/reagents.
    # Resolve them here and store every effective value without dropping inventory.
    options = {**reference, "config": item["config"], "reagents": item["reagents"]}
    name = item.get('task_name', 'titration')
    if name != 'titration':
        options['task'] = name
    identity = 'titration/' + (name+'/' if name != 'titration' else '') + item['variation_id']
    return dict(task_id=identity, description=item["task"],
                backend_options=options, selection=item, split=None)
