"""Task recipe using the frozen classroom catalogue, without constructing a world."""


def resolve(selection):
    from .variations import select, VERSION
    from .tasks import Task
    # An omitted selector explicitly means native catalogue ID zero, not random.
    item = select(**selection) if selection else select(variation_id=0)
    return dict(task_id=f"whiteboard/{VERSION}/{item['variation_id']:04d}",
                description=Task(**item["task"]).description,
                backend_options={"variation_id": item["variation_id"]},
                selection=item, split=item["split"])
