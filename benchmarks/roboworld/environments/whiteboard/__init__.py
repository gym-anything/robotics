"""Newton 1.5 classroom whiteboard environment (no task-solving policy)."""


def create(**kwargs):
    from .env import WhiteboardEnv

    return WhiteboardEnv(**kwargs)
