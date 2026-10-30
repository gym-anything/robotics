"""Complete Newton state copy, including late-created solver namespaces."""


def copy_state(destination, source, wp, newton):
    def layout(dst, src):
        for name, value in vars(src).items():
            current = getattr(dst, name, None)
            if isinstance(value, wp.array):
                if not isinstance(current, wp.array) or current.shape != value.shape or current.dtype != value.dtype:
                    setattr(dst, name, wp.empty_like(value))
            elif isinstance(value, newton.Model.AttributeNamespace):
                if current is None:
                    current = newton.Model.AttributeNamespace()
                    setattr(dst, name, current)
                layout(current, value)
            elif value is None and isinstance(current, wp.array):
                setattr(dst, name, None)
    layout(destination, source)
    destination.assign(source)
