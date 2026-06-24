"""Sama integration exports."""

__all__ = ["create_batch"]


def __getattr__(name):
    if name == "create_batch":
        from active_learning.integrations.sama.seed import create_batch

        return create_batch
    raise AttributeError(name)
