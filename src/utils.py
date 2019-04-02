import os
import logging


_orig_dir = os.path.dirname(os.path.realpath(__file__))


def get_logger(name: str = "custom") -> logging.Logger:
    return logging.getLogger(name)


def resolve_path(*path: str) -> str:
    """Resolve any path based on the project root.
    resolve_path('foo', 'bar') will give an absolute path to your_project_directory/foo/bar
    If the path is already absolute, it will stay absolute
    """
    return os.path.abspath(os.path.join(_orig_dir, '..', *path))


def pretty_duration(t: float) -> str:
    """Return the file size in a pretty way"""
    if t > 1:
        return f"{t:.02f}s"
    t *= 1000
    if t > 1:
        return f"{t:.02f}ms"
    t *= 1000
    if t > 1:
        return f"{t:.02f}us"
    t *= 1000
    return f"{t:.02f}ns"
