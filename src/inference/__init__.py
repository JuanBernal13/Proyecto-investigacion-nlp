"""Inferencia secuencial, sin entrenamiento y con persistencia transaccional."""
from .engine import execute, exclusive_lock
from .runtime import TransformersBackend, resolve_device

__all__ = ['execute', 'exclusive_lock', 'TransformersBackend', 'resolve_device']
