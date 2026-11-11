"""Arithmetic expression evaluator. The behaviour is specified in README.md next to this file.

    from calc import evaluate, CalcError
"""
from .errors import CalcError
from .evaluator import evaluate

__all__ = ["evaluate", "CalcError"]
