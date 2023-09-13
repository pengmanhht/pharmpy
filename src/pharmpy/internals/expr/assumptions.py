from __future__ import annotations

from functools import reduce
from operator import __and__
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    import sympy
else:
    from pharmpy.deps import sympy


def assume_all(predicate: sympy.assumptions.Predicate, expressions: Iterable[sympy.Expr]):
    tautology = sympy.Q.is_true(True)
    return reduce(__and__, map(predicate, expressions), tautology)
