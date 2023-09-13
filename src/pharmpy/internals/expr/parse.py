from __future__ import annotations

from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    import sympy
else:
    from pharmpy.deps import sympy


def parse(expr: Union[int, float, str, sympy.Expr]) -> sympy.Expr:
    ns = {'Q': sympy.Symbol('Q'), 'LT': sympy.Symbol('LT')}
    return sympy.sympify(expr, locals=ns)
