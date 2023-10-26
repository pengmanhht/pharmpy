from functools import partial
from typing import Iterable

from pharmpy.modeling import add_effect_compartment

from ..statement.feature.effect_comp import EffectComp
from ..statement.feature.symbols import Name, Wildcard
from ..statement.statement import Statement
from .feature import Feature


def features(model, statements: Iterable[Statement]) -> Iterable[Feature]:
    for statement in statements:
        if isinstance(statement, EffectComp):
            modes = (
                [Name('LINEAR'), Name('EMAX'), Name('SIGMOID')]
                if isinstance(statement.modes, Wildcard)
                else statement.modes
            )
            for mode in modes:
                yield ('EFFECTCOMP', mode.name), partial(
                    add_effect_compartment, expr=mode.name.lower()
                )
