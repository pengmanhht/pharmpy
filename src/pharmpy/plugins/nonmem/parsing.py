import warnings

import pharmpy.plugins.nonmem
from pharmpy.deps import sympy
from pharmpy.model import Assignment, ODESystem, Parameters, RandomVariables

from .advan import compartmental_model


def parse_parameters(control_stream):
    next_theta = 1
    params = []
    for theta_record in control_stream.get_records('THETA'):
        thetas = theta_record.parameters(next_theta, seen_labels={p.name for p in params})
        params.extend(thetas)
        next_theta += len(thetas)
    next_omega = 1
    previous_size = None
    for omega_record in control_stream.get_records('OMEGA'):
        omegas, next_omega, previous_size = omega_record.parameters(
            next_omega, previous_size, seen_labels={p.name for p in params}
        )
        params.extend(omegas)
    next_sigma = 1
    previous_size = None
    for sigma_record in control_stream.get_records('SIGMA'):
        sigmas, next_sigma, previous_size = sigma_record.parameters(
            next_sigma, previous_size, seen_labels={p.name for p in params}
        )
        params.extend(sigmas)
    return Parameters(params)


def parse_random_variables(control_stream):
    dists = RandomVariables.create([])
    next_omega = 1
    prev_cov = None

    for omega_record in control_stream.get_records('OMEGA'):
        etas, next_omega, prev_cov, _ = omega_record.random_variables(next_omega, prev_cov)
        dists += etas
    dists = _adjust_iovs(dists)
    next_sigma = 1
    prev_cov = None
    for sigma_record in control_stream.get_records('SIGMA'):
        epsilons, next_sigma, prev_cov, _ = sigma_record.random_variables(next_sigma, prev_cov)
        dists += epsilons
    rvs = RandomVariables.create(dists)
    return rvs


def parse_statements(model):
    rec = model.internals.control_stream.get_pred_pk_record()
    statements = rec.statements

    des = model.internals.control_stream.get_des_record()
    error = model.internals.control_stream.get_error_record()
    if error:
        sub = model.internals.control_stream.get_records('SUBROUTINES')[0]
        comp = compartmental_model(model, sub.advan, sub.trans, des)
        trans_amounts = dict()
        if comp is not None:
            cm, link = comp
            statements += [cm, link]
            for i, amount in enumerate(cm.amounts, start=1):
                trans_amounts[sympy.Symbol(f"A({i})")] = amount
        else:
            statements += ODESystem()  # FIXME: Placeholder for ODE-system
            # FIXME: Dummy link statement
            statements += Assignment(sympy.Symbol('F'), sympy.Symbol('F'))
        statements += error.statements
        if trans_amounts:
            statements = statements.subs(trans_amounts)
    return statements


def _adjust_iovs(rvs):
    updated = []
    for i, dist in enumerate(rvs):
        try:
            next_dist = rvs[i + 1]
        except IndexError:
            updated.append(dist)
            break

        if dist.level != 'IOV' and next_dist.level == 'IOV':
            new_dist = dist.derive(level='IOV')
            updated.append(new_dist)
        else:
            updated.append(dist)
    return RandomVariables.create(updated)


def parameter_translation(control_stream, reverse=False, remove_idempotent=False, as_symbols=False):
    """Get a dict of NONMEM name to Pharmpy parameter name
    i.e. {'THETA(1)': 'TVCL', 'OMEGA(1,1)': 'IVCL'}
    """
    d = dict()
    for theta_record in control_stream.get_records('THETA'):
        for key, value in theta_record.name_map.items():
            nonmem_name = f'THETA({value})'
            d[nonmem_name] = key
    for record in control_stream.get_records('OMEGA'):
        for key, value in record.name_map.items():
            nonmem_name = f'OMEGA({value[0]},{value[1]})'
            d[nonmem_name] = key
    for record in control_stream.get_records('SIGMA'):
        for key, value in record.name_map.items():
            nonmem_name = f'SIGMA({value[0]},{value[1]})'
            d[nonmem_name] = key
    if remove_idempotent:
        d = {key: val for key, val in d.items() if key != val}
    if reverse:
        d = {val: key for key, val in d.items()}
    if as_symbols:
        d = {sympy.Symbol(key): sympy.Symbol(val) for key, val in d.items()}
    return d


def create_name_trans(control_stream, rvs, statements):
    conf_functions = {
        'comment': _name_as_comments(control_stream, statements),
        'abbr': _name_as_abbr(control_stream, rvs),
        'basic': _name_as_basic(control_stream),
    }

    abbr = control_stream.abbreviated.replace
    pset_current = {
        **parameter_translation(control_stream, reverse=True),
        **{rv: rv for rv in rvs.names},
    }
    sset_current = {
        **abbr,
        **{
            rv: rv
            for rv in rvs.names
            if rv not in abbr.keys() and sympy.Symbol(rv) in statements.free_symbols
        },
        **{
            p: p
            for p in pset_current.values()
            if p not in abbr.keys() and sympy.Symbol(p) in statements.free_symbols
        },
    }

    trans_sset, trans_pset = dict(), dict()
    names_sset_translated, names_pset_translated, names_basic = [], [], []
    clashing_symbols = set()

    for setting in pharmpy.plugins.nonmem.conf.parameter_names:
        trans_sset_setting, trans_pset_setting = conf_functions[setting]
        if setting != 'basic':
            clashing_symbols.update(
                _clashing_symbols(statements, {**trans_sset_setting, **trans_pset_setting})
            )
        for name_current, name_new in trans_sset_setting.items():
            name_nonmem = sset_current[name_current]

            if sympy.Symbol(name_new) in clashing_symbols or name_nonmem in names_sset_translated:
                continue

            name_in_sset_current = {v: k for k, v in sset_current.items()}[name_nonmem]
            trans_sset[name_in_sset_current] = name_new
            names_sset_translated.append(name_nonmem)

            if name_nonmem in pset_current.values() and name_new in pset_current.keys():
                names_pset_translated.append(name_nonmem)

        for name_current, name_new in trans_pset_setting.items():
            name_nonmem = pset_current[name_current]

            if sympy.Symbol(name_new) in clashing_symbols or name_nonmem in names_pset_translated:
                continue

            trans_pset[name_current] = name_new
            names_pset_translated.append(name_nonmem)

        if setting == 'basic':
            params_left = [k for k in pset_current.keys() if k not in names_pset_translated]
            params_left += [rv for rv in rvs.names if rv not in names_sset_translated]
            names_basic = [name for name in params_left if name not in names_sset_translated]
            break

    if clashing_symbols:
        warnings.warn(
            f'The parameter names {clashing_symbols} are also names of variables '
            f'in the model code. Falling back to the in naming scheme config '
            f'names for these.'
        )

    names_nonmem_all = rvs.names + [key for key in parameter_translation(control_stream).keys()]

    if set(names_nonmem_all) - set(names_sset_translated + names_pset_translated + names_basic):
        raise ValueError(
            'Mismatch in number of parameter names, all have not been accounted for. If basic '
            'NONMEM-names are desired as fallback, double-check that "basic" is included in '
            'config-settings for parameter_names.'
        )
    return trans_sset, trans_pset


def _name_as_comments(control_stream, statements):
    params_current = parameter_translation(control_stream, remove_idempotent=True)
    for name_abbr, name_nonmem in control_stream.abbreviated.replace.items():
        if name_nonmem in params_current.keys():
            params_current[name_abbr] = params_current.pop(name_nonmem)
    trans_params = {
        name_comment: name_comment
        for name_current, name_comment in params_current.items()
        if sympy.Symbol(name_current) not in statements.free_symbols
    }
    trans_statements = {
        name_current: name_comment
        for name_current, name_comment in params_current.items()
        if sympy.Symbol(name_current) in statements.free_symbols
    }
    return trans_statements, trans_params


def _name_as_abbr(control_stream, rvs):
    pharmpy_names = control_stream.abbreviated.translate_to_pharmpy_names()
    params_current = parameter_translation(control_stream, remove_idempotent=True, reverse=True)
    trans_params = {
        name_nonmem: name_abbr
        for name_nonmem, name_abbr in pharmpy_names.items()
        if name_nonmem in parameter_translation(control_stream).keys() or name_nonmem in rvs.names
    }
    for name_nonmem, name_abbr in params_current.items():
        if name_abbr in trans_params.keys():
            trans_params[name_nonmem] = trans_params.pop(name_abbr)
    trans_statements = {
        name_abbr: pharmpy_names[name_nonmem]
        for name_abbr, name_nonmem in control_stream.abbreviated.replace.items()
    }
    return trans_statements, trans_params


def _name_as_basic(control_stream):
    trans_params = {
        name_current: name_nonmem
        for name_current, name_nonmem in parameter_translation(control_stream, reverse=True).items()
        if name_current != name_nonmem
    }
    trans_statements = control_stream.abbreviated.replace
    return trans_statements, trans_params


def _clashing_symbols(statements, trans_statements):
    # Find symbols in the statements that are also in parameters
    parameter_symbols = {sympy.Symbol(symb) for _, symb in trans_statements.items()}
    clashing_symbols = parameter_symbols & statements.free_symbols
    return clashing_symbols


def parse_value_type(control_stream, statements):
    ests = control_stream.get_records('ESTIMATION')
    # Assuming that a model cannot be fully likelihood or fully prediction
    # at the same time
    for est in ests:
        if est.likelihood:
            tp = 'LIKELIHOOD'
            break
        elif est.loglikelihood:
            tp = '-2LL'
            break
    else:
        tp = 'PREDICTION'
    f_flag = sympy.Symbol('F_FLAG')
    if f_flag in statements.free_symbols:
        tp = f_flag
    return tp


def parse_description(control_stream):
    rec = control_stream.get_records('PROBLEM')[0]
    return rec.title