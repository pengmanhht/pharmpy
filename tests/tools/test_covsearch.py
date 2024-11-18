from dataclasses import replace
from functools import partial

import pytest

from pharmpy.modeling import (
    add_allometry,
    add_covariate_effect,
    add_peripheral_compartment,
    get_covariate_effects,
    remove_covariate_effect,
)
from pharmpy.tools import read_modelfit_results
from pharmpy.tools.covsearch.tool import (
    AdaptiveStep,
    AddEffect,
    Candidate,
    ForwardStep,
    SearchState,
    _greedy_search,
    _start,
    create_workflow,
    get_effect_funcs_and_base_model,
    get_exploratory_covariates,
    is_model_in_search_space,
    prepare_mfls,
    task_add_covariate_effect,
    validate_input,
)
from pharmpy.tools.mfl.parse import ModelFeatures, get_model_features
from pharmpy.workflows import ModelEntry, Workflow

MINIMAL_INVALID_MFL_STRING = ''
MINIMAL_VALID_MFL_STRING = 'LET(x, 0)'
LARGE_VALID_MFL_STRING = 'COVARIATE?(@IIV, @CONTINUOUS, *);COVARIATE?(@IIV, @CATEGORICAL, CAT)'


def test_create_workflow():
    assert isinstance(create_workflow(MINIMAL_VALID_MFL_STRING), Workflow)


def test_create_workflow_with_model(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'ruvsearch' / 'mox3.mod')
    assert isinstance(create_workflow(MINIMAL_VALID_MFL_STRING, model=model), Workflow)


def test_validate_input():
    validate_input(MINIMAL_VALID_MFL_STRING)


@pytest.mark.parametrize(
    ('model_path',), ((('nonmem', 'pheno.mod'),), (('nonmem', 'ruvsearch', 'mox3.mod'),))
)
def test_validate_input_with_model(load_model_for_test, testdata, model_path):
    model = load_model_for_test(testdata.joinpath(*model_path))
    validate_input(LARGE_VALID_MFL_STRING, model=model)


@pytest.mark.parametrize(
    'funcs, search_space, is_in_search_space',
    [
        ([], 'COVARIATE?([CL,VC],WT,EXP)', True),
        ([], 'COVARIATE([CL,VC],WT,EXP)', False),
        (
            [partial(add_covariate_effect, parameter='CL', covariate='WT', effect='exp')],
            'COVARIATE?([CL,VC],WT,EXP)',
            False,
        ),
        (
            [partial(add_covariate_effect, parameter='CL', covariate='WT', effect='exp')],
            'COVARIATE([CL,VC],WT,EXP)',
            False,
        ),
        (
            [add_peripheral_compartment, add_allometry],
            'COVARIATE([QP1,CL,VC,VP1],WT,POW);COVARIATE?([CL,VC],AGE,[EXP,LIN]);COVARIATE?([CL,VC],SEX,CAT)',
            True,
        ),
    ],
)
def test_is_model_in_search_space(
    load_model_for_test, testdata, funcs, search_space, is_in_search_space
):
    model = load_model_for_test(testdata / 'nonmem' / 'models' / 'mox2.mod')
    for func in funcs:
        model = func(model)

    ss_mfl = ModelFeatures.create_from_mfl_string(search_space)
    model_mfl = ModelFeatures.create_from_mfl_string(get_model_features(model))
    model_mfl = ModelFeatures.create_from_mfl_statement_list(
        model_mfl.mfl_statement_list(["covariate"])
    )

    assert is_model_in_search_space(model, model_mfl, ss_mfl) == is_in_search_space


@pytest.mark.parametrize(
    'search_space, no_of_exploratory_covs',
    [
        ('COVARIATE?([CL,VC],WT,EXP)', 2),
        ('COVARIATE(CL,WT,EXP);COVARIATE?(VC,WT,EXP)', 1),
    ],
)
def test_get_exploratory_covariates(search_space, no_of_exploratory_covs):
    search_space = ModelFeatures.create_from_mfl_string(search_space)
    assert len(get_exploratory_covariates(search_space)) == no_of_exploratory_covs


@pytest.mark.parametrize(
    'search_space, no_of_covariates',
    [
        ('COVARIATE?([CL,VC],WT,EXP)', 1),
        (
            'LET(CONTINUOUS,[AGE,WT]);LET(CATEGORICAL,SEX)\n'
            'COVARIATE?([CL,VC],@CONTINUOUS,exp,*)\n'
            'COVARIATE?([CL,VC],@CATEGORICAL,cat,*)',
            2,
        ),
    ],
)
def test_prepare_mfls(load_model_for_test, testdata, search_space, no_of_covariates):
    model = load_model_for_test(testdata / 'nonmem' / 'models' / 'mox2.mod')
    ss_mfl, model_mfl = prepare_mfls(model, search_space)
    assert model_mfl.absorption is None
    assert len(ss_mfl.covariate) == no_of_covariates
    assert prepare_mfls(model, ss_mfl)[0] == ss_mfl


@pytest.mark.parametrize(
    (
        'model_path',
        'arguments',
        'exception',
        'match',
    ),
    [
        (
            None,
            dict(p_forward='x'),
            TypeError,
            'Invalid `p_forward`',
        ),
        (
            None,
            dict(p_forward=1.05),
            ValueError,
            'Invalid `p_forward`',
        ),
        (
            None,
            dict(p_backward=[]),
            TypeError,
            'Invalid `p_backward`',
        ),
        (
            None,
            dict(p_backward=1.01),
            ValueError,
            'Invalid `p_backward`',
        ),
        (
            None,
            dict(max_steps=1.2),
            TypeError,
            'Invalid `max_steps`',
        ),
        (None, dict(algorithm=()), ValueError, 'Invalid `algorithm`'),
        (
            None,
            dict(algorithm='scm-backward'),
            ValueError,
            'Invalid `algorithm`',
        ),
        (('nonmem', 'pheno.mod'), dict(search_space=1), TypeError, 'Invalid `search_space`'),
        (
            ('nonmem', 'pheno.mod'),
            dict(search_space=MINIMAL_INVALID_MFL_STRING),
            ValueError,
            'Invalid `search_space`',
        ),
        (
            ('nonmem', 'pheno.mod'),
            dict(search_space='LAGTIME(ON)'),
            ValueError,
            'Invalid `search_space`',
        ),
        (
            ('nonmem', 'pheno.mod'),
            dict(search_space='COVARIATE([CL, VC], WGT, EXP)'),
            ValueError,
            'Invalid `search_space` because of invalid parameter',
        ),
        (
            ('nonmem', 'pheno.mod'),
            dict(search_space='COVARIATE([CL, V], SEX, EXP)'),
            ValueError,
            'Invalid `search_space` because of invalid covariate',
        ),
        (
            ('nonmem', 'pheno.mod'),
            dict(search_space='COVARIATE([CL, V], WGT, [EXP, CUSTOM])'),
            ValueError,
            'Invalid `search_space` because of invalid effect function',
        ),
        (
            ('nonmem', 'pheno.mod'),
            dict(search_space='COVARIATE(CL, WGT, exp, -)'),
            ValueError,
            'Invalid `search_space`',
        ),
        (
            None,
            dict(model=1),
            TypeError,
            'Invalid `model`',
        ),
    ],
)
def test_validate_input_raises(
    load_model_for_test,
    testdata,
    model_path,
    arguments,
    exception,
    match,
):
    model = load_model_for_test(testdata.joinpath(*model_path)) if model_path else None

    harmless_arguments = dict(
        search_space=MINIMAL_VALID_MFL_STRING,
    )

    kwargs = {**harmless_arguments, 'model': model, **arguments}

    with pytest.raises(exception, match=match):
        validate_input(**kwargs)


def test_covariate_filtering(load_model_for_test, testdata):
    search_space = 'COVARIATE(@IIV, @CONTINUOUS, lin);COVARIATE?(@IIV, @CATEGORICAL, CAT)'
    model = load_model_for_test(testdata / 'nonmem' / 'pheno_real.mod')
    orig_cov = get_covariate_effects(model)
    assert len(orig_cov) == 3

    eff, filtered_model = get_effect_funcs_and_base_model(search_space, model)
    assert len(eff) == 2
    expected_cov_eff = set((('CL', 'APGR', 'cat', '*'), ('V', 'APGR', 'cat', '*')))
    assert set(eff.keys()) == expected_cov_eff
    assert len(get_covariate_effects(filtered_model)) == 2

    for cov_effect in get_covariate_effects(model):
        model = remove_covariate_effect(model, cov_effect[0], cov_effect[1].name)

    model = add_covariate_effect(model, 'CL', 'WGT', 'pow', '*')
    assert len(get_covariate_effects(model)) == 1
    search_space = 'COVARIATE([CL, V],WGT,pow,*)'
    eff, filtered_model = get_effect_funcs_and_base_model(search_space, model)
    assert len(get_covariate_effects(filtered_model)) == 2
    assert len(eff) == 0

    # exploratory covariates should be removed before covsearch if present in model
    for cov_effect in get_covariate_effects(model):
        model = remove_covariate_effect(model, cov_effect[0], cov_effect[1].name)

    model = add_covariate_effect(model, 'CL', 'WGT', 'pow', '*')
    assert len(get_covariate_effects(model)) == 1
    search_space = 'COVARIATE?([CL, V], WGT, pow, *)'
    eff, filtered_model = get_effect_funcs_and_base_model(search_space, model)
    assert len(get_covariate_effects(filtered_model)) == 0
    assert len(eff) == 2


def test_max_eval(load_model_for_test, testdata):
    model = load_model_for_test(testdata / 'nonmem' / 'pheno.mod')
    modelres = read_modelfit_results(testdata / 'nonmem' / 'pheno.mod')

    no_max_eval_model_entry = _start(model, modelres, max_eval=False)
    assert no_max_eval_model_entry.model == model

    max_eval_model_entry = _start(model, modelres, max_eval=True)
    assert max_eval_model_entry.model != model
    assert max_eval_model_entry.model.execution_steps[0].maximum_evaluations == round(
        3.1 * modelres.function_evaluations_iterations.loc[1]
    )


def _mock_fit(effect, parent, adaptive_step):
    if effect[0][0] == "CL":
        if effect[0][1] == "WGT" and effect[0][2] == "pow":
            ofv = parent.modelentry.modelfit_results.ofv + 100 * -2
        else:
            if not adaptive_step:
                if (
                    len(parent.steps) != 0
                    and effect[0][0] == "CL"
                    and effect[0][1] == "APGR"
                    and effect[0][2] == "pow"
                ):
                    ofv = parent.modelentry.modelfit_results.ofv + 100 * -2
                else:
                    ofv = parent.modelentry.modelfit_results.ofv + 100 * -1
            else:
                if len(parent.steps) != 0:
                    ofv = parent.modelentry.modelfit_results.ofv + 100
                else:
                    ofv = parent.modelentry.modelfit_results.ofv + 100 * -1
    else:
        ofv = parent.modelentry.modelfit_results.ofv + 100

    return ofv


@pytest.mark.parametrize(('adaptive_step',), ((True,), (False,)))
def test_adaptive_scope_reduction(load_model_for_test, testdata, adaptive_step):
    p_value = 0.01
    search_space = "COVARIATE?([CL,V],[WGT,APGR],[exp,lin,pow])"

    # Mock version of handle_effects() within covsearch
    def _mock_handle_effects(
        step: int,
        parent: Candidate,
        candidate_effect_funcs: dict,
        index_offset: int,
    ):
        new_candidate_modelentries = []
        for i, effect in enumerate(candidate_effect_funcs.items(), 1):
            candidate_model_entry = task_add_covariate_effect(
                parent.modelentry,
                parent,
                effect,
                index_offset + i,
            )

            # Fake new ofv value of model
            ofv = _mock_fit(effect, parent, adaptive_step)
            new_modelfit = replace(parent.modelentry.modelfit_results, ofv=ofv)

            candidate_model_entry = ModelEntry.create(
                model=candidate_model_entry.model,
                modelfit_results=new_modelfit,
                parent=candidate_model_entry.parent,
            )
            new_candidate_modelentries.append(candidate_model_entry)
        return [
            Candidate(modelentry, parent.steps + (ForwardStep(p_value, AddEffect(*effect)),))
            for modelentry, effect in zip(new_candidate_modelentries, candidate_effect_funcs.keys())
        ]

    model = load_model_for_test(testdata / 'nonmem' / 'pheno.mod')
    modelres = read_modelfit_results(testdata / 'nonmem' / 'pheno.mod')

    candidate_effect_funcs, start = get_effect_funcs_and_base_model(search_space, model)
    start = _start(start, modelres, False)
    candidate = Candidate(start, ())
    state = SearchState(start, start, candidate, [candidate])

    res = _greedy_search(
        state,
        _mock_handle_effects,
        candidate_effect_funcs,
        p_value,
        6,
        "minimization_successful or (rounding_errors and sigdigs>=0.1)",
        True,
    )

    best_candidate = res.all_candidates_so_far[0]
    for c in res.all_candidates_so_far:
        if c.modelentry.modelfit_results.ofv < best_candidate.modelentry.modelfit_results.ofv:
            best_candidate = c

    for c in res.all_candidates_so_far:
        if len(c.steps) == 1:
            assert (
                best_candidate.modelentry.modelfit_results.ofv <= c.modelentry.modelfit_results.ofv
            )
        elif len(c.steps) == 2:
            assert best_candidate.steps[0] == c.steps[0]
        elif len(c.steps) == 3 and adaptive_step:
            assert isinstance(c.steps[-2], AdaptiveStep)
        elif len(c.steps) == 3 and not adaptive_step:
            assert c.steps[-1].effect.parameter == "V"
            assert (
                c.modelentry.modelfit_results.ofv > best_candidate.modelentry.modelfit_results.ofv
            )
        elif len(c.steps) == 4:
            assert c.steps[-1].effect.parameter == "V"
            assert (
                c.modelentry.modelfit_results.ofv > best_candidate.modelentry.modelfit_results.ofv
            )
