"""
Score test and Lagrange Multiplier
Score test measures the slope (gradient) difference of likelihood function
at maximum likelihood estimate and the point of null hypothesis.

H0: theta = theta0
H1: theta =/= theta0

Score Statistic = score.T @ COV @ score
    Score statistic follows Chi2 distribution
    Under the null hypothesis, large values of Score statistic providing evidence favor
    the inclusion of the associated covariate effect, i.e. the covariate effect cannot
    be ignored.

Penalized Score Statistic = Score Statistic - num_params * log(num_observations)
    Large values of penalized Score statisitc indicate a potentially true covariate effect
    in the model

NOTE: NONMEM has difficulty evaluating variances and gradients around thetas that are fixed at 0
    or even near 0, so we use 1 as the null positions and modify the covariate effect formula
    from: Parameter_i = Parameter + THETA       * Covariate_i + ETA_i
    to:   Parameter_i = Parameter + (THETA - 1) * Covariate_i + ETA_i
"""

from dataclasses import dataclass, replace
from functools import partial
from itertools import count, product
from typing import Literal, Optional, Union

import pharmpy.tools.covsearch.tool as scm_tool
from pharmpy.deps import numpy as np
from pharmpy.deps import pandas as pd
from pharmpy.deps.scipy import stats
from pharmpy.model import Model
from pharmpy.modeling import (
    add_estimation_step,
    add_parameter_uncertainty_step,
    calculate_bic,
    fix_parameters,
    get_observations,
    get_thetas,
    mu_reference_model,
    remove_estimation_step,
    unfix_parameters,
)
from pharmpy.tools.common import (
    create_plots,
    summarize_tool,
    table_final_eta_shrinkage,
    update_initial_estimates,
)
from pharmpy.tools.covsearch.results import COVSearchResults
from pharmpy.tools.covsearch.samba import (
    _modify_summary_tool,
    samba_effect_funcs_and_start_model,
)
from pharmpy.tools.covsearch.score_covariate_effect import score_test_add_covariate_effect
from pharmpy.tools.covsearch.util import (
    Candidate,
    DummyEffect,
    ForwardStep,
    SearchState,
    StateAndEffect,
    StepResult,
    Test,
    TestResult,
    store_input_model,
)
from pharmpy.tools.mfl.parse import ModelFeatures
from pharmpy.tools.modelfit import create_fit_workflow
from pharmpy.tools.run import (
    summarize_errors_from_entries,
    summarize_modelfit_results_from_entries,
)
from pharmpy.workflows import ModelEntry, Task, Workflow, WorkflowBuilder
from pharmpy.workflows.results import ModelfitResults


class ScoreTest(Test):
    def __init__(self, scores, covmat, num_params, num_obs):
        super().__init__(num_params, num_obs)
        self.scores = scores
        self.covmat = covmat

    @property
    def statistic(self):
        if self.scores is not None and self.covmat is not None:
            try:
                stat = self.scores.T @ self.covmat @ self.scores
                stat = float(stat.squeeze())
            except np.linalg.LinAlgError:
                raise ValueError("Failed to compute score statistic: singular covariance matrix")
        else:
            stat = 0
        return stat

    @property
    def pval(self):
        if (stat := self.statistic) is not None and self.scores is not None:
            try:
                pval = stats.chi2.sf(stat, len(self.scores))
                pval = float(pval)
            except Exception as e:
                raise ValueError(f"Failed to compute p-value: {str(e)}")
        else:
            pval = np.nan
        return pval


@dataclass
class ScoreSearchState(SearchState):
    aux: Optional[StepResult] = None


@dataclass
class ScoreInput:
    scores: np.ndarray
    covmat: np.ndarray
    num_params: int
    num_covars: int
    num_obs: int


def score_workflow(
    model: Model,
    results: ModelfitResults,
    search_space: Union[str, ModelFeatures],
    p_forward: float = 0.05,
    rank: int = 1,
    max_steps: int = -1,
    strictness: str = "",
):
    wb = WorkflowBuilder(name="covsearch")

    store_task = Task("store_input_model", store_input_model, model, results)
    wb.add_task(store_task)

    init_task = Task("init", score_init_state_and_effect, search_space)
    wb.add_task(init_task, predecessors=store_task)

    # Score forward search task
    score_forward_task = Task(
        "score_search",
        score_forward,
        rank,
        max_steps,
        p_forward,
    )
    wb.add_task(score_forward_task, predecessors=init_task)
    search_output = wb.output_tasks

    # result task
    result_task = Task("result", score_task_results, p_forward, strictness)
    wb.add_task(result_task, predecessors=search_output)

    return Workflow(wb)


def score_forward(
    context,
    rank: int,
    max_steps: int,
    p_forward: float,
    state_and_effect: StateAndEffect,
):
    init_effect_funcs = state_and_effect.effect_funcs
    search_state = state_and_effect.search_state

    steps = range(1, max_steps + 1) if max_steps >= 1 else count(1)
    for step in steps:
        search_state = score_step(
            context,
            state_and_effect,
            rank,
            step,
        )
        search_state, remaining_effect_funcs = score_nonlinear_model_selection(
            context, step, search_state, init_effect_funcs, p_forward
        )

        if search_state is state_and_effect.search_state:
            break
        else:
            state_and_effect = replace(state_and_effect, search_state=search_state)

        if not remaining_effect_funcs:
            break
        else:
            state_and_effect = replace(state_and_effect, effect_funcs=remaining_effect_funcs)
            init_effect_funcs = remaining_effect_funcs

    return search_state


def score_init_state_and_effect(context, search_space, input_modelentry):
    model = input_modelentry.model
    effect_funcs, null_model = samba_effect_funcs_and_start_model(search_space, model)

    null_modelentry = prepare_null_model(context, null_model, effect_funcs)
    assert isinstance(null_modelentry, ModelEntry)

    candidate = Candidate(null_modelentry, ())

    search_state = ScoreSearchState(
        user_input_modelentry=input_modelentry,
        start_modelentry=null_modelentry,
        best_candidate_so_far=candidate,
        all_candidates_so_far=[candidate],
    )
    return StateAndEffect(search_state=search_state, effect_funcs=effect_funcs)


def set_null_estimation_step(model):
    model = mu_reference_model(model)
    for i in range(len(model.execution_steps)):
        model = remove_estimation_step(model, i)

    model = add_estimation_step(
        model,
        method="ITS",
        idx=0,
        interaction=True,
        auto=True,
        niter=5,
    )
    model = add_estimation_step(
        model,
        method="SAEM",
        idx=1,
        interaction=True,
        niter=200,
        auto=True,
        isample=2,
        keep_every_nth_iter=50,
        tool_options={"NOABORT": 0},
    )

    model = add_estimation_step(
        model,
        method="IMP",
        idx=3,
        interaction=True,
        niter=20,
        auto=True,
        isample=1000,
        tool_options={
            "EONLY": "1",
            "NOABORT": 0,
            "CTYPE": "3",
            "RANMETHOD": "3S2",
        },
    )

    return model


def prepare_null_model(context, model, effect_funcs):
    model = set_null_estimation_step(model)
    score_effect_funcs = _process_effect_funcs(effect_funcs)
    for cov_func in score_effect_funcs.values():
        model = cov_func(model)
    model = model.replace(name="null_model", description="start_model")

    # fix covaraite effect parameters
    covar_names = _get_covar_names(effect_funcs)
    model = fix_parameters(model, covar_names)

    null_me = ModelEntry.create(model=model, parent=None)
    fit_workflow = create_fit_workflow(modelentries=[null_me])
    null_me = context.call_workflow(fit_workflow, "fit_null_model")
    return null_me


def _process_effect_funcs(effect_funcs):
    score_effect_funcs = {
        cov_effect: partial(score_test_add_covariate_effect, *cov_func.args, **cov_func.keywords)
        for cov_effect, cov_func in effect_funcs.items()
    }
    return score_effect_funcs


def _get_covar_names(effect_funcs):
    covar_names = [f"POP_{cov[0]}{cov[1]}" for cov in effect_funcs.keys()]
    return covar_names


def score_step(context, state_and_effect, rank, step) -> ScoreSearchState:
    effect_funcs = state_and_effect.effect_funcs
    search_state = state_and_effect.search_state
    score_result = search_state.aux
    null_me = search_state.best_candidate_so_far.modelentry

    results = [] if score_result is None else score_result.results
    effect_fetcher, score_fetcher = {}, {}
    score_input = _prepare_test_input(context, null_me, effect_funcs, step)
    combinations = _get_combination(score_input.num_covars)

    for comb in combinations:
        test_result, inclusion, inclusion_idx = run_score_test(comb, score_input)
        assert inclusion_idx.size >= 0
        # covariate coefficient to unfix
        effect_subset = dict(
            item for i, item in enumerate(effect_funcs.items()) if i in inclusion_idx
        )
        effect_fetcher[inclusion] = effect_subset
        score_fetcher[inclusion] = test_result.penalized_stat
        results.append(
            [
                step,
                _get_covar_names(effect_subset),
                test_result.stat,
                test_result.pval,
                test_result.penalized_stat,
            ]
        )
    rank = min(len(score_fetcher), rank) if rank else len(score_fetcher)
    # NOTE: aux table's lines may scale up as search_space increases
    step_res = StepResult(rank, results, score_fetcher, effect_fetcher)
    search_state = replace(search_state, aux=step_res)

    return search_state


def run_score_test(comb, score_input):
    if not isinstance(comb, np.ndarray):
        comb = np.array(comb)
    exclusion_idx = np.where(comb == 0)[0]
    inclusion_idx = np.where(comb == 1)[0]
    num_params = score_input.num_params - len(exclusion_idx)

    if len(exclusion_idx) >= 0:
        sub_scores = score_input.scores[inclusion_idx].reshape(-1, 1)
        sub_covmat = score_input.covmat[np.ix_(inclusion_idx, inclusion_idx)]
        score_result = ScoreTest(sub_scores, sub_covmat, num_params, score_input.num_obs).run()

    else:
        score_result = TestResult(np.nan, 1, np.nan)

    inclusion = ",".join(map(str, inclusion_idx))
    return score_result, inclusion, inclusion_idx


def _get_combination(num_covars: int, min_inclusion: bool = True):
    if min_inclusion:
        return np.eye(num_covars)
    else:
        return product([0, 1], repeat=num_covars)


def _set_score_estimation_step(model):
    for i in range(len(model.execution_steps)):
        model = remove_estimation_step(model, 0)

    model = add_estimation_step(
        model,
        method="SAEM",
        idx=0,
        interaction=True,
        niter=200,
        auto=True,
        isample=2,
        keep_every_nth_iter=50,
        tool_options={"NOABORT": 0, "EONLY": "1"},
    )

    model = add_parameter_uncertainty_step(model, "RMAT")
    return model


def _prepare_test_input(context, null_modelentry, effect_fucns, step) -> ScoreInput:
    covar_names = _get_covar_names(effect_fucns)
    num_covars = len(covar_names)

    # get gradients and covariance matrix
    model = update_initial_estimates(null_modelentry.model, null_modelentry.modelfit_results)
    model = unfix_parameters(model, covar_names)
    model = _set_score_estimation_step(model)
    score_model = model.replace(name=f"score_step{step}", description=f"score_step{step}")
    score_me = ModelEntry.create(model=score_model, parent=None)
    fit_workflow = create_fit_workflow(modelentries=[score_me])
    score_me = context.call_workflow(fit_workflow, "fit_score_model")

    scores = score_me.modelfit_results.gradients.loc[covar_names].values
    covmat = score_me.modelfit_results.covariance_matrix.loc[covar_names, covar_names].values

    # get number of parameters and observations
    thetas = get_thetas(score_model).nonfixed.symbols
    num_thetas = len(thetas)
    num_obs = len(get_observations(score_model))

    return ScoreInput(scores, covmat, num_params=num_thetas, num_covars=num_covars, num_obs=num_obs)


def score_nonlinear_model_selection(context, step, search_state, effect_funcs, p_forward):
    best_me = search_state.best_candidate_so_far.modelentry
    best_bic = calculate_bic(best_me.model, best_me.modelfit_results.ofv, "mixed")
    score_result = search_state.aux
    assert isinstance(score_result, StepResult)

    # prepare nonlinear model selection
    remaining_effect_funcs, new_models, candidate_steps = {}, {}, {}
    new_modelentries = []
    score_fetcher = score_result.sorted_score_fetcher(reverse=True)
    rank = score_result.rank
    effect_fetcher = score_result.effect_func_fetcher

    for r in range(rank):
        inc = score_fetcher[r][0]
        selection = effect_fetcher[inc]
        model = best_me.model
        desc = model.description
        cand_steps = search_state.best_candidate_so_far.steps
        covar_names = _get_covar_names(selection)
        model = unfix_parameters(model, covar_names)

        for cov_effect in selection.keys():
            desc = desc + f";({'-'.join(cov_effect[:3])})"
            cand_steps += (ForwardStep(p_forward, DummyEffect(*cov_effect)),)
        model = model.replace(name=f"score_step{step}_rank#{r + 1}", description=desc)

        candidate_steps[inc] = cand_steps
        updated_modelentry = ModelEntry.create(model=model, parent=best_me.model)
        new_models[inc] = model
        new_modelentries.append(updated_modelentry)

    fit_wf = create_fit_workflow(modelentries=new_modelentries)
    wb = WorkflowBuilder(fit_wf)
    task_gather = Task("gather", lambda *models: models)
    wb.add_task(task_gather, predecessors=wb.output_tasks)
    new_modelentries = context.call_workflow(Workflow(wb), "fit_nonlinear_models")

    model_map = {me.model: me for me in new_modelentries}
    new_mes = {inc: model_map[model] for inc, model in new_models.items() if model in model_map}
    nonlin_bic = {
        inc: calculate_bic(me.model, me.modelfit_results.ofv, "mixed")
        for inc, me in new_mes.items()
    }
    candidates = {inc: Candidate(me, candidate_steps[inc]) for inc, me in new_mes.items()}
    search_state.all_candidates_so_far.extend(candidates.values())

    best_candidate_key = min(
        nonlin_bic, key=lambda x: nonlin_bic[x] if not np.isnan(nonlin_bic[x]) else np.inf
    )
    if nonlin_bic[best_candidate_key] < best_bic:
        search_state = replace(search_state, best_candidate_so_far=candidates[best_candidate_key])
        remaining_effect_funcs = {
            cov_eff: cov_func
            for cov_eff, cov_func in effect_funcs.items()
            if cov_eff not in effect_fetcher[best_candidate_key]
        }

    return search_state, remaining_effect_funcs


# ========== Score Method Results ==============
def score_task_results(
    context,
    p_forward,
    strictness,
    state,
):
    candidates = state.all_candidates_so_far
    modelentries = list(map(lambda candidate: candidate.modelentry, candidates))
    base_modelentry, *rest_modelentries = modelentries
    best_modelentry = state.best_candidate_so_far.modelentry
    user_input_modelentry = state.user_input_modelentry
    score_results = state.aux.processed_results()
    tables = _score_create_result_tables(
        candidates,
        best_modelentry,
        user_input_modelentry,
        base_modelentry,
        rest_modelentries,
        cutoff=p_forward,
        strictness=strictness,
    )
    plots = create_plots(best_modelentry.model, best_modelentry.modelfit_results)

    res = COVSearchResults(
        final_model=best_modelentry.model,
        final_results=best_modelentry.modelfit_results,
        summary_models=tables["summary_models"],
        summary_tool=tables["summary_tool"],
        summary_errors=tables["summary_errors"],
        final_model_dv_vs_ipred_plot=plots["dv_vs_ipred"],
        final_model_dv_vs_pred_plot=plots["dv_vs_pred"],
        final_model_cwres_vs_idv_plot=plots["cwres_vs_idv"],
        final_model_abs_cwres_vs_ipred_plot=plots["abs_cwres_vs_ipred"],
        final_model_eta_distribution_plot=plots["eta_distribution"],
        final_model_eta_shrinkage=table_final_eta_shrinkage(
            best_modelentry.model, best_modelentry.modelfit_results
        ),
        linear_covariate_screening_summary=score_results,
        steps=tables["steps"],
        ofv_summary=None,
        candidate_summary=None,
    )
    context.store_final_model_entry(best_modelentry)
    context.log_info("Finishing tool covsearch")
    return res


def _score_create_result_tables(
    candidates,
    best_modelentry,
    input_modelentry,
    base_modelentry,
    rest_modelentries,
    cutoff,
    strictness,
):
    model_entries = [base_modelentry] + rest_modelentries
    if input_modelentry != base_modelentry:
        model_entries.insert(0, input_modelentry)
    sum_tool = summarize_tool(
        model_entries,
        base_modelentry,
        rank_type="bic",
        cutoff=cutoff,
        strictness=strictness,
    )
    sum_tool = sum_tool.drop(["rank"], axis=1)

    sum_models = summarize_modelfit_results_from_entries(model_entries)
    sum_errors = summarize_errors_from_entries(model_entries)
    steps = scm_tool._make_df_steps(best_modelentry, candidates)
    steps = steps.reset_index().rename(columns={"pvalue": "lrt_pval", "goal_pvalue": "goal_pval"})
    sum_tool = _modify_summary_tool(sum_tool, steps)
    return {
        "summary_tool": sum_tool,
        "summary_models": sum_models,
        "summary_errors": sum_errors,
        "steps": steps,
    }
