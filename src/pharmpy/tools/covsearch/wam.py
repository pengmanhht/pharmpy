"""
"Wald approximation method is an efficient algorithm for screening covariate in population model
building using Wald approximation to the likelihood ratio test statistics in conjunction with
Schwarz's Bayesian ceriterion. The algorithm can be aplied to a full model fit of k covariates
parameters to calculate the approximate LRT and SBC for all possible restricted models."

H0: the restricted covariate effects are 0 in hypothesized submodel
H1: the restricted covariate effects are not 0 in hypothesized submodel

Wald Statistic = theta.T @ inv(COV) @ theta
    NOTE: Wald statistic itself follows Chi2 distribution.
    Kowalski et al. described it as an approximation to -2LLR
    Under the hypothesized submodel (H0), large values of Wald statistic providing evidence against
    the submodel

Schwarz's Bayesian criterion (SBC, in reference paper):
    SBC = LL - (p-q)/2 * log(n)
    SBC' = -0.5 * (-2LLR + (p-q) * log(n))
    Large values of SBC indicate a better fit and more probable model

Wald statistic that penalizes the number of parameters and observations
    Keep consistent with the use of BIC in Pharmpy
    BIC = -2LL + n_estimated_parameters * log(n_obs)
    Penalzied Wald Statistic = Wald Statisitc + n_estimated_parameters * log(n_obs)
    Small values of penalized Wald statistic indicate a better fit and more probable model

Reference:
    Kowalski KG, Hutmacher MM. Efficient Screening of Covariates in Population Models Using
        Wald’s Approximation to the Likelihood Ratio Test. J Pharmacokinet Pharmacodyn.
        2001 Jun 1;28(3):253–75.
    Zou Y, Tang F, Ng CM. A Modified Hybrid Wald’s Approximation Method for Efficient Covariate
        Selection in Population Pharmacokinetic Analysis. AAPS J. 2021 Mar;23(2):37.

wam_workflow
    |- wam_init_state_and_effect
    |   |- get_effect_funcs_and_start_model
    |   |- prepare_wam_full_model
    |- wam_backward
    |   |- wam_step
    |   |   |- run_wald_test
    |   |- nonlinear_model_selection_step
    |- results
"""

from collections import Counter
from dataclasses import astuple, dataclass, replace
from itertools import product
from typing import Optional, Union

from pharmpy.deps import numpy as np
from pharmpy.deps import pandas as pd
from pharmpy.deps.scipy import stats
from pharmpy.model import Model
from pharmpy.modeling import (
    add_estimation_step,
    add_parameter_uncertainty_step,
    calculate_bic,
    get_observations,
    remove_estimation_step,
)
from pharmpy.modeling.parameters import get_thetas
from pharmpy.tools.common import create_plots, summarize_tool, table_final_eta_shrinkage
from pharmpy.tools.covsearch.results import COVSearchResults
from pharmpy.tools.covsearch.samba import (
    _modify_summary_tool,
    _nonlinear_step_lrt,
    samba_effect_funcs_and_start_model,
)
from pharmpy.tools.covsearch.util import (
    Candidate,
    DummyEffect,
    SearchState,
    StateAndEffect,
    Step,
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


class Test:
    def __init__(self):
        pass

    def statistic(self) -> Optional[float]:
        pass

    def p_value(self) -> Optional[float]:
        pass

    def penalized_stat(self) -> Optional[float]:
        pass


class WaldTest(Test):
    def __init__(
        self,
        thetas: Optional[np.ndarray],
        covariance_matrix: Optional[np.ndarray],
        num_params: int,
        num_obs: Optional[int] = None,
    ) -> None:
        """
        Initialize Wald test class.
        num_params: number of parameters remaining in the submodel, used for
            penalized_stat calculation
        num_obs: number of observations in the dataset, used for penalized_stat
            calculation
        """
        self.thetas = thetas
        self.covmat = covariance_matrix
        self.num_params = num_params
        self.num_obs = num_obs

    def statistic(self) -> float:
        """perform Wald test and calculate Wald Statistic"""
        if self.thetas is not None and self.covmat is not None:
            try:
                statistic = self.thetas.T @ np.linalg.inv(self.covmat) @ self.thetas
                statistic = float(statistic.squeeze())
            except np.linalg.LinAlgError:
                raise ValueError("Failed to compute Wald statistic: singular covariance matrix")
        else:
            statistic = 0

        return statistic

    def p_value(self) -> float:
        """
        calculate Wald test p-value
        """
        if (wald_stat := self.statistic()) is not None and self.thetas is not None:
            try:
                # df: len(thetas), the number of excluded parameters
                pval = stats.chi2.sf(wald_stat, len(self.thetas))
                pval = float(pval)
            except Exception as e:
                raise ValueError(f"Failed to compute p-value: {str(e)}")
        else:
            pval = np.nan

        return pval

    def penalized_stat(self) -> Optional[float]:
        """
        calculate penalized Wald statistic
        """

        if self.num_obs is None:
            raise ValueError(
                "Number of observations (num_obs) required for calculating penalized_stat of type 'bic'"
            )
        wald_stat = self.statistic()
        try:
            penalized_stat = wald_stat + self.num_params * np.log(self.num_obs)
        except Exception as e:
            raise ValueError(f"Failed to compute penalized statistic: {str(e)}")

        return penalized_stat

    def wald_results(self):
        return WaldResult(
            stat=self.statistic(),
            pval=self.p_value(),
            penalized_stat=self.penalized_stat(),
        )


@dataclass
class WaldResult:
    stat: Optional[float]
    pval: Optional[float]
    penalized_stat: Optional[float]


@dataclass
class WaldInput:
    thetas: np.ndarray  # covariate coefficient estimates
    covmat: np.ndarray  # covariance matrix
    num_obs: int
    num_thetas: int  # number of all thetas
    num_covars: int  # number of covariate parameters


class BackwardStep(Step):
    pass


class WAMStep(BackwardStep):
    pass


@dataclass
class WAMResult:
    rank: int
    results: list
    score_fetcher: dict
    effect_func_fetcher: dict

    @property
    def processed_results(self):
        res_table = pd.DataFrame(
            self.results,
            columns=["inclusion", "Wald_Stat", "Wald_Test_pvalue", "Penalized_Wald_Stat"],
        )
        res_table = res_table.sort_values(
            by="Penalized_Wald_Stat",
            ascending=True,
        ).reset_index(drop=True)
        return res_table

    @property
    def sorted_score_fetcher(self):
        sorted_sf = sorted(self.score_fetcher.items(), key=lambda item: item[1])
        return sorted_sf


@dataclass
class WAMSearchState(SearchState):
    wam_full: Model
    wam_result: Optional[WAMResult] = None

    def __eq__(self, other):
        if not isinstance(other, SearchState):
            return NotImplemented
        return (self.best_candidate_so_far, self.all_candidates_so_far) == (
            other.best_candidate_so_far,
            other.all_candidates_so_far,
        )


def wam_workflow(
    model: Model,
    results: ModelfitResults,
    search_space: Union[str, ModelFeatures],
    p_backward: float = 0.05,
    rank: int = 3,
    strictness: str = "",
):
    wb = WorkflowBuilder(name="covsearch")

    # initiate model and search space
    store_task = Task("store_input_model", store_input_model, model, results)
    wb.add_task(store_task)

    init_task = Task("init", wam_init_state_and_effect, search_space)
    wb.add_task(init_task, predecessors=store_task)

    # WAM backward search task
    wam_search_task = Task(
        "wam_search",
        wam_backward,
        rank,
        p_backward,
    )
    wb.add_task(wam_search_task, predecessors=init_task)
    search_output = wb.output_tasks

    # result task
    result_task = Task("result", wam_task_result, p_backward, strictness)
    wb.add_task(result_task, predecessors=search_output)

    return Workflow(wb)


def wam_init_state_and_effect(context, search_space, input_modelentry):
    # prepare effect functions and model for wam covsearch
    model = input_modelentry.model
    effect_funcs, filtered_model = samba_effect_funcs_and_start_model(search_space, model)
    assert filtered_model is not None

    # create input modelentry (filtered_model)
    input_me = ModelEntry.create(model=filtered_model)

    # prepare full model (call fit workflow inside the function)
    full_me, _ful_me = prepare_wam_full_model(context, filtered_model, effect_funcs)

    # init candiate
    candidate = Candidate(full_me, steps=())

    # init search state
    search_state = WAMSearchState(
        user_input_modelentry=input_me,
        start_modelentry=full_me,
        best_candidate_so_far=candidate,
        all_candidates_so_far=[candidate],
        wam_full=_ful_me,
    )

    return StateAndEffect(search_state=search_state, effect_funcs=effect_funcs)


def prepare_wam_full_model(context, model, effect_funcs):
    """
    prepare full model for wam covsearch
     full_model: model with all covariate effects, fitted with SAEM + IMP for robust OFV
     _ful_model: model for wam step
    """
    # add covariate effects
    desc = "full_model"
    for covfuncs in effect_funcs.values():
        model = covfuncs(model)
    full_model = model.replace(name="full_model", description=desc)
    full_model, _ful_model = set_wam_estimation_step(full_model)

    # fit full models
    full_me = ModelEntry.create(model=full_model, parent=None)
    _ful_me = ModelEntry.create(model=_ful_model, parent=None)
    full_me, _ful_me = _fit_many(context, [full_me, _ful_me])

    return full_me, _ful_me


def _fit_many(context, modelentries):
    # assuming the return order is the same as the input order
    fit_wf = create_fit_workflow(modelentries=modelentries)
    wb = WorkflowBuilder(fit_wf)
    task_gather = Task("gather", lambda *models: models)
    wb.add_task(task_gather, predecessors=wb.output_tasks)
    return context.call_workflow(Workflow(wb), "fit_nonlinear_models")


def set_wam_estimation_step(model):
    # model for robust OFV, whereas _model for covariance matrix
    for i in range(len(model.execution_steps)):
        model = remove_estimation_step(model, i)

    # ITS + SAEM step
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

    # COV step
    _model = add_parameter_uncertainty_step(model, "RMAT")
    _model = _model.replace(name="wam_" + model.name)

    # full model
    model = add_estimation_step(
        model,
        method="IMP",
        idx=2,
        interaction=True,
        niter=20,
        isample=1000,
        tool_options={"EONLY": "1"},
    )
    return model, _model


def _get_covar_names(effect_funcs):
    covar_names = [f"POP_{coveffect[0]}{coveffect[1]}" for coveffect in effect_funcs.keys()]
    return covar_names


def wam_backward(
    context,
    rank: Optional[int],
    p_backward: float,
    state_and_effect: StateAndEffect,
):
    # Wald Approximation
    search_state = wam_step(
        context,
        state_and_effect,
        rank,
    )

    search_state = wam_nonlinear_model_selection(context, search_state, p_backward)
    return search_state


def run_wald_test(
    combination,
    wald_input,
):
    if not isinstance(combination, np.ndarray):
        combination = np.array(combination)
    exclusion_idx = np.where(combination == 0)[0]
    inclusion_idx = np.where(combination == 1)[0]
    num_params = wald_input.num_thetas - len(exclusion_idx)

    if len(exclusion_idx) >= 0:
        sub_thetas = wald_input.thetas[exclusion_idx].reshape(-1, 1)
        sub_covmat = wald_input.covmat[np.ix_(exclusion_idx, exclusion_idx)]
        wald_result = WaldTest(
            sub_thetas, sub_covmat, num_params, wald_input.num_obs
        ).wald_results()
    else:
        wald_result = WaldResult(np.inf, 1.0, np.inf)

    inclusion = ",".join(map(str, inclusion_idx))

    return wald_result, inclusion, inclusion_idx


def wam_step(
    context,
    state_and_effect,
    rank,
) -> WAMSearchState:
    effect_funcs = state_and_effect.effect_funcs
    search_state = state_and_effect.search_state
    wam_full = search_state.wam_full

    results = []
    effect_func_fetcher, score_fetcher = {}, {}

    # wald approximation
    wald_input = prepare_wald_input(wam_full, effect_funcs)
    combinations = product([0, 1], repeat=wald_input.num_covars)

    for comb in combinations:
        wald_result, inclusion, inclusion_idx = run_wald_test(comb, wald_input)
        results.append(
            [
                inclusion,
                wald_result.stat,
                wald_result.pval,
                wald_result.penalized_stat,
            ]
        )
        # get corresponding add_covariate_effect partial functions
        if inclusion_idx.size >= 0:
            effect_funcs_subset = dict(
                item for i, item in enumerate(effect_funcs.items()) if i in inclusion_idx
            )
            effect_func_fetcher[inclusion] = effect_funcs_subset
            score_fetcher[inclusion] = wald_result.penalized_stat

    rank = min(len(score_fetcher), rank) if rank else len(score_fetcher)
    wam_result = WAMResult(rank, results, score_fetcher, effect_func_fetcher)
    search_state = replace(search_state, wam_result=wam_result)

    _wam_loginfo(context, wam_result.processed_results, rank)

    return search_state


def _wam_loginfo(context, results, rank):
    context.log_info(f"WAM MODEL SELECTION\n {results.head(5 if rank < 5 else rank)}")


def prepare_wald_input(modelentry: ModelEntry, effect_funcs: dict) -> WaldInput:
    """
    prepare inputs for Wald test
    """
    assert modelentry.modelfit_results is not None
    assert modelentry.modelfit_results.covariance_matrix is not None
    assert modelentry.modelfit_results.parameter_estimates is not None
    # number of observations
    num_obs = len(get_observations(modelentry.model))
    # name and number of THETAs
    theta_names = get_thetas(modelentry.model).nonfixed.symbols
    theta_names = [t.name for t in theta_names]
    num_thetas = len(theta_names)
    # name and number of covariate parameters
    covar_names = _get_covar_names(effect_funcs)
    num_covars = len(covar_names)
    # estimates of covariate parameters
    covtheta_values = modelentry.modelfit_results.parameter_estimates.loc[covar_names].values
    # estimates of covariance matrix (matrix corresponding to covariates of interesets)
    covarmat_values = modelentry.modelfit_results.covariance_matrix.loc[
        covar_names, covar_names
    ].values

    return WaldInput(
        thetas=covtheta_values,
        covmat=covarmat_values,
        num_obs=num_obs,
        num_thetas=num_thetas,
        num_covars=num_covars,
    )


def wam_nonlinear_model_selection(
    context,
    search_state: WAMSearchState,
    p_backward: float,
) -> tuple:
    best_me = search_state.best_candidate_so_far.modelentry
    best_bic = calculate_bic(best_me.model, best_me.modelfit_results.ofv, "mixed")
    wam_result = search_state.wam_result
    assert isinstance(wam_result, WAMResult), "Expected WAMResult instance"

    # GENERATE CANDIDATE MODELS
    candidate_steps = []
    new_modelentries = []

    score_fetcher = wam_result.sorted_score_fetcher
    effect_func_fetcher = wam_result.effect_func_fetcher

    for r in range(wam_result.rank):
        # get the effect functions for the r-th rank
        inc = score_fetcher[r][0]
        selection = effect_func_fetcher[inc]

        # start with the base model
        updated_model = search_state.user_input_modelentry.model
        desc = updated_model.description
        steps = ()

        # apply the effect functions to the base model
        for cov_effect, cov_func in selection.items():
            updated_model = cov_func(updated_model)
            desc = desc + f";({'-'.join(cov_effect[:3])})"
            steps += (WAMStep(p_backward, DummyEffect(*cov_effect)),)

        # update model metadata
        updated_model = updated_model.replace(name=f"wam_rank#{r + 1}", description=desc)
        updated_model, _ = set_wam_estimation_step(updated_model)

        # fit the updated_model
        candidate_steps.append(steps)
        updated_modelentry = ModelEntry.create(
            model=updated_model,
            parent=search_state.best_candidate_so_far.modelentry.model,
        )

        new_modelentries.append(updated_modelentry)
    fitted_modelentries = _fit_many(context, new_modelentries)

    # EVALUATE CANDIDATE MODELS
    candidates = []
    bic_values = []

    for i, me in enumerate(fitted_modelentries):
        # skip models that failed to fit properly
        if me.modelfit_results is None:
            continue

        # calculate BIC
        bic = calculate_bic(me.model, me.modelfit_results.ofv, "mixed")

        # create candidate and store with its BIC value
        candidate = Candidate(me, candidate_steps[i])
        candidates.append(candidate)
        bic_values.append((candidate, bic))

    # register all candidates
    search_state.all_candidates_so_far.extend(candidates)

    # find the best candidate if any exist
    valid_bics = [(candidate, bic) for candidate, bic in bic_values if not np.isnan(bic)]
    if valid_bics:
        # get the best candidate based on lowest BIC
        best_candidate, best_cand_bic = min(valid_bics, key=lambda x: x[1])

        # update the search state if finding a better model
        if best_cand_bic < best_bic:
            search_state = replace(search_state, best_candidate_so_far=best_candidate)

    _wam_nonlin_loginfo(context, best_bic, bic_values, wam_result)

    return search_state


def _wam_nonlin_loginfo(context, best_bic, bic_values, wam_result):
    score_fetcher = wam_result.sorted_score_fetcher

    log_info = [f"NONLINEAR MODEL RANK\n FULL MODEL: BIC {best_bic:.3f}\n"]
    for r in range(wam_result.rank):
        log_info.append(
            f"    RANK#{r + 1}:\n   BIC {bic_values[r][1]:.3f} | Penalized Wald Stat {score_fetcher[r][1]:.3f}\n"
        )
    context.log_info("\n".join(log_info))


# ============= WAM RESULTS ===============
def wam_task_result(context, p_backward: float, strictness: str, state: WAMSearchState):
    if isinstance(state.wam_result, WAMResult):
        wam_result_table = state.wam_result.processed_results
    else:
        wam_result_table = None
    candidates = state.all_candidates_so_far
    modelentries = list(map(lambda candidate: candidate.modelentry, candidates))
    full_modelentry, *rest_modelentries = modelentries
    assert full_modelentry is state.start_modelentry
    best_modelentry = state.best_candidate_so_far.modelentry
    user_input_modelentry = state.user_input_modelentry
    tables = _wam_create_result_tables(
        candidates,
        best_modelentry,
        user_input_modelentry,
        full_modelentry,
        rest_modelentries,
        cutoff=p_backward,
        strictness=strictness,
    )
    assert best_modelentry.modelfit_results is not None
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
        linear_covariate_screening_summary=wam_result_table,
        steps=tables["steps"],
        ofv_summary=tables["ofv_summary"],
        candidate_summary=tables["candidate_summary"],
    )
    context.store_final_model_entry(best_modelentry)
    context.log_info("Finishing tool covsearch")
    return res


def _wam_create_result_tables(
    candidates,
    best_modelentry,
    input_modelentry,
    full_modelentry,
    res_modelentries,
    cutoff,
    strictness,
):
    model_entries = [full_modelentry] + res_modelentries
    if input_modelentry != full_modelentry:
        model_entries.insert(0, input_modelentry)
    sum_tool_lrt = summarize_tool(
        model_entries,
        full_modelentry,
        rank_type="lrt",
        cutoff=cutoff,
        strictness=strictness,
    )
    sum_tool_lrt = sum_tool_lrt.drop(["rank"], axis=1)
    sum_tool_bic = summarize_tool(
        model_entries,
        full_modelentry,
        rank_type="bic",
        cutoff=cutoff,
        strictness=strictness,
        bic_type="mixed",
    )
    sum_tool_bic = sum_tool_bic.drop(["rank"], axis=1)

    sum_tool = sum_tool_lrt.merge(sum_tool_bic[["dbic", "bic"]], on="model")
    sum_models = summarize_modelfit_results_from_entries(model_entries)
    sum_errors = summarize_errors_from_entries(model_entries)
    steps = _make_wam_steps(best_modelentry, candidates)
    sum_tool = _modify_summary_tool(sum_tool, steps)
    return {
        "summary_tool": sum_tool,
        "summary_models": sum_models,
        "summary_errors": sum_errors,
        "steps": steps,
        "ofv_summary": None,
        "candidate_summary": None,
    }


def _make_wam_steps(best_mdoelentry, candidates):
    best_model = best_mdoelentry.model

    me_dict = {candidate.modelentry.model.name: candidate.modelentry for candidate in candidates}
    children_count = Counter(
        candidate.modelentry.parent.name for candidate in candidates if candidate.modelentry.parent
    )
    data = (
        _make_wam_step_row(me_dict, children_count, best_model, candidate)
        for candidate in candidates
    )
    return pd.DataFrame(data)


def _make_wam_step_row(me_dict, children_count, best_model, candidate):
    candidate_me = candidate.modelentry
    candidate_model = candidate_me.model
    step = _get_step_index(candidate_model.name)

    parent_name = candidate_me.parent.name if candidate_me.parent else candidate_model.name
    parent_me = me_dict[parent_name]

    if candidate.steps:
        steps = candidate.steps
        effects = ["-".join(astuple(st.effect)) for st in steps]
        alpha = steps[-1].alpha
        # LRT
        lrt_res = _nonlinear_step_lrt(candidate_me, parent_me)
        reduced_ofv, extended_ofv, dofv, lrt_pval = (
            lrt_res.parent_ofv,
            lrt_res.child_ofv,
            lrt_res.dofv,
            lrt_res.lrt_pval,
        )
        lrt_significant = lrt_pval < alpha
        # BIC
        reduced_bic = (
            np.nan
            if np.isnan(reduced_ofv)
            else calculate_bic(candidate_model, reduced_ofv, "mixed")
        )
        extended_bic = (
            np.nan
            if np.isnan(extended_ofv)
            else calculate_bic(parent_me.model, extended_ofv, "mixed")
        )
        dbic = reduced_bic - extended_bic
    else:
        effects = ""
        reduced_ofv = np.nan if (mfr := candidate_me.modelfit_results) is None else mfr.ofv
        extended_ofv = np.nan if (mfr := parent_me.modelfit_results) is None else mfr.ofv
        dofv = reduced_ofv - extended_ofv
        reduced_bic = (
            np.nan
            if (reduced_ofv) is None
            else calculate_bic(candidate_model, reduced_ofv, "mixed")
        )
        extended_bic = (
            np.nan
            if (extended_ofv) is None
            else calculate_bic(parent_me.model, extended_ofv, "mixed")
        )
        dbic = reduced_bic - extended_bic
        alpha, lrt_significant, lrt_pval = np.nan, np.nan, np.nan

    selected = children_count[candidate_model.name] >= 1 or candidate_model.name == best_model.name
    return {
        "step": step,
        "covariate_effects": effects,
        "reduced_ofv": reduced_ofv,
        "extended_ofv": extended_ofv,
        "dofv": dofv,
        "lrt_pval": lrt_pval,
        "goal_pval": alpha,
        "lrt_significant": lrt_significant,
        "reduced_bic": reduced_bic,
        "extended_bic": extended_bic,
        "dbic": dbic,
        "selected": selected,
        "model": candidate_model.name,
    }


def _get_step_index(input_string):
    import re

    match = re.search(r"step(\d+)", input_string)
    if match:
        return int(match.group(1))
    return 0
