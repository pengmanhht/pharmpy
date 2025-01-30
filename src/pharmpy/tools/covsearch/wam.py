"""
"Wald approximation method is an efficient algorithm for screening covariate in population model
building using Wald approximation to the likelihood ratio test statistics in conjunction with
Schwarz's Bayesian ceriterion. The algorithm can be aplied to a full model fit of k covariates
parameters to calculate the approximate LRT and SBC for all possible restricted models."

H0: the restricted covariate effects are 0 in hypothesized submodel
H1: the restricted covariate effects are not 0 in hypothesized submodel

Wald Approximation to -2LLR = theta.T @ inv(COV) @ theta
    Under the hypothesized submodel (H0), large values of dOFV providing evidence against
    the submodel

Schwarz's Bayesian criterion (SBC, in reference paper):
    SBC = LL - (p-q)/2 * log(n)
    SBC' = -0.5 * (-2LLR + (p-q) * log(n))
    Large values of SBC indicate a better fit and more probable model

Bayesian Information Criterion in Pharmpy
    BIC = -2LL + n_estimated_parameters * log(n_obs) # calculate_bic(model, likelihood, 'fixed')
        = -2 * SBC
        ~ -2 * SBC'
        ~ -2LLR + n_estimated_parameters * log(n_obs) # use this in implementation
    Small values of BIC indicate a better fit and more probable model # use this in implementation

Reference:
    Kowalski KG, Hutmacher MM. Efficient Screening of Covariates in Population Models Using
        Wald’s Approximation to the Likelihood Ratio Test. J Pharmacokinet Pharmacodyn.
        2001 Jun 1;28(3):253–75.

wam_workflow
    |- wam_init_state_and_effect
    |   |- get_effect_funcs_and_start_model
    |   |- prepare_wam_full_model
    |- wam_backward
    |   |- wam_approx_step
    |   |   |- wald_approx
    |   |- nonlinear_model_selection_step
    |- results
"""

from itertools import product
from dataclasses import dataclass, replace
import sys
from typing import Optional
from scipy import stats

from pharmpy.tools.covsearch.util import (
    Candidate,
    SearchState,
    StateAndEffect,
    Step,
    DummyEffect,
)
from pharmpy.tools.modelfit import create_fit_workflow
from pharmpy.tools.run import Workflow
from pharmpy.workflows import ModelEntry, Task, WorkflowBuilder

sys.path.insert(0, "/users/peng/pharmpy/src/")

import numpy as np
import pandas as pd
from pharmpy.modeling import (
    add_parameter_uncertainty_step,
    calculate_bic,
    get_thetas,
    get_observations,
)
from pharmpy.tools import fit
from pharmpy.tools.covsearch.samba import (
    samba_effect_funcs_and_start_model,
)


@dataclass
class WaldInputs:
    num_observations: int
    num_thetas: int
    num_covariates: int
    covariate_estimates: np.ndarray
    covariance_matrix: np.ndarray


@dataclass
class WaldResults:
    approx_n2llr: float
    wald_pval: float
    sbc: float
    inclusion: str | None
    inclusion_idx: np.ndarray | None


@dataclass(frozen=True)
class WAMStep(Step):
    pass


@dataclass
class WAMStepResults:
    rank: int
    results: pd.DataFrame
    score_fetcher: list
    effect_func_fetcher: dict


def wam_init_state_and_effect(search_space, model, input_me=None):
    # both start model and full model are required
    effect_funcs, filtered_model = samba_effect_funcs_and_start_model(
        search_space, model
    )
    assert filtered_model is not None
    input_me = ModelEntry.create(model=filtered_model)
    full_me = prepare_wam_full_model(filtered_model, effect_funcs)
    candidate = Candidate(full_me, steps=())
    search_state = SearchState(
        user_input_modelentry=input_me,
        start_modelentry=full_me,
        best_candidate_so_far=candidate,
        all_candidates_so_far=[candidate],
    )
    return StateAndEffect(search_state=search_state, effect_funcs=effect_funcs)


def prepare_wam_full_model(model, effect_funcs):
    # TODO: make sure use SAEM as estimation method
    desc = model.description
    for coveffect, covfuncs in effect_funcs.items():
        model = covfuncs(model)
        desc = desc + f";({'-'.join(coveffect[:3])})"
    full_model = model.replace(name="full_model", description=desc)
    full_model = add_parameter_uncertainty_step(model, "RMAT")

    # full_me = ModelEntry.create(model=full_model)
    # fit_workflow = create_fit_workflow(modelentries=[full_me])
    # full_me = context.call_workflow(fit_workflow, "fit_full_model")
    full_modelfit = fit(full_model, path="full_model")  # pyright: ignore
    full_me = ModelEntry.create(model=full_model, modelfit_results=full_modelfit)

    return full_me


def _get_covar_names(effect_funcs):
    covar_names = [
        f"POP_{coveffect[0]}{coveffect[1]}" for coveffect in effect_funcs.keys()
    ]
    return covar_names


def wam_backward(
    state_and_effect: StateAndEffect,
    rank: Optional[int] = None,
    p_backward: float = 0.01,
):
    effect_funcs = state_and_effect.effect_funcs
    search_state = state_and_effect.search_state
    full_modelentry = search_state.best_candidate_so_far.modelentry
    full_model_ofv = full_modelentry.modelfit_results.ofv  # pyright: ignore
    full_model_bic = calculate_bic(
        full_modelentry.model,
        full_model_ofv,  # pyright: ignore
        "fixed",
    )

    wam_results = wam_approx_step(
        full_modelentry,
        effect_funcs,
        rank,
    )

    search_state, nonlin_bic = wam_nonlinear_model_selection(
        wam_results, search_state, p_backward
    )
    state_and_effect = replace(state_and_effect, search_state=search_state)

    # show approx_bic and true_bic in ranks
    rank, score_fetcher = wam_results.rank, wam_results.score_fetcher
    print(f"NONLINEAR MODEL RANK\n    FULL MODEL: BIC {full_model_bic:.3f}\n")
    for r in range(rank):
        inc = score_fetcher[r][0]
        print(
            f"    RANK#{r + 1} MODEL:\n",
            f"    BIC {nonlin_bic[inc]:.3f} | approx. BIC {score_fetcher[r][1] + full_model_ofv:.3f}\n",
        )

    return state_and_effect


def prepare_wald_inputs(modelentry: ModelEntry, effect_funcs: dict) -> WaldInputs:
    """ """
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
    covtheta_values = modelentry.modelfit_results.parameter_estimates.loc[
        covar_names
    ].values
    # estimates of covariance matrix (matrix corresponding to covariates of interesets)
    covarmat_values = modelentry.modelfit_results.covariance_matrix.loc[
        covar_names, covar_names
    ].values

    wald_inputs = WaldInputs(
        num_observations=num_obs,
        num_thetas=num_thetas,
        num_covariates=num_covars,
        covariate_estimates=covtheta_values,
        covariance_matrix=covarmat_values,
    )
    return wald_inputs


def wald_approx(
    combination: np.ndarray,
    covariance_matrix: np.ndarray,
    covariate_thetas: np.ndarray,
    num_thetas: int,
    num_obs: int,
) -> WaldResults:
    exclusion_idx = np.where(combination == 0)[0]
    inclusion_idx = np.where(combination == 1)[0]

    if len(exclusion_idx) > 0:
        sub_covmat = covariance_matrix[np.ix_(exclusion_idx, exclusion_idx)]
        sub_thetas = covariate_thetas[exclusion_idx].reshape(-1, 1)
        approx_n2llr = wald_stat(sub_thetas, sub_covmat)
    else:
        approx_n2llr = 0

    # number of remaining parameters
    num_params = num_thetas - len(exclusion_idx)
    # calculate Schwarz's Bayesian criterion (SBC)
    sbc = schwarz_bayes(approx_n2llr, num_params, num_obs)
    # index of covariates included in the model (covariate_thetas[inclusion_idx] to get the keys)
    inclusion = ",".join(map(str, inclusion_idx))

    wald_pval = stats.chi2.sf(approx_n2llr, len(exclusion_idx))
    return WaldResults(
        approx_n2llr=approx_n2llr,
        wald_pval=wald_pval,  # pyright: ignore
        sbc=sbc,
        inclusion=inclusion,
        inclusion_idx=inclusion_idx,
    )


def wald_stat(thetas, covmat):
    """
    Wald approximation to -2 * log-likelihood ratio based on
    parameter estiamtes and covariance matrix
    Delta = -2 * log (L1 / L2)
    Delta' = thetas.T @ covmat @ thetas
    """
    approx_n2llr = thetas.T @ np.linalg.inv(covmat) @ thetas

    return approx_n2llr.squeeze()


def schwarz_bayes(approx_n2llr, num_params, num_obs):
    """
    BIC based on approximated LRT statistics
    SBC' in original paper:
        -0.5 * (approx_n2llr + num_params * np.log(num_obs))
    SBC' here is adjusted to make it align with Pharmpy's BIC (fixed):
        approx_n2llr + num_params * np.log(num_obs)
    """
    return approx_n2llr + num_params * np.log(num_obs)


def wam_approx_step(
    full_modelentry,
    effect_funcs,
    rank,
) -> WAMStepResults:
    results = []
    effect_func_fetcher, score_fetcher = {}, {}

    # wald approximation
    wald_inputs = prepare_wald_inputs(full_modelentry, effect_funcs)
    combinations = np.array(list(product([0, 1], repeat=wald_inputs.num_covariates)))
    # reassign rank values
    rank = min(combinations.shape[0], rank) if rank else combinations.shape[0]
    for comb in combinations:
        wald_result = wald_approx(
            comb,
            wald_inputs.covariance_matrix,
            wald_inputs.covariate_estimates,
            wald_inputs.num_thetas,
            wald_inputs.num_observations,
        )
        results.append(
            [
                wald_result.inclusion,
                wald_result.approx_n2llr,
                wald_result.wald_pval,
                wald_result.sbc,
            ]
        )
        # get corresponding add_covariate_effect partial functions
        idx = wald_result.inclusion_idx
        if idx is not None:
            effect_funcs_subset = dict(
                item for i, item in enumerate(effect_funcs.items()) if i in idx
            )
            effect_func_fetcher[wald_result.inclusion] = effect_funcs_subset
            score_fetcher[wald_result.inclusion] = wald_result.sbc

    # sort the score_fetcher by BIC values
    score_fetcher = sorted(score_fetcher.items(), key=lambda item: item[1])
    # results can be returned in the results.csv
    results = pd.DataFrame(
        results,
        columns=["inclusion", "approx_n2llr", "wald_pval", "approx_BIC"],  # pyright: ignore
    )
    # sort BIC in descending order, i.e. small values indicate a better fit
    results = results.sort_values(by="approx_BIC", ascending=True).reset_index(
        drop=True
    )
    print("WAM MODEL SELECTION\n", results.head(10 if rank < 10 else rank))

    return WAMStepResults(rank, results, score_fetcher, effect_func_fetcher)


def wam_nonlinear_model_selection(
    wam_results: WAMStepResults,
    search_state: SearchState,
    p_backward: float,
) -> tuple:
    # candidate models
    new_models, candidate_steps = {}, {}
    new_modelentries = []
    score_fetcher = wam_results.score_fetcher
    best_inc = score_fetcher[1][0]
    rank = wam_results.rank
    effect_func_fetcher = wam_results.effect_func_fetcher
    for r in range(rank):
        inc = score_fetcher[r][0]
        selection = effect_func_fetcher[inc]
        updated_model = search_state.user_input_modelentry.model  # filtered model
        desc = updated_model.description
        for cov_effect, cov_func in selection.items():
            updated_model = cov_func(updated_model)
            desc = desc + f";({'-'.join(cov_effect[:3])})"
            updated_model = updated_model.replace(
                name=f"rank#{r + 1}", description=desc
            )
            updated_model = add_parameter_uncertainty_step(updated_model, "RMAT")
            candidate_steps[inc] = WAMStep(p_backward, DummyEffect(*cov_effect))

        # fit the updated_model
        updated_modelfit = fit(updated_model, path=f"reduced_model_rank#{r + 1}")  # pyright: ignore
        updated_modelentry = ModelEntry.create(
            model=updated_model,
            modelfit_results=updated_modelfit,
            # parent=full_modelentry.model,
        )
        new_models[inc] = updated_model
        new_modelentries.append(updated_modelentry)
    # fit_wf = create_fit_workflow(modelentries=new_modelentries)
    # wb = WorkflowBuilder(fit_wf)
    # task_gather = Task("gather", lambda: *models, models)
    # wb.add_task(task_gather, predecessors=wb.output_tasks)
    # new_modelentries = context.call_workflow(Workflow(wb), "fit_nonlinear_models")
    model_map = {me.model: me for me in new_modelentries}
    new_mes = {
        inc: model_map[model] for inc, model in new_models.items() if model in model_map
    }
    nonlin_bic = {
        inc: calculate_bic(me.model, me.modelfit_results.ofv, "fixed")
        for inc, me in new_mes.items()
    }
    candidates = {
        inc: Candidate(me, candidate_steps[inc]) for inc, me in new_mes.items()
    }
    search_state.all_candidates_so_far.extend(candidates)
    search_state = replace(search_state, best_candidate_so_far=candidates[best_inc])

    return search_state, nonlin_bic
