import pytest

from pharmpy.tools.estmethod.algorithms import _create_est_model
from pharmpy.tools.estmethod.tool import SOLVERS, create_workflow


@pytest.mark.parametrize(
    'algorithm, methods, solvers, no_of_models',
    [
        ('exhaustive', ['foce'], None, 1),
        ('exhaustive', ['foce', 'laplace'], None, 2),
        ('exhaustive', ['foce', 'imp'], ['lsoda'], 2),
        ('exhaustive', ['foce'], 'all', len(SOLVERS)),
        ('exhaustive_with_update', ['foce'], None, 2),
        ('exhaustive_with_update', ['foce', 'laplace'], None, 4),
        ('exhaustive_with_update', ['laplace'], None, 3),
        ('exhaustive_with_update', ['foce'], ['lsoda'], 3),
        ('exhaustive_with_update', ['foce'], 'all', len(SOLVERS) * 2 + 1),
    ],
)
def test_algorithm(algorithm, methods, solvers, no_of_models):
    wf = create_workflow(algorithm, methods=methods, solvers=solvers)
    fit_tasks = [task.name for task in wf.tasks if task.name.startswith('run')]

    assert len(fit_tasks) == no_of_models


@pytest.mark.parametrize(
    'method, est_rec, eval_rec',
    [
        (
            'FO',
            '$ESTIMATION METHOD=ZERO INTER MAXEVAL=9999 AUTO=1 PRINT=10',
            '$ESTIMATION METHOD=IMP INTER EONLY=1 MAXEVAL=9999 ISAMPLE=10000 NITER=10 PRINT=10',
        ),
        (
            'LAPLACE',
            '$ESTIMATION METHOD=COND LAPLACE INTER MAXEVAL=9999 AUTO=1 PRINT=10',
            '$ESTIMATION METHOD=IMP LAPLACE INTER EONLY=1 MAXEVAL=9999 ISAMPLE=10000 '
            'NITER=10 PRINT=10',
        ),
    ],
)
def test_create_est_model(load_model_for_test, pheno_path, method, est_rec, eval_rec):
    model = load_model_for_test(pheno_path)
    assert len(model.estimation_steps) == 1
    est_model = _create_est_model(method, None, model=model, update=False)
    assert len(est_model.estimation_steps) == 2
    assert est_model.model_code.split('\n')[-5] == est_rec
    assert est_model.model_code.split('\n')[-4] == eval_rec
