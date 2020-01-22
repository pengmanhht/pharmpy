import math

import numpy as np

import pharmpy.math
from pharmpy.model import ModelFormatError
from pharmpy.parameter import Parameter, ParameterSet

from .record import Record


class OmegaRecord(Record):
    def parameters(self, start_omega):
        """Get a ParameterSet for this omega record
        """
        row = start_omega
        block = self.root.find('block')
        same = bool(self.root.find('same'))
        parameters = ParameterSet()
        if not block:
            for node in self.root.all('diag_item'):
                init = node.init.NUMERIC
                fixed = bool(node.find('FIX'))
                sd = bool(node.find('SD'))
                var = bool(node.find('VAR'))
                n = node.n.INT if node.find('n') else 1
                if sd and var:
                    raise ModelFormatError(f'Initial estimate for {self.name.upper} cannot be both'
                                           f' on SD and VAR scale\n{self.root}')
                if init == 0 and not fixed:
                    raise ModelFormatError(f'If initial estimate for {self.name.upper} is 0 it'
                                           f' must be set to FIX')
                if sd:
                    init = init ** 2
                if fixed:
                    lower = None
                else:
                    lower = 0
                for _ in range(n):
                    name = f'{self.name}({row},{row})'
                    param = Parameter(name, init, lower=lower, fix=fixed)
                    parameters.add(param)
                    row += 1
            next_omega = row
        else:
            inits = []
            size = self.root.block.size.INT
            fix, sd, corr, cholesky = self._block_flags()
            for node in self.root.all('omega'):
                init = node.init.NUMERIC
                n = node.n.INT if node.find('n') else 1
                inits += [init] * n
            if not same:
                if size != pharmpy.math.triangular_root(len(inits)):
                    raise ModelFormatError('Wrong number of inits in BLOCK')
                if not cholesky:
                    A = pharmpy.math.flattened_to_symmetric(inits)
                    if corr:
                        for i in range(size):
                            for j in range(size):
                                if i != j:
                                    if sd:
                                        A[i, j] = A[i, i] * A[j, j] * A[i, j]
                                    else:
                                        A[i, j] = math.sqrt(A[i, i]) * math.sqrt(A[j, j]) * A[i, j]
                    if sd:
                        np.fill_diagonal(A, A.diagonal()**2)
                else:
                    L = np.zeros((size, size))
                    inds = np.tril_indices_from(L)
                    L[inds] = inits
                    A = L @ L.T
                for i in range(size):
                    for j in range(i, size):
                        name = f'{self.name}({j + start_omega},{i + start_omega})'
                        init = A[j, i]
                        lower = None if i != j or fix else 0
                        param = Parameter(name, init, lower=lower, fix=fix)
                        parameters.add(param)
            next_omega = start_omega + size
        return parameters, next_omega

    def _block_flags(self):
        """Get a tuple of all interesting flags for block
        """
        fix = bool(self.root.find('FIX'))
        var = bool(self.root.find('VAR'))
        sd = bool(self.root.find('SD'))
        cov = bool(self.root.find('COV'))
        corr = bool(self.root.find('CORR'))
        cholesky = bool(self.root.find('CHOLESKY'))
        for node in self.root.all('omega'):
            if node.find('FIX'):
                if fix:
                    raise ModelFormatError('Cannot specify option FIX more than once')
                else:
                    fix = True
            if node.find('VAR'):
                if var or sd or cholesky:
                    raise ModelFormatError('Cannot specify either option VARIANCE, SD or '
                                           'CHOLESKY more than once')
                else:
                    var = True
            if node.find('SD'):
                if sd or var or cholesky:
                    raise ModelFormatError('Cannot specify either option VARIANCE, SD or '
                                           'CHOLESKY more than once')
                else:
                    sd = True
            if node.find('COV'):
                if cov or corr:
                    raise ModelFormatError('Cannot specify either option COVARIANCE or '
                                           'CORRELATION more than once')
                else:
                    cov = True
            if node.find('CORR'):
                if corr or cov:
                    raise ModelFormatError('Cannot specify either option COVARIANCE or '
                                           'CORRELATION more than once')
                else:
                    corr = True
            if node.find('CHOLESKY'):
                if cholesky or var or sd:
                    raise ModelFormatError('Cannot specify either option VARIANCE, SD or '
                                           'CHOLESKY more than once')
                else:
                    cholesky = True
        return fix, sd, corr, cholesky

    def update(self, parameters, first_omega):
        """From a ParameterSet update the OMEGAs in this record
            returns the next omega number
        """
        i = first_omega
        block = self.root.find('block')
        if not block:
            for node in self.root.all('diag_item'):
                sd = bool(node.find('SD'))
                name = f'{self.name}({i},{i})'
                if not sd:
                    value = parameters[name].init
                else:
                    value = parameters[name].init ** 0.5
                node.init.tokens[0].value = str(value)
                n = node.n.INT if node.find('n') else 1
                i += n
        else:
            fix, sd, corr, cholesky = self._block_flags()
            size = self.root.block.size.INT
            row = first_omega
            col = first_omega
            inits = []
            for row in range(first_omega, first_omega + size):
                for col in range(first_omega, row - 1):
                    name = f'{self.name}({row},{col})'
                    inits.append(parameters[name].init)

            A = pharmpy.math.flattened_to_symmetric(inits)
            try:
                L = np.linalg.cholesky(A)
            except np.LinAlgError:
                raise ValueError("Cannot set initial estimates as covariance matrix would not be positive definite.")
            # FIXME: Write initial covariance matrix of (EPS(1), EPS(2)) here...
            if corr:
                for i in range(size):
                    for j in range(size):
                        if i != j:
                            A[i, j] = A[i, j] / (math.sqrt(A[i, i]) * math.sqrt(A[j, j]))
            if sd:
                np.fill_diagonal(A, A.diagonal()**0.5)

            if cholesky:
                A = L

            for node in self.root.all('omega'):
                #node.init.tokens[0].value
                n = node.n.INT if node.find('n') else 1
                inits += [init] * n

        return i

    def random_variables(self, start_omega):
        """Get a RandomVariableSet for this omega record
        """
        pass
