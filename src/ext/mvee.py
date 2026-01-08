from __future__ import division

import numpy as np
import numpy.linalg as la
import scipy.stats
import scipy.linalg as sla
import scipy.sparse.linalg as ssla
import copy
import warnings
import time

from scipy.optimize import minimize_scalar
import scipy.optimize
from argparse import Namespace

import choldate as chol

from tube_generation.settings import *



# {{{ Special sparse matrix

class NullSpaceMatrix:
    '''
    Matrix representing a basis for vectors orthogonal to constraints.
    Takes advantage of special structure to save on storage and matvec costs
    even more so than sparse matrices.
    '''
    def __init__(self, constraints):
        self._constraints = copy.copy(constraints)
        self.shape = (constraints.lenw + constraints.lennw,
                      constraints.lennw - 1)


    def dot(self, x):
        '''
        Calculate matrix-vector or matrix-matrix product
        '''
        if x.shape[0] != self.shape[1]:
            info = (self.shape, x.shape, self.shape[1], x.shape[0])
            raise ValueError('shapes %s and %s not aligned: %d != %d' % info)

        if len(x.shape) > 1:
            b = np.zeros((self.shape[0], x.shape[1]))
        else:
            b = np.zeros(self.shape[0])
        b[self._constraints.basic] = -x.sum(axis=0)
        setting_inds = np.argsort(self._constraints.order)
        b[self._constraints.superbasic] = x[setting_inds]

        return b


    def Tdot(self, x):
        '''
        Calculate matrix-vector or matrix-matrix product using
        transpose matrix
        '''
        if x.shape[0] != self.shape[0]:
            info = (self.shape, x.shape, self.shape[0], x.shape[0])
            raise ValueError('shapes %s and %s not aligned: %d != %d' % info)

        if len(x.shape) > 1:
            b = -np.ones((self.shape[1], x.shape[1]))
        else:
            b = -np.ones(self.shape[1])
        b = b*x[self._constraints.basic]
        b = b + x[self._constraints.superbasic][self._constraints.order]

        return b


    def Tdotsmall(self, X_superbasic, X_basic):
        '''
        Calculate product with cached matrix
        '''
        b = -np.ones((self.shape[1], X_basic.shape[0]))
        b = b*X_basic
        b = b + X_superbasic.T

        return b


    def hatTdot(self, x):
        '''
        Calculate sort-of matrix-vector or matrix-matrix product using
        transpose matrix
        '''
        if x.shape[0] != self.shape[0]:
            info = (self.shape, x.shape, self.shape[0], x.shape[0])
            raise ValueError('shapes %s and %s not aligned: %d != %d' % info)

        if len(x.shape) > 1:
            b = np.ones((self._constraints.lennw - 1, x.shape[1]))
        else:
            b = np.ones((self._constraints.lennw - 1))
        b = b*x[self._constraints.basic]
        b = b + x[self._constraints.superbasic][self._constraints.order]

        return b


    def hatTdotsmall(self, X_superbasic, X_basic):
        '''
        Calculate product with cached matrix
        '''
        b = np.ones((self._constraints.lennw - 1,
                     X_basic.shape[0]))
        b = b*X_basic
        b = b + X_superbasic.T

        return b


    def project_z(self, x, ortho=True):
        '''
        Orthogonal projection
        Gives coordinates in basis Z instead of standard basis
        Equivalent to multiplication by (Z^T Z)^-1 Z^T for ortho=True.

        With ortho=False, a non-orthogonal projection equivalent to Z.Tdot(x).
        '''
        vec = self.Tdot(x)
        if ortho:
            vec = vec - vec.sum()/(1 + self.shape[1])

        return vec


    def project(self, x, ortho=True):
        '''
        Orthogonal projection onto Z
        Equivalent to multiplication by Z (Z^T Z)^-1 Z^T

        With ortho=False, a non-orthogonal projection equivalent to
        Z.dot(Z.Tdot(x)).

        The orthogonal projection matrix is symmetric, so there is no need
        for a separate routine for multiplication by the transpose.
        '''
        vec = self.project_z(x, ortho)
        return self.dot(vec)

# }}}


# {{{ Constraints class

class Constraints:
    '''
    Bitmask of length m representing active equality constraints.
    '''
    def __init__(self, working, n_c, cache):
        '''
        Creates new working set.

        Parameters
        ----------
        working : list
            Indices of working inequality constraints.
        n_c : int
            Total number of constraints, equality and inequality,
            including non-working ones.
        '''
        self.active = np.zeros(n_c - 1, dtype=bool)
        self.active[working] = 1
        self.lenw = len(working)
        self.lennw = n_c - 1 - self.lenw
        self.basic = np.argmax(self.active < 1)
        self.basic_ind = 0
        self.superbasic = ~self.active
        self.superbasic[self.basic] = 0
        self.order = np.arange(self.superbasic.sum())
        self.cache = cache
        self.cache.set_constraints(self)


    def add_constraint(self, i_new_constraint):
    # {{{
        '''
        Add constraint to working set.

        Parameters
        ----------
        i_new_constraint : int
            Index of constraint being added to active set.

        Returns
        -------
        Z : NullSpaceMatrix
            Basis for subspace of vectors orthogonal to rows of constraint
            matrix.
        '''
    # }}}

        if self.active[i_new_constraint]:
            raise ValueError(('Cannot add constraint %d because it is'
                              ' already in active set' % i_new_constraint))

        self.active[i_new_constraint] = True
        self.superbasic[i_new_constraint] = False
        self.lenw += 1
        self.lennw -= 1
        if self.basic == i_new_constraint:
            self.basic = np.argmax(self.active < 1)
            self.superbasic[self.basic] = 0
            order_ind = np.sum(self.superbasic[:self.basic])
            self.basic_ind = order_ind
        else:
            order_ind = np.sum(self.superbasic[:i_new_constraint])
            self.basic_ind = np.sum(self.superbasic[:self.basic])
        order_loc = np.where(self.order == order_ind)[0][0]
        self.order[self.order > order_ind] -= 1
        self.order = np.delete(self.order, order_loc)
        Z = NullSpaceMatrix(self)
        self.cache.reset()

        return Z


    def remove_constraint(self, i_gone_constraint):
    # {{{
        '''
        Remove constraint from working set.

        Parameters
        ----------
        i_gone_constraint : int
            Index of constraint being moved from working to not working.

        Returns
        -------
        Z : NullSpaceMatrix
            Basis for subspace of vectors orthogonal to rows of constraint
            matrix.
        '''
    # }}}

        if not self.active[i_gone_constraint]:
            raise ValueError(('Cannot remove constraint %d because it is'
                              ' not in active set' % i_gone_constraint))

        self.active[i_gone_constraint] = False
        self.superbasic[i_gone_constraint] = True
        if i_gone_constraint < self.basic:
            self.basic_ind += 1
        self.lenw -= 1
        self.lennw += 1
        order_ind = np.sum(self.superbasic[:i_gone_constraint])
        self.order[self.order >= order_ind] += 1
        self.order = np.append(self.order, order_ind)
        Z = NullSpaceMatrix(self)
        self.cache.reset()

        return Z

# }}}


# {{{ Cache

class Cache:
    '''
    Lazily instantiate expensive variables

    The variables (except big_X) must be cleared when constraints is updated.
    '''
    def __init__(self, X):
        self.big_X = X
        self.reset()


    def set_constraints(self, constraints):
        self.constraints = constraints
        self.reset()


    def reset(self):
        self.X = None
        self.X_basic = None
        self.X_superbasic = None
        self.x_k = None
        self.zTxT = None
        self.zhatTxT = None


    def get_X(self):
        if self.X is None:
            self.X = self.big_X[:, ~self.constraints.active]
        return self.X


    def get_X_basic(self):
        if self.X_basic is None:
            X = self.get_X()
            self.X_basic = X[:, self.constraints.basic_ind]
        return self.X_basic


    def get_X_superbasic(self):
        if self.X_superbasic is None:
            X = self.get_X()
            self.X_superbasic = np.delete(X, self.constraints.basic_ind,
                                          axis=1)[:, self.constraints.order]
        return self.X_superbasic


    def get_zTxT(self, Z):
        if self.zTxT is None:
            X = self.get_X()
            self.ZTXT = Z.Tdotsmall(self.get_X_superbasic(),
                                    self.get_X_basic())
        return self.ZTXT


    def get_zhatTxT(self, Z):
        if self.zhatTxT is None:
            X = self.get_X()
            self.ZhatTXT = Z.hatTdotsmall(self.get_X_superbasic(),
                                          self.get_X_basic())
        return self.ZhatTXT

# }}}


# {{{ Objective

def objective(L, factor=1.0):
    '''
    Objective function of the optimization problem. Takes as input the
    scaled Cholesky factorization of the shape matrix. Let H be the test
    matrix:

    H = inverse(1/factor * L @ L.T)

    that satisfies

    x.T @ H @ x <= n for all x.

    The objective function is

    f(H) = -lndet(H) ,

    which is a strictly convex function of H that the primal problem seeks to
    minimize. An equivlent objective function is

    g(u) = lndet(XUX^T) ,

    which is a strictly concave function of u that the dual problem seeks to
    maximize.

    Note that the two are equivalent because H = (XUX^T)^-1.
    '''
    return 2*(np.sum(np.log(np.diag(L)))) - len(L)*np.log(factor)


def objective_primal(H):
    return -np.log(la.det(H))

# }}}


# {{{ BFGS helpers

def _bfgs_remove_constraint(HessL, constraints, old_constraints, Z, X, xk, L,
                            g_z, all_indices, bfgs_restart, ortho_bfgs,
                            gpu_args=None):
    '''
    g_z is projected gradient at xk under *old* constraints
    '''

    if bfgs_restart == 'update':
        # Do not increase size if Z was empty before
        if HessL.shape[1] == Z.shape[1]:
            return HessL

        q = np.argmax(constraints.order)
        inds = ~constraints.active
        z = xk[inds]

        # Determine step size relative to current iterate
        relative_step_norm = 1e-8
        current_iter_norm = la.norm(z, np.inf)
        h = relative_step_norm*current_iter_norm

        # Take step in direction of new constraint
        z[constraints.basic_ind] -= h
        z[q] += h
        L2 = cholesky_small(constraints, z, less_stable=True)

        # Computing L2 requires new constraints to get correct values of
        # gradient entries, but we want to select elements for the projected
        # gradient based on the old constraints
        g_z_plus_h = projected_gradient_small(old_constraints, L2,
                                              ortho=ortho_bfgs)

        # projected (onto Z) difference-of-gradients
        # this is actually Z.T v in Murtagh paper
        # Order of subtraction is reversed because we use negative Hessian
        v = (g_z - g_z_plus_h)/h

        dmax = np.max(np.diag(HessL))
        dmin = np.min(np.diag(HessL))

        if la.norm(v, np.inf) > 100*dmax or la.norm(v, np.inf) < 0.01*dmin:
            r = np.zeros(len(HessL))
            rho = 1
        else:
            r = sla.solve(HessL, v, lower=True)
            v1 = (gradient_elements(X, L, constraints.basic, gpu_args=gpu_args) -
                  gradient_elements(X, L2, constraints.basic, gpu_args=gpu_args))/h
            ind2 = all_indices[constraints.superbasic][q]
            v2 = (gradient_elements(X, L, ind2, gpu_args=gpu_args) -
                  gradient_elements(X, L2, ind2, gpu_args=gpu_args))/h
            zTv = v2 - v1
            sigma = zTv - la.norm(r, np.inf)**2

            if sigma > 0:
                rho = np.sqrt(sigma)
            else:
                rho = np.inf

            if la.norm(v, np.inf) > 100*rho or la.norm(v, np.inf) < 0.01*rho:
                r = np.zeros(len(HessL))
                rho = 1

        n = len(HessL)
        new_HessL = np.zeros((n + 1, n + 1))
        new_HessL[:n, :n] = HessL
        new_HessL[-1, :n] = r
        new_HessL[-1, -1] = rho
    else:
        new_HessL = np.eye(len(HessL) + 1)

    return new_HessL


def _bfgs_swap_basic(HessL, q):
    '''
    Swap the basic variable for the qth superbasic variable.

    q is index into superbasic variables
    g_b is gradient evaluated at previous basic variable
    '''
    HessL = HessL.copy().astype(float)
    l_q = HessL[q].copy()
    l_qq = l_q[q]

    v = -np.ones(len(HessL))
    v[q] -= 1

    # Givens rotations to reduce l_q to a multiple of e_q
    for i in range(q - 1, -1, -1):
        x = l_qq
        y = l_q[i]
        r = np.sqrt(x**2 + y**2)
        c = x/r
        s = -y/r

        l_qq = c*x - s*y

        # Also fills up qth col of L
        col_q = c*HessL[:, q] - s*HessL[:, i]
        col_i = s*HessL[:, q] + c*HessL[:, i]
        HessL[:, q] = col_q
        HessL[:, i] = col_i

    # add multiple of v to qth col of L
    HessL[:, q] += v*l_qq

    # More Givens rotations to restore L to triangular form
    for i in range(q):
        x = HessL[i, i]
        y = HessL[i, q]
        r = np.sqrt(x**2 + y**2)
        c = x/r
        s = -y/r

        col_i = c*HessL[:, i] - s*HessL[:, q]
        col_q = s*HessL[:, i] + c*HessL[:, q]
        HessL[:, q] = col_q
        HessL[:, i] = col_i

    return HessL


def _bfgs_add_constraint(HessL, q, bfgs_restart):
    '''
    Remove qth row from HessL due to new constraint.

    q is index into superbasic variables
    '''

    if len(HessL) < 3:
        return np.array([[1]])

    if bfgs_restart == 'update':
        n = len(HessL) - 1

        # Delete qth row of L
        new_HessL = np.zeros((n, n + 1))
        new_HessL[:q] = HessL[:q]
        new_HessL[q:] = HessL[(q + 1):]

        # Restore to triangular form by Givens rotations on right
        for i in range(q, n):
            j = i + 1
            x = new_HessL[i, i]
            y = new_HessL[i, j]
            r = np.sqrt(x**2 + y**2)
            c = x/r
            s = -y/r

            col_i = c*new_HessL[:, i] - s*new_HessL[:, j]
            col_j = s*new_HessL[:, i] + c*new_HessL[:, j]
            new_HessL[:, i] = col_i
            new_HessL[:, j] = col_j

        return new_HessL[:, :-1]
    else:
        return np.eye(len(HessL) - 1)


def _bfgs_updated_hessian(HessL, pk, ak, gk, gkp1, bfgs_method):
# {{{
    '''
    Calculate rank-two update of approximate Hessian according to
    BFGS formula. For description below, let B be the approximate
    Hessian and Bbar be the updated Hessian.

    Note: Since we are keeping a Cholesky factorization, we expect a positive
    definite matrix. However, the actual optimization process is maximizing,
    so the matrix is negative definite. To get around this, we negate the
    gradients for all of our calculations in this function and compute the
    negative Hessian.  We need to be careful elsewhere in the code to remember
    that the Hessian is -(HessL @ HessL.T), not (HessL @ HessL.T).

    Parameters
    ----------
    HessL : 2d array
        Unit lower-triangular matrix such that
        `-HessL @ HessL.T = B`
    pk : 1d array
        Step direction calculated by solving `B @ pk = -gk`
    ak : scalar
        Length of step along pk as determined by line search.
        If actual step taken is `sk`, then `sk = ak*pk`
    gk : 1d array
        Gradient at step k
    gkp1 : 1d array
        Gradient at step k+1

    Returns
    -------
    HessL : 2d array
        Unit lower-triangular matrix such that
        `-HessL @ HessL.T = Bbar`
    '''
# }}}

    # Skip update if step was too small or in bad direction
    if ak < 1e-15 or gkp1.dot(pk)/np.abs(gk.dot(pk)) > 0.9:
        return HessL

    yk = gk - gkp1

    if bfgs_method == 'BFGS':
        Hess_p = HessL.dot(HessL.T.dot(pk))

    b = ak*yk.dot(pk)
    v = yk/np.sqrt(b)
    chol.cholupdate(HessL.T, v.copy())

    if bfgs_method == 'BFGS':
        a = ak**2*pk.dot(Hess_p)
        u = ak*Hess_p/np.sqrt(a)
    else:
        a = gk.dot(pk)
        u = gk/np.sqrt(a)

    chol.choldowndate(HessL.T, u.copy())

    return HessL


def _lbfgs_updated_hessian(pk, ak, gk, gkp1, sks, yks, rhos, HessInv, nsteps,
                           i, first):
# {{{
    '''
    Store vectors needed for L-BFGS approximation

    Note: Since we are keeping a Cholesky factorization, we expect a positive
    definite matrix. However, the actual optimization process is maximizing,
    so the matrix is negative definite. To get around this, we negate the
    gradients for all of our calculations in this function and compute the
    negative Hessian.  We need to be careful elsewhere in the code to remember
    that the Hessian is -(HessL @ HessL.T), not (HessL @ HessL.T).

    Parameters
    ----------
    pk : 1d array
        Step direction calculated by solving `B @ pk = -gk`
    ak : scalar
        Length of step along pk as determined by line search.
        If actual step taken is `sk`, then `sk = ak*pk`
    gk : 1d array
        Gradient at step k
    gkp1 : 1d array
        Gradient at step k+1
    sks : 2d array
        Each row of sks is a previous step. See notes in _lbfgs_matmult.
        Modified in place.
    yks : 2d array
        Each row of yks is a previous difference of gradients.
        See notes in _lbfgs_matmult. Modified in place.
    rhos : 1d array
        Array satisfying rhos[k] = 1/(y[k].dot(s[k])). Modified in place.
    nsteps : scalar
        Number of previous steps used in L-BFGS computations
    i : scalar
        Number of iterations since last restart
    first : scalar
        See notes in _lbfgs_matmult

    Returns
    -------
    first : scalar
        See notes in _lbfgs_matmult
    '''
# }}}
    # Skip update if step was too small or in bad direction
    if ak < 1e-15 or gkp1.dot(pk)/np.abs(gk.dot(pk)) > 0.9:
        return first, HessInv

    s = ak*pk
    y = gk - gkp1
    rho = 1/(y.dot(s))

    sks[i % nsteps] = s
    yks[i % nsteps] = y
    rhos[i % nsteps] = rho

    HessInv = np.eye(len(s))*s.dot(y)/la.norm(y)**2

    if i >= nsteps:
        first += 1

    return first, HessInv


def _lbfgs_restart_hessian(nsteps, Z):
    sks = np.zeros((nsteps, Z.shape[1]))
    yks = np.zeros((nsteps, Z.shape[1]))
    rhos = np.zeros(nsteps)
    HessInv = np.eye(Z.shape[1])
    return 0, 0, sks, yks, rhos, HessInv


def _lbfgs_matmult(hinv, vec, sks, yks, rhos, first):
# {{{
    '''
    Perform matrix-vector multiplication with the approximate Hessian
    according to the L-BFGS algorithm.

    Parameters
    ----------
    hinv : 2d array
        Inverse of initial approximate (negative) Hessian
    vec : 1d array
        Vector to multiply by inverse approximate Hessian
    sks : 2d array
        Each row of sks is a previous step. See notes below.
    yks : 2d array
        Each row of yks is a previous difference of gradients.
        See notes below.
    rhos : 1d array
        Array satisfying rhos[k] = 1/(y[k].dot(s[k]))
    first : scalar
        Index of oldest value in other arrays. See notes below.

    Returns
    -------
    r : 1d array
        Matrix-vector product of vec with approximate Hessian

    Notes
    -----
    The arrays sks, yks, and rhos hold all elements that we want and no more.

    The values corresonding to the earliest iteration are stored at element
    `first`. The rest of the elements are stored in order from earliest to
    latest iteration starting from element at index `first`, moving rightward
    through the array, and cycling back to element at index 0 once the end of
    the array is hit.

    For example, if values from iterations 50-53 are stored, one possible
    ordering would be elements from iterations

    [52, 53, 50, 51],

    in which case `first = 2`.
    '''
# }}}

    n = len(sks)

    r = vec.copy()
    alphas = np.zeros(n)
    for j in range(n - 1, -1, -1):
        i = (j + first) % n
        alpha = rhos[i]*np.dot(sks[i], r)
        alphas[i] = alpha
        r = r - alpha*yks[i]

    r = hinv.dot(r)

    for j in range(n):
        i = (j + first) % n
        beta = rhos[i]*yks[i].dot(r)
        r = r + (alphas[i] - beta)*sks[i]

    return r

# }}}


# {{{ Calculation of gradient and Hessian

if DO_CUDA:
    import pycuda.autoinit
    import pycuda.driver as cuda
    import pycuda.gpuarray as gpuarray
    from pycuda.compiler import SourceModule
    import skcuda
    import skcuda.cublas
    import skcuda.cusolver


def gradient(X, L, factor=1.0, gpu_args=None):
    if DO_CUDA and gpu_args is not None:
        n, m = X.shape
        if not L.flags['F_CONTIGUOUS']:
            L = np.asfortranarray(L)
        if not X.flags['F_CONTIGUOUS']:
            X = np.asfortranarray(X)

        handle, L_gpu, X_gpu, square_gpu = gpu_args
        side = 'L'
        uplo = 'L'
        trans = 'n'
        diag = 'n'
        m_gpu = n
        n_gpu = m
        alpha = np.float32(1.)
        cuda.memcpy_htod(L_gpu, L)
        lda = n
        cuda.memcpy_htod(X_gpu, X)
        ldb = n

        skcuda.cublas.cublasDtrsm(handle, side, uplo, trans, diag,
                                  m_gpu, n_gpu, alpha,
                                  int(L_gpu.gpudata), lda,
                                  int(X_gpu.gpudata), ldb)

        square_gpu(X_gpu, block=(256, 1, 1), grid=((n*m + 255)//256, 1))
        sol = np.empty((n, m), order='F')
        cuda.memcpy_dtoh(sol, X_gpu)

        sol = sol.sum(axis=0)
        return factor*sol
    else:
        return factor*\
               (sla.solve_triangular(L, X,
                                     lower=True,
                                     check_finite=False)**2).sum(axis=0)


def gradient_elements(X, L, inds, gpu_args=None):
    return gradient(X[:, inds], L, gpu_args=gpu_args)


def small_gradient(constraints, L):
    X_basic = constraints.cache.get_X_basic()
    X_superbasic = constraints.cache.get_X_superbasic()

    g_z = (sla.solve_triangular(L, X_superbasic,
                                lower=True,
                                check_finite=False)**2).sum(axis=0)
    tmp = (sla.solve_triangular(L, X_basic,
                                lower=True,
                                check_finite=False)**2).sum()

    return np.hstack([tmp, g_z])


def projected_gradient_small(constraints, L, ortho=True):
    X_basic = constraints.cache.get_X_basic()
    X_superbasic = constraints.cache.get_X_superbasic()

    g_z = (sla.solve_triangular(L, X_superbasic,
                                lower=True,
                                check_finite=False)**2).sum(axis=0)
    tmp = (sla.solve_triangular(L, X_basic,
                                lower=True,
                                check_finite=False)**2).sum()

    g_z = g_z - tmp

    if ortho:
        # (constraints.lennw - 1) is Z.shape[1]
        g_z = g_z - g_z.sum()/(1 + (constraints.lennw - 1))

    return g_z


def projected_gradient(X, L, Z, ortho=True):
    '''
    Directly compute projected gradient instead of forming g_xk first.
    If ortho is true, calculate (Z^T Z)^-1 Z^T g_xk.
    Otherwise, calculate Z^T g_xk.

    Note that this is different from Z.project(g_xk) because this will return
    a smaller vector. To recover Z.project(g_xk) from the g_z returned by this
    function (with ortho=True), simply compute Z.dot(g_z).
    '''
    basic = Z._constraints.basic
    superbasic = Z._constraints.superbasic
    order = Z._constraints.order

    g_z = (sla.solve_triangular(L, X[:, superbasic][:, order],
                                lower=True,
                                check_finite=False)**2).sum(axis=0)
    tmp = (sla.solve_triangular(L, X[:, basic],
                                lower=True,
                                check_finite=False)**2).sum()

    g_z = g_z - tmp

    if ortho:
        g_z = g_z - g_z.sum()/(1 + Z.shape[1])

    return g_z


def gradient_primal(H, is_cholesky=False, upper_inds=None, diag_inds=None):
    '''
    Return the primal gradient assuming that the function is treated as a
    function of the upper triangle of the matrix H rather than as a function
    of the entire symmetric matrix.

    If is_cholesky is True, assume we have been passed the Cholesky factor
    of the negative gradient.
    '''
    if is_cholesky:
        grad = -H@H.T
    else:
        grad = -la.inv(H)
    return upper_triangle_to_vec_scaled(grad, upper_inds=upper_inds,
                                        diag_inds=diag_inds)


def gradient_primal_fd(H, orig_L=None):
    '''
    Return the finite-difference approximation of the primal gradient assuming
    that the function is treated as a function of the upper triangle of the
    matrix H rather than as a function of the entire symmetric matrix.

    If orig_L is not None, we have been passed the Cholesky factor of the
    negative gradient.
    '''
    step_len = 1e-8*H.max()

    if orig_L is None:
        orig_L = la.cholesky(la.inv(H))

    f_xk = objective(orig_L)

    n = len(H)
    fd_grad = np.zeros(int(n*(n + 1)/2))
    H2 = H.copy()
    entry = 0
    for i in range(len(H)):
        for j in range(i, len(H)):
            if i == j:
                H2[i, j] += step_len
            else:
                H2[i, j] += step_len/np.sqrt(2)
                H2[j, i] += step_len/np.sqrt(2)

            temp_L = la.cholesky(la.inv(H2))
            new_obj = objective(temp_L)

            if i == j:
                H2[i, j] -= step_len
            if i != j:
                H2[i, j] -= step_len/np.sqrt(2)
                H2[j, i] -= step_len/np.sqrt(2)

            fd_grad[entry] = (new_obj - f_xk)/step_len
            entry += 1

    return fd_grad


def Hessian(X, L):
    warnings.warn(("Calculating entire Hessian is slow and typically "
                   "not necessary. Newton's method uses projected "
                   "Hessian instead."), RuntimeWarning)
    v = sla.solve_triangular(L, X, lower=True, check_finite=False)
    ret = -np.dot(v.T, v)**2
    return ret


def Hessian_primal(H, upper_inds=None, diag_inds=None):
    Hinv = la.inv(H)

    n = H.shape[0]
    r = int(n*(n + 1)/2)
    Hess = np.zeros((r, r))
    entry = 0
    for i in range(n):
        for j in range(i, n):
            hess_row_mat = np.outer(Hinv[:, i], Hinv[j])
            hess_row = upper_triangle_to_vec_scaled(hess_row_mat,
                                                    upper_inds=upper_inds,
                                                    diag_inds=diag_inds)
            Hess[entry] = hess_row
            if i != j:
                Hess[entry] *= np.sqrt(2)
            entry += 1

    return Hess


def Hessian_primal_fd(H, upper_inds=None, diag_inds=None):

    grad = gradient_primal(H, upper_inds=upper_inds, diag_inds=diag_inds)

    n = H.shape[0]
    r = len(grad)
    Hess = np.zeros((r, r))
    H2 = H.copy()
    entry = 0
    for i in range(n):
        for j in range(i, n):
            eps = 1e-8 + 1e-8*H2[i, j]
            if i == j:
                H2[i, j] += eps
            else:
                H2[i, j] += eps/np.sqrt(2)
                H2[j, i] += eps/np.sqrt(2)

            grad_new = gradient_primal(H2, upper_inds=upper_inds,
                                       diag_inds=diag_inds)
            grad_change = (grad_new - grad)

            if i == j:
                H2[i, j] -= eps
            else:
                H2[i, j] -= eps/np.sqrt(2)
                H2[j, i] -= eps/np.sqrt(2)

            Hess[entry] = grad_change/eps
            entry += 1
    return Hess


def projected_Hessian(X, L, Z):
    '''
    Compute projected Hessian (Z^T Hess Z) without first computing
    full Hessian.
    '''
    zTxT = Z.Tdot(X.T)
    zhatTxT = Z.hatTdot(X.T)

    solved_zT = solve_cholesky(L, zTxT.T)
    solved_zhatT = solve_cholesky(L, zhatTxT.T)

    A = zTxT.dot(solved_zT)
    B = zhatTxT.dot(solved_zT)
    C = zTxT.dot(solved_zhatT)
    D = zhatTxT.dot(solved_zhatT)

    P = ((B - A)/2)[0]
    Phat = ((D - C)/2)[0]
    N = -(((A + B) - (C + D))/4)[:, 0]
    M = A + N[:, np.newaxis] + P[np.newaxis, :]

    return -(M**2 - (N[:, np.newaxis])**2 - (P*Phat)[np.newaxis, :])


def do_cuda_matmat(A, B, C, m, n, k, handle):
    transa = 'n'
    transb = 'n'
    alpha = 1
    lda = m
    ldb = k
    beta = 0
    ldc = m
    skcuda.cublas.cublasDgemm(handle, transa, transb, m, n, k,
                              alpha, A, lda, B, ldb, beta, C, ldc)


def projected_Hessian_small(constraints, L, Z, gpu_args=None):
    zTxT = constraints.cache.get_zTxT(Z)
    zhatTxT = constraints.cache.get_zhatTxT(Z)

    solved_zT = solve_cholesky(L, zTxT.T)
    solved_zhatT = solve_cholesky(L, zhatTxT.T)
    if DO_CUDA and (gpu_args is not None):
        handle, _, _, _ = gpu_args

        if not solved_zT.flags['F_CONTIGUOUS']:
            solved_zT = np.asfortranarray(solved_zT)
        if not solved_zhatT.flags['F_CONTIGUOUS']:
            solved_zhatT = np.asfortranarray(solved_zhatT)
        if not zTxT.flags['F_CONTIGUOUS']:
            zTxT = np.asfortranarray(zTxT)
        if not zhatTxT.flags['F_CONTIGUOUS']:
            zhatTxT = np.asfortranarray(zhatTxT)

        solved_zT_gpu = gpuarray.to_gpu(solved_zT)
        solved_zhatT_gpu = gpuarray.to_gpu(solved_zhatT)
        zTxT_gpu = gpuarray.to_gpu(zTxT)
        zhatTxT_gpu = gpuarray.to_gpu(zhatTxT)

        m, k = zTxT.shape
        n = solved_zT.shape[1]
        A = np.empty((m, n), order='F')
        B = np.empty((m, n), order='F')
        C = np.empty((m, n), order='F')
        D = np.empty((m, n), order='F')
        res_gpu = gpuarray.empty((m, n), dtype=np.double, order='F')
        do_cuda_matmat(zTxT_gpu.gpudata, solved_zT_gpu.gpudata,
                       res_gpu.gpudata, m, n, k, handle)
        A = res_gpu.get()
        do_cuda_matmat(zhatTxT_gpu.gpudata, solved_zT_gpu.gpudata,
                       res_gpu.gpudata, m, n, k, handle)
        B = res_gpu.get()
        do_cuda_matmat(zTxT_gpu.gpudata, solved_zhatT_gpu.gpudata,
                       res_gpu.gpudata, m, n, k, handle)
        C = res_gpu.get()
        do_cuda_matmat(zhatTxT_gpu.gpudata, solved_zhatT_gpu.gpudata,
                       res_gpu.gpudata, m, n, k, handle)
        D = res_gpu.get()
    else:
        A = zTxT.dot(solved_zT)
        B = zhatTxT.dot(solved_zT)
        C = zTxT.dot(solved_zhatT)
        D = zhatTxT.dot(solved_zhatT)

    P = ((B - A)/2)[0]
    Phat = ((D - C)/2)[0]
    N = -(((A + B) - (C + D))/4)[:, 0]
    M = A + N[:, np.newaxis] + P[np.newaxis, :]

    return -(M**2 - (N[:, np.newaxis])**2 - (P*Phat)[np.newaxis, :])


def _updated_gradient(g_xk, X, L, factor, idx, lmda):
    xHat = solve_cholesky(L, X[:, idx], factor)
    return (1 + lmda)*(g_xk - lmda/(1 + lmda*g_xk[idx])*(xHat.dot(X)**2))

# }}}


# {{{ Cholesky routines

def cholesky_small(constraints, u, less_stable=False):
    X = constraints.cache.get_X()

    if less_stable:
        L = sla.cholesky((X*u[np.newaxis, :]).dot(X.T),
                         lower=True, check_finite=False)
    else:
        if np.min(u) < -1e-10:
            raise ValueError('Variable u cannot have negative values')
        else:
            # Ignore small negative values
            u = np.maximum(u, 0)

        L = la.qr(X.T*np.sqrt(u[:, np.newaxis]))[1].T
        L = L*np.sign(np.diag(L))
    return L


def cholesky(X, u, constraints, less_stable=False):

    inds = ~constraints.active

    if less_stable:
        # Special option for FD newton, which sometimes goes out of bounds
        L = sla.cholesky((X[:, inds]*u[np.newaxis, inds]).dot(X[:, inds].T),
                         lower=True, check_finite=False)
    else:
        if np.min(u) < -1e-10:
            raise ValueError('Variable u cannot have negative values')
        else:
            # Ignore small negative values
            u = np.maximum(u, 0)

        L = la.qr(X[:, inds].T*np.sqrt(u[inds, np.newaxis]))[1].T
        L = L*np.sign(np.diag(L))  # enforce positive diagonal
    return L


def cholesky_primal(X, u):
    inds = (u > 0)

    if np.min(u) < -1e-10:
        raise ValueError('Variable u cannot have negative values')

    L = la.qr(X[:, inds].T*np.sqrt(u[inds, np.newaxis]))[1].T
    L = L*np.sign(np.diag(L))  # enforce positive diagonal
    return L


def solve_cholesky(L, x, factor=1.0):
    return (factor*
            sla.solve_triangular(L.T,
                                 sla.solve_triangular(L, x, lower=True,
                                                      check_finite=False),
                                 check_finite=False))


def _updated_cholesky(L, factor, x, lmda):
    mat = L.copy()
    v = np.sqrt(abs(lmda)*factor)*x
    if np.sign(lmda) > 0:
        chol.cholupdate(mat.T, v.copy())
    else:
        chol.choldowndate(mat.T, v.copy())
    factor = factor*(1 + lmda)
    return mat, factor

# }}}


# {{{ Orthogonalization

def orthogonalize(basis, vec):
    '''
    Orthogonalize vector against basis with modified GS
    '''
    vec = vec.copy()

    for k in range(basis.shape[1]):
        r = basis[:, k].dot(vec)
        vec = vec - r*basis[:, k]/la.norm(basis[:, k])**2

    return vec


def m_orthogonalize(basis, vec, M):
    '''
    Orthogonalize vector against basis with modified GS using M inner product
    '''
    vec = vec.copy()

    for k in range(basis.shape[1]):
        r = basis[:, k].dot(M.dot(vec))
        m_norm_squared = basis[:, k].dot(M.dot(basis[:, k]))
        vec = vec - r*basis[:, k]/m_norm_squared

    return vec

# }}}


# {{{ Line search

def trust_region(aBar, p_k, x_k, HessL, f_xk, constraints, bfgs_ortho, delta):
    '''
    Using trust region rather than line search to compute step length seems to
    be a promising avenue for speeding-up non-Newton methods because it may
    require fewer evaluations of the (expensive) objective function, but
    algorithm currently does not converge with trust-region search as
    implemented here.

    Trust-region search is not referenced in the paper and is included here
    only in case there is future interest in modifying (correcting) the
    implementation to possibly produce a speed-up to non-Newton methods.
    It is currently broken and should not be used.
    '''
    inds = ~constraints.active
    x = x_k[inds]
    p_orig = p_k[inds].copy()
    norm_p_orig = la.norm(p_orig, np.inf)
    Zshape1 = constraints.lennw - 1

    if aBar < 1:
        p_orig *= aBar

    delta_max = max(min(aBar, 10), 1)
    delta = min(delta_max, delta)
    eta = 0.2

    count = 0
    x_new = None
    while x_new is None and count < 100:

        if la.norm(p_orig) > delta:
            p = p_orig*delta/la.norm(p_orig)
        else:
            p = p_orig

        rho_L = cholesky_small(constraints, x + p)
        rho_val = objective(rho_L)
        rho_grad = projected_gradient_small(constraints, rho_L, bfgs_ortho)

        # px is same as Z.project_z(p) if p were full size
        pz = np.delete(p, constraints.basic_ind)
        pz = pz[constraints.order] - p[constraints.basic_ind]
        pz = pz - pz.sum()/(1 + Zshape1)
        model_obj = f_xk + rho_grad@pz + 0.5*pz@HessL@(HessL.T@pz)

        model_diff = model_obj - f_xk
        if model_diff == 0:
            rho = 1
        else:
            rho = (rho_val - f_xk)/model_diff

        if rho < 0.25:
            delta = 0.25*la.norm(p)
        else:
            if rho > 0.75 and abs(la.norm(p) - delta) < 1e-12:
                delta = min(2*delta, delta_max)

        if rho > eta:
            x_new = x + p
            new_L = rho_L
            new_val = rho_val
            a_k = la.norm(p, np.inf)/norm_p_orig

        count += 1

    if x_new is None:
        raise Exception

    new_x = np.zeros_like(x_k)
    new_x[inds] = x_new
    return new_x, new_val, new_L, delta, a_k


def line_search(a, b, p_k, x_k, constraints, options):
    '''
    Use line search to find the step length along p_k, bounded
    by a and b, that minimizes the objective function.

    Parameters
    ----------
    a, b : integers
        Lower and upper bounds, respectively, of step length
    p_k : numpy array
        Direction of step
    x_k : numpy array
        Current iterate
    X : 2-d numpy array
        Data points from MVEE problem
    constraints: Constraints
        Active constraints
    options: dict
        Options for solver, usually including tolerance and maximum number
        of iterations

    Returns
    -------
    a_k : integer
        Approximately optimal step length
    max_val : integer
        Objective function value at x_k + a_k*p_k
    max_L : 2d numpy array
        Matrix L at x_k + a_k*p_k
    '''
    inds = ~constraints.active
    x_k_small = x_k[inds]
    p_k_small = p_k[inds]


    def myobj(a_k):
        L = cholesky_small(constraints, x_k_small + a_k*p_k_small)
        return -objective(L)


    a_k = minimize_scalar(myobj, bounds=[a, b], method='Bounded',
                          options=options).x

    max_L = cholesky_small(constraints, x_k_small + a_k*p_k_small)
    max_val = objective(max_L)

    return a_k, max_val, max_L


def simple_line_search_primal(H, p_k, max_step, options, times_failed=0):
    '''
    Use line search to find the optimal step length along p_k

    Parameters
    ----------
    H : 2-d numpy array
        Current iterate
    p_k : numpy array
        Direction of step
    max_step : scalar
        Maximum step that can be taken without violating constraint
    options: dict
        Options for solver, usually including tolerance and maximum number
        of iterations
    times_failed: int
        Number of times line search has converged to non-positive-definite
        matrix

    Returns
    -------
    a_k : integer
        Approximately optimal step length
    '''


    def myobj(a_k):
        a_k /= (10**times_failed)
        H_k = H + a_k*p_k
        eigvs = la.eigvalsh(H_k)
        if (eigvs > 0).all():
            obj = -np.sum(np.log(eigvs))
        else:
            obj = 1e8
        return obj


    a_k = minimize_scalar(myobj, bounds=[0, max_step], method='Bounded',
                          options=options)

    if a_k.fun == 1e8:
        # line search converged to non-positive-definite matrix
        times_failed += 1
        a_k = simple_line_search_primal(H, p_k, max_step, options,
                                        times_failed=times_failed)
        a_k = a_k/(10**times_failed)
    else:
        a_k = a_k.x

    return a_k

# }}}


# {{{ Initialization routines

def initialize_ky(X, n_desired=None):
    n, m = X.shape

    if n_desired is not None and n_desired != n:
        raise Exception('KY init with more than n points is not implemented')

    Q = np.zeros((n, n))
    us = []
    i = 0
    c = orthogonalize(Q[:, :i], np.random.rand(n))
    us.append(np.argmax(np.abs(c.dot(X))))
    Q[:, i] = orthogonalize(Q[:, :i], X[:, us[-1]])
    # print("us", us)

    while len(us) < n:
        # print("i", i)
        # for i in range(n):
        rand_n = np.random.rand(n)
        # print(rand_n)
        # c = orthogonalize(Q[:, :i], np.random.rand(n))
        c = orthogonalize(Q[:, :i], rand_n)
        # print("c", c)
        # print(np.abs(c.dot(X)))
        # print(X)
        arg_max = np.argmax(np.abs(c.dot(X)))
        # print("arg_max", arg_max)
        # print("us", us)
        while arg_max in us:
            arg_max = np.random.randint(m)
            # arg_max += 1
        us.append(arg_max)
        i += 1
        Q[:, i] = orthogonalize(Q[:, :i], X[:, us[-1]])
    

    # print("us", us)
    u = np.zeros(m)
    u[us] = 1.0/n
    return u


def initialize_sci(X, n_desired=None):
    n, m = X.shape

    if n_desired is None:
        n_desired = int(n**1.5)

    M = np.cov(X)
    center = np.mean(X, axis=1)
    X_center = (X - center[:, np.newaxis])
    dists = (X_center*(M@X_center)).sum(axis=0)

    inds = np.argpartition(dists, m - n_desired)[-n_desired:]
    u = np.zeros(m)
    u[inds] = 1.0/n_desired

    return u


def initialize_sci_ortho(X):
    n, m = X.shape

    M = np.cov(X)
    center = np.mean(X, axis=1)
    X_center = (X - center[:, np.newaxis])

    Q = np.zeros((n, n))
    us = []
    for i in range(n):
        c = m_orthogonalize(Q[:, :i], np.random.rand(n), M)
        us.append(np.argmax(np.abs(c.dot(M).dot(X_center))))
        Q[:, i] = m_orthogonalize(Q[:, :i], X_center[:, us[-1]], M)

    # Ensure that there are at least n nonzeros
    while len(set(us)) < n:
        new_ind = np.random.randint(m)
        if new_ind not in us:
            us.append(new_ind)

    u = np.zeros(m)
    u[us] = 1.0/n

    return u


def initialize_eig_ky(X, upproject=False):
    from stats import cardoso_kurtosis_matrix

    n, m = X.shape

    if upproject:
        small_n = n - 1
        X = X[:-1]
    else:
        small_n = n

    mat = cardoso_kurtosis_matrix(X)
    _, vecs = la.eigh(mat)

    Q = np.zeros((small_n, small_n))
    us = []
    for i in range(small_n):
        c = orthogonalize(Q[:, :i], vecs[i])
        us.append(np.argmax(np.abs(c.dot(X))))
        Q[:, i] = orthogonalize(Q[:, :i], X[:, us[-1]])

    u = np.zeros(m)
    u[us] = 1.0/n

    if upproject:
        u[0] = 1.0/n

    return u


def initialize_qr(X, n_desired=None, recursive=True, use_R=True,
                  centered=False, just_inds=False):
    if n_desired is None:
        n_desired = X.shape[0]
    if centered:
        X = X.copy() - X.mean(axis=1)[:, np.newaxis]
    if use_R:
        return initialize_qr_with_R(X, n_desired, recursive, just_inds)
    else:
        return initialize_qr_with_X(X, n_desired, recursive, just_inds)


def initialize_qr_with_R(X, n_desired, recursive, just_inds):
    '''
    When performing more than one QR because n_desired > n,
    use the updated matrix R from the QR factorization instead of the original
    matrix X.

    Note: this will give the exact same result as initialize_qr_with_X
    '''
    n, m = X.shape
    all_inds = np.arange(m, dtype=int)

    if just_inds:
        ret_inds = []

    R = X
    u = np.zeros(m)
    total = 0
    first = True
    while first or (recursive and total < n_desired):
        _, R, P = sla.qr(R, mode='economic', pivoting=True, check_finite=False)
        all_inds[total:] = all_inds[total:][P]
        new_nonzeros = all_inds[total:][:min(n_desired - total, n)]
        if just_inds:
            ret_inds.extend(new_nonzeros)
        R = R[:, n:]
        u[new_nonzeros] = 1.0
        total += n
        first = False

    if just_inds:
        return ret_inds

    u /= u.sum()
    return u


def initialize_qr_with_X(X, n_desired, recursive, just_inds):
    '''
    When performing more than one QR because n_desired > n,
    use the original matrix X

    Note: this will give the exact same result as initialize_qr_with_R
    '''
    n, m = X.shape
    all_inds = np.arange(m, dtype=int)
    inds = np.ones(m, dtype=bool)

    if just_inds:
        ret_inds = []

    u = np.zeros(m)
    total = 0
    first = True
    while first or (recursive and total < n_desired):
        _, _, P = sla.qr(X[:, all_inds[inds]], mode='economic', pivoting=True,
                         check_finite=False)
        new_nonzeros = all_inds[inds][P[:min(n_desired - total, n)]]
        if just_inds:
            ret_inds.extend(new_nonzeros)

        u[new_nonzeros] = 1.0
        inds[new_nonzeros] = 0
        total += n
        first = False

    if just_inds:
        return ret_inds

    u /= u.sum()
    return u


def initialize_values_qr_random(X, n_desired=None):
    n, m = X.shape

    if n_desired is None:
        n_desired = n

    u = initialize_qr_with_X(X, n_desired, True)

    u[u > 0] = np.random.rand(n_desired)
    u /= u.sum()

    div_count = 0
    while div_count < 5 and u.max() > 1/n + 1e-6:
        u[u > 1/n] = 1/n
        u /= u.sum()
        div_count += 1


    return u


def initialize_values_qr(X, n_desired=None):
    '''
    Use QR to get not just points, but values too.
    '''
    n, m = X.shape

    if n_desired is None:
        n_desired = n

    all_inds = np.arange(m, dtype=int)
    inds = np.ones(m, dtype=bool)

    u = np.zeros(m)
    total = 0
    iter_count = 1
    while total < n_desired:
        _, R, P = sla.qr(X[:, all_inds[inds]], mode='economic', pivoting=True,
                         check_finite=False)
        new_nonzeros = all_inds[inds][P[:min(n_desired - total, n)]]
        for i, ind in enumerate(new_nonzeros):
            u[ind] = abs(R[i, i])/iter_count
        inds[new_nonzeros] = 0
        total += n
        iter_count += 1

    u /= u.sum()
    div_count = 0
    while div_count < 5 and u.max() > 1/n + 1e-6:
        u[u > 1/n] = 1/n
        u /= u.sum()
        div_count += 1

    return u


def initialize_qr_n(X, n_desired=None, subdim_size=None, n_repeats=None):
    '''
    Split the data by dimension.
    Initialize in lower dimension.
    Add nonzero indices from lower dimension to initial guess.
    '''
    n, m = X.shape

    if n_desired is None:
        n_desired = n

    if subdim_size is None:
        subdim_size = int(n/2)

    if n_repeats is None:
        n_repeats = int((n/subdim_size)**2/2)

    inds = np.zeros(m)
    for i in range(n_repeats):
        shuffled = np.random.permutation(n).astype(int)
        ui = initialize_qr(X[shuffled[:subdim_size], :])
        inds[ui > 1e-8] += 1

    max_inds = np.argpartition(inds, -n_desired)[-n_desired:]

    u = np.zeros(m)
    u[max_inds] = 1/n_desired

    return u


def initialize_qr_n_all(X):
    '''
    Same idea as initialize_qr_n(), but split into subsets so that elements
    are not missed due to taking independent random samples.

    This routine is not necessarily meant to be used. It is left around as
    example code that could be combined with initialize_qr_n() and used
    as an option with a flag (e.g. split_subsets=True). Or, with a few changes
    to make it more flexible, this could potentially be called as a subroutine
    of initialize_qr_n() when a certain flag is set.
    '''
    n, m = X.shape
    half_n = int(n/2)
    shuffled = np.random.permutation(n).astype(int)
    ind_splits = np.zeros((2, n), dtype=bool)
    ind_splits[0][shuffled[:half_n]] = 1
    ind_splits[1][shuffled[half_n:]] = 1
    inds = np.zeros(m)

    for i in range(ind_splits.shape[0]):
        ui = initialize_qr(X[ind_splits[i], :])
        inds[ui > 1e-8] += 1

    n_desired = n
    max_inds = np.argpartition(inds, -n_desired)[-n_desired:]

    u = np.zeros(m)
    u[max_inds] = 1/n_desired
    u /= u.sum()

    return u


def initialize_ky_n(X):
    n, m = X.shape
    half_n = int(n/2)
    shuffled = np.random.permutation(n).astype(int)
    ind_splits = np.zeros((2, n), dtype=bool)
    ind_splits[0][shuffled[:half_n]] = 1
    ind_splits[1][shuffled[half_n:]] = 1
    u = np.zeros(m)

    for i in range(ind_splits.shape[0]):
        ui = initialize_ky(X[ind_splits[i], :])
        u[ui > 1e-8] = 1

    u /= u.sum()

    return u


def initialize_ellipsoids_n(X, n_desired=None, subdim_size=None,
                            n_repeats=None, max_iter=10):
    '''
    Split the data by dimension.
    Solve in lower dimension.
    Add nonzero indices from lower dimension to initial guess.
    '''
    n, m = X.shape

    if n_desired is None:
        n_desired = n

    if subdim_size is None:
        subdim_size = int(n/2)

    if n_repeats is None:
        n_repeats = int((n/subdim_size)**2/2)

    inds = np.zeros(m)
    for i in range(n_repeats):
        shuffled = np.random.permutation(n).astype(int)
        ui = mvee2(X[shuffled[:subdim_size], :], method='newton',
                   full_output=True, verbose=False, drop_every=np.inf,
                   initialize='qr', max_iter=max_iter, silent=True)['u']

        inds[ui > 1e-8] += 1

    max_inds = np.argpartition(inds, -n_desired)[-n_desired:]

    u = np.zeros(m)
    u[max_inds] = 1/n_desired

    return u


def initialize_ellipsoids_n_all(X):
    '''
    Same idea as initialize_ellipsoids_n(), but split into subsets so that
    elements are not missed due to taking independent random samples.

    See documentation for initialize_qr_n_all() for more details.
    '''
    n, m = X.shape
    half_n = int(n/2)
    shuffled = np.random.permutation(n).astype(int)
    ind_splits = np.zeros((2, n), dtype=bool)
    ind_splits[0][shuffled[:half_n]] = 1
    ind_splits[1][shuffled[half_n:]] = 1
    inds = np.zeros(m)

    for i in range(ind_splits.shape[0]):
        ui = mvee2(X[ind_splits[i], :], method='newton', full_output=True,
                   verbose=False, drop_every=np.inf, initialize='qr',
                   max_iter=10, silent=True)['u']

        inds[ui > 1e-8] += 1

    n_desired = 2*n
    max_inds = np.argpartition(inds, -n_desired)[-n_desired:]

    u = np.zeros(m)
    u[max_inds] = 1/n_desired

    return u


def initialize_ellipsoids(X):
    track_valid = False
    n, m = X.shape

    weights = np.ones(m)
    n_inds = n**2
    n_desired = 2*n
    n_iter = 100
    if track_valid:
        plotweights = np.zeros((n_iter, m), dtype=int)
    for i in range(n_iter):
        if False:
            # Works better at moving all points in the correct direction
            inds = np.random.choice(m, size=n_inds, replace=False)
        else:
            # Works better at narrowing down best few points
            inds = np.random.choice(m, size=n_inds, replace=False,
                                    p=weights/weights.sum())
        ui = mvee2(X[:, inds], method='newton', full_output=True,
                      verbose=False, drop_every=np.inf, initialize='ky',
                      max_iter=20, silent=True)['u']

        improved = np.zeros(m, dtype=bool)
        improved[inds[ui > 1e-5]] = 1
        weights[improved] = weights[improved]*2

        if track_valid:
            plotweights[i] = np.argsort(weights)[::-1]

    if track_valid:
        np.save('data/weightsarr', weights)
        np.save('data/allweightsarr', plotweights)
    u = np.zeros(m)
    max_inds = np.argpartition(weights, -n_desired)[-n_desired:]
    u[max_inds] = 1.0/n_desired

    return u


def initialize_extrema(X):
    n, m = X.shape

    X = X.copy() - np.average(X, axis=1)[:, np.newaxis]

    maxima = X.argmax(axis=1)
    minima = X.argmin(axis=1)

    extrema = set(maxima).union(minima)

    while len(extrema) < n:
        extrema = extrema.union(np.random.randint(0, m, n - len(extrema)))

    extrema = list(extrema)

    # Procedure usually results in more than n nonzeros. However, if we
    # uncomment this line to drop the number to n, the performance
    # of the method as a whole is worse.
    #extrema = extrema[:n]

    u = np.zeros(m)
    u[extrema] = 1.0/len(extrema)

    return u


def initialize_norm(X, p=2, n_desired=None, just_inds=False):
    n, m = X.shape

    if n_desired is None:
        n_desired = n

    X = X.copy() - np.average(X, axis=1)[:, np.newaxis]

    if m > n_desired:
        maxima = np.argpartition(la.norm(X, ord=p, axis=0), m - n_desired)
        maxima = list(maxima[-n_desired:])
    else:
        maxima = list(range(m))

    if just_inds:
        order = np.argsort(la.norm(X, ord=p, axis=0))[::-1]
        return order[:n_desired]

    u = np.zeros(m)
    u[maxima] = 1.0/len(maxima)

    return u


def initialize_norms(X, ps=[1, 2, np.inf], n_desired=None):
    n, m = X.shape

    if n_desired is None:
        n_desired = len(ps)*n

    u = np.zeros(m)
    for p in ps:
        u += initialize_norm(X, ord=p, n_desired=int(n_desired/len(ps)))

    nonzeros = (u > 0)
    u[nonzeros] = 1.0/nonzeros.sum()

    return u


def initialize_random(X, n_desired=None):
    n, m = X.shape

    if n_desired is None:
        n_desired = n

    x_k = np.zeros(m)
    rand_inds = np.random.choice(m, n_desired, replace=False)
    x_k[rand_inds] = 1.0/len(rand_inds)

    return x_k

# }}}


# {{{ Calculation of next step

def _newton_step(Z, g_xk, X, L):
    '''
    Compute Newton step on feasible subspace.
    '''
    g_z = Z.Tdot(g_xk)
    Hess_z = projected_Hessian(X, L, Z)
    return Z.dot(la.solve(Hess_z, -g_z))


def _newton_step_small(Z, g_z, constraints, L, gpu_args=None):
    '''
    Compute Newton step on feasible subspace.
    '''
    Hess_z = projected_Hessian_small(constraints, L, Z, gpu_args=gpu_args)
    if DO_CUDA and (gpu_args is not None):
        _, _, A_gpu, _ = gpu_args

        r, r = Hess_z.shape
        if not g_z.flags['F_CONTIGUOUS']:
            g_z = np.asfortranarray(g_z)
        if not Hess_z.flags['F_CONTIGUOUS']:
            Hess_z = np.asfortranarray(Hess_z)

        trans = 'n'
        n_gpu = r
        m_gpu = r
        nrhs = 1
        cuda.memcpy_htod(A_gpu, Hess_z)
        lda = r
        B_gpu = gpuarray.to_gpu(-g_z)
        ldb = r
        devInfo = 0

        handle = skcuda.cusolver.cusolverDnCreate()
        bufsize = skcuda.cusolver.cusolverDnDgetrf_bufferSize(handle, m_gpu,
                                             n_gpu, A_gpu, lda)
        workspace = gpuarray.empty(bufsize, np.double)
        devIpiv = gpuarray.empty(m_gpu, np.int)

        skcuda.cusolver.cusolverDnDgetrf(handle, m_gpu, n_gpu,
                                         A_gpu, lda,
                                         int(workspace.gpudata),
                                         int(devIpiv.gpudata),
                                         devInfo)
        skcuda.cusolver.cusolverDnDgetrs(handle, trans, n_gpu, nrhs,
                                         A_gpu, lda, int(devIpiv.gpudata),
                                         int(B_gpu.gpudata), ldb,
                                         devInfo)
        skcuda.cusolver.cusolverDnDestroy(handle)

        sol = B_gpu.get()

        return Z.dot(sol)
    else:
        try:
            step_z = la.solve(Hess_z, -g_z)
        except la.LinAlgError:
            step_z = la.lstsq(Hess_z, -g_z, rcond=None)[0]
        return Z.dot(step_z)


def _truncated_newton_step(Z, g_z, X, xk, L, **kwargs):
    '''
    Compute approximate Newton step.
    '''
    Hess_z = projected_Hessian(X, L, Z)
    x, _ = ssla.gmres(Hess_z, -g_z, **kwargs)

    return Z.dot(x)


def _truncated_newton_step_small(Z, g_z, constraints, L, **kwargs):
    '''
    Compute approximate Newton step.
    '''
    Hess_z = projected_Hessian_small(constraints, L, Z, gpu_args=gpu_args)
    x, _ = ssla.gmres(Hess_z, -g_z, **kwargs)

    return Z.dot(x)


def _truncated_newton_fd_step(Z, g_z, x_k, constraints, L, **kwargs):
    '''
    Compute approximate Newton step. Use finite differences to approximate
    Hessian-vector product and solve system using GMRES.
    '''
    inds = ~constraints.active
    x_k = x_k[inds]

    relative_step_norm = 1e-8
    current_iter_norm = la.norm(x_k, np.inf)
    desired_norm_step = relative_step_norm*current_iter_norm


    def matvec_hessian(vec):
        step = Z.dot(vec)
        step_small = step[inds]
        norm_step = la.norm(step_small, np.inf)

        if norm_step < 1e-14:
            return np.zeros(len(g_z))

        h = desired_norm_step/norm_step

        L = cholesky_small(constraints, x_k + h*step_small, less_stable=True)
        g_z_plus_h = projected_gradient_small(constraints, L, ortho=False)

        return (g_z_plus_h - g_z)/h


    n = Z.shape[1]
    Hess_z = ssla.LinearOperator((n, n), matvec_hessian)
    x, _ = ssla.gmres(Hess_z, -g_z, **kwargs)

    return Z.dot(x)


def _bfgs_step(Z, g_z, HessL):
    '''
    Use Cholesky factorization of approximate projected Hessian from BFGS
    to solve for next step

    Note that we are solving for the gradient, instead of the negative
    gradient, because our HessL is the Cholesky factor of a positive definite
    matrix, but the actual Hessian is a negative definite matrix.
    '''
    y = sla.solve_triangular(HessL, g_z, lower=True, check_finite=False)
    x = sla.solve_triangular(HessL.T, y, lower=False, check_finite=False)

    return Z.dot(x), x


def _lbfgs_step(Z, g_z, HessInv, sks, yks, rhos, first):
    '''
    Use L-BFGS approximation to projected Hessian to solve for next step

    Note that we are solving for the gradient, instead of the negative
    gradient, because our HessInv is the inverse of a positive definite
    matrix, but the actual Hessian is a negative definite matrix.
    '''
    s = _lbfgs_matmult(HessInv, g_z, sks, yks, rhos, first)

    return Z.dot(s), s


def _primal_newton_step(H, g_xk, Z, upper_inds=None, diag_inds=None):

    Hess = Hessian_primal(H, upper_inds=upper_inds, diag_inds=diag_inds)

    Hess_z = Z.T.dot(Hess).dot(Z)
    g_z = Z.T.dot(g_xk)

    step_z = la.solve(Hess_z, -g_z)
    step = Z.dot(step_z)

    return step


def _sqp_step(Y, Z, R, H, u_k, w_k, upper_inds=None, diag_inds=None):

    Hess = Hessian_primal(H, upper_inds=upper_inds, diag_inds=diag_inds)

    Hess_z = Z.T.dot(Hess).dot(Z)
    rhs = -Z.T.dot(w_k + Hess@Y@u_k)

    v_k = la.solve(Hess_z, rhs)

    p_k = Y@u_k + Z@v_k

    delta_k = la.solve(R, -Y.T@(w_k + Hess@p_k))

    return p_k, delta_k

# }}}


# {{{ General helpers

def upper_triangle_to_vec_scaled(H, upper_inds=None, diag_inds=None):
    '''
    Convert a symmetric matrix into a vector containing its upper triangle in
    row-major order. Scale the off-diagonal entries by sqrt(2).
    '''
    if diag_inds is None:
        diag_inds = np.diag_indices_from(H)
    if upper_inds is None:
        upper_inds = np.triu_indices_from(H)

    vec_H = H*np.sqrt(2)
    vec_H[diag_inds] = H[diag_inds]
    vec_H = vec_H[upper_inds]
    return vec_H.flatten()


def vec_to_upper_triangle_scaled(v, shape, upper_inds=None, diag_inds=None):
    '''
    Convert a vector representing the upper triangle of a symmetric matrix in
    row-major order back into a symmetric matrix. The vector entries are scaled
    by sqrt(2) on the off-diagonal entries of the matrix.

    Undoes the effect of upper_triangle_to_vec_scaled().
    '''
    H = np.zeros(shape)
    if upper_inds is None:
        upper_inds = np.triu_indices_from(H)
    if diag_inds is None:
        diag_inds = np.diag_indices_from(H)

    H[upper_inds] = v/np.sqrt(2)
    H[diag_inds] *= (np.sqrt(2)/2)
    H = H + H.T

    return H


def upper_triangle_to_vec(H):
    '''
    Convert a symmetric matrix into a vector containing its upper triangle in
    row-major order
    '''
    vec_H = H[np.triu_indices_from(H)]
    return vec_H.flatten()


# }}}


# {{{ Measuring problem difficulty with kurtosis

def kurtosis(X, aggregate='mean', do_log=True):
    if aggregate.lower() == 'mean':
        agg_fun = np.mean
    elif aggregate.lower() == 'min':
        agg_fun = np.min
    elif aggregate.lower() == 'max':
        agg_fun = np.max
    elif aggregate.lower() == 'median':
        agg_fun = np.median
    else:
        raise ValueError('%s is not a valid aggregation option' % aggregate)

    bigX = np.hstack([-X, X])
    row_kurs = scipy.stats.kurtosis(bigX, axis=1, fisher=False)

    if do_log:
        row_kurs = np.log10(row_kurs)
        overall_kur = 10**agg_fun(row_kurs)
    else:
        overall_kur = agg_fun(row_kurs)

    return overall_kur

# }}}


# {{{ Choose initialization

def choose_init(X, pr):
    n, m = X.shape
    # print("n", n)

    init_size = pr.init_size
    if init_size is None:
        init_size = n

    if pr.initialize == 'khachiyan':
        x_k = np.ones(m)/m
    elif pr.initialize == 'given':
        x_k = pr.update_data
    elif pr.initialize == 'ellipsoids':
        x_k = initialize_ellipsoids(X)
    elif pr.initialize == 'random':
        x_k = initialize_random(X, n_desired=init_size)
    elif pr.initialize == 'qr':
        x_k = initialize_qr(X, n_desired=init_size)
    elif pr.initialize == 'extrema':
        x_k = initialize_extrema(X, n_desired=init_size)
    elif pr.initialize == 'eig_ky':
        x_k = initialize_eig_ky(X, pr.upproject)
    elif pr.initialize == 'sci':
        x_k = initialize_sci(X, n_desired=init_size)
    elif pr.initialize == '2norm':
        x_k = initialize_norm(X, p=2, n_desired=init_size)
    else:
        if pr.initialize != 'ky':
            warnings.warn(('Did not recognize initialization %s.\n'
                           'Defaulting to KY initialization.'))
        # print("init_size", init_size)
        x_k = initialize_ky(X, n_desired=init_size)
    return x_k

# }}}


# {{{ Default parameters

# defaults = {'initialize': 'ky', 'epsilon': 1e-5, 'max_iter': 1000,
#     'verbose': True, 'update_data': None, 'full_output': False,
#     'init_size': None,
#     'timing': False, 'method': 'newton', 'track_objs': False,
#     'track_ellipses': False, 'track_iters': False, 'track_angles': False,
#     'track_gradients': False, 'track_count': False, 'track_epsilons': False,
#     'track_differences_obj': False, 'track_core_set_size': False,
#     'track_lambdas': True, 'track_ress': True,
#     'track_all_iters': False, 'constraint_add_inds': False,
#     'constraint_rem_inds': False, 'track_stepsizes': False,
#     'drop_every': 100, 'count_inds': False, 'silent': False, 'stop_obj': None,
#     'algorithm': 'wa', 'upproject': True, 'bfgs_method': 'BFGS',
#     'bfgs_restart': 'update', 'bfgs_warm_start': False, 'lbfgs_n_vecs': 10,
#     'hybrid': None, 'Hessian': False, 'converged': True, 'search': 'line',
#     'large_output': False, 'true_solution': None, 'track_errors': False}

defaults = {'initialize': 'ky', 'epsilon': 1e-5, 'max_iter': 5000,
    'verbose': True, 'update_data': None, 'full_output': True,
    'init_size': None,
    'timing': False, 'method': 'newton', 'track_objs': False,
    'track_ellipses': False, 'track_iters': False, 'track_angles': False,
    'track_gradients': False, 'track_count': False, 'track_epsilons': False,
    'track_differences_obj': False, 'track_core_set_size': False,
    'track_lambdas': True, 'track_ress': True,
    'track_all_iters': False, 'constraint_add_inds': False,
    'constraint_rem_inds': False, 'track_stepsizes': False,
    'drop_every': 100, 'count_inds': False, 'silent': False, 'stop_obj': None,
    'algorithm': 'wa', 'upproject': True, 'bfgs_method': 'BFGS',
    'bfgs_restart': 'update', 'bfgs_warm_start': False, 'lbfgs_n_vecs': 10,
    'hybrid': None, 'Hessian': False, 'converged': True, 'search': 'line',
    'large_output': False, 'true_solution': None, 'track_errors': False}

# }}}


# {{{ Calculate MVEE via primal algorithm

def maximize_over_eigvals(X, V, solver=None):
    '''
    Given approximate eigenvectors as columns of V, solve the maximization
    problem

    maximize sum[ln(lmda[i])]
    such that X[:, j] @ V @ diag(lmda) @ V.T @ X[:, j] <= n for all j

    This is a concave maximization problem with linear constraints.
    '''
    import cvxpy as cvx

    if solver is None:
        solver = cvx.CVXOPT

    lmda = cvx.Variable(X.shape[0])
    Z = V.T.dot(X).T**2
    eig_constraints = [lmda >= 0, Z*lmda <= X.shape[0]]
    cvx_objective = cvx.Maximize(sum(cvx.log(lmda)))

    prob = cvx.Problem(cvx_objective, eig_constraints)
    prob.solve(solver=solver)

    if prob.status != cvx.OPTIMAL:
        raise Exception('Could not find optimal solution')

    return prob.value, np.array(lmda.value).flatten()


def mvee_eig(X, epsilon=1e-5, max_iter=1000, verbose=True, full_output=False, use_svd=False, use_qr=False, use_rand=False, track_objs=False, track_ellipses=False):

    n, m = X.shape

    V = np.eye(X.shape[0])

    if track_objs:
        obj, _ = maximize_over_eigvals(X, V)
        objs = [obj]
    elif track_ellipses:
        Vs = []
        lmdas = []

    for i in range(len(V)):
        for j in range(i + 1, len(V)):
            Vi, Vj = V[i].copy(), V[j].copy()

            def obj_after_rotate(angle):
                # Rotate V
                V[i] = np.cos(angle)*Vi + -np.sin(angle)*Vj
                V[j] = np.sin(angle)*Vi + np.cos(angle)*Vj

                obj, lmda = maximize_over_eigvals(X, V)
                return -obj


            res = minimize_scalar(obj_after_rotate,
                                  method='bounded', bounds=(0, np.pi/2))

            # Recompute best V from angle
            angle = res.x
            V[i] = np.cos(angle)*Vi + -np.sin(angle)*Vj
            V[j] = np.sin(angle)*Vi + np.cos(angle)*Vj

            if track_objs:
                obj, _ = maximize_over_eigvals(X, V)
                objs = [obj]
            elif track_ellipses:
                Vs.append(V.copy())
                _, lmda = maximize_over_eigvals(X, V)
                lmdas.append(lmda.copy())

    best_V = V
    _, best_lmda = maximize_over_eigvals(X, best_V)

    if verbose:
        ellipse = best_V.dot(np.diag(best_lmda)).dot(best_V.T)
        test = np.einsum('ij, ij->j', X, ellipse.dot(X))
        print('Feasible: ', (test < (n*(1 + epsilon))).all())

    retvals = []
    if full_output:
        retvals.append(best_V)
        retvals.append(best_lmda)
    if track_objs:
        retvals.append(np.array(objs))
    if track_ellipses:
        retvals.append(Vs)
        retvals.append(lmdas)

    if len(retvals) == 0:
        retvals = best_V.dot(np.diag(best_lmda)).dot(best_V.T)
    elif len(retvals) == 1:
        retvals = retvals[0]
    else:
        retvals = tuple(retvals)

    return retvals

# }}}


# {{{ Calculate MVEE via other primal algorithm

def mvee_primal(X, **kwargs):

    # Read arguments and supply defaults as necessary
    params = copy.deepcopy(defaults)
    params.update(kwargs)
    pr = Namespace(**params)

    epsilon = pr.epsilon

    if pr.method not in ('gradient', 'newton'):
        raise Exception(('You specified method=%s, which does not exist'
                         ' or is not a primal method') %
                        pr.method)

    if pr.upproject:
        X = np.vstack([X, np.ones(X.shape[1])])

    n, m = X.shape

    x_k = choose_init(X, pr)

    # Calculate initial state
    if len(x_k.shape) < 2:
        nz_inds = x_k > 0
        H = (X[:, nz_inds]*x_k[np.newaxis, nz_inds])@X[:, nz_inds].T
    else:
        # an initial matrix, rather than initial vector, was passed in
        H = x_k

    # Compute indices for transforming matrices to vectors
    upper_inds = np.triu_indices_from(H)
    diag_inds = np.diag_indices_from(H)

    # Generate matrix of constraints
    LM_mat = np.zeros((int(n*(n + 1)/2), m))
    for i in range(m):
        outer_prod = np.outer(X[:, i], X[:, i])
        LM_mat[:, i] = upper_triangle_to_vec_scaled(outer_prod,
                                                    upper_inds=upper_inds,
                                                    diag_inds=diag_inds)

    # Make initial state feasible
    bounds = LM_mat.T@upper_triangle_to_vec_scaled(H, upper_inds=upper_inds,
                                                   diag_inds=diag_inds)
    too_big = np.max(bounds)/n
    H = H/too_big
    bounds = bounds/too_big
    f_xk = objective_primal(H)

    # Determine initial active set
    active = np.where(np.abs(bounds - n) < 1e-8)[0]

    # Compute initial gradient
    g_k = gradient_primal(H, upper_inds=upper_inds,
                          diag_inds=diag_inds)

    # Estimate Lagrange multipliers
    try:
        lmda, res = scipy.optimize.nnls(LM_mat[:, active], -g_k)
    except RuntimeError:
        lmda = np.zeros(len(active))
        res = np.inf

    retvals = {}
    if pr.track_objs:
        objs = [objective_primal(H)]
        retvals['objs'] = objs
    if pr.track_ellipses:
        L = la.cholesky(la.inv(H))
        Ls = [L.copy()]
        retvals['Ls'] = Ls
    if pr.track_epsilons:
        epss = []
        retvals['epss'] = epss
    if pr.track_errors:
        errors = []
        retvals['errors'] = errors
        if pr.true_solution is None:
            raise Exception(('Parameter track_errors does not make sense '
                             'without parameter true_solution.'))
        truesol = pr.true_solution
    if pr.track_lambdas:
        lmdas = []
        retvals['lambdas'] = lmdas
    if pr.track_ress:
        ress = []
        retvals['ress'] = ress
    if pr.track_stepsizes:
        stepsizes = []
        retvals['stepsizes'] = stepsizes

    iter_count = 0
    all_indices = np.arange(m)
    constraint_changed = True
    t1 = time.time()
    while iter_count < pr.max_iter:

        iter_count = iter_count + 1

        g_k = gradient_primal(H, upper_inds=upper_inds, diag_inds=diag_inds)

        if pr.track_ress:
            ress.append(res)
        if pr.track_lambdas:
            lmdas.append(lmda)
        if pr.track_errors:
            if pr.upproject:
                test_H = H[:-1, :-1]
            errors.append(np.sum(np.abs((test_H - pr.true_solution)/
                                        (pr.true_solution + 1))))

        # basis for feasible steps
        if constraint_changed:
            Q, R = la.qr(LM_mat[:, active], mode='complete')
            R = R[:len(active)]
            Y = Q[:, :len(active)]
            Z = Q[:, len(active):]
            constraint_changed = False

        if pr.method == 'gradient':
            p_k = -Z@Z.T@g_k
        elif pr.method == 'newton':
            u_k = la.solve(R.T, -(bounds[active] - n))
            w_k = g_k + LM_mat[:, active].dot(lmda)
            p_k, delta_k = _sqp_step(Y, Z, R, H, u_k, w_k,
                                     upper_inds=upper_inds,
                                     diag_inds=diag_inds)

        # Compute maximum step length in this direction while taking into
        # account constraints
        bounds_steps = LM_mat.T@p_k
        pos_inds = bounds_steps > 1e-10
        step_lengths = (X.shape[0] - bounds[pos_inds])/bounds_steps[pos_inds]
        try:
            max_step = np.min(step_lengths)
        except:
            max_step = 1e6

        # Compute maximum step length based on LMs
        neg_inds = delta_k < -1e-10
        delta_lengths = -lmda[neg_inds]/delta_k[neg_inds]
        try:
            max_delta = np.min(delta_lengths)
        except:
            max_delta = 1e6

        p_k = vec_to_upper_triangle_scaled(p_k, H.shape,
                                           upper_inds=upper_inds,
                                           diag_inds=diag_inds)

        if pr.method == 'gradient':
            # Line search for best step length
            search_options = {'maxiter': 40}
            if max_step > 1e-15 and la.norm(p_k) > 1e-15:
                a_k = simple_line_search_primal(H, p_k, max_step, search_options)
            else:
                a_k = 0
        elif pr.method == 'newton':
            a_k = max(min(1, max_step, max_delta), 0)

        # Add constraint if we cannot step
        if a_k < 1e-14:
            if max_step < 1e-14:
                bad_ind = all_indices[pos_inds][step_lengths.argmin()]
                active = np.append(active, bad_ind)
                lmda = np.append(lmda, 0)
            if max_delta < 1e-14:
                full_inds = np.arange(len(delta_k))
                bad_ind_loc = full_inds[neg_inds][delta_lengths.argmin()]
                active = np.delete(active, bad_ind_loc)
                lmda = np.delete(lmda, bad_ind_loc)

            constraint_changed = True

            if pr.track_ellipses:
                L = la.cholesky(la.inv(H))
                Ls.append(L.copy())
            if pr.track_objs:
                objs.append(f_xk)
            if pr.track_stepsizes:
                stepsizes.append(0)

            continue

        # Take step
        H = H + a_k*p_k
        lmda = lmda + a_k*delta_k
        f_xk = objective_primal(H)
        bounds = bounds + a_k*bounds_steps

        if pr.track_ellipses:
            L = la.cholesky(la.inv(H))
            Ls.append(L.copy())
        if pr.track_objs:
            objs.append(f_xk)
        if pr.track_stepsizes:
            stepsizes.append(la.norm(a_k*p_k))

        # If close enough to known solution, consider converged
        if pr.stop_obj is not None and f_xk < pr.stop_obj*(1 + pr.epsilon):
            break

    L = None
    retvals['H'] = H
    if pr.verbose or pr.converged:
        if L is None:
            L = la.cholesky(la.inv(H))
        test = np.einsum('ij, ij->j', X, solve_cholesky(L, X))
        feasible = (test < (n*(1 + epsilon))).all()
        if pr.verbose:
            print('Feasible: ', feasible)
        if pr.converged:
            retvals['converged'] = feasible

    if iter_count >= pr.max_iter and not pr.silent:
        print("Maximum number of iterations reached. (%s)" % pr.method)

    if pr.full_output:
        if L is None:
            L = la.cholesky(la.inv(H))
        if pr.upproject:
            # dual gradient, which computes primal bounds
            x_k = gradient(X, la.cholesky(la.inv(H)))

            small_H = solve_cholesky(L, np.eye(len(L)))[:-1, :-1]
            retvals['L'] = la.cholesky(la.inv(small_H))
            retvals['c'] = ((X @ x_k)[:-1])
        else:
            retvals['L'] = L
    if pr.track_count:
        retvals['iter_count'] = iter_count

    if len(retvals) == 0:
        if L is None:
            L = la.cholesky(la.inv(H))
        retvals['mat'] = solve_cholesky(L, np.eye(len(L)))

    return retvals

# }}}


# {{{ Calculate MVEE via active-set method

def mvee2(X, **kwargs):
    '''
    Implementation of Algorithm 2

    Specify initialization with keyword 'initialize' --
    currently-available options can be found in function 'choose_init'.

    Specify method as one of the following:
    ['newton', 'BFGS', 'truncated', 'gradient', 'cg', 'todd', 'truncated_fd',
     'L-BFGS']

    Comments in code indicate which sections match which lines of pseudocode
    from paper introducing this method. Lines are not always in exactly the
    same order as pseudocode because the most clear exposition was not always
    the most convenient or efficient in this Python implementation.
    '''

    # Options to control behavior of (L)BFGS, CG, or trust-region methods
    # These are currently set to values that were found empirically to be
    # most effective and should not be changed without a good reason
    ortho_bfgs = False
    ortho_lbfgs = False
    ortho_cg = False
    tr_delta = 0.25  # for trust region methods

    # Begin parse arguments -----

    for key in kwargs.keys():
        if key not in defaults.keys():
            raise Exception('Received unknown parameter %s' % key)

    # Read arguments and supply defaults as necessary
    params = copy.deepcopy(defaults)
    params.update(kwargs)
    pr = Namespace(**params)

    if pr.track_all_iters:
        pr.track_iters = True

    epsilon = pr.epsilon

    if pr.method not in ('newton', 'BFGS', 'truncated', 'gradient', 'cg', 'todd',
                      'truncated_fd', 'L-BFGS'):
        raise Exception('You specified method=%s, which does not exist' %
                        pr.method)

    # Trust-region search is not currently implemented
    # See trust_region() for details
    if pr.search == 'trust':
        raise Exception('You specified search=trust, but trust-region search '
                        'is not currently implemented.')

    # End parse arguments -----

    # General MVEE can be thought of as centered MVEE in one higher dimension
    # If specified by user, add dimension so that we can find centered
    # ellipsoid
    if pr.upproject:
        X = np.vstack([X, np.ones(X.shape[1])])

    # Todd's coordinate-ascent algorithm (a variant of Frank-Wolfe) does
    # not fall under the general template of constrained algorithms
    # implemented here, so is treated in a separate method
    if pr.method == 'todd':
        return mvee(X, **params)

    n, m = X.shape

    # Note about CUDA -- the CUDA implementation is not highly optimized or
    # thoroughly tested, and is not referenced in the paper describing this
    # algorithm. However, brief testing did not lead to any noticeable errors
    # and produced a good speed-up, so the CUDA implementation is included
    # in case it is of interest to users.
    if DO_CUDA:
        X = np.asfortranarray(X)
        handle = skcuda.cublas.cublasCreate()
    else:
        handle = None

    # Initialize -- line (2) of Algorithm 2
    # Variable 'x_k' (denoting kth iterate of x) is called 'u' in Algorithm 2.
    # At some point, the code may be changed to refer to this as 'u' or 'u_k',
    # either of which would be more clear and consistent.
    x_k = choose_init(X, pr)
    # print(pr.initialize)
    # print(pr.init_size)

    # Initialize working set -- line (5) of Algorithm 2
    # Line (5) is included before lines (3) and (4) in this implementation
    # because our Cholesky routine is more efficient when using the
    # 'constraints' array directly rather than looking for nonzeros in 'u'
    working = list(np.where(x_k < 1e-25)[0])
    constraints = Constraints(working, m + 1, Cache(X))
    Z = NullSpaceMatrix(constraints)
    try:
        fake_working = x_k[constraints.active].max()
    except ValueError:
        fake_working = 0
    all_indices = np.arange(m)
    
    # if len(x_k[~constraints.active]) == 2:
    # print(m)
    # print(x_k)
    # print("fake working", fake_working)
    # print(constraints.active)
    print(x_k[~constraints.active].shape)
    # print(x_k[~constraints.active])
    # Calculate initial state -- line (3) of Algorithm 2
    L = cholesky_small(constraints, x_k[~constraints.active])

    # Calculate projected gradient
    # This is computed here only because it is sometimes of interest to track
    # the initial projected gradient.
    print(L.shape)  # (3, 2)
    g_z = projected_gradient_small(constraints, L)

    if DO_CUDA:
        L_gpu = cuda.mem_alloc(L.nbytes)
        size_double = 8
        X_gpu = cuda.mem_alloc(((X.size + 255)//256)*256*size_double)  # padded for 256 blocksize
        mod = SourceModule("""
            __global__ void square(double* a)
            {
                int idx = threadIdx.x + blockIdx.x*256;
                a[idx] = a[idx]*a[idx];
            }
            """)
        square_gpu = mod.get_function("square")
        gpu_args = (handle, L_gpu, X_gpu, square_gpu)
    else:
        gpu_args = None

    # Starting guesses for BFGS and L-BFGS -- not applicable to other methods
    if pr.method == 'BFGS':
        if pr.bfgs_warm_start:
            HessL = sla.cholesky(-projected_Hessian(X, L, Z),
                                 lower=True, check_finite=False)
        else:
            HessL = np.eye(Z.shape[1])
    elif pr.method == 'L-BFGS':
        HessInv = np.eye(Z.shape[1])
        sks = np.zeros((pr.lbfgs_n_vecs, Z.shape[1]))
        yks = np.zeros((pr.lbfgs_n_vecs, Z.shape[1]))
        rhos = np.zeros(pr.lbfgs_n_vecs)
        lbfgs_first = 0
        lbfgs_iter = 0

    # Initial objective value is not currently used as part of convergence
    # condition or step choice, but is cheap to compute once L is known and
    # is often of interest when monitoring the algorithm's behavior
    f_xk = objective(L)

    # For hybrid methods only, determine how many iterations to continue after
    # constraints stop changing before switching method
    if pr.hybrid:
        constraints_unchanged_count = 0
        if 'constraints_unchanged_limit' not in pr.hybrid:
            pr.hybrid['constraints_unchanged_limit'] = 30
        elif pr.hybrid['constraints_unchanged_limit'] == -1:
            pr.hybrid['constraints_unchanged_limit'] = np.inf

        if pr.hybrid['constraints_unchanged_limit'] < np.inf:
            # keep track of inactive constraints because the number of active
            # constraints can decrease spuriously when drop_every is applied
            new_constraint_count = constraints.lennw

    # Begin tracking info -----------
    # Values in this section are not part of the algorithm and are used for
    # profiling and tracking the algorithm

    retvals = {}
    if pr.track_objs:
        objs = [-objective(L)]
        retvals['objs'] = objs
    if pr.track_ellipses:
        Ls = [L.copy()]
        retvals['Ls'] = Ls
    if pr.track_iters:
        xs = [x_k.copy()]
        retvals['us'] = xs
    if pr.track_gradients:
        gradients = [g_z.copy()]
        retvals['gradients'] = gradients
    if pr.track_angles:
        angles = []
        retvals['angles'] = angles
    if pr.track_epsilons:
        epss = []
        retvals['epss'] = epss
    if pr.constraint_add_inds:
        constraint_add_inds = []
        retvals['constraint_add_inds'] = constraint_add_inds
    if pr.constraint_rem_inds:
        constraint_rem_inds = []
        retvals['constraint_rem_inds'] = constraint_rem_inds
    if pr.track_stepsizes:
        stepsizes = []
        retvals['stepsizes'] = stepsizes
    if pr.track_errors:
        errors = []
        retvals['errors'] = errors
        if pr.true_solution is None:
            raise Exception(('Parameter track_errors does not make sense '
                             'without parameter true_solution.'))
        truesol = pr.true_solution
    if pr.track_core_set_size:
        core_set_sizes = [(~constraints.active).sum()]
        retvals['core_set_sizes'] = core_set_sizes

    orig = set(np.where(x_k > 0)[0])
    added = set([])
    remd = set([])
    acount = 0
    rcount = 0

    # End tracking info -----------

    iter_count = 0
    t1 = time.time()
    while iter_count < pr.max_iter:

        iter_count = iter_count + 1

        if pr.track_core_set_size:
            core_set_sizes.append((~constraints.active).sum())

        if pr.hybrid and (
          iter_count > pr.hybrid['step_count'] or
          constraints_unchanged_count >= pr.hybrid['constraints_unchanged_limit']):
            new_params = copy.deepcopy(params)
            new_params['max_iter'] = pr.max_iter - pr.hybrid['step_count']
            new_params['update_data'] = x_k.copy()
            new_params['initialize'] = 'given'
            new_params['method'] = pr.hybrid['method']
            new_params['hybrid'] = None
            new_retvals = mvee2(X, **new_params)

            for key in retvals.keys():
                retvals[key].extend(new_retvals[key])
            if DO_CUDA:
                skcuda.cublas.cublasDestroy(handle)
                pass
            for key in new_retvals.keys():
                if key not in retvals.keys():
                    retvals[key] = new_retvals[key]
            return retvals

        # Compute gradient -- lines (4) and (20) from Algorithm 2
        # It was slightly more efficient in practice not to compute the
        # gradient at the end of the loop in case the loop terminated and the
        # gradient did not need to be computed the last time
        g_xk = gradient(X, L, gpu_args=gpu_args)

        # Begin check for convergence of entire problem --------
        # Lines (7) - (12) of Algorithm 2
        eps_plus = (np.max(g_xk) - n)/n
        nonzeros = np.where(x_k > 1e-8)[0]
        eps_minus = (n - np.min(g_xk[nonzeros]))/n
        eps_worst = max(eps_plus, eps_minus)
        if pr.track_epsilons:
            epss.append(eps_worst)

        if eps_worst < epsilon:
            break
        # End check for convergence of entire problem --------

        # Begin remove bad constraint if one exists ----------

        # This section is part of line (21) of Algorithm 2. In practice,
        # removing constraints is more efficient to do before trying to step
        # because a bad constraint can force us to take an unnecessarily small
        # step. Adding constraints is done later, after the step is computed.
        active_grad = g_xk[constraints.active]
        try_ind = min(3, len(active_grad))
        if try_ind > 0:
            max_lmdas_i = np.sort(np.argpartition(active_grad,
                                                  -try_ind)[-try_ind:])
        else:
            max_lmdas_i = []

        for max_lmda_i in max_lmdas_i:
            if constraints.lenw > 0:
                try:
                    max_lmda = g_xk[constraints.active][max_lmda_i] - \
                               np.average(g_xk[~constraints.active])
                except ValueError:
                    max_lmda = 0
                    max_lmda_i = None

                norm_g_p = max(abs(np.sum(g_z)), np.max(np.abs(g_z)))
                if max_lmda > norm_g_p:
                    bad_ind = all_indices[constraints.active][max_lmda_i]
                    added.add(bad_ind)
                    acount += 1
                elif fake_working > norm_g_p:
                    max_ind = x_k[constraints.active].argmax()
                    bad_ind = all_indices[constraints.active][max_ind]
                else:
                    bad_ind = -1

                if bad_ind > -1:
                    if pr.method == 'BFGS':
                        old_constraints = copy.deepcopy(constraints)
                        g_zz = projected_gradient_small(constraints, L,
                                                        ortho=ortho_bfgs)

                    Z = constraints.remove_constraint(bad_ind)
                    try:
                        fake_working = x_k[constraints.active].max()
                    except ValueError:
                        fake_working = 0

                    if pr.method == 'BFGS':
                        HessL = _bfgs_remove_constraint(HessL, constraints,
                                                        old_constraints, Z,
                                                        X, x_k, L, g_zz,
                                                        all_indices,
                                                        pr.bfgs_restart,
                                                        ortho_bfgs,
                                                        gpu_args=gpu_args)
                    elif pr.method == 'L-BFGS':
                        lbfgs_first, lbfgs_iter, sks, yks, rhos, HessInv = \
                            _lbfgs_restart_hessian(pr.lbfgs_n_vecs, Z)
                    elif pr.method == 'cg':
                        g_zkm1 = None

                    max_lmdas_i -= 1
                    g_z = projected_gradient_small(constraints, L)

                    if pr.constraint_rem_inds:
                        if not iter_count in constraint_rem_inds:
                            constraint_rem_inds.append(iter_count)

        # End remove bad constraint if one exists ----------

        # If active set spans entire space, we are done
        if Z.shape[1] == 0:
            break

        # Begin calculate direction of next step -------------

        # Various implementations of lines (13) - (15) of Algorithm 2 depending
        # on which method is in use
        if pr.method == 'BFGS':
            g_z = projected_gradient_small(constraints, L, ortho=ortho_bfgs)
            p_k, bfgs_diff = _bfgs_step(Z, g_z, HessL)
        elif pr.method == 'L-BFGS':
            g_z = projected_gradient_small(constraints, L, ortho=ortho_lbfgs)
            p_k, lbfgs_diff = _lbfgs_step(Z, g_z, HessInv, sks, yks, rhos,
                                          lbfgs_first)
        elif pr.method == 'truncated':
            g_z = projected_gradient_small(constraints, L, ortho=False)
            # Note: In scipy, maximum iterations is actually maxiter*restart
            p_k = _truncated_newton_step_small(Z, g_z, constraints, L,
                                               tol=1e-8, maxiter=1, restart=10)
        elif pr.method == 'truncated_fd':
            g_z = projected_gradient_small(constraints, L, ortho=False)
            p_k = _truncated_newton_fd_step(Z, g_z, x_k, constraints, L,
                                            tol=1e-8, maxiter=1, restart=10)
        elif pr.method == 'newton':
            g_z = projected_gradient_small(constraints, L, ortho=False)
            p_k = _newton_step_small(Z, g_z, constraints, L, gpu_args)
        elif pr.method == 'gradient':
            g_z = projected_gradient_small(constraints, L, ortho=True)
            p_k = Z.dot(g_z)
        elif pr.method == 'cg':
            g_z = projected_gradient_small(constraints, L, ortho=ortho_cg)
            try:
                beta = (g_z - g_zkm1).dot(g_z)/la.norm(g_zkm1)**2
            except:
                beta = 0
                g_zkm1 = np.zeros(Z.shape[1])
            p_k = g_z + beta*g_zkm1
            p_k = Z.dot(p_k)

        # Compare angles between Newton and other methods -- for tracking only
        if pr.track_angles:
            step = p_k
            g_z2 = projected_gradient_small(constraints, L, ortho=False)
            step2 = _newton_step_small(Z, g_z2, constraints, L, gpu_args)
            cos_t = (step@step2)/(la.norm(step)*la.norm(step2))
            angles.append(180/np.pi*np.arccos(cos_t))

        last_nw = all_indices[~constraints.active]
        if x_k[last_nw][-1] == 0 and p_k[last_nw[-1]] <= 0:
            # Sometimes truncated fails to find a good step.
            # In this case, fall back to gradient ascent so we can make
            # progress.
            p_k = Z.dot(g_z)

        # End calculate direction of next step -------------

        # Begin maximum step length --------------

        # Determine maximum possible step length until running into a
        # constraint not in the active set
        # Line (16) of Algorithm 2
        neg_inds = np.where(p_k[~constraints.active] < 0)[0]
        neg_not_working = all_indices[~constraints.active][neg_inds]
        if len(neg_inds) == 0:
            aBar = np.inf
        else:
            stops = -x_k[neg_not_working]/p_k[neg_not_working]
            aBar_i = np.argmin(stops)
            aBar = stops[aBar_i]

            # If we cannot step, add the offending index to constraints and
            # ensure that we do not immediately remove it next iteration
            if aBar < 1e-16:
                if pr.method == 'BFGS':
                    old_constraints = copy.deepcopy(constraints)

                add_ind = neg_not_working[aBar_i]
                Z = constraints.add_constraint(add_ind)

                try:
                    fake_working = x_k[constraints.active].max()
                except ValueError:
                    fake_working = 0

                # Updating constraints requires updating information for
                # methods that track approximate Hessian and CG
                if pr.method == 'BFGS':
                    do_swap = False
                    if add_ind == old_constraints.basic:
                        add_ind = constraints.basic
                        do_swap = True

                    bfgs_add_ind = np.sum(old_constraints.superbasic[:add_ind])
                    bfgs_add_order = np.where(old_constraints.order ==
                                              bfgs_add_ind)[0][0]

                    if do_swap and pr.bfgs_restart == 'update':
                        HessL = _bfgs_swap_basic(HessL, bfgs_add_order)
                    HessL = _bfgs_add_constraint(HessL, bfgs_add_order,
                                                 pr.bfgs_restart)
                elif pr.method == 'L-BFGS':
                    lbfgs_first, lbfgs_iter, sks, yks, rhos, HessInv = \
                        _lbfgs_restart_hessian(pr.lbfgs_n_vecs, Z)
                elif pr.method == 'cg':
                    g_zkm1 = None

                # These lines are solely for tracking purposes -----
                if pr.track_all_iters:
                    xs.append(x_k.copy())

                if pr.track_errors:
                    errors.append(la.norm(x_k - truesol))

                if pr.constraint_add_inds:
                    if not iter_count in constraint_add_inds:
                        constraint_add_inds.append(iter_count)

                if pr.track_stepsizes and pr.track_all_iters:
                    stepsizes.append(0)
                # End tracking lines -----

                continue

        # End maximum step length --------------

        # Begin actual step length -------------

        # Line search for best step length
        # Line (17) from Algorithm 2
        # Variable a_k below is equivalent to lambda in pseudocode
        if pr.method in ('newton', 'truncated', 'truncated_fd'):
            a_k = min(aBar, 1)
            inds = ~constraints.active
            max_L = cholesky_small(constraints, x_k[inds] + a_k*p_k[inds])
            max_val = objective(max_L)
        else:
            search_options = {'maxiter': 40, 'xatol': 1e-12}
            if pr.search == 'trust' and pr.method == 'BFGS':
                new_xk, max_val, max_L, tr_delta, a_k = \
                  trust_region(aBar, p_k, x_k, HessL, f_xk, constraints,
                               ortho_bfgs, tr_delta)
            else:
                a_k, max_val, max_L = line_search(0, aBar, p_k,
                                                  x_k, constraints,
                                                  search_options)

        if pr.track_stepsizes:
            stepsizes.append(a_k)

        # End actual step length -------------

        # Before stepping, BFGS and related methods require us to keep track
        # of values that will be used to update approximate curvature
        if pr.method in ('BFGS', 'cg', 'L-BFGS'):
            if pr.method == 'BFGS':
                ortho_all = ortho_bfgs
            elif pr.method == 'L-BFGS':
                ortho_all = ortho_lbfgs
            elif pr.method == 'cg':
                ortho_all = ortho_cg
            g_zkm1 = projected_gradient_small(constraints, L, ortho=ortho_all)

        # Begin perform step and update related values ----------

        # Step to next iterate -- line (18) from Algorithm 2
        old_xk = x_k.copy()
        if pr.search == 'trust' and pr.method == 'BFGS':
            x_k = new_xk
        else:
            x_k = x_k + a_k*p_k

        # In theory, elements of x_k already sum to 1 due to constraints, but
        # to avoid rounding errors as much as possilbe, we force the sum to 1
        x_k /= x_k.sum()

        # Abort if step is exactly same as last iteration
        if len(old_xk) == len(x_k) and (old_xk == x_k).all():
            break

        # Update values that depend on current iterate
        # Line (19) from Algorithm 2, though, in practice, some values are
        # actually computed earlier as part of line search
        L = max_L
        g_z = projected_gradient_small(constraints, L)
        f_xk = max_val

        # End perform step and update related values ----------

        # Update approximate Hessian
        # Example of method-specific code in line (22) of Algorithm 2
        if pr.method == 'BFGS':
            g_zk = projected_gradient_small(constraints, L, ortho=ortho_bfgs)
            HessL = _bfgs_updated_hessian(HessL, bfgs_diff, a_k,
                                          g_zkm1, g_zk, pr.bfgs_method)
        elif pr.method == 'L-BFGS':
            g_zk = projected_gradient_small(constraints, L, ortho=ortho_lbfgs)
            lbfgs_first, HessInv = _lbfgs_updated_hessian(lbfgs_diff, a_k,
                              g_zkm1, g_zk, sks, yks, rhos, HessInv,
                              pr.lbfgs_n_vecs, lbfgs_iter, lbfgs_first)
            lbfgs_iter += 1

        # Begin add constraint ----------

        # Add constraint to working set
        # This section is part of line (21) of Algorithm 2. Constraints are
        # added if it is found that they limited the length of the most recent
        # step. In practice, removing constraints is done earlier, before the
        # step is computed.
        if np.abs(a_k - aBar) < 0.01*aBar:
            all_small = [neg_not_working[aBar_i]]
            for add_ind in all_small:
                if pr.method == 'BFGS':
                    old_constraints = copy.deepcopy(constraints)

                remd.add(add_ind)
                Z = constraints.add_constraint(add_ind)
                if x_k[add_ind] < 1e-8:
                    x_k[add_ind] = 0

                try:
                    fake_working = x_k[constraints.active].max()
                except ValueError:
                    fake_working = 0

                if pr.method == 'BFGS':
                    do_swap = False
                    if add_ind == old_constraints.basic:
                        add_ind = constraints.basic
                        do_swap = True

                    bfgs_add_ind = np.sum(old_constraints.superbasic[:add_ind])
                    bfgs_add_order = np.where(old_constraints.order ==
                                              bfgs_add_ind)[0][0]

                    if do_swap and pr.bfgs_restart == 'update':
                        HessL = _bfgs_swap_basic(HessL, bfgs_add_order)
                    HessL = _bfgs_add_constraint(HessL, bfgs_add_order,
                                                 pr.bfgs_restart)
                elif pr.method == 'L-BFGS':
                    lbfgs_first, lbfgs_iter, sks, yks, rhos, HessInv = \
                        _lbfgs_restart_hessian(pr.lbfgs_n_vecs, Z)
                elif pr.method == 'cg':
                    g_zkm1 = None

                if pr.constraint_add_inds:
                    if not iter_count in constraint_add_inds:
                        constraint_add_inds.append(iter_count)

            x_k = x_k/x_k.sum()

        # End add constraint ----------

        # Hybrid method must track whether constraints changed to determine
        # when to transition
        if pr.hybrid and pr.hybrid['constraints_unchanged_limit'] < np.inf:
            old_constraint_count = new_constraint_count
            new_constraint_count = constraints.lennw
            if new_constraint_count == old_constraint_count:
                constraints_unchanged_count += 1
            else:
                constraints_unchanged_count = 0

        # Next few lines are solely for tracking purposes
        if pr.track_ellipses:
            Ls.append(L.copy())
        if pr.track_iters:
            xs.append(x_k.copy())
        if pr.track_objs:
            objs.append(-f_xk)
        if pr.track_gradients:
            gradients.append(g_z.copy())
        if pr.track_errors:
            errors.append(la.norm(x_k - truesol))

    # Undo upproject to get final answer if necessary
    if pr.upproject:
        small_H = solve_cholesky(L, np.eye(len(L)))[:-1, :-1]
        L = sla.cholesky(la.inv(small_H),
                         lower=True, check_finite=False)
        c = ((X @ x_k)[:-1])
    else:
        c = np.zeros(n)

    # Compute epsilon-primal feasibility and optimality of final answer if
    # requested
    if pr.verbose or pr.converged:
        if pr.upproject:
            Xc = X[:-1] - c[:, np.newaxis]
            n = n - 1
        else:
            Xc = X
        test = np.einsum('ij, ij->j', Xc, solve_cholesky(L, Xc))
        upos = np.where(x_k > 1e-15)[0]
        feasible = (test < (n*(1 + epsilon))).all()
        optimal = (test[upos] > (n*(1 - epsilon))).all()
        if pr.verbose:
            print('Feasible: ', feasible)
            print('Optimal: ', optimal)
        if pr.converged:
            retvals['converged'] = eps_worst < epsilon

    if iter_count >= pr.max_iter and not pr.silent:
        print("Maximum number of iterations reached. (%s)" % pr.method)

    # User may want various outputs -- return what they requested in a dict
    # Same dict may include tracking information from above if requested
    if pr.full_output:
        retvals['L'] = L
        retvals['c'] = c
        retvals['u'] = x_k
    if pr.count_inds:
        inds = np.where(x_k > 1e-10)[0]
        correct = orig.intersection(inds)
        retvals['correct'] = len(correct)
        retvals['diff'] = len(orig) - len(correct)
    if pr.Hessian:
        retvals['Hess'] = projected_Hessian(X, L, Z)
    if pr.large_output:
        retvals['L'] = L
        retvals['c'] = c
        retvals['u'] = x_k
        retvals['Z'] = Z
        retvals['g_z'] = g_z
        retvals['X'] = X
        retvals['constraints'] = constraints
    if pr.track_count:
        retvals['iter_count'] = iter_count

    # If no values were requested by user, default to returning matrix
    # representing ellipsoid
    if len(retvals) == 0:
        retvals['mat'] = solve_cholesky(L, np.eye(len(L)))

    if DO_CUDA:
        skcuda.cublas.cublasDestroy(handle)
        pass

    return retvals

# }}}


# {{{ Calculate MVEE via Todd's algorithm

def mvee(X, **kwargs):
    '''
    Python implementation of Todd's coordinate ascent algorithm

    Other than addition of substantial amounts of tracking code to help profile
    method and compare it to new methods, code follows MATLAB version presented
    in appendix of reference [18]. Please see original code for details on
    how method is implemented.
    '''

    # Read arguments and supply defaults as necessary
    params = copy.deepcopy(defaults)
    params.update(kwargs)
    pr = Namespace(**params)

    epsilon = pr.epsilon
    if pr.track_all_iters:
        pr.track_iters = True

    retvals = {}
    if pr.track_epsilons:
        epss = []
        retvals['epss'] = epss

    if pr.track_differences_obj:
        diffs = []
        retvals['diffs'] = diffs
        pr.track_objs = True

    import time
    start_init = time.time()
    n, m = X.shape
    if pr.update_data is None or pr.initialize == 'given':
        u = choose_init(X, pr)

        end_init = time.time()

        working = list(np.where(u == 0)[0])
        constraints = Constraints(working, m + 1, Cache(X))
        L = cholesky(X, u, constraints)
        factor = 1.0
    else:
        # update_data is a dictionary with 'new_x', 'L', 'factor', and 'u'
        X = np.hstack([X, pr.update_data['new_x'][:, np.newaxis]])
        n, m = X.shape

        u = np.hstack([pr.update_data['u'], 0])
        L = pr.update_data['L']
        factor = pr.update_data['factor']
        end_init = time.time()

    if pr.algorithm is None:
        pr.algorithm = 'wa'

    g_xk = gradient(X, L, factor)
    e_i = np.zeros(m)
    iter_count = 0

    if pr.hybrid:
        constraints_unchanged_count = 0
        if 'constraints_unchanged_limit' not in pr.hybrid:
            pr.hybrid['constraints_unchanged_limit'] = 30
        elif pr.hybrid['constraints_unchanged_limit'] == -1:
            pr.hybrid['constraints_unchanged_limit'] = np.inf

        if pr.hybrid['constraints_unchanged_limit'] < np.inf:
            # keep track of inactive constraints because the number of active
            # constraints can decrease spuriously when drop_every is applied
            new_constraint_count = m - (u > 1e-8).sum()

    if pr.track_objs:
        objs = [-objective(L, factor)]
        retvals['objs'] = objs
    if pr.track_ellipses:
        Ls = [L.copy()]
        factors = [factor]
        retvals['Ls'] = Ls
        retvals['factors'] = factors
    if pr.track_iters:
        us = [u.copy()]
        retvals['us'] = us
    if pr.constraint_add_inds:
        constraint_add_inds = []
        retvals['constraint_add_inds'] = constraint_add_inds
    if pr.constraint_rem_inds:
        constraint_rem_inds = []
        retvals['constraint_rem_inds'] = constraint_rem_inds
        maybe_rem_ind = False
    if pr.track_core_set_size:
        core_set_sizes = [(u > 1e-8).sum()]
        retvals['core_set_sizes'] = core_set_sizes
    if pr.track_errors:
        errors = []
        retvals['errors'] = errors
        if pr.true_solution is None:
            raise Exception(('Parameter track_errors does not make sense '
                             'without parameter true_solution.'))
        truesol = pr.true_solution

    t1 = time.time()
    while iter_count < pr.max_iter:

        iter_count = iter_count + 1

        if pr.hybrid and (
          iter_count > pr.hybrid['step_count'] or
          constraints_unchanged_count >= pr.hybrid['constraints_unchanged_limit']):
            new_params = copy.deepcopy(params)
            new_params['max_iter'] = pr.max_iter - pr.hybrid['step_count']
            new_params['update_data'] = u.copy()
            new_params['initialize'] = 'given'
            new_params['method'] = pr.hybrid['method']
            new_params['hybrid'] = None
            new_params['upproject'] = False
            new_retvals = mvee2(X, **new_params)

            for key in retvals.keys():
                retvals[key].extend(new_retvals[key])
            for key in new_retvals.keys():
                if key not in retvals.keys():
                    retvals[key] = new_retvals[key]
            return retvals

        # Remove inessential points every 100 iterations
        if pr.drop_every != np.inf and iter_count % pr.drop_every == 0:
            delta = n*np.log(1 + eps_plus)
            dn = delta*n
            lwr_bound = n*(1 + dn/2 - np.sqrt(dn - delta + dn**2/4))
            above = np.where(np.logical_or(g_xk >= lwr_bound,
                                           u > 1e-8))[0]

            removed = m - len(above)
            m = len(above)
            g_xk = g_xk[above]
            X = X[:, above]
            u = u[above]
            e_i = np.zeros(m)

            if pr.track_differences_obj:
                all_indices = np.arange(m)
                working = list(np.where(u < 1e-8)[0])
                constraints = Constraints(working, m + 1, Cache(X))

        idx_plus = np.argmax(g_xk)
        eps_plus = (g_xk[idx_plus] - n)/n

        if pr.algorithm == 'wa':
            nonzeros = np.where(u != 0)[0]
            idx_minus = nonzeros[np.argmin(g_xk[nonzeros])]
            eps_minus = (n - g_xk[idx_minus])/n
        else:
            idx_minus = -1
            eps_minus = 0

        go_minus = eps_minus > eps_plus
        if go_minus:
            idx_worst = idx_minus
            eps_worst = eps_minus
        else:
            idx_worst = idx_plus
            eps_worst = eps_plus

        if pr.track_epsilons:
            epss.append(eps_worst)
        if pr.constraint_rem_inds:
            if u[idx_worst] < 1e-8:
                maybe_rem_ind = True
        if pr.constraint_add_inds:
            add_inds_nonzeros = (u > 1e-8).astype(int)

        if eps_worst < epsilon:
            break

        lmda = (g_xk[idx_worst] - n)/((n - 1)*g_xk[idx_worst])
        if go_minus:
            lmda = max(-u[idx_worst], lmda)

        if pr.track_differences_obj:
            oldu = u.copy()

        e_i[idx_worst] = lmda
        u = 1/(1 + lmda)*(u + e_i)
        u = u/u.sum()
        e_i[idx_worst] = 0

        g_xk = _updated_gradient(g_xk, X, L, factor, idx_worst, lmda)

        L, factor = _updated_cholesky(L, factor, X[:, idx_worst], lmda)

        if pr.hybrid and pr.hybrid['constraints_unchanged_limit'] < np.inf:
            old_constraint_count = new_constraint_count
            new_constraint_count = m - (u > 1e-8).sum()
            if new_constraint_count == old_constraint_count:
                constraints_unchanged_count += 1
            else:
                constraints_unchanged_count = 0

        if pr.track_objs:
            objs.append(-objective(L, factor))
        if pr.track_ellipses:
            Ls.append(L.copy())
            factors.append(factor)
        if pr.track_iters:
            us.append(u)
        if pr.constraint_rem_inds and maybe_rem_ind:
            if u[idx_worst] > 1e-8:
                constraint_rem_inds.append(iter_count)
            maybe_rem_ind = False
        if pr.constraint_add_inds:
            add_inds_post_nonzeros = (u > 1e-8).astype(int)
            if (add_inds_nonzeros - add_inds_post_nonzeros > 0).any():
                constraint_add_inds.append(iter_count)
        if pr.track_core_set_size:
            core_set_sizes.append((u > 1e-8).sum())
        if pr.track_errors:
            errors.append(la.norm(u - truesol))
        if pr.track_differences_obj:
            # very slow
            x_k = u.copy()
            all_indices = np.arange(m)
            working = list(np.where(u < 1e-8)[0])
            constraints = Constraints(working, m + 1, constraints.cache)
            Z = NullSpaceMatrix(constraints)

            g_z2 = projected_gradient_small(constraints, L, ortho=False)
            p_k = _newton_step_small(Z, g_z2, constraints, L)

            neg_inds = np.where(p_k[~constraints.active] < 0)[0]
            neg_not_working = all_indices[~constraints.active][neg_inds]
            if len(neg_inds) == 0:
                aBar = np.inf
            else:
                stops = -x_k[neg_not_working]/p_k[neg_not_working]
                aBar_i = np.argmin(stops)
                aBar = stops[aBar_i]
            a_k = min(aBar, 1)

            inds = ~constraints.active
            max_L = cholesky_small(constraints, x_k[inds] + a_k*p_k[inds])
            obj2 = objective(max_L)

            try:
                diff = (obj2 + objs[-2])/(objs[-1] - objs[-2])
                if diff == np.inf:
                    diffs.append(50)
                else:
                    diffs.append(diff)
            except:
                pass

    end_all = time.time()

    if iter_count >= pr.max_iter and not pr.silent:
        print("Maximum number of iterations reached.")

    if pr.verbose or pr.converged:
        test = np.einsum('ij, ij->j', X, solve_cholesky(L, X, factor))
        upos = np.where(u > 1e-15)[0]
        feasible = (test < (n*(1 + epsilon))).all()
        optimal = (test[upos] > (n*(1 - epsilon))).all()
        if pr.verbose:
            print('Feasible: ', feasible)
            print('Optimal: ', optimal)
        if pr.converged:
            retvals['converged'] = eps_worst < epsilon

    if pr.full_output:
        retvals['L'] = L
        retvals['factor'] = factor
        retvals['u'] = u
    if pr.timing:
        t_init = end_init - start_init
        t_all = end_all - start_init
        retvals['t_all'] = t_all
        retvals['t_init'] = t_init
        retvals['iter_count'] = iter_count
    if pr.track_count:
        retvals['iter_count'] = iter_count

    if len(retvals) == 0:
        retvals['mat'] = solve_cholesky(L, np.eye(len(L)), factor)

    return retvals


def add_point(new_x, X, L, factor, u, **kwargs):
    return mvee(X, update_data={'new_x': new_x, 'L': L, 'factor': factor,
                                'u': u}, **kwargs)


# }}}


# vim: foldmethod=marker
