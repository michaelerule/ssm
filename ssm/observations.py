import copy
import warnings

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.misc import logsumexp
from autograd.scipy.special import gammaln, digamma
from autograd.scipy.stats import norm, gamma
from autograd.misc.optimizers import sgd, adam
from autograd import grad

from ssm.util import random_rotation, ensure_args_are_lists, ensure_args_not_none, \
    logistic, logit, one_hot, generalized_newton_studentst_dof
from ssm.preprocessing import interpolate_data
from ssm.cstats import robust_ar_statistics


class _Observations(object):
    def __init__(self, K, D, M=0):
        self.K, self.D, self.M = K, D, M

    @property
    def params(self):
        raise NotImplementedError

    @params.setter
    def params(self, value):
        raise NotImplementedError

    def permute(self, perm):
        pass

    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None):
        pass

    def log_prior(self):
        return 0

    def log_likelihoods(self, data, input, mask, tag):
        raise NotImplementedError

    def sample_x(self, z, xhist, input=None, tag=None, with_noise=True):
        raise NotImplementedError

    def m_step(self, expectations, datas, inputs, masks, tags,
               optimizer="adam", **kwargs):
        """
        If M-step cannot be done in closed form for the transitions, default to SGD.
        """
        optimizer = dict(sgd=sgd, adam=adam)[optimizer]

        # expected log joint
        def _expected_log_joint(expectations):
            elbo = self.log_prior()
            for data, input, mask, tag, (expected_states, expected_joints, _) \
                in zip(datas, inputs, masks, tags, expectations):
                lls = self.log_likelihoods(data, input, mask, tag)
                elbo += np.sum(expected_states * lls)
            return elbo

        # define optimization target
        T = sum([data.shape[0] for data in datas])
        def _objective(params, itr):
            self.params = params
            obj = _expected_log_joint(expectations)
            return -obj / T

        self.params = \
            optimizer(grad(_objective), self.params, **kwargs)

    def smooth(self, expectations, data, input, tag):
        raise NotImplementedError


class GaussianObservations(_Observations):
    def __init__(self, K, D, M=0):
        super(GaussianObservations, self).__init__(K, D, M)
        self.mus = npr.randn(K, D)
        self.inv_sigmas = -2 + npr.randn(K, D)

    @property
    def params(self):
        return self.mus, self.inv_sigmas

    @params.setter
    def params(self, value):
        self.mus, self.inv_sigmas = value

    def permute(self, perm):
        self.mus = self.mus[perm]
        self.inv_sigmas = self.inv_sigmas[perm]

    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None):
        # Initialize with KMeans
        from sklearn.cluster import KMeans
        data = np.concatenate(datas)
        km = KMeans(self.K).fit(data)
        self.mus = km.cluster_centers_
        sigmas = np.array([np.var(data[km.labels_ == k], axis=0)
                           for k in range(self.K)])
        self.inv_sigmas = np.log(sigmas + 1e-16)

    def log_likelihoods(self, data, input, mask, tag):
        mus, sigmas = self.mus, np.exp(self.inv_sigmas) + 1e-16
        mask = np.ones_like(data, dtype=bool) if mask is None else mask
        return -0.5 * np.sum(
            (np.log(2 * np.pi * sigmas) + (data[:, None, :] - mus)**2 / sigmas)
            * mask[:, None, :], axis=2)

    def sample_x(self, z, xhist, input=None, tag=None, with_noise=True):
        D, mus = self.D, self.mus
        sigmas = np.exp(self.inv_sigmas) if with_noise else np.zeros((self.K, self.D))
        return mus[z] + np.sqrt(sigmas[z]) * npr.randn(D)

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
        x = np.concatenate(datas)
        weights = np.concatenate([Ez for Ez, _, _ in expectations])
        for k in range(self.K):
            self.mus[k] = np.average(x, axis=0, weights=weights[:,k])
            sqerr = (x - self.mus[k])**2
            self.inv_sigmas[k] = np.log(np.average(sqerr, weights=weights[:,k], axis=0))

    def smooth(self, expectations, data, input, tag):
        """
        Compute the mean observation under the posterior distribution
        of latent discrete states.
        """
        return expectations.dot(self.mus)


class StudentsTObservations(_Observations):
    def __init__(self, K, D, M=0):
        super(StudentsTObservations, self).__init__(K, D, M)
        self.mus = npr.randn(K, D)
        self.inv_sigmas = -2 + npr.randn(K, D)
        # Student's t distribution also has a degrees of freedom parameter
        self.inv_nus = np.log(4) * np.ones(K)

    @property
    def params(self):
        return self.mus, self.inv_sigmas, self.inv_nus

    @params.setter
    def params(self, value):
        self.mus, self.inv_sigmas, self.inv_nus = value

    def permute(self, perm):
        self.mus = self.mus[perm]
        self.inv_sigmas = self.inv_sigmas[perm]
        self.inv_nus = self.inv_nus[perm]

    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None):
        # Initialize with KMeans
        from sklearn.cluster import KMeans
        data = np.concatenate(datas)
        km = KMeans(self.K).fit(data)
        self.mus = km.cluster_centers_
        sigmas = np.array([np.var(data[km.labels_ == k], axis=0)
                           for k in range(self.K)])
        self.inv_sigmas = np.log(sigmas + 1e-16)
        self.inv_nus = np.log(4) * np.ones(self.K)

    def log_likelihoods(self, data, input, mask, tag):
        D, mus, sigmas, nus = self.D, self.mus, np.exp(self.inv_sigmas), np.exp(self.inv_nus)
        # mask = np.ones_like(data, dtype=bool) if mask is None else mask

        resid = data[:, None, :] - mus
        z = resid / sigmas
        return -0.5 * (nus + D) * np.log(1.0 + (resid * z).sum(axis=2) / nus) + \
            gammaln((nus + D) / 2.0) - gammaln(nus / 2.0) - D / 2.0 * np.log(nus) \
            -D / 2.0 * np.log(np.pi) - 0.5 * np.sum(np.log(sigmas), axis=1)

    def sample_x(self, z, xhist, input=None, tag=None, with_noise=True):
        D, mus, sigmas, nus = self.D, self.mus, np.exp(self.inv_sigmas), np.exp(self.inv_nus)
        tau = npr.gamma(nus[z] / 2.0, 2.0 / nus[z])
        sigma = sigmas[z] / tau if with_noise else 0
        return mus[z] + np.sqrt(sigma) * npr.randn(D)

    def smooth(self, expectations, data, input, tag):
        """
        Compute the mean observation under the posterior distribution
        of latent discrete states.
        """
        return expectations.dot(self.mus)

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
        """
        Student's t is a scale mixture of Gaussians.  We can estimate its
        parameters using the EM algorithm. See the notebook in doc/students_t for
        complete details.
        """
        self._m_step_mu_sigma(expectations, datas, inputs, masks, tags)
        self._m_step_nu(expectations, datas, inputs, masks, tags)

    def _m_step_mu_sigma(self, expectations, datas, inputs, masks, tags):
        K, D = self.K, self.D

        # Estimate the precisions w for each data point
        E_taus = []
        for y in datas:
            # nu: (K,)  mus: (K, D)  sigmas: (K, D)  y: (T, D)  -> w: (T, K, D)
            nus = np.exp(self.inv_nus[:, None])
            alpha = nus/2 + 1/2
            beta = nus/2 + 1/2 * (y[:, None, :] - self.mus)**2 / np.exp(self.inv_sigmas)
            E_taus.append(alpha / beta)

        # Update the mean (notation from natural params of Gaussian)
        J = np.zeros((K, D))
        h = np.zeros((K, D))
        for E_tau, (Ez, _, _), y in zip(E_taus, expectations, datas):
            J += np.sum(Ez[:, :, None] * E_tau, axis=0)
            h += np.sum(Ez[:, :, None] * E_tau * y[:, None, :], axis=0)
        self.mus = h / J

        # Update the variance
        sqerr = np.zeros((K, D))
        weight = np.zeros((K, D))
        for E_tau, (Ez, _, _), y in zip(E_taus, expectations, datas):
            sqerr += np.sum(Ez[:, :, None] * E_tau * (y[:, None, :] - self.mus)**2, axis=0)
            weight += np.sum(Ez[:, :, None], axis=0)
        self.inv_sigmas = np.log(sqerr / weight + 1e-16)

    def _m_step_nu(self, expectations, datas, inputs, masks, tags):
        """
        The shape parameter nu determines a gamma prior.  We have

            tau_n ~ Gamma(nu/2, nu/2)
            y_n ~ N(mu, sigma^2 / tau_n)

        To update nu, we do EM and optimize the expected log likelihood using
        a generalized Newton's method.  See the notebook in doc/students_t for
        complete details.
        """
        K, D = self.K, self.D

        # Compute the precisions w for each data point
        E_taus = np.zeros(K)
        E_logtaus = np.zeros(K)
        weights = np.zeros(K)
        for y, (Ez, _, _) in zip(datas, expectations):
            # nu: (K,)  mus: (K, D)  sigmas: (K, D)  y: (T, D)  -> alpha/beta: (T, K, D)
            nus = np.exp(self.inv_nus[:, None])
            alpha = nus/2 + 1/2
            beta = nus/2 + 1/2 * (y[:, None, :] - self.mus)**2 / np.exp(self.inv_sigmas)

            E_taus += np.sum(Ez[:, :, None] * alpha / beta, axis=(0, 2))
            E_logtaus += np.sum(Ez[:, :, None] * (digamma(alpha) - np.log(beta)), axis=(0, 2))
            weights += np.sum(Ez, axis=0) * D

        E_taus /= weights
        E_logtaus /= weights

        for k in range(K):
            self.inv_nus[k] = np.log(generalized_newton_studentst_dof(E_taus[k], E_logtaus[k]))


class BernoulliObservations(_Observations):
    def __init__(self, K, D, M=0):
        super(BernoulliObservations, self).__init__(K, D, M)
        self.logit_ps = npr.randn(K, D)

    @property
    def params(self):
        return self.logit_ps

    @params.setter
    def params(self, value):
        self.logit_ps = value

    def permute(self, perm):
        self.logit_ps = self.logit_ps[perm]

    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None):

        # Initialize with KMeans
        from sklearn.cluster import KMeans
        data = np.concatenate(datas)
        km = KMeans(self.K).fit(data)
        ps = np.clip(km.cluster_centers_, 1e-3, 1-1e-3)
        self.logit_ps = logit(ps)

    def log_likelihoods(self, data, input, mask, tag):
        assert (data.dtype == int or data.dtype == bool)
        assert data.ndim == 2 and data.shape[1] == self.D
        assert data.min() >= 0 and data.max() <= 1
        ps = logistic(self.logit_ps)
        mask = np.ones_like(data, dtype=bool) if mask is None else mask
        lls = data[:, None, :] * np.log(ps) + (1 - data[:, None, :]) * np.log(1 - ps)
        return np.sum(lls * mask[:, None, :], axis=2)

    def sample_x(self, z, xhist, input=None, tag=None, with_noise=True):
        ps = 1 / (1 + np.exp(self.logit_ps))
        return npr.rand(self.D) < ps[z]

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
        x = np.concatenate(datas)
        weights = np.concatenate([Ez for Ez, _, _ in expectations])
        for k in range(self.K):
            ps = np.clip(np.average(x, axis=0, weights=weights[:,k]), 1e-3, 1-1e-3)
            self.logit_ps[k] = logit(ps)

    def smooth(self, expectations, data, input, tag):
        """
        Compute the mean observation under the posterior distribution
        of latent discrete states.
        """
        ps = 1 / (1 + np.exp(self.logit_ps))
        return expectations.dot(ps)


class PoissonObservations(_Observations):
    def __init__(self, K, D, M=0):
        super(PoissonObservations, self).__init__(K, D, M)
        self.log_lambdas = npr.randn(K, D)

    @property
    def params(self):
        return self.log_lambdas

    @params.setter
    def params(self, value):
        self.log_lambdas = value

    def permute(self, perm):
        self.log_lambdas = self.log_lambdas[perm]

    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None):

        # Initialize with KMeans
        from sklearn.cluster import KMeans
        data = np.concatenate(datas)
        km = KMeans(self.K).fit(data)
        self.log_lambdas = np.log(km.cluster_centers_ + 1e-3)

    def log_likelihoods(self, data, input, mask, tag):
        assert data.dtype == int
        lambdas = np.exp(self.log_lambdas)
        mask = np.ones_like(data, dtype=bool) if mask is None else mask
        lls = -gammaln(data[:,None,:] + 1) - lambdas + data[:,None,:] * np.log(lambdas)
        assert lls.shape == (data.shape[0], self.K, self.D)
        return np.sum(lls * mask[:, None, :], axis=2)

    def sample_x(self, z, xhist, input=None, tag=None, with_noise=True):
        lambdas = np.exp(self.log_lambdas)
        return npr.poisson(lambdas[z])

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
        x = np.concatenate(datas)
        weights = np.concatenate([Ez for Ez, _, _ in expectations])
        for k in range(self.K):
            self.log_lambdas[k] = np.log(np.average(x, axis=0, weights=weights[:,k]) + 1e-16)

    def smooth(self, expectations, data, input, tag):
        """
        Compute the mean observation under the posterior distribution
        of latent discrete states.
        """
        return expectations.dot(np.exp(self.log_lambdas))


class CategoricalObservations(_Observations):
    def __init__(self, K, D, M=0, C=2):
        """
        @param C:  number of classes in the categorical observations
        """
        super(CategoricalObservations, self).__init__(K, D, M)
        self.C = C
        self.logits = npr.randn(K, D, C)

    @property
    def params(self):
        return self.logits

    @params.setter
    def params(self, value):
        self.logits = value

    def permute(self, perm):
        self.logits = self.logits[perm]

    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None):
        pass

    def log_likelihoods(self, data, input, mask, tag):
        assert (data.dtype == int or data.dtype == bool)
        assert data.ndim == 2 and data.shape[1] == self.D
        assert data.min() >= 0 and data.max() < self.C
        logits = self.logits - logsumexp(self.logits, axis=2, keepdims=True)  # K x D x C
        x = one_hot(data, self.C)                                             # T x D x C
        lls = np.sum(x[:, None, :, :] * logits, axis=3)                       # T x K x D
        mask = np.ones_like(data, dtype=bool) if mask is None else mask       # T x D
        return np.sum(lls * mask[:, None, :], axis=2)                         # T x K

    def sample_x(self, z, xhist, input=None, tag=None, with_noise=True):
        ps = np.exp(self.logits - logsumexp(self.logits, axis=2, keepdims=True))
        return np.array([npr.choice(self.C, p=ps[z, d]) for d in range(self.D)])

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
        x = np.concatenate(datas)
        weights = np.concatenate([Ez for Ez, _, _ in expectations])
        for k in range(self.K):
            # compute weighted histogram of the class assignments
            xoh = one_hot(x, self.C)                                          # T x D x C
            ps = np.average(xoh, axis=0, weights=weights[:, k]) + 1e-3        # D x C
            ps /= np.sum(ps, axis=-1, keepdims=True)
            self.logits[k] = np.log(ps)

    def smooth(self, expectations, data, input, tag):
        """
        Compute the mean observation under the posterior distribution
        of latent discrete states.
        """
        raise NotImplementedError


class AutoRegressiveObservations(_Observations):
    def __init__(self, K, D, M=0, lags=1,reg_type="none",reg_coef=0,D_vec=0,W_inv=0,W=0):
        super(AutoRegressiveObservations, self).__init__(K, D, M)

        #Regularization parameters of dynamics
        self.reg_type=reg_type
        self.reg_coef=reg_coef
        self.D_vec=D_vec
        self.W_inv=W_inv
        self.W=W

        # Distribution over initial point
        self.mu_init = np.zeros(D)
        self.inv_sigma_init = np.zeros(D)

        # AR parameters
        assert lags > 0
        self.lags = lags
        self.As = .95 * np.array([
                np.column_stack([random_rotation(D), np.zeros((D, (lags-1) * D))])
            for _ in range(K)])
        self.bs = npr.randn(K, D)
        self.Vs = npr.randn(K, D, M)
        self.inv_sigmas = -4 + npr.randn(K, D)

    @property
    def params(self):
        return self.As, self.bs, self.Vs, self.inv_sigmas

    @params.setter
    def params(self, value):
        self.As, self.bs, self.Vs, self.inv_sigmas = value

    def permute(self, perm):
        self.As = self.As[perm]
        self.bs = self.bs[perm]
        self.Vs = self.Vs[perm]
        self.inv_sigmas = self.inv_sigmas[perm]

    def initialize(self, datas, inputs=None, masks=None, tags=None):
        # Initialize with linear regressions
        from sklearn.linear_model import LinearRegression
        data = np.concatenate(datas)
        input = np.concatenate(inputs)
        T = data.shape[0]

        for k in range(self.K):
            ts = npr.choice(T-self.lags, replace=False, size=(T-self.lags)//self.K)
            x = np.column_stack([data[ts + l] for l in range(self.lags)] + [input[ts]])
            y = data[ts+self.lags]
            lr = LinearRegression().fit(x, y)
            # print(x.shape)
            # lr2 = LinearRegression().fit(x[:,:6], y[:,:6]) #new
            # lr3 = LinearRegression().fit(x[:,6:], y[:,6:])#new
            # self.As[k] = np.zeros([self.D,self.D])
            # self.As[k][:6,:6]=lr2.coef_
            # self.As[k][6:,6:]=lr3.coef_
            self.As[k] = lr.coef_[:, :self.D * self.lags]
            self.Vs[k] = lr.coef_[:, self.D * self.lags:]
            self.bs[k] = lr.intercept_

            resid = y - lr.predict(x)
            sigmas = np.var(resid, axis=0)
            self.inv_sigmas[k] = np.log(sigmas + 1e-16)

    def _compute_mus(self, data, input, mask, tag):
        assert np.all(mask), "ARHMM cannot handle missing data"
        T, D = data.shape
        As, bs, Vs = self.As, self.bs, self.Vs

        # Instantaneous inputs
        mus = np.matmul(Vs[None, ...], input[self.lags:, None, :, None])[:, :, :, 0]

        # Lagged data
        for l in range(self.lags):
            mus = mus + np.matmul(As[None, :, :, l*D:(l+1)*D],
                                  data[self.lags-l-1:-l-1, None, :, None])[:, :, :, 0]

        # Bias
        mus = mus + bs

        # Pad with the initial condition
        mus = np.concatenate((self.mu_init * np.ones((self.lags, self.K, self.D)), mus))

        assert mus.shape == (T, self.K, D)
        return mus

    def _compute_sigmas(self, data, input, mask, tag):
        T, D = data.shape
        inv_sigmas = self.inv_sigmas

        sigma_init = np.exp(self.inv_sigma_init) * np.ones((self.lags, self.K, self.D))
        sigma_ar = np.repeat(np.exp(inv_sigmas)[None, :, :], T-self.lags, axis=0)
        sigmas = np.concatenate((sigma_init, sigma_ar))
        assert sigmas.shape == (T, self.K, D)
        return sigmas

    def log_likelihoods(self, data, input, mask, tag):
        mus = self._compute_mus(data, input, mask, tag)
        sigmas = self._compute_sigmas(data, input, mask, tag)
        return -0.5 * np.sum(
            (np.log(2 * np.pi * sigmas) + (data[:, None, :] - mus)**2 / sigmas)
            * mask[:, None, :], axis=2)
        # return -0.5 * np.sum(
        #     (np.log(2 * np.pi ) + (data[:, None, :] - mus)**2 )
        #     * mask[:, None, :], axis=2)

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
        K, D, M, lags = self.K, self.D, self.M, self.lags
        # Collect data for this dimension
        xs, ys, Ezs = [], [], []
        for (Ez, _, _), data, input, mask, tag in zip(expectations, datas, inputs, masks, tags):
            # Only use data if it is complete
            if not np.all(mask):
                raise Exception("Encountered missing data in AutoRegressiveObservations!")

            xs.append(
                np.hstack([data[self.lags-l-1:-l-1] for l in range(self.lags)]
                          + [input[self.lags:, :self.M], np.ones((data.shape[0]-self.lags, 1))]))
            ys.append(data[self.lags:])
            Ezs.append(Ez[self.lags:])

        # Fit a weighted linear regression for each discrete state
        for k in range(K):
            # Check for zero weights (singular matrix)
            # if np.sum(weights[:, k]) < D * lags + M + 1:
            #     self.As[k] = 0
            #     self.Vs[k] = 0
            #     self.bs[k] = 0
            #     self.inv_sigmas[k] = 0
            #     continue

            # Update each row of the AR matrix
            for d in range(D):
                # This is a weak prior centered on zero
                Jk = 1e-16 * np.eye(D * lags + M + 1)
                hk = np.zeros((D * lags + M + 1,))
                for x, y, Ez in zip(xs, ys, Ezs):
                    scale = Ez[:, k]
                    Jk += np.sum(scale[:, None, None] * x[:,:,None] * x[:, None,:], axis=0)
                    hk += np.sum(scale[:, None] * x * y[:, d:d+1], axis=0)

                muk = np.linalg.solve(Jk, hk)
                self.As[k, d] = muk[:D*lags]
                self.Vs[k, d] = muk[D*lags:D*lags+M]
                self.bs[k, d] = muk[-1]

                # Update the variance
                sqerr = 0
                weight = 0
                for x, y, Ez in zip(xs, ys, Ezs):
                    yhat = np.dot(x, muk)
                    sqerr += np.sum(Ez[:, k] * (y[:, d] - yhat)**2)
                    weight += np.sum(Ez[:, k])
                self.inv_sigmas[k, d] = np.log(sqerr / weight + 1e-16)

    def sample_x(self, z, xhist, input=None, tag=None, with_noise=True):
        D, As, bs, sigmas = self.D, self.As, self.bs, np.exp(self.inv_sigmas)
        if xhist.shape[0] < self.lags:
            sigma_init = np.exp(self.inv_sigma_init) if with_noise else 0
            return self.mu_init + np.sqrt(sigma_init) * npr.randn(D)
        else:
            mu = bs[z].copy()
            for l in range(self.lags):
                mu += As[z][:,l*D:(l+1)*D].dot(xhist[-l-1])

            sigma = sigmas[z] if with_noise else 0
            return mu + np.sqrt(sigma) * npr.randn(D)

    def smooth(self, expectations, data, input, tag):
        """
        Compute the mean observation under the posterior distribution
        of latent discrete states.
        """
        T = expectations.shape[0]
        mask = np.ones((T, self.D), dtype=bool)
        mus = self._compute_mus(data, input, mask, tag)
        return (expectations[:, :, None] * mus).sum(1)

    def log_prior(self):
        alpha=-self.reg_coef
        if self.reg_type=="none":
            return 0
        elif self.reg_type=='L1' or self.reg_type=='l1':
            return alpha*np.sum(np.abs((self.As-np.identity(self.D))[:]))
        elif self.reg_type=='L2' or self.reg_type=='l2':
            return alpha*np.sum((self.As-np.identity(self.D))[:]**2)
        elif self.reg_type=='group_lasso':
            D_vec=self.D_vec
            P=len(D_vec)
            D_vec_cumsum=np.concatenate(([0],np.cumsum(D_vec)))
            group_sums = [np.sqrt(D_vec[i]*D_vec[j])*np.linalg.norm((self.As[k]-np.identity(self.D))[D_vec_cumsum[i]:D_vec_cumsum[i+1],D_vec_cumsum[j]:D_vec_cumsum[j+1]]) for i in range(P) for j in range(P) for k in range(self.K)]
            return alpha*np.sum(group_sums)
        elif self.reg_type=='con_L1' or self.reg_type=='con_l1':
            return alpha*np.sum(np.abs((self.W_inv*(self.As-np.identity(self.D)))[:]))
        elif self.reg_type=='con_L2' or self.reg_type=='con_l2':
            return alpha*np.sum((self.W_inv*(self.As-np.identity(self.D)))[:]**2)
        elif self.reg_type=='con_group_lasso':
            D_vec=self.D_vec
            P=len(D_vec)
            D_vec_cumsum=np.concatenate(([0],np.cumsum(D_vec)))
            group_sums = [self.W_inv[i,j]*np.sqrt(D_vec[i]*D_vec[j])*np.linalg.norm((self.As[k]-np.identity(self.D))[D_vec_cumsum[i]:D_vec_cumsum[i+1],D_vec_cumsum[j]:D_vec_cumsum[j+1]]) for i in range(P) for j in range(P) for k in range(self.K)]
            return alpha*np.sum(group_sums)
        elif self.reg_type=='con_matching_L2' or self.reg_type=='con_matching_l2':
            return alpha*np.sum((self.As-self.W)[:]**2)
        elif self.reg_type=='con_matching_L1' or self.reg_type=='con_matching_l1':
            return alpha*np.sum(np.abs((self.As-self.W)[:]))
        else:
            #Error: This is not a valid regularization type
            raise NotImplementedError('This is not a valid regularization type')

class IndependentAutoRegressiveObservations(_Observations):
    def __init__(self, K, D, M=0, lags=1):
        super(IndependentAutoRegressiveObservations, self).__init__(K, D, M)

        # Distribution over initial point
        self.mu_init = np.zeros(D)
        self.inv_sigma_init = np.zeros(D)

        # AR parameters
        assert lags > 0
        self.lags = lags
        self.As = .95 * np.ones((K, D, lags))
        self.bs = npr.randn(K, D)
        self.Vs = npr.randn(K, D, M)
        self.inv_sigmas = -4 + npr.randn(K, D)

    @property
    def params(self):
        return self.As, self.bs, self.Vs, self.inv_sigmas

    @params.setter
    def params(self, value):
        self.As, self.bs, self.Vs, self.inv_sigmas = value

    def permute(self, perm):
        self.As = self.As[perm]
        self.bs = self.bs[perm]
        self.Vs = self.Vs[perm]
        self.inv_sigmas = self.inv_sigmas[perm]

    def initialize(self, datas, inputs=None, masks=None, tags=None):
        # Initialize with linear regressions
        from sklearn.linear_model import LinearRegression
        data = np.concatenate(datas)
        input = np.concatenate(inputs)
        T = data.shape[0]

        for k in range(self.K):
            for d in range(self.D):
                ts = npr.choice(T-self.lags, replace=False, size=(T-self.lags)//self.K)
                x = np.column_stack([data[ts + l, d:d+1] for l in range(self.lags)] + [input[ts, :self.M]])
                y = data[ts+self.lags, d:d+1]
                lr = LinearRegression().fit(x, y)

                self.As[k, d] = lr.coef_[:, :self.lags]
                self.Vs[k, d] = lr.coef_[:, self.lags:self.lags+self.M]
                self.bs[k, d] = lr.intercept_

                resid = y - lr.predict(x)
                sigmas = np.var(resid, axis=0)
                self.inv_sigmas[k, d] = np.log(sigmas + 1e-16)

    def _compute_mus(self, data, input, mask, tag):
        T, D = data.shape
        As, bs, Vs = self.As, self.bs, self.Vs

        # Instantaneous inputs, lagged data, and bias
        mus = np.matmul(Vs[None, ...], input[self.lags:, None, :self.M, None])[:, :, :, 0]
        for l in range(self.lags):
            mus += As[:, :, l] * data[self.lags-l-1:-l-1, None, :]
        mus += bs

        # Pad with the initial condition
        mus = np.concatenate((self.mu_init * np.ones((self.lags, self.K, self.D)), mus))

        assert mus.shape == (T, self.K, D)
        return mus

    def _compute_sigmas(self, data, input, mask, tag):
        T, D = data.shape

        sigma_init = np.exp(self.inv_sigma_init) * np.ones((self.lags, self.K, self.D))
        sigma_ar = np.repeat(np.exp(self.inv_sigmas)[None, :, :], T-self.lags, axis=0)
        sigmas = np.concatenate((sigma_init, sigma_ar))
        assert sigmas.shape == (T, self.K, D)
        return sigmas

    def log_likelihoods(self, data, input, mask, tag):
        mus = self._compute_mus(data, input, mask, tag)
        sigmas = self._compute_sigmas(data, input, mask, tag)
        ll = -0.5 * (np.log(2 * np.pi * sigmas) + (data[:, None, :] - mus)**2 / sigmas)
        return np.sum(ll * mask[:, None, :], axis=2)

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
        from sklearn.linear_model import LinearRegression
        D, M = self.D, self.M

        for d in range(self.D):
            # Collect data for this dimension
            xs, ys, weights = [], [], []
            for (Ez, _, _), data, input, mask in zip(expectations, datas, inputs, masks):
                # Only use data if it is complete
                if np.all(mask[:, d]):
                    xs.append(
                        np.hstack([data[self.lags-l-1:-l-1, d:d+1] for l in range(self.lags)]
                                  + [input[self.lags:, :M], np.ones((data.shape[0]-self.lags, 1))]))
                    ys.append(data[self.lags:, d])
                    weights.append(Ez[self.lags:])

            xs = np.concatenate(xs)
            ys = np.concatenate(ys)
            weights = np.concatenate(weights)

            # If there was no data for this dimension then skip it
            if len(xs) == 0:
                self.As[:, d, :] = 0
                self.Vs[:, d, :] = 0
                self.bs[:, d] = 0
                continue

            # Otherwise, fit a weighted linear regression for each discrete state
            for k in range(self.K):
                # Check for zero weights (singular matrix)
                if np.sum(weights[:, k]) < self.lags + M + 1:
                    self.As[k, d] = 1.0
                    self.Vs[k, d] = 0
                    self.bs[k, d] = 0
                    self.inv_sigmas[k, d] = 0
                    continue

                # Solve for the most likely A,V,b (no prior)
                Jk = np.sum(weights[:, k][:, None, None] * xs[:,:,None] * xs[:, None,:], axis=0)
                hk = np.sum(weights[:, k][:, None] * xs * ys[:, None], axis=0)
                muk = np.linalg.solve(Jk, hk)

                self.As[k, d] = muk[:self.lags]
                self.Vs[k, d] = muk[self.lags:self.lags+M]
                self.bs[k, d] = muk[-1]

                # Update the variances
                yhats = xs.dot(np.concatenate((self.As[k, d], self.Vs[k, d], [self.bs[k, d]])))
                sqerr = (ys - yhats)**2
                sigma = np.average(sqerr, weights=weights[:, k], axis=0) + 1e-16
                self.inv_sigmas[k, d] = np.log(sigma)

    def sample_x(self, z, xhist, input=None, tag=None, with_noise=True):
        D, As, bs, sigmas = self.D, self.As, self.bs, np.exp(self.inv_sigmas)
        if xhist.shape[0] < self.lags:
            sigma_init = np.exp(self.inv_sigma_init) if with_noise else 0
            return self.mu_init + np.sqrt(sigma_init) * npr.randn(D)
        else:
            mu = bs[z].copy()
            for l in range(self.lags):
                mu += As[z,:,l] * xhist[-l-1]

            sigma = sigmas[z] if with_noise else 0
            return mu + np.sqrt(sigma) * npr.randn(D)

    def smooth(self, expectations, data, input, tag):
        """
        Compute the mean observation under the posterior distribution
        of latent discrete states.
        """
        T = expectations.shape[0]
        mask = np.ones((T, self.D), dtype=bool)
        mus = self._compute_mus(data, input, mask, tag)
        return (expectations[:, :, None] * mus).sum(1)


# Robust autoregressive models with Student's t noise
class RobustAutoRegressiveObservations(AutoRegressiveObservations):
    def __init__(self, K, D, M=0, lags=1):
        super(RobustAutoRegressiveObservations, self).__init__(K, D, M=M, lags=lags)
        self.inv_nus = np.log(4) * np.ones(K)

    @property
    def params(self):
        return self.As, self.bs, self.Vs, self.inv_sigmas, self.inv_nus

    @params.setter
    def params(self, value):
        self.As, self.bs, self.Vs, self.inv_sigmas, self.inv_nus = value

    def permute(self, perm):
        super(RobustAutoRegressiveObservations, self).permute(perm)
        self.inv_nus = self.inv_nus[perm]

    def log_likelihoods(self, data, input, mask, tag):
        D = self.D
        mus = self._compute_mus(data, input, mask, tag)
        sigmas = self._compute_sigmas(data, input, mask, tag)
        nus = np.exp(self.inv_nus)

        resid = data[:, None, :] - mus
        z = resid / sigmas
        return -0.5 * (nus + D) * np.log(1.0 + (resid * z).sum(axis=2) / nus) + \
            gammaln((nus + D) / 2.0) - gammaln(nus / 2.0) - D / 2.0 * np.log(nus) \
            -D / 2.0 * np.log(np.pi) - 0.5 * np.sum(np.log(sigmas), axis=-1)

    def m_step(self, expectations, datas, inputs, masks, tags,
               num_em_iters=1, optimizer="adam", num_iters=10, **kwargs):
        """
        Student's t is a scale mixture of Gaussians.  We can estimate its
        parameters using the EM algorithm. See the notebook in doc/students_t
        for complete details.
        """
        self._m_step_ar(expectations, datas, inputs, masks, tags, num_em_iters)
        self._m_step_nu(expectations, datas, inputs, masks, tags, optimizer, num_iters, **kwargs)

    def _m_step_ar(self, expectations, datas, inputs, masks, tags, num_em_iters):
        K, D, M, lags = self.K, self.D, self.M, self.lags

        # Collect data for this dimension
        xs, ys, Ezs = [], [], []
        for (Ez, _, _), data, input, mask, tag in zip(expectations, datas, inputs, masks, tags):
            # Only use data if it is complete
            if not np.all(mask):
                raise Exception("Encountered missing data in AutoRegressiveObservations!")

            xs.append(
                np.hstack([data[self.lags-l-1:-l-1] for l in range(self.lags)]
                          + [input[self.lags:, :self.M], np.ones((data.shape[0]-self.lags, 1))]))
            ys.append(data[self.lags:])
            Ezs.append(Ez[self.lags:])

        for itr in range(num_em_iters):
            # Compute expected precision for each data point given current parameters
            taus = []
            for x, y in zip(xs, ys):
                # mus = self._compute_mus(data, input, mask, tag)
                # sigmas = self._compute_sigmas(data, input, mask, tag)
                Afull = np.concatenate((self.As, self.Vs, self.bs[:, :, None]), axis=2)
                mus = np.matmul(Afull[None, :, :, :], x[:, None, :, None])[:, :, :, 0]
                sigmas = np.exp(self.inv_sigmas)

                # nu: (K,)  mus: (T, K, D)  sigmas: (K, D)  y: (T, D)  -> tau: (T, K, D)
                alpha = np.exp(self.inv_nus[:, None])/2 + 1/2
                beta = np.exp(self.inv_nus[:, None])/2 + 1/2 * (y[:, None, :] - mus)**2 / sigmas
                taus.append(alpha / beta)

            # Fit the weighted linear regressions for each K and D
            J = np.tile(np.eye(D * lags + M + 1)[None, None, :, :], (K, D, 1, 1))
            h = np.zeros((K, D,  D*lags + M + 1,))
            for x, y, Ez, tau in zip(xs, ys, Ezs, taus):
                robust_ar_statistics(Ez, tau, x, y, J, h)

            mus = np.linalg.solve(J, h)
            self.As = mus[:, :, :D*lags]
            self.Vs = mus[:, :, D*lags:D*lags+M]
            self.bs = mus[:, :, -1]

            # Fit the variance
            sqerr = 0
            weight = 0
            for x, y, Ez, tau in zip(xs, ys, Ezs, taus):
                yhat = np.matmul(x[None, :, :], np.swapaxes(mus, -1, -2))
                sqerr += np.einsum('tk, tkd, ktd -> kd', Ez, tau, (y - yhat)**2)
                weight += np.sum(Ez, axis=0)
            self.inv_sigmas = np.log(sqerr / weight[:, None] + 1e-16)

    def _m_step_nu(self, expectations, datas, inputs, masks, tags, optimizer, num_iters, **kwargs):
        K, D = self.K, self.D
        E_taus = np.zeros(K)
        E_logtaus = np.zeros(K)
        weights = np.zeros(K)
        for (Ez, _, _,), data, input, mask, tag in zip(expectations, datas, inputs, masks, tags):
            # nu: (K,)  mus: (K, D)  sigmas: (K, D)  y: (T, D)  -> w: (T, K, D)
            mus = self._compute_mus(data, input, mask, tag)
            sigmas = self._compute_sigmas(data, input, mask, tag)
            nus = np.exp(self.inv_nus[:, None])

            alpha = nus/2 + 1/2
            beta = nus/2 + 1/2 * (data[:, None, :] - mus)**2 / sigmas

            E_taus += np.sum(Ez[:, :, None] * alpha / beta, axis=(0, 2))
            E_logtaus += np.sum(Ez[:, :, None] * (digamma(alpha) - np.log(beta)), axis=(0, 2))
            weights += np.sum(Ez, axis=0) * D

        E_taus /= weights
        E_logtaus /= weights

        for k in range(K):
            self.inv_nus[k] = np.log(generalized_newton_studentst_dof(E_taus[k], E_logtaus[k]))

    def sample_x(self, z, xhist, input=None, tag=None, with_noise=True):
        D, As, bs, sigmas, nus = self.D, self.As, self.bs, np.exp(self.inv_sigmas), np.exp(self.inv_nus)
        if xhist.shape[0] < self.lags:
            mu_init = self.mu_init
            sigma_init = np.exp(self.inv_sigma_init) if with_noise else 0
            return mu_init + np.sqrt(sigma_init) * npr.randn(D)
        else:
            mu = bs[z].copy()
            for l in range(self.lags):
                mu += As[z][:,l*D:(l+1)*D].dot(xhist[-l-1])

            tau = npr.gamma(nus[z] / 2.0, 2.0 / nus[z])
            sigma = sigmas[z] / tau if with_noise else 0
            return mu + np.sqrt(sigma) * npr.randn(D)


class _RecurrentAutoRegressiveObservationsMixin(AutoRegressiveObservations):
    """
    A simple mixin to allow for smarter initialization.
    """
    def initialize(self, datas, inputs=None, masks=None, tags=None):
        data = np.concatenate(datas)
        ddata = np.concatenate([np.gradient(d, axis=0) for d in datas])
        ddata = (ddata - ddata.mean(0)) / ddata.std(0)
        input = np.concatenate(inputs)
        T = data.shape[0]

        # Cluster the data and its gradient before initializing
        from sklearn.cluster import KMeans
        km = KMeans(self.K)
        # km.fit(np.column_stack((data, ddata)))
        km.fit(data)
        z = km.labels_[:-self.lags]

        from sklearn.linear_model import LinearRegression

        for k in range(self.K):
            ts = np.where(z == k)[0]
            x = np.column_stack([data[ts + l] for l in range(self.lags)] + [input[ts]])
            y = data[ts+self.lags]
            lr = LinearRegression().fit(x, y)
            self.As[k] = lr.coef_[:, :self.D * self.lags]
            self.Vs[k] = lr.coef_[:, self.D * self.lags:]
            self.bs[k] = lr.intercept_

            resid = y - lr.predict(x)
            sigmas = np.var(resid, axis=0)
            self.inv_sigmas[k] = np.log(sigmas + 1e-16)
            assert np.all(np.isfinite(self.inv_sigmas))



class RecurrentAutoRegressiveObservations(
    _RecurrentAutoRegressiveObservationsMixin,
    AutoRegressiveObservations):
    pass


class RecurrentRobustAutoRegressiveObservations(
    _RecurrentAutoRegressiveObservationsMixin,
    RobustAutoRegressiveObservations):
    pass


class VonMisesObservations(_Observations):
    def __init__(self, K, D, M=0):
        super(VonMisesObservations, self).__init__(K, D, M)
        self.mus = npr.randn(K, D)
        max_k = 9
        self.log_kappas = np.log(-1*npr.uniform(low=-1*max_k, high=0, size=(K, D)))

    @property
    def params(self):
        return self.mus, self.log_kappas

    @params.setter
    def params(self, value):
        self.mus, self.log_kappas = value

    def permute(self, perm):
        self.mus = self.mus[perm]
        self.log_kappas = self.log_kappas[perm]

    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None):
        # TODO: add spherical k-means for initialization
        pass

    def log_likelihoods(self, data, input, mask, tag):
        from autograd.scipy.special import i0
        # Compute the log likelihood of the data under each of the K classes
        # Return a TxK array of probability of data[t] under class k
        mus, kappas = self.mus, np.exp(self.log_kappas)
        mask = np.ones_like(data, dtype=bool) if mask is None else mask

        return np.sum(
            (kappas*(np.cos(data[:, None, :] - mus)) - np.log(2 * np.pi)
             - np.log(i0(kappas)))
            * mask[:, None, :], axis=2)

    def sample_x(self, z, xhist, input=None, tag=None, with_noise=True):
        D, mus, kappas = self.D, self.mus, np.exp(self.log_kappas)
        return npr.vonmises(self.mus[z], kappas[z], D)

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
        from autograd.scipy.special import i0, i1
        x = np.concatenate(datas)

        weights = np.concatenate([Ez for Ez, _, _ in expectations])

        # convert angles to 2D representation and employ closed form solutions
        x_k = np.stack((np.sin(x), np.cos(x)), axis=1)

        r_k = np.tensordot(weights.T, x_k, (-1, 0))

        r_norm = np.sqrt(np.sum(r_k ** 2, 1))
        mus_k = r_k / r_norm[:, None]
        r_bar = r_norm / weights.sum(0)[:, None]

        # truncated newton approximation with 2 iterations
        kappa_0 = r_bar * (2 - r_bar ** 2) / (1 - r_bar ** 2)

        kappa_1 = kappa_0 - ((i1(kappa_0)/i0(kappa_0)) - r_bar) / \
                  (1 - (i1(kappa_0)/i0(kappa_0)) ** 2 - (i1(kappa_0)/i0(kappa_0)) / kappa_0)
        kappa_2 = kappa_1 - ((i1(kappa_1)/i0(kappa_1)) - r_bar) / \
                  (1 - (i1(kappa_1)/i0(kappa_1)) ** 2 - (i1(kappa_1)/i0(kappa_1)) / kappa_1)

        for k in range(self.K):
            self.mus[k] = np.arctan2(*mus_k[k])
            self.log_kappas[k] = np.log(kappa_2[k])

    def smooth(self, expectations, data, input, tag):
        mus = self.mus
        return expectations.dot(mus)
