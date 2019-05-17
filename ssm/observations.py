import copy
import warnings

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.misc import logsumexp
from autograd.scipy.special import gammaln, digamma

from ssm.util import random_rotation, ensure_args_are_lists, \
    logistic, logit, one_hot, generalized_newton_studentst_dof, fit_linear_regression
from ssm.preprocessing import interpolate_data
from ssm.cstats import robust_ar_statistics
from ssm.optimizers import adam, bfgs, rmsprop, sgd
import ssm.stats as stats


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
               optimizer="bfgs", **kwargs):
        """
        If M-step cannot be done in closed form for the transitions, default to SGD.
        """
        optimizer = dict(adam=adam, bfgs=bfgs, rmsprop=rmsprop, sgd=sgd)[optimizer]

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

        self.params = optimizer(_objective, self.params, **kwargs)

    def smooth(self, expectations, data, input, tag):
        raise NotImplementedError


class GaussianObservations(_Observations):
    def __init__(self, K, D, M=0):
        super(GaussianObservations, self).__init__(K, D, M)
        self.mus = npr.randn(K, D)
        self._sqrt_Sigmas = npr.randn(K, D, D)

    @property
    def params(self):
        return self.mus, self._sqrt_Sigmas

    @params.setter
    def params(self, value):
        self.mus, self._sqrt_Sigmas = value

    def permute(self, perm):
        self.mus = self.mus[perm]
        self._sqrt_Sigmas = self._sqrt_Sigmas[perm]

    @property
    def Sigmas(self):
        return np.matmul(self._sqrt_Sigmas, np.swapaxes(self._sqrt_Sigmas, -1, -2))

    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None):
        # Initialize with KMeans
        from sklearn.cluster import KMeans
        data = np.concatenate(datas)
        km = KMeans(self.K).fit(data)
        self.mus = km.cluster_centers_
        Sigmas = np.array([np.atleast_2d(np.cov(data[km.labels_ == k].T))
                           for k in range(self.K)])
        assert np.all(np.isfinite(Sigmas))
        self._sqrt_Sigmas = np.linalg.cholesky(Sigmas + 1e-8 * np.eye(self.D))

    def log_likelihoods(self, data, input, mask, tag):
        mus, Sigmas = self.mus, self.Sigmas
        if mask is not None and np.any(~mask) and not isinstance(mus, np.ndarray):
            raise Exception("Current implementation of multivariate_normal_logpdf for masked data"
                            "does not work with autograd because it writes to an array. "
                            "Use DiagonalGaussian instead if you need to support missing data.")

        return stats.multivariate_normal_logpdf(data[:, None, :], mus, Sigmas)

    def sample_x(self, z, xhist, input=None, tag=None, with_noise=True):
        D, mus = self.D, self.mus
        sqrt_Sigmas = self._sqrt_Sigmas if with_noise else np.zeros((self.K, self.D, self.D))
        return mus[z] + np.dot(sqrt_Sigmas[z], npr.randn(D))

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
        K, D = self.K, self.D
        J = np.zeros((K, D))
        h = np.zeros((K, D))
        for (Ez, _, _), y in zip(expectations, datas):
            J += np.sum(Ez[:, :, None], axis=0)
            h += np.sum(Ez[:, :, None] * y[:, None, :], axis=0)
        self.mus = h / J

        # Update the variance
        sqerr = np.zeros((K, D, D))
        weight = np.zeros((K,))
        for (Ez, _, _), y in zip(expectations, datas):
            resid = y[:, None, :] - self.mus
            sqerr += np.sum(Ez[:, :, None, None] * resid[:, :, None, :] * resid[:, :, :, None], axis=0)
            weight += np.sum(Ez, axis=0)
        self._sqrt_Sigmas = np.linalg.cholesky(sqerr / weight[:, None, None] + 1e-8 * np.eye(self.D))

    def smooth(self, expectations, data, input, tag):
        """
        Compute the mean observation under the posterior distribution
        of latent discrete states.
        """
        return expectations.dot(self.mus)


class DiagonalGaussianObservations(_Observations):
    def __init__(self, K, D, M=0):
        super(DiagonalGaussianObservations, self).__init__(K, D, M)
        self.mus = npr.randn(K, D)
        self._log_sigmasq = -2 + npr.randn(K, D)

    @property
    def sigmasq(self):
        return np.exp(self._log_sigmasq)

    @sigmasq.setter
    def sigmasq(self, value):
        assert np.all(value > 0) and value.shape == (self.K, self.D)
        self._log_sigmasq = np.log(value)

    @property
    def params(self):
        return self.mus, self._log_sigmasq

    @params.setter
    def params(self, value):
        self.mus, self._log_sigmasq = value

    def permute(self, perm):
        self.mus = self.mus[perm]
        self._log_sigmasq = self._log_sigmasq[perm]

    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None):
        # Initialize with KMeans
        from sklearn.cluster import KMeans
        data = np.concatenate(datas)
        km = KMeans(self.K).fit(data)
        self.mus = km.cluster_centers_
        sigmas = np.array([np.var(data[km.labels_ == k], axis=0)
                           for k in range(self.K)])
        self._log_sigmasq = np.log(sigmas + 1e-16)

    def log_likelihoods(self, data, input, mask, tag):
        mus, sigmas = self.mus, np.exp(self._log_sigmasq) + 1e-16
        mask = np.ones_like(data, dtype=bool) if mask is None else mask
        return stats.diagonal_gaussian_logpdf(data[:, None, :], mus, sigmas, mask=mask[:, None, :])

    def sample_x(self, z, xhist, input=None, tag=None, with_noise=True):
        D, mus = self.D, self.mus
        sigmas = np.exp(self._log_sigmasq) if with_noise else np.zeros((self.K, self.D))
        return mus[z] + np.sqrt(sigmas[z]) * npr.randn(D)

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
        x = np.concatenate(datas)
        weights = np.concatenate([Ez for Ez, _, _ in expectations])
        for k in range(self.K):
            self.mus[k] = np.average(x, axis=0, weights=weights[:,k])
            sqerr = (x - self.mus[k])**2
            self._log_sigmasq[k] = np.log(np.average(sqerr, weights=weights[:,k], axis=0))

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
        self._log_sigmasq = -2 + npr.randn(K, D)
        # Student's t distribution also has a degrees of freedom parameter
        self._log_nus = np.log(4) * np.ones((K, D))

    @property
    def sigmasq(self):
        return np.exp(self._log_sigmasq)

    @property
    def nus(self):
        return np.exp(self._log_nus)

    @property
    def params(self):
        return self.mus, self._log_sigmasq, self._log_nus

    @params.setter
    def params(self, value):
        self.mus, self._log_sigmasq, self._log_nus = value

    def permute(self, perm):
        self.mus = self.mus[perm]
        self._log_sigmasq = self._log_sigmasq[perm]
        self._log_nus = self._log_nus[perm]

    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None):
        # Initialize with KMeans
        from sklearn.cluster import KMeans
        data = np.concatenate(datas)
        km = KMeans(self.K).fit(data)
        self.mus = km.cluster_centers_
        sigmas = np.array([np.var(data[km.labels_ == k], axis=0) for k in range(self.K)])
        self._log_sigmasq = np.log(sigmas + 1e-16)
        self._log_nus = np.log(4) * np.ones((self.K, self.D))

    def log_likelihoods(self, data, input, mask, tag):
        D, mus, sigmas, nus = self.D, self.mus, self.sigmasq, self.nus
        mask = np.ones_like(data, dtype=bool) if mask is None else mask
        return stats.independent_studentst_logpdf(data[:, None, :], mus, sigmas, nus, mask=mask[:, None, :])

    def sample_x(self, z, xhist, input=None, tag=None, with_noise=True):
        D, mus, sigmas, nus = self.D, self.mus, self.sigmasq, self.nus
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
            # nu: (K,D)  mus: (K, D)  sigmas: (K, D)  y: (T, D)  -> tau: (T, K, D)
            alpha = self.nus/2 + 1/2
            beta = self.nus/2 + 1/2 * (y[:, None, :] - self.mus)**2 / self.sigmasq
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
        self._log_sigmasq = np.log(sqerr / weight + 1e-16)

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
        E_taus = np.zeros((K, D))
        E_logtaus = np.zeros((K, D))
        weights = np.zeros(K)
        for y, (Ez, _, _) in zip(datas, expectations):
            # nu: (K,D)  mus: (K, D)  sigmas: (K, D)  y: (T, D)  -> alpha/beta: (T, K, D)
            alpha = self.nus/2 + 1/2
            beta = self.nus/2 + 1/2 * (y[:, None, :] - self.mus)**2 / self.sigmasq

            E_taus += np.sum(Ez[:, :, None] * (alpha / beta), axis=0)
            E_logtaus += np.sum(Ez[:, :, None] * (digamma(alpha) - np.log(beta)), axis=0)
            weights += np.sum(Ez, axis=0)

        E_taus /= weights[:, None]
        E_logtaus /= weights[:, None]

        for k in range(K):
            for d in range(D):
                self._log_nus[k, d] = np.log(generalized_newton_studentst_dof(E_taus[k, d], E_logtaus[k, d]))


class MultivariateStudentsTObservations(_Observations):
    def __init__(self, K, D, M=0):
        super(MultivariateStudentsTObservations, self).__init__(K, D, M)
        self.mus = npr.randn(K, D)
        self._sqrt_Sigmas = npr.randn(K, D, D)
        self._log_nus = np.log(4) * np.ones((K,))

    @property
    def Sigmas(self):
        return np.matmul(self._sqrt_Sigmas, np.swapaxes(self._sqrt_Sigmas, -1, -2))

    @property
    def nus(self):
        return np.exp(self._log_nus)

    @property
    def params(self):
        return self.mus, self._sqrt_Sigmas, self._log_nus

    @params.setter
    def params(self, value):
        self.mus, self._sqrt_Sigmas, self._log_nus = value

    def permute(self, perm):
        self.mus = self.mus[perm]
        self._sqrt_Sigmas = self._sqrt_Sigmas[perm]
        self._log_nus = self._log_nus[perm]

    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None):
        # Initialize with KMeans
        from sklearn.cluster import KMeans
        data = np.concatenate(datas)
        km = KMeans(self.K).fit(data)
        self.mus = km.cluster_centers_
        Sigmas = np.array([np.atleast_2d(np.cov(data[km.labels_ == k].T))
                           for k in range(self.K)])
        assert np.all(np.isfinite(Sigmas))
        self._sqrt_Sigmas = np.linalg.cholesky(Sigmas + 1e-8 * np.eye(self.D))
        self._log_nus = np.log(4) * np.ones((self.K,))

    def log_likelihoods(self, data, input, mask, tag):
        assert np.all(mask), "MultivariateStudentsTObservations does not support missing data"
        D, mus, Sigmas, nus = self.D, self.mus, self.Sigmas, self.nus
        return stats.multivariate_studentst_logpdf(data[:, None, :], mus, Sigmas, nus)

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
            # nu: (K,)  mus: (K, D)  Sigmas: (K, D, D)  y: (T, D)  -> tau: (T, K)
            alpha = self.nus/2 + D/2
            beta = self.nus/2 + 1/2 * stats.batch_mahalanobis(self._sqrt_Sigmas, y[:, None, :] - self.mus)
            E_taus.append(alpha / beta)

        # Update the mean (notation from natural params of Gaussian)
        J = np.zeros((K,))
        h = np.zeros((K, D))
        for E_tau, (Ez, _, _), y in zip(E_taus, expectations, datas):
            J += np.sum(Ez * E_tau, axis=0)
            h += np.sum(Ez[:, :, None] * E_tau[:, :, None] * y[:, None, :], axis=0)
        self.mus = h / J[:, None]

        # Update the variance
        sqerr = np.zeros((K, D, D))
        weight = np.zeros((K,))
        for E_tau, (Ez, _, _), y in zip(E_taus, expectations, datas):
            # sqerr += np.sum(Ez[:, :, None] * E_tau * (y[:, None, :] - self.mus)**2, axis=0)
            resid = y[:, None, :] - self.mus
            sqerr += np.einsum('tk,tk,tki,tkj->kij', Ez, E_tau, resid, resid)
            weight += np.sum(Ez, axis=0)

        self._sqrt_Sigmas = np.linalg.cholesky(sqerr / weight[:, None, None] + 1e-8 * np.eye(D))

    def _m_step_nu(self, expectations, datas, inputs, masks, tags):
        """
        The shape parameter nu determines a gamma prior.  We have

            tau_n ~ Gamma(nu/2, nu/2)
            y_n ~ N(mu, Sigma / tau_n)

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
            # nu: (K,)  mus: (K, D)  Sigmas: (K, D, D)  y: (T, D)  -> alpha/beta: (T, K)
            alpha = self.nus/2 + D/2
            beta = self.nus/2 + 1/2 * stats.batch_mahalanobis(self._sqrt_Sigmas, y[:, None, :] - self.mus)

            E_taus += np.sum(Ez * (alpha / beta), axis=0)
            E_logtaus += np.sum(Ez * (digamma(alpha) - np.log(beta)), axis=0)
            weights += np.sum(Ez, axis=0)

        E_taus /= weights
        E_logtaus /= weights

        for k in range(K):
            self._log_nus[k] = np.log(generalized_newton_studentst_dof(E_taus[k], E_logtaus[k]))

    def sample_x(self, z, xhist, input=None, tag=None, with_noise=True):
        D, mus, Sigmas, nus = self.D, self.mus, self.Sigmas, self.nus
        tau = npr.gamma(nus[z] / 2.0, 2.0 / nus[z])
        sqrt_Sigma = np.linalg.cholesky(Sigmas[z] / tau) if with_noise else 0
        return mus[z] + np.dot(sqrt_Sigma, npr.randn(D))

    def smooth(self, expectations, data, input, tag):
        """
        Compute the mean observation under the posterior distribution
        of latent discrete states.
        """
        return expectations.dot(self.mus)


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
        mask = np.ones_like(data, dtype=bool) if mask is None else mask
        return stats.bernoulli_logpdf(data[:, None, :], self.logit_ps, mask=mask[:, None, :])

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
        lambdas = np.exp(self.log_lambdas)
        mask = np.ones_like(data, dtype=bool) if mask is None else mask
        return stats.poisson_logpdf(data[:, None, :], lambdas, mask=mask[:, None, :])

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
        mask = np.ones_like(data, dtype=bool) if mask is None else mask
        return stats.categorical_logpdf(data[:, None, :], self.logits, mask=mask[:, None, :])

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


class _AutoRegressiveObservationsBase(_Observations):
    """
    Base class for autoregressive observations of the form,

    E[x_t | x_{t-1}, z_t=k, u_t]
        = \sum_{l=1}^{L} A_k^{(l)} x_{t-l} + b_k + V_k u_t.

    where L is the number of lags and u_t is the input.
    """
    def __init__(self, K, D, M=0, lags=1):
        super(_AutoRegressiveObservationsBase, self).__init__(K, D, M)

        # Distribution over initial point
        self.mu_init = np.zeros((K, D))

        # AR parameters
        assert lags > 0
        self.lags = lags
        self.bs = npr.randn(K, D)
        self.Vs = npr.randn(K, D, M)

        # Inheriting classes may treat _As differently
        self._As = None

    @property
    def As(self):
        return self._As

    @As.setter
    def As(self, value):
        self._As = value

    @property
    def params(self):
        return self.As, self.bs, self.Vs

    @params.setter
    def params(self, value):
        self.As, self.bs, self.Vs = value

    def permute(self, perm):
        self.mu_init = self.mu_init[perm]
        self.As = self.As[perm]
        self.bs = self.bs[perm]
        self.Vs = self.Vs[perm]

    def _compute_mus(self, data, input, mask, tag):
        assert np.all(mask), "ARHMM cannot handle missing data"
        T, D = data.shape
        As, bs, Vs = self.As, self.bs, self.Vs

        # Instantaneous inputs
        mus = np.matmul(Vs[None, ...], input[self.lags:, None, :self.M, None])[:, :, :, 0]

        # Lagged data
        for l in range(self.lags):
            Als = As[None, :, :, l*D:(l+1)*D]
            lagged_data = data[self.lags-l-1:-l-1, None, :, None]
            mus = mus + np.matmul(Als, lagged_data)[:, :, :, 0]

        # Bias
        mus = mus + bs

        # Pad with the initial condition
        mus = np.concatenate((self.mu_init * np.ones((self.lags, self.K, self.D)), mus))

        assert mus.shape == (T, self.K, D)
        return mus

    def smooth(self, expectations, data, input, tag):
        """
        Compute the mean observation under the posterior distribution
        of latent discrete states.
        """
        T = expectations.shape[0]
        mask = np.ones((T, self.D), dtype=bool)
        mus = self._compute_mus(data, input, mask, tag)
        return (expectations[:, :, None] * mus).sum(1)


class AutoRegressiveObservations(_AutoRegressiveObservationsBase):
    """
    AutoRegressive observation model with Gaussian noise.

        (x_t | z_t = k, u_t) ~ N(A_k x_{t-1} + b_k + V_k u_t, S_k)

    where S_k is a positive definite covariance matrix.

    The parameters are fit via maximum likelihood estimation.
    """
    def __init__(self, K, D, M=0, lags=1, regularization_params=None):

        super(AutoRegressiveObservations, self).\
            __init__(K, D, M, lags=lags)

        # Initialize the dynamics and the noise covariances
        self._As = .95 * np.array([
                np.column_stack([random_rotation(D), np.zeros((D, (lags-1) * D))])
            for _ in range(K)])

        self._sqrt_Sigmas_init = np.tile(np.eye(D)[None, ...], (K, 1, 1))
        self._sqrt_Sigmas = npr.randn(K, D, D)

        # Regularization penalties on A, b, and V
        if regularization_params is None:
            self.regularization_params = dict(type="l2", lambda_A=1e-8, lambda_V=1e-8)

        elif isinstance(regularization_params, dict):
            # TODO: Validate the dictionary key/value pairs
            reg_type = regularization_params["type"]
            if reg_type.lower() in ("l1","l2","l2_diff","group_lasso",'group_lasso_nodiag','con_group_lasso'):
                prms = dict(lambda_A=0, lambda_V=0)
                prms.update(regularization_params)

                self.regularization_params = prms


        else:
            raise Exception("regularization_params must be a None or a dictionary.")

    @property
    def Sigmas_init(self):
        return np.matmul(self._sqrt_Sigmas_init, np.swapaxes(self._sqrt_Sigmas_init, -1, -2))

    @property
    def Sigmas(self):
        return np.matmul(self._sqrt_Sigmas, np.swapaxes(self._sqrt_Sigmas, -1, -2))

    @Sigmas.setter
    def Sigmas(self, value):
        assert value.shape == (self.K, self.D, self.D)
        self._sqrt_Sigmas = np.linalg.cholesky(value)

    @property
    def params(self):
        return super(AutoRegressiveObservations, self).params + (self._sqrt_Sigmas,)

    @params.setter
    def params(self, value):
        self._sqrt_Sigmas = value[-1]
        super(AutoRegressiveObservations, self.__class__).params.fset(self, value[:-1])

    def permute(self, perm):
        super(AutoRegressiveObservations, self).permute(perm)
        self._sqrt_Sigmas = self._sqrt_Sigmas[perm]

    def initialize(self, datas, inputs=None, masks=None, tags=None, localize=False):
        from sklearn.linear_model import LinearRegression

        # Sample time bins for each discrete state
        # Use the data to cluster the time bins if specified.
        Ts = [data.shape[0] for data in datas]
        if localize:
            from sklearn.cluster import KMeans
            km = KMeans(self.K)
            km.fit(np.vstack(datas))
            zs = np.split(km.labels_, np.cumsum(Ts)[:-1])
            zs = [z[:-self.lags] for z in zs]               # remove the ends
        else:
            zs = [npr.choice(self.K, size=T-self.lags) for T in Ts]

        # Initialize the weights with linear regression
        Sigmas = []
        for k in range(self.K):
            ts = [np.where(z == k)[0] for z in zs]
            Xs = [np.column_stack([data[t + l] for l in range(self.lags)] + [input[t, :self.M]])
                  for t, data, input in zip(ts, datas, inputs)]
            ys = [data[t+self.lags] for t, data in zip(ts, datas)]

            # Solve the linear regression
            coef_, intercept_, Sigma = fit_linear_regression(Xs, ys)
            self.As[k] = coef_[:, :self.D * self.lags]
            self.Vs[k] = coef_[:, self.D * self.lags:]
            self.bs[k] = intercept_
            Sigmas.append(Sigma)

        # Set the variances all at once to use the setter
        self.Sigmas = np.array(Sigmas)

    def log_prior(self):
        if self.regularization_params is None:
            return 0

        As, Vs, K, D = self.As, self.Vs, self.K, self.D
        regularization_params = self.regularization_params
        reg_type = regularization_params["type"]

        if reg_type.lower() =='l1':
            lambda_A = -regularization_params["lambda_A"]
            return lambda_A * np.sum(np.abs(As - np.identity(D)))

        elif reg_type.lower() =='l2':
            lp = -regularization_params["lambda_A"] * np.sum((As)**2)
            lp += -regularization_params["lambda_V"] * np.sum(Vs**2)
            return lp

        elif reg_type.lower() =='l2_diff':
            lp = -regularization_params["lambda_A"] * np.sum((As - np.identity(D))**2)
            lp += -regularization_params["lambda_V"] * np.sum(Vs**2)
            return lp

        elif reg_type.lower() == 'group_lasso':
            lambda_A = -regularization_params["lambda_A"]
            D_vec = regularization_params["D_vec"]
            P = len(D_vec) #Number of populations
            D_vec_cumsum = np.concatenate(([0], np.cumsum(D_vec)))
            group_sums = [np.sqrt(D_vec[i] * D_vec[j]) * np.linalg.norm((As[k]-np.identity(D))[D_vec_cumsum[i]:D_vec_cumsum[i+1],D_vec_cumsum[j]:D_vec_cumsum[j+1]]) for i in range(P) for j in range(P) for k in range(K)]
            return lambda_A*np.sum(group_sums)

        elif reg_type.lower() == 'group_lasso_nodiag':
            lambda_A = -regularization_params["lambda_A"]
            D_vec = regularization_params["D_vec"]
            P = len(D_vec) #Number of populations
            D_vec_cumsum = np.concatenate(([0], np.cumsum(D_vec)))
            group_sums = [np.sqrt(D_vec[i] * D_vec[j]) * np.linalg.norm((As[k]-np.diag(np.diag(As[k])))[D_vec_cumsum[i]:D_vec_cumsum[i+1],D_vec_cumsum[j]:D_vec_cumsum[j+1]]) for i in range(P) for j in range(P) for k in range(K)]
            return lambda_A*np.sum(group_sums)

        elif reg_type.lower() == 'con_group_lasso':
            lambda_A = -regularization_params["lambda_A"]
            D_vec = regularization_params["D_vec"]
            W_inv = regularization_params["W_inv"]
            P=len(D_vec)
            D_vec_cumsum=np.concatenate(([0],np.cumsum(D_vec)))
            group_sums = [W_inv[i,j]*np.sqrt(D_vec[i]*D_vec[j])*np.linalg.norm((As[k]-np.identity(D))[D_vec_cumsum[i]:D_vec_cumsum[i+1],D_vec_cumsum[j]:D_vec_cumsum[j+1]]) for i in range(P) for j in range(P) for k in range(K)]
            return lambda_A*np.sum(group_sums)

    #     elif reg_type.lower() == 'nuclear':
    #         norms = [np.linalg.norm((self.As[k]-np.identity(self.D)),'nuc') for k in range(self.K)]
    #         return alpha*np.sum(norms)

    #     elif reg_type.lower() == 'group_nuclear':
    #         D_vec=self.D_vec
    #         P=len(D_vec)
    #         D_vec_cumsum=np.concatenate(([0],np.cumsum(D_vec)))
    #         group_norms = [np.linalg.norm((self.As[k]-np.identity(self.D))[D_vec_cumsum[i]:D_vec_cumsum[i+1],D_vec_cumsum[j]:D_vec_cumsum[j+1]],'nuc') for i in range(P) for j in range(P) for k in range(self.K)]
    #         return alpha*np.sum(group_norms)

    #     elif reg_type.lower() =='con_l1':
    #         return alpha*np.sum(np.abs((self.W_inv*(self.As-np.identity(self.D)))[:]))

    #     elif reg_type.lower() == 'con_l2':
    #         return alpha*np.sum((self.W_inv*(self.As-np.identity(self.D)))[:]**2)


    #     elif reg_type.lower() == 'con_matching_l2':
    #         return alpha*np.sum((self.As-self.W)[:]**2)

    #     elif reg_type.lower() =='con_matching_l1':
    #         return alpha*np.sum(np.abs((self.As-self.W)[:]))

    #     else:
    #         #Error: This is not a valid regularization type
    #         raise NotImplementedError('{} is not a valid regularization type'.format(reg_type))


    def log_likelihoods(self, data, input, mask, tag=None):
        assert np.all(mask), "Cannot compute likelihood of autoregressive obsevations with missing data."
        L = self.lags
        mus = self._compute_mus(data, input, mask, tag)

        # Compute the likelihood of the initial data and remainder separately
        ll_init = stats.multivariate_normal_logpdf(data[:L, None, :], mus[:L], self.Sigmas_init)
        ll_ar = stats.multivariate_normal_logpdf(data[L:, None, :], mus[L:], self.Sigmas)
        return np.row_stack((ll_init, ll_ar))

    def m_step(self, expectations, datas, inputs, masks, tags, J0=None, h0=None):
        K, D, M, lags = self.K, self.D, self.M, self.lags

        i_flag=0
        if self.regularization_params["type"].lower() in ("l2_diff",):
            i_flag=1

        # Collect all the data
        xs, ys, Ezs = [], [], []
        for (Ez, _, _), data, input, mask, tag in zip(expectations, datas, inputs, masks, tags):
            # Only use data if it is complete
            if not np.all(mask):
                raise Exception("Encountered missing data in AutoRegressiveObservations!")

            xs.append(
                np.hstack([data[self.lags-l-1:-l-1] for l in range(self.lags)]
                          + [input[self.lags:, :self.M], np.ones((data.shape[0]-self.lags, 1))]))
            if i_flag==0:
                ys.append(data[self.lags:])
            if i_flag==1:
                ys.append(data[self.lags:]-data[self.lags-1:-1])
            Ezs.append(Ez[self.lags:])

        # M step: Fit the weighted linear regressions for each K and D
        if not self.regularization_params["type"].lower() in ("l2","l2_diff"):
            warnings.warn("M-step only supports L2 regularization for now.")

        if J0 is None and h0 is None:
            lambda_A = self.regularization_params["lambda_A"]
            lambda_V = self.regularization_params["lambda_V"]
            J_diag = np.concatenate((lambda_A * np.ones(D * lags),
                                     lambda_V * np.ones(M),
                                     1e-8 * np.ones(1)))
            J = np.tile(np.diag(J_diag)[None, :, :], (K, 1, 1))
            h = np.zeros((K, D * lags + M + 1, D))
        else:
            assert J0.shape == (K, D*lags + M + 1, D*lags + M + 1)
            assert h0.shape == (K, D*lags + M + 1, D)
            J = J0
            h = h0

        for x, y, Ez in zip(xs, ys, Ezs):
            # Einsum is concise but slow!
            # J += np.einsum('tk, ti, tj -> kij', Ez, x, x)
            # h += np.einsum('tk, ti, td -> kid', Ez, x, y)
            # Do weighted products for each of the k states
            for k in range(K):
                weighted_x = x * Ez[:, k:k+1]
                J[k] += np.dot(weighted_x.T, x)
                h[k] += np.dot(weighted_x.T, y)

        mus = np.linalg.solve(J, h)

        self.As = np.swapaxes(mus[:, :D*lags, :], 1, 2)
        if i_flag==1:
            self.As=self.As+np.identity(D)

        self.Vs = np.swapaxes(mus[:, D*lags:D*lags+M, :], 1, 2)
        self.bs = mus[:, -1, :]

        # Update the covariance
        sqerr = np.zeros((K, D, D))
        weight = 1e-8 * np.ones(K)
        for x, y, Ez in zip(xs, ys, Ezs):
            yhat = np.matmul(x[None, :, :], mus)
            resid = y[None, :, :] - yhat
            sqerr += np.einsum('tk,kti,ktj->kij', Ez, resid, resid)
            weight += np.sum(Ez, axis=0)

        self.Sigmas = sqerr / weight[:, None, None] + 1e-8 * np.eye(D)

    def sample_x(self, z, xhist, input=None, tag=None, with_noise=True):
        D, As, bs, Vs = self.D, self.As, self.bs, self.Vs

        if xhist.shape[0] < self.lags:
            # Sample from the initial distribution
            S = np.linalg.cholesky(self.Sigmas_init[z]) if with_noise else 0
            return self.mu_init[z] + np.dot(S, npr.randn(D))
        else:
            # Sample from the autoregressive distribution
            mu = Vs[z].dot(input[:self.M]) + bs[z]
            for l in range(self.lags):
                Al = As[z][:,l*D:(l+1)*D]
                mu += Al.dot(xhist[-l-1])

            S = np.linalg.cholesky(self.Sigmas[z]) if with_noise else 0
            return mu + np.dot(S, npr.randn(D))


class AutoRegressiveObservationsNoInput(AutoRegressiveObservations):
    """
    AutoRegressive observation model without the inputs.
    """
    def __init__(self, K, D, M=0, lags=1, regularization_params=None):

        super(AutoRegressiveObservationsNoInput, self).\
            __init__(K, D, M=0, lags=lags, regularization_params=regularization_params)


class AutoRegressiveDiagonalNoiseObservations(AutoRegressiveObservations):
    """
    AutoRegressive observation model with diagonal Gaussian noise.

        (x_t | z_t = k, u_t) ~ N(A_k x_{t-1} + b_k + V_k u_t, S_k)

    where

        S_k = diag([sigma_{k,1}, ..., sigma_{k, D}])

    The parameters are fit via maximum likelihood estimation.
    """
    def __init__(self, K, D, M=0, lags=1, regularization_params=None):

        super(AutoRegressiveDiagonalNoiseObservations, self).\
            __init__(K, D, M, lags=lags, regularization_params=regularization_params)

        # Initialize the dynamics and the noise covariances
        self._As = .95 * np.array([
                np.column_stack([random_rotation(D), np.zeros((D, (lags-1) * D))])
            for _ in range(K)])

        # Get rid of the square root parameterization and replace with log diagonal
        del self._sqrt_Sigmas_init
        del self._sqrt_Sigmas
        self._log_sigmasq_init = np.zeros((K, D))
        self._log_sigmasq = np.zeros((K, D))

    @property
    def sigmasq_init(self):
        return np.exp(self._log_sigmasq_init)

    @sigmasq_init.setter
    def sigmasq_init(self, value):
        assert value.shape == (self.K, self.D)
        assert np.all(value > 0)
        self._log_sigmasq_init = np.log(value)

    @property
    def sigmasq(self):
        return np.exp(self._log_sigmasq)

    @sigmasq.setter
    def sigmasq(self, value):
        assert value.shape == (self.K, self.D)
        assert np.all(value > 0)
        self._log_sigmasq = np.log(value)

    @property
    def Sigmas_init(self):
        return np.array([np.diag(np.exp(log_s)) for log_s in self._log_sigmasq_init])

    @property
    def Sigmas(self):
        return np.array([np.diag(np.exp(log_s)) for log_s in self._log_sigmasq])

    @Sigmas.setter
    def Sigmas(self, value):
        assert value.shape == (self.K, self.D, self.D)
        sigmasq = np.array([np.diag(S) for S in value])
        assert np.all(sigmasq > 0)
        self._log_sigmasq = np.log(sigmasq)

    @property
    def params(self):
        return super(AutoRegressiveObservations, self).params + (self._log_sigmasq,)

    @params.setter
    def params(self, value):
        self._log_sigmasq = value[-1]
        super(AutoRegressiveObservations, self.__class__).params.fset(self, value[:-1])

    def permute(self, perm):
        super(AutoRegressiveObservations, self).permute(perm)
        self._log_sigmasq_init = self._log_sigmasq_init[perm]
        self._log_sigmasq = self._log_sigmasq[perm]

    def log_likelihoods(self, data, input, mask, tag):
        assert np.all(mask), "Cannot compute likelihood of autoregressive obsevations with missing data."
        mus = self._compute_mus(data, input, mask, tag)

        # Compute the likelihood of the initial data and remainder separately
        L = self.lags
        ll_init = stats.diagonal_gaussian_logpdf(data[:L, None, :], mus[:L], self.sigmasq_init)
        ll_ar = stats.diagonal_gaussian_logpdf(data[L:, None, :], mus[L:], self.sigmasq)
        return np.row_stack((ll_init, ll_ar))


class IdentityAutoRegressiveObservations(_Observations):
    pass

    def __init__(self, K, D, M):
        super(IdentityAutoRegressiveObservations, self).__init__(K, D, M)
        self.inv_sigmas = -2 + npr.randn(K, D) #inv_sigma is the log of the variance
        self.inv_sigma_init= np.zeros(D) #inv_sigma_init is the log of the variance of the initial data point
        self.K=K
        self.D=D

    @property
    def params(self):
        return self.inv_sigmas

    @params.setter
    def params(self, value):
        self.inv_sigmas = value

    def permute(self, perm):
        self.inv_sigmas = self.inv_sigmas[perm]

    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None):
        # sigmas=np.var(datas[1:]-datas[:-1])
        self.inv_sigmas = -2 + npr.randn(self.K, self.D)#np.log(sigmas + 1e-16)

    # def _compute_sigmas(self, data, input, mask, tag):
    #     T, D = data.shape
    #     inv_sigmas = self.inv_sigmas
    #
    #     sigma_init = np.exp(self.inv_sigma_init) * np.ones((self.lags, self.K, self.D))
    #     sigma_ar = np.repeat(np.exp(inv_sigmas)[None, :, :], T-self.lags, axis=0)
    #     sigmas = np.concatenate((sigma_init, sigma_ar))
    #     assert sigmas.shape == (T, self.K, D)
    #     return sigmas
    #
    def log_likelihoods(self, data, input, mask, tag):
        sigmas = np.exp(self.inv_sigmas) + 1e-16
        sigma_init=np.exp(self.inv_sigma_init)+1e-16

        #Log likelihood of data (except for first point)
        ll1 = -0.5 * np.sum(
            (np.log(2 * np.pi * sigmas) + (data[1:, None, :] - data[:-1, None, :])**2 / sigmas)
            * mask[1:, None, :], axis=2)

        # ll2= -0.5 * np.sum(
        #     (np.log(2 * np.pi * sigma_init) + (data[0, None, :]) / sigma_init), axis=1) #Make 0 instead of data[0...]???

        #Log likelihood of first point (just setting to 0, since we're not fitting sigma_init)
        ll2=np.array([0])

        #Log likelihood of all data points (including first)
        # print(sigmas.shape)
        # print(data.shape)
        # print(ll1.shape)
        # print(ll2.shape)
        ll=np.concatenate((ll1,ll2[:,np.newaxis]),axis=0)



        # print(ll.shape)
        return ll

        # return -0.5 * np.sum(
        #     (np.log(2 * np.pi * sigmas) + (data[1:, None, :] - data[:-1, None, :])**2 / sigmas)
        #     * mask[1:, None, :], axis=2)

        # ll_init = stats.diagonal_gaussian_logpdf(data[0, None, :], 0, self.inv_sigma_init)
        # ll_ar = stats.diagonal_gaussian_logpdf(data[1:, None, :], data[0:-1, :], self.inv_sigma)
        # return np.row_stack((ll_init, ll_ar))


    def sample_x(self, z, xhist, input=None, tag=None, with_noise=True):
        D = self.D
        sigmas = np.exp(self.inv_sigmas) if with_noise else np.zeros((self.K, self.D))
        sigma_init = np.exp(self.inv_sigma_init) if with_noise else 0

        if xhist.shape[0] == 0:
            return np.sqrt(sigma_init[z]) * npr.randn(D)
        else:
            return xhist[-1] + np.sqrt(sigmas[z]) * npr.randn(D)

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
        x = np.concatenate(datas)
        weights = np.concatenate([Ez for Ez, _, _ in expectations])
        for k in range(self.K):
            # self.mus[k] = np.average(x, axis=0, weights=weights[:,k])
            sqerr = (x[1:] - x[:-1])**2
            d2=np.average(sqerr, weights=weights[1:,k], axis=0)


            self.inv_sigmas[k] = np.log(d2)

            #
            # print("is",self.inv_sigmas)

    def smooth(self, expectations, data, input, tag):
        """
        Compute the mean observation under the posterior distribution
        of latent discrete states.
        """
        raise NotImplementedError
        # return expectations.dot(data)




class IndependentAutoRegressiveObservations(_AutoRegressiveObservationsBase):
    def __init__(self, K, D, M=0, lags=1):
        super(IndependentAutoRegressiveObservations, self).__init__(K, D, M, lags=lags)

        self._As = np.concatenate((.95 * np.ones((K, D, 1)), np.zeros((K, D, lags-1))), axis=2)
        self._log_sigmasq_init = np.zeros((K, D))
        self._log_sigmasq = np.zeros((K, D))

    @property
    def sigmasq_init(self):
        return np.exp(self._log_sigmasq_init)

    @property
    def sigmasq(self):
        return np.exp(self._log_sigmasq)

    @property
    def As(self):
        return np.array([
                np.column_stack([np.diag(Ak[:,l]) for l in range(self.lags)])
            for Ak in self._As
        ])

    @As.setter
    def As(self, value):
        # TODO: extract the diagonal components
        raise NotImplementedError

    @property
    def params(self):
        return self._As, self.bs, self.Vs, self._log_sigmasq

    @params.setter
    def params(self, value):
        self._As, self.bs, self.Vs, self._log_sigmasq = value

    def permute(self, perm):
        self.mu_init = self.mu_init[perm]
        self._As = self._As[perm]
        self.bs = self.bs[perm]
        self.Vs = self.Vs[perm]
        self._log_sigmasq_init = self._log_sigmasq_init[perm]
        self._log_sigmasq = self._log_sigmasq[perm]

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
                self._log_sigmasq[k, d] = np.log(sigmas + 1e-16)

    def _compute_mus(self, data, input, mask, tag):
        """
        Re-implement compute_mus for this class since we can do it much
        more efficiently than in the general AR case.
        """
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

    def log_likelihoods(self, data, input, mask, tag):
        mus = self._compute_mus(data, input, mask, tag)

        # Compute the likelihood of the initial data and remainder separately
        L = self.lags
        ll_init = stats.diagonal_gaussian_logpdf(data[:L, None, :], mus[:L], self.sigmasq_init)
        ll_ar = stats.diagonal_gaussian_logpdf(data[L:, None, :], mus[L:], self.sigmasq)
        return np.row_stack((ll_init, ll_ar))

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
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
                    self._log_sigmasq[k, d] = 0
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
                self._log_sigmasq[k, d] = np.log(sigma)

    def sample_x(self, z, xhist, input=None, tag=None, with_noise=True):
        D, As, bs, sigmas = self.D, self.As, self.bs, self.sigmasq

        # Sample the initial condition
        if xhist.shape[0] < self.lags:
            sigma_init = self.sigmasq_init[z] if with_noise else 0
            return self.mu_init[z] + np.sqrt(sigma_init) * npr.randn(D)

        # Otherwise sample the AR model
        muz = bs[z].copy()
        for lag in range(self.lags):
            muz += As[z, :, lag] * xhist[-lag - 1]

        sigma = sigmas[z] if with_noise else 0
        return muz + np.sqrt(sigma) * npr.randn(D)


# Robust autoregressive models with diagonal Student's t noise
class _RobustAutoRegressiveObservationsMixin(object):
    """
    Mixin for AR models where the noise is distributed according to a
    multivariate t distribution,

        epsilon ~ t(0, Sigma, nu)

    which is equivalent to,

        tau ~ Gamma(nu/2, nu/2)
        epsilon | tau ~ N(0, Sigma / tau)

    We use this equivalence to perform the M step (update of Sigma and tau)
    via an inner expectation maximization algorithm.

    This mixin mus be used in conjunction with either AutoRegressiveObservations or
    AutoRegressiveDiagonalNoiseObservations, which provides the parameterization for
    Sigma.  The mixin does not capitalize on structure in Sigma, so it will pay
    a small complexity penalty when used in conjunction with the diagonal noise model.
    """
    def __init__(self, K, D, M=0, lags=1, regularization_params=None):

        super(_RobustAutoRegressiveObservationsMixin, self).\
            __init__(K, D, M=M, lags=lags, regularization_params=regularization_params)
        self._log_nus = np.log(4) * np.ones(K)

    @property
    def nus(self):
        return np.exp(self._log_nus)

    @property
    def params(self):
        return super(_RobustAutoRegressiveObservationsMixin, self).params + (self._log_nus,)

    @params.setter
    def params(self, value):
        self._log_nus = value[-1]
        super(_RobustAutoRegressiveObservationsMixin, self.__class__).params.fset(self, value[:-1])

    def permute(self, perm):
        super(_RobustAutoRegressiveObservationsMixin, self).permute(perm)
        self._log_nus = self._log_nus[perm]

    def log_likelihoods(self, data, input, mask, tag):
        assert np.all(mask), "Cannot compute likelihood of autoregressive obsevations with missing data."
        mus = self._compute_mus(data, input, mask, tag)

        # Compute the likelihood of the initial data and remainder separately
        L = self.lags
        ll_init = stats.multivariate_normal_logpdf(data[:L, None, :], mus[:L], self.Sigmas_init)
        ll_ar = stats.multivariate_studentst_logpdf(data[L:, None, :], mus[L:], self.Sigmas, self.nus)
        return np.row_stack((ll_init, ll_ar))

    def m_step(self, expectations, datas, inputs, masks, tags, num_em_iters=1, J0=None, h0=None):
        """
        Student's t is a scale mixture of Gaussians.  We can estimate its
        parameters using the EM algorithm. See the notebook in doc/students_t
        for complete details.
        """
        self._m_step_ar(expectations, datas, inputs, masks, tags, num_em_iters, J0=J0, h0=h0)
        self._m_step_nu(expectations, datas, inputs, masks, tags)

    def _m_step_ar(self, expectations, datas, inputs, masks, tags, num_em_iters, J0=None, h0=None):
        K, D, M, lags = self.K, self.D, self.M, self.lags

        # Collect data for this dimension
        xs, ys, Ezs = [], [], []
        for (Ez, _, _), data, input, mask, tag in zip(expectations, datas, inputs, masks, tags):
            # Only use data if it is complete
            if not np.all(mask):
                raise Exception("Encountered missing data in AutoRegressiveObservations!")

            xs.append(
                np.hstack([data[lags-l-1:-l-1] for l in range(lags)]
                          + [input[lags:, :self.M], np.ones((data.shape[0]-lags, 1))]))
            ys.append(data[lags:])
            Ezs.append(Ez[lags:])

        for itr in range(num_em_iters):
            # E Step: compute expected precision for each data point given current parameters
            taus = []
            for x, y in zip(xs, ys):
                Afull = np.concatenate((self.As, self.Vs, self.bs[:, :, None]), axis=2)
                mus = np.matmul(Afull[None, :, :, :], x[:, None, :, None])[:, :, :, 0]

                # nu: (K,)  mus: (T, K, D)  sigmas: (K, D, D)  y: (T, D)  -> tau: (T, K)
                alpha = self.nus / 2 + D/2
                sqrt_Sigmas = np.linalg.cholesky(self.Sigmas)
                beta = self.nus / 2 + 1/2 * stats.batch_mahalanobis(sqrt_Sigmas, y[:, None, :] - mus)
                taus.append(alpha / beta)

            # M step: Fit the weighted linear regressions for each K and D
            # This is exactly the same as the M-step for the AutoRegressiveObservations,
            # but it has an extra scaling factor of tau applied to the weight.
            if not self.regularization_params["type"].lower() in ("l2",):
                warnings.warn("M-step only supports L2 regularization for now.")

            if J0 is None and h0 is None:
                lambda_A = self.regularization_params["lambda_A"]
                lambda_V = self.regularization_params["lambda_V"]
                J_diag = np.concatenate((lambda_A * np.ones(D * lags),
                                         lambda_V * np.ones(M),
                                         1e-8 * np.ones(1)))
                J = np.tile(np.diag(J_diag)[None, :, :], (K, 1, 1))
                h = np.zeros((K, D * lags + M + 1, D))
            else:
                assert J0.shape == (K, D*lags + M + 1, D*lags + M + 1)
                assert h0.shape == (K, D*lags + M + 1, D)
                J = J0
                h = h0

            for x, y, Ez, tau in zip(xs, ys, Ezs, taus):
                weight = Ez * tau
                # Einsum is concise but slow!
                # J += np.einsum('tk, ti, tj -> kij', weight, x, x)
                # h += np.einsum('tk, ti, td -> kid', weight, x, y)
                # Do weighted products for each of the k states
                for k in range(K):
                    weighted_x = x * weight[:, k:k+1]
                    J[k] += np.dot(weighted_x.T, x)
                    h[k] += np.dot(weighted_x.T, y)

            mus = np.linalg.solve(J, h)
            self.As = np.swapaxes(mus[:, :D*lags, :], 1, 2)
            self.Vs = np.swapaxes(mus[:, D*lags:D*lags+M, :], 1, 2)
            self.bs = mus[:, -1, :]

            # Update the covariance
            sqerr = np.zeros((K, D, D))
            weight = np.zeros(K)
            for x, y, Ez, tau in zip(xs, ys, Ezs, taus):
                yhat = np.matmul(x[None, :, :], mus)
                resid = y[None, :, :] - yhat
                sqerr += np.einsum('tk,kti,ktj->kij', Ez * tau, resid, resid)
                weight += np.sum(Ez, axis=0)

            self.Sigmas = sqerr / weight[:, None, None] + 1e-8 * np.eye(D)

    def _m_step_nu(self, expectations, datas, inputs, masks, tags):
        """
        Update the degrees of freedom parameter of the multivariate t distribution
        using a generalized Newton update. See notes in the ssm repo.
        """
        K, D, L = self.K, self.D, self.lags
        E_taus = np.zeros(K)
        E_logtaus = np.zeros(K)
        weights = np.zeros(K)
        for (Ez, _, _,), data, input, mask, tag in zip(expectations, datas, inputs, masks, tags):
            # nu: (K,)  mus: (K, D)  Sigmas: (K, D, D)  y: (T, D)  -> tau: (T, K)
            mus = self._compute_mus(data, input, mask, tag)
            alpha = self.nus/2 + D/2
            sqrt_Sigma = np.linalg.cholesky(self.Sigmas)
            beta = self.nus/2 + 1/2 * stats.batch_mahalanobis(sqrt_Sigma, data[L:, None, :] - mus[L:])

            E_taus += np.sum(Ez[L:, :] * alpha / beta, axis=0)
            E_logtaus += np.sum(Ez[L:, :] * (digamma(alpha) - np.log(beta)), axis=0)
            weights += np.sum(Ez, axis=0)

        E_taus /= weights
        E_logtaus /= weights

        for k in range(K):
            self._log_nus[k] = np.log(generalized_newton_studentst_dof(E_taus[k], E_logtaus[k]))

    def sample_x(self, z, xhist, input=None, tag=None, with_noise=True):
        D, As, bs, Vs, Sigmas, nus = self.D, self.As, self.bs, self.Vs, self.Sigmas, self.nus
        if xhist.shape[0] < self.lags:
            S = np.linalg.cholesky(self.Sigmas_init[z]) if with_noise else 0
            return self.mu_init[z] + np.dot(S, npr.randn(D))
        else:
            mu = Vs[z].dot(input[:self.M]) + bs[z]
            for l in range(self.lags):
                mu += As[z][:,l*D:(l+1)*D].dot(xhist[-l-1])

            tau = npr.gamma(nus[z] / 2.0, 2.0 / nus[z])
            S = np.linalg.cholesky(Sigmas[z] / tau) if with_noise else 0
            return mu + np.dot(S, npr.randn(D))


class RobustAutoRegressiveObservations(_RobustAutoRegressiveObservationsMixin, AutoRegressiveObservations):
    """
    AR model where the noise is distributed according to a multivariate t distribution,

        epsilon ~ t(0, Sigma, nu)

    which is equivalent to,

        tau ~ Gamma(nu/2, nu/2)
        epsilon | tau ~ N(0, Sigma / tau)

    Here, Sigma is a general covariance matrix.
    """
    pass


class RobustAutoRegressiveObservationsNoInput(RobustAutoRegressiveObservations):
    """
    RobusAutoRegressiveObservations model without the inputs.
    """
    def __init__(self, K, D, M=0, lags=1, regularization_params=None):

        super(RobustAutoRegressiveObservationsNoInput, self).\
            __init__(K, D, M=0, lags=lags, regularization_params=regularization_params)



class RobustAutoRegressiveDiagonalNoiseObservations(
    _RobustAutoRegressiveObservationsMixin, AutoRegressiveDiagonalNoiseObservations):
    """
    AR model where the noise is distributed according to a multivariate t distribution,

        epsilon ~ t(0, Sigma, nu)

    which is equivalent to,

        tau ~ Gamma(nu/2, nu/2)
        epsilon | tau ~ N(0, Sigma / tau)

    Here, Sigma is a diagonal covariance matrix.
    """
    pass

# Robust autoregressive models with diagonal Student's t noise
class AltRobustAutoRegressiveDiagonalNoiseObservations(AutoRegressiveDiagonalNoiseObservations):
    """
    An alternative formulation of the robust AR model where the noise is
    distributed according to a independent scalar t distribution,

    For each output dimension d,

        epsilon_d ~ t(0, sigma_d^2, nu_d)

    which is equivalent to,

        tau_d ~ Gamma(nu_d/2, nu_d/2)
        epsilon_d | tau_d ~ N(0, sigma_d^2 / tau_d)

    """
    def __init__(self, K, D, M=0, lags=1):
        super(AltRobustAutoRegressiveDiagonalNoiseObservations, self).__init__(K, D, M=M, lags=lags)
        self._log_nus = np.log(4) * np.ones((K, D))

    @property
    def nus(self):
        return np.exp(self._log_nus)

    @property
    def params(self):
        return self.As, self.bs, self.Vs, self._log_sigmasq, self._log_nus

    @params.setter
    def params(self, value):
        self.As, self.bs, self.Vs, self._log_sigmasq, self._log_nus = value

    def permute(self, perm):
        super(AltRobustAutoRegressiveDiagonalNoiseObservations, self).permute(perm)
        self.inv_nus = self.inv_nus[perm]

    def log_likelihoods(self, data, input, mask, tag):
        assert np.all(mask), "Cannot compute likelihood of autoregressive obsevations with missing data."
        mus = self._compute_mus(data, input, mask, tag)

        # Compute the likelihood of the initial data and remainder separately
        L = self.lags
        ll_init = stats.diagonal_gaussian_logpdf(data[:L, None, :], mus[:L], self.sigmasq_init)
        ll_ar = stats.independent_studentst_logpdf(data[L:, None, :], mus[L:], self.sigmasq, self.nus)
        return np.row_stack((ll_init, ll_ar))

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
                np.hstack([data[lags-l-1:-l-1] for l in range(lags)]
                          + [input[lags:, :self.M], np.ones((data.shape[0]-lags, 1))]))
            ys.append(data[lags:])
            Ezs.append(Ez[lags:])

        for itr in range(num_em_iters):
            # E Step: compute expected precision for each data point given current parameters
            taus = []
            for x, y in zip(xs, ys):
                # mus = self._compute_mus(data, input, mask, tag)
                # sigmas = self._compute_sigmas(data, input, mask, tag)
                Afull = np.concatenate((self.As, self.Vs, self.bs[:, :, None]), axis=2)
                mus = np.matmul(Afull[None, :, :, :], x[:, None, :, None])[:, :, :, 0]

                # nu: (K,D)  mus: (T, K, D)  sigmas: (K, D)  y: (T, D)  -> tau: (T, K, D)
                alpha = self.nus / 2 + 1/2
                beta = self.nus / 2 + 1/2 * (y[:, None, :] - mus)**2 / self.sigmasq
                taus.append(alpha / beta)

            # M step: Fit the weighted linear regressions for each K and D
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
            self._log_sigmasq = np.log(sqerr / weight[:, None] + 1e-16)

    def _m_step_nu(self, expectations, datas, inputs, masks, tags, optimizer, num_iters, **kwargs):
        K, D, L = self.K, self.D, self.lags
        E_taus = np.zeros((K, D))
        E_logtaus = np.zeros((K, D))
        weights = np.zeros(K)
        for (Ez, _, _,), data, input, mask, tag in zip(expectations, datas, inputs, masks, tags):
            # nu: (K,D)  mus: (K, D)  sigmas: (K, D)  y: (T, D)  -> w: (T, K, D)
            mus = self._compute_mus(data, input, mask, tag)
            alpha = self.nus/2 + 1/2
            beta = self.nus/2 + 1/2 * (data[L:, None, :] - mus[L:])**2 / self.sigmasq
            E_taus += np.sum(Ez[L:, :, None] * alpha / beta, axis=0)
            E_logtaus += np.sum(Ez[L:, :, None] * (digamma(alpha) - np.log(beta)), axis=0)
            weights += np.sum(Ez, axis=0)

        E_taus /= weights[:, None]
        E_logtaus /= weights[:, None]

        for k in range(K):
            for d in range(D):
                self._log_nus[k, d] = np.log(generalized_newton_studentst_dof(E_taus[k, d], E_logtaus[k, d]))

    def sample_x(self, z, xhist, input=None, tag=None, with_noise=True):
        D, As, bs, sigmasq, nus = self.D, self.As, self.bs, self.sigmasq, self.nus
        if xhist.shape[0] < self.lags:
            sigma_init = self.sigmasq_init[z] if with_noise else 0
            return self.mu_init[z] + np.sqrt(sigma_init) * npr.randn(D)
        else:
            mu = bs[z].copy()
            for l in range(self.lags):
                mu += As[z][:,l*D:(l+1)*D].dot(xhist[-l-1])

            tau = npr.gamma(nus[z] / 2.0, 2.0 / nus[z])
            var = sigmasq[z] / tau if with_noise else 0
            return mu + np.sqrt(var) * npr.randn(D)


class VonMisesObservations(_Observations):
    def __init__(self, K, D, M=0):
        super(VonMisesObservations, self).__init__(K, D, M)
        self.mus = npr.randn(K, D)
        self.log_kappas = np.log(-1*npr.uniform(low=-1, high=0, size=(K, D)))

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
        mus, kappas = self.mus, np.exp(self.log_kappas)

        mask = np.ones_like(data, dtype=bool) if mask is None else mask
        return stats.vonmises_logpdf(data[:, None, :], mus, kappas, mask=mask[:, None, :])

    def sample_x(self, z, xhist, input=None, tag=None, with_noise=True):
        D, mus, kappas = self.D, self.mus, np.exp(self.log_kappas)
        return npr.vonmises(self.mus[z], kappas[z], D)

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):

        x = np.concatenate(datas)
        weights = np.concatenate([Ez for Ez, _, _ in expectations])  # T x D
        assert x.shape[0] == weights.shape[0]

        # convert angles to 2D representation and employ closed form solutions
        x_k = np.stack((np.sin(x), np.cos(x)), axis=1)  # T x 2 x D

        r_k = np.tensordot(weights.T, x_k, axes=1)  # K x 2 x D
        r_norm = np.sqrt(np.sum(np.power(r_k, 2), axis=1))  # K x D

        mus_k = np.divide(r_k, r_norm[:, None])  # K x 2 x D
        r_bar = np.divide(r_norm, np.sum(weights, 0)[:, None])  # K x D

        mask = (r_norm.sum(1) == 0)
        mus_k[mask] = 0
        r_bar[mask] = 0

        # Approximation
        kappa0 = r_bar * (self.D + 1 - np.power(r_bar, 2)) / (1 - np.power(r_bar, 2))  # K,D

        kappa0[kappa0 == 0] += 1e-6

        for k in range(self.K):
            self.mus[k] = np.arctan2(*mus_k[k])  #
            self.log_kappas[k] = np.log(kappa0[k])  # K, D

    def smooth(self, expectations, data, input, tag):
        mus = self.mus
        return expectations.dot(mus)
