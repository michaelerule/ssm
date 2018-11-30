import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.special import gammaln

from ssm.util import ensure_args_are_lists, ensure_args_not_none, \
    ensure_slds_args_not_none, logistic, logit, softplus, inv_softplus
from ssm.preprocessing import interpolate_data, pca_with_imputation


# Observation models for SLDS
class _Emissions(object):
    def __init__(self, N, K, D, M=0, single_subspace=True):
        self.N, self.K, self.D, self.M, self.single_subspace = \
            N, K, D, M, single_subspace

    @property
    def params(self):
        raise NotImplementedError

    @params.setter
    def params(self, value):
        raise NotImplementedError

    def permute(self, perm):
        pass

    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None, num_em_iters=25):
        pass

    def initialize_from_arhmm(self, arhmm, pca):
        pass

    def log_prior(self):
        return 0

    def log_likelihoods(self, data, input, mask, tag, x):
        raise NotImplementedError

    def forward(self, x, input=None, tag=None):
        raise NotImplemented

    def invert(self, data, input=None, mask=None, tag=None):
        raise NotImplemented

    def sample(self, z, x, input=None, tag=None):
        raise NotImplementedError

    def smooth(self, expected_states, variational_mean, data, input=None, mask=None, tag=None):
        """
        Compute the mean observation under the posterior distribution
        of latent discrete states.
        """
        raise NotImplementedError


# Many emissions models start with a linear layer
class _LinearEmissions(_Emissions):
    def __init__(self, N, K, D, M=0, single_subspace=True):
        super(_LinearEmissions, self).__init__(N, K, D, M=M, single_subspace=single_subspace)

        # Initialize linear layer
        # Use the rational Cayley transform to parameterize an orthogonal emission matrix
        assert N > D
        self._Ms = npr.randn(1, D, D) if single_subspace else npr.randn(K, D, D)
        self._As = npr.randn(1, N-D, D) if single_subspace else npr.randn(K, N-D, D)
        self.Fs = npr.randn(1, N, M) if single_subspace else npr.randn(K, N, M)
        self.ds = npr.randn(1, N) if single_subspace else npr.randn(K, N)

        # Set the emission matrix to be a random orthogonal matrix
        C0 = npr.randn(1, N, D) if single_subspace else npr.randn(K, N, D)
        for k in range(C0.shape[0]):
            C0[k] = np.linalg.svd(C0[k], full_matrices=False)[0]
        self.Cs = C0

    @property
    def Cs(self):
        # See https://pubs.acs.org/doi/pdf/10.1021/acs.jpca.5b02015
        # for a derivation of the rational Cayley transform.
        D = self.D
        T = lambda X: np.swapaxes(X, -1, -2)

        Bs = 0.5 * (self._Ms - T(self._Ms))    # Bs is skew symmetric
        Fs = np.matmul(T(self._As), self._As) - Bs
        trm1 = np.concatenate((np.eye(D) - Fs, 2 * self._As), axis=1)
        trm2 = np.eye(D) + Fs
        Cs = T(np.linalg.solve(T(trm2), T(trm1)))
        assert np.allclose(
            np.matmul(T(Cs), Cs),
            np.tile(np.eye(D)[None, :, :], (Cs.shape[0], 1, 1))
            )
        return Cs

    @Cs.setter
    def Cs(self, value):
        N, D = self.N, self.D
        T = lambda X: np.swapaxes(X, -1, -2)

        # Make sure value is the right shape and orthogonal
        Keff = 1 if self.single_subspace else self.K
        assert value.shape == (Keff, N, D)
        assert np.allclose(
            np.matmul(T(value), value),
            np.tile(np.eye(D)[None, :, :], (Keff, 1, 1))
            )

        Q1s, Q2s = value[:, :D, :], value[:, D:, :]
        Fs = T(np.linalg.solve(T(np.eye(D) + Q1s), T(np.eye(D) - Q1s)))
        # Bs = 0.5 * (T(Fs) - Fs) = 0.5 * (self._Ms - T(self._Ms)) -> _Ms = T(Fs)
        self._Ms = T(Fs)
        self._As = 0.5 * np.matmul(Q2s, np.eye(D) + Fs)
        assert np.allclose(self.Cs, value)

    @property
    def params(self):
        return self._As, self._Ms, self.Fs, self.ds

    @params.setter
    def params(self, value):
        self._As, self._Ms, self.Fs, self.ds = value

    def permute(self, perm):
        if not self.single_subspace:
            self._As = self._As[perm]
            self._Ms = self._Bs[perm]
            self.Fs = self.Fs[perm]
            self.ds = self.ds[perm]

    def _invert(self, data, input=None, mask=None, tag=None):
        """
        Approximate invert the linear emission model with the pseudoinverse

        y = Cx + d + noise; C orthogonal.
        xhat = (C^T C)^{-1} C^T (y-d)
        """
        assert self.single_subspace, "Can only invert with a single emission model"

        T = data.shape[0]
        C, F, d = self.Cs[0], self.Fs[0], self.ds[0]
        C_pseudoinv = np.linalg.solve(C.T.dot(C), C.T).T

        # Account for the bias
        bias = input.dot(F.T) + d

        if not np.all(mask):
            data = interpolate_data(data, mask)
            # We would like to find the PCA coordinates in the face of missing data
            # To do so, alternate between running PCA and imputing the missing entries
            for itr in range(25):
                q_mu = (data - bias).dot(C_pseudoinv)
                data[:, ~mask[0]] = (q_mu.dot(C.T) + bias)[:, ~mask[0]]

        # Project data to get the mean
        return (data - bias).dot(C_pseudoinv)

    def forward(self, x, input, tag):
        return np.matmul(self.Cs[None, ...], x[:, None, :, None])[:, :, :, 0] \
             + np.matmul(self.Fs[None, ...], input[:, None, :, None])[:, :, :, 0] \
             + self.ds

    @ensure_args_are_lists
    def _initialize_with_pca(self, datas, inputs=None, masks=None, tags=None, num_iters=20):
        Keff = 1 if self.single_subspace else self.K

        # First solve a linear regression for data given input
        if self.M > 0:
            from sklearn.linear_model import LinearRegression
            lr = LinearRegression(fit_intercept=False)
            lr.fit(np.vstack(inputs), np.vstack(datas))
            self.Fs = np.tile(lr.coef_[None, :, :], (Keff, 1, 1))

        # Compute residual after accounting for input
        resids = [data - np.dot(input, self.Fs[0].T) for data, input in zip(datas, inputs)]

        # Run PCA to get a linear embedding of the data
        pca, xs = pca_with_imputation(self.D, resids, masks, num_iters=num_iters)

        self.Cs = np.tile(pca.components_.T[None, :, :], (Keff, 1, 1))
        self.ds = np.tile(pca.mean_[None, :], (Keff, 1))

        return pca





# Many emissions models start with a linear layer
class _ConstrainedEmissions(_Emissions):
    def __init__(self, N, K, D, M=0, single_subspace=True, N_vec=0, D_vec=0, **kwargs):
        super(_ConstrainedEmissions, self).__init__(N, K, D, M=M, single_subspace=single_subspace, **kwargs)

        self.N_vec=N_vec
        print(self.N_vec)
        self.D_vec=D_vec
        print(self.D_vec)

        self.emissions_models = [_LinearEmissions(n, K, d) for n, d in zip(N_vec, D_vec)]
        # self.emissions_models = [_LinearEmissions(N_vec, K, D_vec)]
        self.np=len(N_vec)


        # print("params",self.emissions_models[0].params)


        self.Cs = [self.emissions_models[i].Cs for i in range(2)]

        self.Fs = npr.randn(1, N, M) if single_subspace else npr.randn(K, N, M)

        # print(self.Cs)
        # self._Ms = self.emissions_models[0]._Ms
        # self._As = self.emissions_models[0]._As
        # self.Fs = self.emissions_models[0].Fs
        # self.ds = self.emissions_models[0].ds
        # self.Cs = self.emissions_models[0].Cs
        # print(self.emissions_models[1].Cs)

        # print((emissions.params for emissions in self.emissions_models))

        # Initialize linear layer
        # Use the rational Cayley transform to parameterize an orthogonal emission matrix
        # assert N > D
        # self._Ms = npr.randn(1, D, D) if single_subspace else npr.randn(K, D, D)
        # self._As = npr.randn(1, N-D, D) if single_subspace else npr.randn(K, N-D, D)
        # self.Fs = npr.randn(1, N, M) if single_subspace else npr.randn(K, N, M)
        # self.ds = npr.randn(1, N) if single_subspace else npr.randn(K, N)
        #
        # # Set the emission matrix to be a random orthogonal matrix
        # C0 = npr.randn(1, N, D) if single_subspace else npr.randn(K, N, D)
        # for k in range(C0.shape[0]):
        #     C0[k] = np.linalg.svd(C0[k], full_matrices=False)[0]
        # self.Cs = C0

    @property
    def Cs(self):
        # See https://pubs.acs.org/doi/pdf/10.1021/acs.jpca.5b02015
        # for a derivation of the rational Cayley transform.
        Cs=[]
        for i in range(self.np):
            D = self.D
            T = lambda X: np.swapaxes(X, -1, -2)

            Bs = 0.5 * (self.emissions_models[i]._Ms - T(self.emissions_models[i]._Ms))    # Bs is skew symmetric
            Fs = np.matmul(T(self.emissions_models[i]._As), self.emissions_models[i]._As) - Bs
            trm1 = np.concatenate((np.eye(D) - Fs, 2 * self.emissions_models[i]._As), axis=1)
            trm2 = np.eye(D) + Fs
            Cs_temp = T(np.linalg.solve(T(trm2), T(trm1)))
            assert np.allclose(
                np.matmul(T(Cs_temp), Cs_temp),
                np.tile(np.eye(D)[None, :, :], (Cs_temp.shape[0], 1, 1))
                )
            Cs.append(Cs_temp)
        return Cs

    @Cs.setter
    def Cs(self, values):
        for i in range(2):
            value=values[i]
            N, D = self.N, self.D
            T = lambda X: np.swapaxes(X, -1, -2)

            # Make sure value is the right shape and orthogonal
            Keff = 1 if self.single_subspace else self.K
            assert value.shape == (Keff, N, D)
            assert np.allclose(
                np.matmul(T(value), value),
                np.tile(np.eye(D)[None, :, :], (Keff, 1, 1))
                )

            Q1s, Q2s = value[:, :D, :], value[:, D:, :]
            Fs = T(np.linalg.solve(T(np.eye(D) + Q1s), T(np.eye(D) - Q1s)))
            # Bs = 0.5 * (T(Fs) - Fs) = 0.5 * (self._Ms - T(self._Ms)) -> _Ms = T(Fs)
            self.emissions_models[i]._Ms = T(Fs)
            self.emissions_models[i]._As = 0.5 * np.matmul(Q2s, np.eye(D) + Fs)
            assert np.allclose(self.Cs[i], value)

    @property
    def params(self):

        # return self._As, self._Ms, self.Fs, self.ds

        temp_tuple=(self.Fs,self.emissions_models[0]._As,self.emissions_models[0]._Ms,self.emissions_models[0].ds)
        for i in range(self.np-1):
            temp_tuple= temp_tuple + (self.emissions_models[i+1]._As,self.emissions_models[i+1]._Ms,self.emissions_models[i+1].ds)

        return temp_tuple

    @params.setter
    def params(self, value):
        # self._As, self._Ms, self.Fs, self.ds = value

        self.Fs = value[0]
        for i in range(self.np):
            self.emissions_models[i]._As=value[1+3*i]
            self.emissions_models[i]._Ms=value[2+3*i]
            self.emissions_models[i].ds=value[3+3*i]

    def permute(self, perm):
        if not self.single_subspace:
            print("need to redo function")
            # self._As = self._As[perm]
            # self._Ms = self._Bs[perm]
            # self.Fs = self.Fs[perm]
            # self.ds = self.ds[perm]

    def _invert(self, data, input=None, mask=None, tag=None):
        """
        Approximate invert the linear emission model with the pseudoinverse

        y = Cx + d + noise; C orthogonal.
        xhat = (C^T C)^{-1} C^T (y-d)
        """
        assert self.single_subspace, "Can only invert with a single emission model"

        T = data.shape[0]
        C, F, d = self.Cs[0][0], self.Fs[0], self.emissions_models[0].ds[0] #Used to be self.Cs[0] - NEED TO UPDATE FOR SPECIFIC POPULATIONS
        C_pseudoinv = np.linalg.solve(C.T.dot(C), C.T).T

        # Account for the bias
        bias = input.dot(F.T) + d

        if not np.all(mask):
            data = interpolate_data(data, mask)
            # We would like to find the PCA coordinates in the face of missing data
            # To do so, alternate between running PCA and imputing the missing entries
            for itr in range(25):
                q_mu = (data - bias).dot(C_pseudoinv)
                data[:, ~mask[0]] = (q_mu.dot(C.T) + bias)[:, ~mask[0]]

        # Project data to get the mean
        return (data - bias).dot(C_pseudoinv)

    def forward(self, x, input, tag):
        return np.matmul(self.Cs[0][None, ...], x[:, None, :, None])[:, :, :, 0] \
             + np.matmul(self.Fs[None, ...], input[:, None, :, None])[:, :, :, 0] \
             + self.emissions_models[0].ds #ADDED [0] AFTER CS (NEED TO UPDATE)

    @ensure_args_are_lists
    def _initialize_with_pca(self, datas, inputs=None, masks=None, tags=None, num_iters=20):
        Keff = 1 if self.single_subspace else self.K

        # First solve a linear regression for data given input
        if self.M > 0:
            from sklearn.linear_model import LinearRegression
            lr = LinearRegression(fit_intercept=False)
            lr.fit(np.vstack(inputs), np.vstack(datas))
            self.Fs = np.tile(lr.coef_[None, :, :], (Keff, 1, 1))

        for i in range(self.np):
        # Compute residual after accounting for input
            resids = [data - np.dot(input, self.Fs[0].T) for data, input in zip(datas, inputs)]

            # Run PCA to get a linear embedding of the data
            pca, xs = pca_with_imputation(self.D, resids, masks, num_iters=num_iters)

            self.Cs[i] = np.tile(pca.components_.T[None, :, :], (Keff, 1, 1))
            self.emissions_models[i].ds = np.tile(pca.mean_[None, :], (Keff, 1))

            return pca






# # Many emissions models start with a linear layer
# class _ConstrainedEmissions(_Emissions):
#     def __init__(self, N, K, D, M=0, single_subspace=True, N_vec=0, **kwargs):
#         super(_ConstrainedEmissions, self).__init__(N, K, D, M=M, single_subspace=single_subspace, **kwargs)
#
#         self.N_vec=N_vec
#
#         print(self.N_vec)
#
#         # Initialize linear layer
#         # Use the rational Cayley transform to parameterize an orthogonal emission matrix
#         assert N > D
#         self._Ms = npr.randn(1, D, D) if single_subspace else npr.randn(K, D, D)
#         self._As = npr.randn(1, N-D, D) if single_subspace else npr.randn(K, N-D, D)
#         self.Fs = npr.randn(1, N, M) if single_subspace else npr.randn(K, N, M)
#         self.ds = npr.randn(1, N) if single_subspace else npr.randn(K, N)
#
#         # Set the emission matrix to be a random orthogonal matrix
#         C0 = npr.randn(1, N, D) if single_subspace else npr.randn(K, N, D)
#         for k in range(C0.shape[0]):
#             C0[k] = np.linalg.svd(C0[k], full_matrices=False)[0]
#         self.Cs = C0
#
#     @property
#     def Cs(self):
#         # See https://pubs.acs.org/doi/pdf/10.1021/acs.jpca.5b02015
#         # for a derivation of the rational Cayley transform.
#         D = self.D
#         T = lambda X: np.swapaxes(X, -1, -2)
#
#         Bs = 0.5 * (self._Ms - T(self._Ms))    # Bs is skew symmetric
#         Fs = np.matmul(T(self._As), self._As) - Bs
#         trm1 = np.concatenate((np.eye(D) - Fs, 2 * self._As), axis=1)
#         trm2 = np.eye(D) + Fs
#         Cs = T(np.linalg.solve(T(trm2), T(trm1)))
#         assert np.allclose(
#             np.matmul(T(Cs), Cs),
#             np.tile(np.eye(D)[None, :, :], (Cs.shape[0], 1, 1))
#             )
#         return Cs
#
#     @Cs.setter
#     def Cs(self, value):
#         N, D = self.N, self.D
#         T = lambda X: np.swapaxes(X, -1, -2)
#
#         # Make sure value is the right shape and orthogonal
#         Keff = 1 if self.single_subspace else self.K
#         assert value.shape == (Keff, N, D)
#         assert np.allclose(
#             np.matmul(T(value), value),
#             np.tile(np.eye(D)[None, :, :], (Keff, 1, 1))
#             )
#
#         Q1s, Q2s = value[:, :D, :], value[:, D:, :]
#         Fs = T(np.linalg.solve(T(np.eye(D) + Q1s), T(np.eye(D) - Q1s)))
#         # Bs = 0.5 * (T(Fs) - Fs) = 0.5 * (self._Ms - T(self._Ms)) -> _Ms = T(Fs)
#         self._Ms = T(Fs)
#         self._As = 0.5 * np.matmul(Q2s, np.eye(D) + Fs)
#         assert np.allclose(self.Cs, value)
#
#     @property
#     def params(self):
#         return self._As, self._Ms, self.Fs, self.ds
#
#     @params.setter
#     def params(self, value):
#         self._As, self._Ms, self.Fs, self.ds = value
#
#     def permute(self, perm):
#         if not self.single_subspace:
#             self._As = self._As[perm]
#             self._Ms = self._Bs[perm]
#             self.Fs = self.Fs[perm]
#             self.ds = self.ds[perm]
#
#     def _invert(self, data, input=None, mask=None, tag=None):
#         """
#         Approximate invert the linear emission model with the pseudoinverse
#
#         y = Cx + d + noise; C orthogonal.
#         xhat = (C^T C)^{-1} C^T (y-d)
#         """
#         assert self.single_subspace, "Can only invert with a single emission model"
#
#         T = data.shape[0]
#         C, F, d = self.Cs[0], self.Fs[0], self.ds[0]
#         C_pseudoinv = np.linalg.solve(C.T.dot(C), C.T).T
#
#         # Account for the bias
#         bias = input.dot(F.T) + d
#
#         if not np.all(mask):
#             data = interpolate_data(data, mask)
#             # We would like to find the PCA coordinates in the face of missing data
#             # To do so, alternate between running PCA and imputing the missing entries
#             for itr in range(25):
#                 q_mu = (data - bias).dot(C_pseudoinv)
#                 data[:, ~mask[0]] = (q_mu.dot(C.T) + bias)[:, ~mask[0]]
#
#         # Project data to get the mean
#         return (data - bias).dot(C_pseudoinv)
#
#     def forward(self, x, input, tag):
#         return np.matmul(self.Cs[None, ...], x[:, None, :, None])[:, :, :, 0] \
#              + np.matmul(self.Fs[None, ...], input[:, None, :, None])[:, :, :, 0] \
#              + self.ds
#
#     @ensure_args_are_lists
#     def _initialize_with_pca(self, datas, inputs=None, masks=None, tags=None, num_iters=20):
#         Keff = 1 if self.single_subspace else self.K
#
#         # First solve a linear regression for data given input
#         if self.M > 0:
#             from sklearn.linear_model import LinearRegression
#             lr = LinearRegression(fit_intercept=False)
#             lr.fit(np.vstack(inputs), np.vstack(datas))
#             self.Fs = np.tile(lr.coef_[None, :, :], (Keff, 1, 1))
#
#         # Compute residual after accounting for input
#         resids = [data - np.dot(input, self.Fs[0].T) for data, input in zip(datas, inputs)]
#
#         # Run PCA to get a linear embedding of the data
#         pca, xs = pca_with_imputation(self.D, resids, masks, num_iters=num_iters)
#
#         self.Cs = np.tile(pca.components_.T[None, :, :], (Keff, 1, 1))
#         self.ds = np.tile(pca.mean_[None, :], (Keff, 1))
#
#         return pca
#






# Sometimes we just want a bit of additive noise on the observations
class _IdentityEmissions(_Emissions):
    def __init__(self, N, K, D, M=0, single_subspace=True):
        super(_IdentityEmissions, self).__init__(N, K, D, M=M, single_subspace=single_subspace)
        assert N == D

    def forward(self, x, input):
        return x

    def _invert(self, data, input=None, mask=None, tag=None):
        """
        Inverse is just the data
        """
        return np.copy(data)


# Allow general nonlinear emission models with neural networks
class _NeuralNetworkEmissions(_Emissions):
    def __init__(self, N, K, D, M=0, hidden_layer_sizes=(50,), single_subspace=True):
        assert single_subspace, "_NeuralNetworkEmissions only supports `single_subspace=True`"
        super(_NeuralNetworkEmissions, self).__init__(N, K, D, M=M, single_subspace=True)

        # Initialize the neural network weights
        assert N > D
        layer_sizes = (D + M,) + hidden_layer_sizes + (N,)
        self.weights = [npr.randn(m, n) for m, n in zip(layer_sizes[:-1], layer_sizes[1:])]
        self.biases = [npr.randn(n) for n in layer_sizes[1:]]

    @property
    def params(self):
        return self.weights, self.biases

    @params.setter
    def params(self, value):
        self.weights, self.biases = value

    def permute(self, perm):
        pass

    def forward(self, x, input, tag):
        inputs = np.column_stack((x, input))
        for W, b in zip(self.weights, self.biases):
            outputs = np.dot(inputs, W) + b
            inputs = np.tanh(outputs)
        return outputs[:, None, :]

    def _invert(self, data, input=None, mask=None, tag=None):
        """
        Inverse is... who knows!
        """
        return npr.randn(data.shape[0], self.D)


# Observation models for SLDS
class _GaussianEmissionsMixin(object):
    def __init__(self, N, K, D, M=0, single_subspace=True, **kwargs):
        super(_GaussianEmissionsMixin, self).__init__(N, K, D, M=M, single_subspace=single_subspace, **kwargs)
        self.inv_etas = -4 + npr.randn(1, N) if single_subspace else npr.randn(K, N)

    @property
    def params(self):
        return super(_GaussianEmissionsMixin, self).params + (self.inv_etas,)

    @params.setter
    def params(self, value):
        self.inv_etas = value[-1]
        super(_GaussianEmissionsMixin, self.__class__).params.fset(self, value[:-1])

    def permute(self, perm):
        super(_GaussianEmissionsMixin, self).permute(perm)
        if not self.single_subspace:
            self.inv_etas = self.inv_etas[perm]

    def log_likelihoods(self, data, input, mask, tag, x):
        mus = self.forward(x, input, tag)
        etas = np.exp(self.inv_etas)
        lls = -0.5 * np.log(2 * np.pi * etas) - 0.5 * (data[:, None, :] - mus)**2 / etas
        return np.sum(lls * mask[:, None, :], axis=2)

    def invert(self, data, input=None, mask=None, tag=None):
        return self._invert(data, input=input, mask=mask, tag=tag)

    def sample(self, z, x, input=None, tag=None):
        T = z.shape[0]
        z = np.zeros_like(z, dtype=int) if self.single_subspace else z
        mus = self.forward(x, input, tag)
        etas = np.exp(self.inv_etas)
        return mus[np.arange(T), z, :] + np.sqrt(etas[z]) * npr.randn(T, self.N)

    def smooth(self, expected_states, variational_mean, data, input=None, mask=None, tag=None):
        mus = self.forward(variational_mean, input, tag)
        return mus[:, 0, :] if self.single_subspace else np.sum(mus * expected_states[:,:,None], axis=1)


class GaussianEmissions(_GaussianEmissionsMixin, _LinearEmissions):

    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None):
        datas = [interpolate_data(data, mask) for data, mask in zip(datas, masks)]
        pca = self._initialize_with_pca(datas, inputs=inputs, masks=masks, tags=tags)
        self.inv_etas[:,...] = np.log(pca.noise_variance_)


class GaussianIdentityEmissions(_GaussianEmissionsMixin, _IdentityEmissions):
    pass


class GaussianNeuralNetworkEmissions(_GaussianEmissionsMixin, _NeuralNetworkEmissions):
    pass


class _StudentsTEmissionsMixin(object):
    def __init__(self, N, K, D, M=0, single_subspace=True, **kwargs):
        super(_StudentsTEmissionsMixin, self).__init__(N, K, D, M, single_subspace=single_subspace, **kwargs)
        self.inv_etas = -4 + npr.randn(1, N) if single_subspace else npr.randn(K, N)
        self.inv_nus = np.log(4) * np.ones(1, N) if single_subspace else np.log(4) * np.ones(K, N)

    @property
    def params(self):
        return super(_StudentsTEmissionsMixin, self).params + (self.inv_etas, self.inv_nus)

    @params.setter
    def params(self, value):
        super(_StudentsTEmissionsMixin, self.__class__).params.fset(self, value[:-2])

    def permute(self, perm):
        super(_StudentsTEmissionsMixin, self).permute(perm)
        if not self.single_subspace:
            self.inv_etas = self.inv_etas[perm]
            self.inv_nus = self.inv_nus[perm]

    def log_likelihoods(self, data, input, mask, tag, x):
        N, etas, nus = self.N, np.exp(self.inv_etas), np.exp(self.inv_nus)
        mus = self.forward(x, input, tag)

        resid = data[:, None, :] - mus
        z = resid / etas
        return -0.5 * (nus + N) * np.log(1.0 + (resid * z).sum(axis=2) / nus) + \
            gammaln((nus + N) / 2.0) - gammaln(nus / 2.0) - N / 2.0 * np.log(nus) \
            -N / 2.0 * np.log(np.pi) - 0.5 * np.sum(np.log(etas), axis=1)

    def invert(self, data, input=None, mask=None, tag=None):
        return self._invert(data, input=input, mask=mask, tag=tag)

    def sample(self, z, x, input=None, tag=None):
        T = z.shape[0]
        z = np.zeros_like(z, dtype=int) if self.single_subspace else z
        mus = self.forward(x, input, tag)
        etas = np.exp(self.inv_etas)
        taus = npr.gamma(nus[z] / 2.0, 2.0 / nus[z])
        return mus[np.arange(T), z, :] + np.sqrt(etas[z] / taus) * npr.randn(T, self.N)

    def smooth(self, expected_states, variational_mean, data, input=None, mask=None, tag=None):
        mus = self.forward(variational_mean, input, tag)
        return mus[:,0,:] if self.single_subspace else np.sum(mus * expected_states[:,:,None], axis=1)


class StudentsTEmissions(_StudentsTEmissionsMixin, _LinearEmissions):

    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None):
        # datas = [interpolate_data(data, mask) for data, mask in zip(datas, masks)]
        pca = self._initialize_with_pca(datas, inputs=inputs, masks=masks, tags=tags)
        self.inv_etas[:,...] = np.log(pca.noise_variance_)


class StudentsTIdentityEmissions(_StudentsTEmissionsMixin, _IdentityEmissions):
    pass


class StudentsTNeuralNetworkEmissions(_StudentsTEmissionsMixin, _NeuralNetworkEmissions):
    pass


class _BernoulliEmissionsMixin(object):
    def __init__(self, N, K, D, M=0, single_subspace=True, link="logit", **kwargs):
        super(BernoulliEmissions, self).__init__(N, K, D, M, single_subspace=single_subspace, **kwargs)

        mean_functions = dict(
            logit=logistic
            )
        self.mean = mean_functions[link]

        link_functions = dict(
            logit=logit
            )
        self.link = link_functions[link]

    def log_likelihoods(self, data, input, mask, tag, x):
        assert data.dtype == int and data.min() >= 0 and data.max() <= 1
        ps = self.mean(self.forward(x, input, tag))
        mask = np.ones_like(data, dtype=bool) if mask is None else mask
        lls = data[:, None, :] * np.log(ps) + (1 - data[:, None, :]) * np.log(1 - ps)
        return np.sum(lls * mask[:, None, :], axis=2)

    def invert(self, data, input=None, mask=None, tag=None):
        yhat = self.link(np.clip(data, .1, .9))
        return self._invert(yhat, input=input, mask=mask, tag=tag)

    def sample(self, z, x, input=None, tag=None):
        T = z.shape[0]
        z = np.zeros_like(z, dtype=int) if self.single_subspace else z
        ps = self.mean(self.forward(x, input, tag))
        return npr.rand(T, self.N) < ps[np.arange(T), z,:]

    def smooth(self, expected_states, variational_mean, data, input=None, mask=None, tag=None):
        ps = self.mean(self.forward(variational_mean, input, tag))
        return ps[:,0,:] if self.single_subspace else np.sum(ps * expected_states[:,:,None], axis=1)


class BernoulliEmissions(_BernoulliEmissionsMixin, _LinearEmissions):
    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None):
        datas = [interpolate_data(data, mask) for data, mask in zip(datas, masks)]
        yhats = [self.link(np.clip(d, .1, .9)) for d in datas]
        self._initialize_with_pca(logits, inputs=inputs, masks=masks, tags=tags)


class BernoulliIdentityEmissions(_BernoulliEmissionsMixin, _IdentityEmissions):
    pass


class BernoulliNeuralNetworkEmissions(_BernoulliEmissionsMixin, _NeuralNetworkEmissions):
    pass


class _PoissonEmissionsMixin(object):
    def __init__(self, N, K, D, M=0, single_subspace=True, link="log", **kwargs):
        super(_PoissonEmissionsMixin, self).__init__(N, K, D, M, single_subspace=single_subspace, **kwargs)

        mean_functions = dict(
            log=lambda x: np.exp(x),
            softplus=softplus
            )
        self.mean = mean_functions[link]

        link_functions = dict(
            log=lambda rate: np.log(rate),
            softplus=inv_softplus
            )
        self.link = link_functions[link]

    def log_likelihoods(self, data, input, mask, tag, x):
        assert data.dtype == int
        lambdas = self.mean(self.forward(x, input, tag))
        mask = np.ones_like(data, dtype=bool) if mask is None else mask
        lls = -gammaln(data[:,None,:] + 1) -lambdas + data[:,None,:] * np.log(lambdas)
        return np.sum(lls * mask[:, None, :], axis=2)

    def invert(self, data, input=None, mask=None, tag=None):
        yhat = self.link(np.clip(data, .1, np.inf))
        return self._invert(yhat, input=input, mask=mask, tag=tag)

    def sample(self, z, x, input=None, tag=None):
        T = z.shape[0]
        z = np.zeros_like(z, dtype=int) if self.single_subspace else z
        lambdas = self.mean(self.forward(x, input, tag))
        y = npr.poisson(lambdas[np.arange(T), z, :])
        return y

    def smooth(self, expected_states, variational_mean, data, input=None, mask=None, tag=None):
        lambdas = self.mean(self.forward(variational_mean, input, tag))
        return lambdas[:,0,:] if self.single_subspace else np.sum(lambdas * expected_states[:,:,None], axis=1)

class PoissonEmissions(_PoissonEmissionsMixin, _LinearEmissions):
    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None):
        datas = [interpolate_data(data, mask) for data, mask in zip(datas, masks)]
        yhats = [self.link(np.clip(d, .1, np.inf)) for d in datas]
        self._initialize_with_pca(yhats, inputs=inputs, masks=masks, tags=tags)

class PoissonConstrainedEmissions(_PoissonEmissionsMixin, _ConstrainedEmissions):
    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None):
        datas = [interpolate_data(data, mask) for data, mask in zip(datas, masks)]
        yhats = [self.link(np.clip(d, .1, np.inf)) for d in datas]
        self._initialize_with_pca(yhats, inputs=inputs, masks=masks, tags=tags)

class PoissonIdentityEmissions(_PoissonEmissionsMixin, _IdentityEmissions):
    pass


class PoissonNeuralNetworkEmissions(_PoissonEmissionsMixin, _NeuralNetworkEmissions):
    pass


class _AutoRegressiveEmissionsMixin(object):
    """
    Include past observations as a covariate in the SLDS emissions.
    The AR model is restricted to be diagonal.
    """
    def __init__(self, N, K, D, M=0, single_subspace=True, **kwargs):
        super(_AutoRegressiveEmissionsMixin, self).__init__(N, K, D, M=M, single_subspace=single_subspace, **kwargs)

        # Initialize AR component of the model
        self.As = npr.randn(1, N) if single_subspace else npr.randn(K, N)
        self.inv_etas = -4 + npr.randn(1, N) if single_subspace else npr.randn(K, N)

    @property
    def params(self):
        return super(_AutoRegressiveEmissionsMixin, self).params + (self.As, self.inv_etas)

    @params.setter
    def params(self, value):
        self.As, self.inv_etas = value[-2:]
        super(_AutoRegressiveEmissionsMixin, self.__class__).params.fset(self, value[:-2])

    def permute(self, perm):
        super(_AutoRegressiveEmissionsMixin, self).permute(perm)
        if not self.single_subspace:
            self.As = self.inv_nus[perm]
            self.inv_etas = self.inv_etas[perm]

    def log_likelihoods(self, data, input, mask, tag, x):
        mus = self.forward(x, input, tag)
        pad = np.zeros((1, 1, self.N)) if self.single_subspace else np.zeros((1, self.K, self.N))
        mus = mus + np.concatenate((pad, self.As[None, :, :] * data[:-1, None, :]))

        etas = np.exp(self.inv_etas)
        lls = -0.5 * np.log(2 * np.pi * etas) - 0.5 * (data[:, None, :] - mus)**2 / etas
        return np.sum(lls * mask[:, None, :], axis=2)

    def invert(self, data, input=None, mask=None, tag=None):
        pad = np.zeros((1, 1, self.N)) if self.single_subspace else np.zeros((1, self.K, self.N))
        resid = data - np.concatenate((pad, self.As[None, :, :] * data[:-1, None, :]))
        return self._invert(resid, input=input, mask=mask, tag=tag)

    def sample(self, z, x, input=None, tag=None):
        T, N = z.shape[0], self.N
        z = np.zeros_like(z, dtype=int) if self.single_subspace else z
        mus = self.forward(x, input, tag)
        etas = np.exp(self.inv_etas)

        y = np.zeros((T, N))
        y[0] = mus[0, z[0], :] + np.sqrt(etas[z[0]]) * npr.randn(N)
        for t in range(1, T):
            y[t] = mus[t, z[t], :] + self.As[z[t]] * y[t-1] + np.sqrt(etas[z[0]]) * npr.randn(N)
        return y

    def smooth(self, expected_states, variational_mean, data, input=None, mask=None, tag=None):
        mus = self.forward(variational_mean, input, tag)
        mus[1:] += self.As[None, :, :] * data[:-1, None, :]
        return mus[:,0,:] if self.single_subspace else np.sum(mus * expected_states, axis=1)


class AutoRegressiveEmissions(_AutoRegressiveEmissionsMixin, _LinearEmissions):
    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None, num_em_iters=25):
        # Initialize the subspace with PCA
        from sklearn.decomposition import PCA
        datas = [interpolate_data(data, mask) for data, mask in zip(datas, masks)]

        # Solve a linear regression for the AR coefficients.
        from sklearn.linear_model import LinearRegression
        for n in range(self.N):
            lr = LinearRegression()
            lr.fit(np.concatenate([d[:-1, n] for d in datas])[:,None],
                   np.concatenate([d[1:, n] for d in datas]))
            self.As[:,n] = lr.coef_[0]

        # Compute the residuals of the AR model
        pad = np.zeros((1,self.N))
        mus = [np.concatenate((pad, self.As[0] * d[:-1])) for d in datas]
        residuals = [data - mu for data, mu in zip(datas, mus)]

        # Run PCA on the residuals to initialize C and d
        pca = self._initialize_with_pca(residuals, inputs=inputs, masks=masks, tags=tags)
        self.inv_etas[:,...] = np.log(pca.noise_variance_)


class AutoRegressiveIdentityEmissions(_AutoRegressiveEmissionsMixin, _IdentityEmissions):
    pass


class AutoRegressiveNeuralNetworkEmissions(_AutoRegressiveEmissionsMixin, _NeuralNetworkEmissions):
    pass
