import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.special import gammaln
from autograd.scipy.linalg import block_diag

from sklearn.decomposition import PCA

from ssm.emissions import _Emissions, _GaussianEmissionsMixin, _PoissonEmissionsMixin, \
    _LinearEmissions, _OrthogonalLinearEmissions, _NeuralNetworkEmissions
from ssm.util import ensure_args_are_lists, ensure_args_not_none, \
    ensure_slds_args_not_none, logistic, logit, softplus, inv_softplus
from ssm.preprocessing import interpolate_data, pca_with_imputation


class _CompoundLinearEmissions(_Emissions):
    def __init__(self, N, K, D, M=0, single_subspace=True,
                 N_vec=None, D_vec=None, **kwargs):
        """
        N_vec, D_vec are the sizes of the constituent emission models.
        Assume N_vec and D_vec are lists/tuples/arrays of length G and

        N_vec = [N_1, ..., N_P] indicates that the first group of neurons
        is size N_1, the P-th populations is size N_P.  Likewise for D_vec.
        We will assume that the data is grouped in the same way.

        We require sum(N_vec) == N and sum(D_vec) == D.
        """
        super(_CompoundLinearEmissions, self).__init__(N, K, D, M=M, single_subspace=single_subspace)

        assert isinstance(N_vec, (np.ndarray, list, tuple))
        N_vec = np.array(N_vec, dtype=int)
        assert np.sum(N_vec) == N

        assert isinstance(D_vec, (np.ndarray, list, tuple)) and len(D_vec) == len(N_vec)
        D_vec = np.array(D_vec, dtype=int)
        assert np.sum(D_vec) == D

        self.N_vec, self.D_vec = N_vec, D_vec

        # Save the number of subpopulations
        self.P = len(self.N_vec)

        # The main purpose of this class is to wrap a bunch of emissions instances
        self.emissions_models = [_LinearEmissions(n, K, d) for n, d in zip(N_vec, D_vec)]

    @property
    def Cs(self):
        if self.single_subspace:
            return np.array([block_diag(*[em.Cs[0] for em in self.emissions_models])])
        else:
            return np.array([block_diag(*[em.Cs[k] for em in self.emissions_models])
                             for k in range(self.K)])

    @property
    def ds(self):
        return np.concatenate([em.ds for em in self.emissions_models], axis=1)

    @property
    def Fs(self):
        return np.concatenate([em.Fs for em in self.emissions_models], axis=1)

    @property
    def params(self):
        return tuple(em.params for em in self.emissions_models)

    @params.setter
    def params(self, value):
        assert len(value) == self.P
        for em, v in zip(self.emissions_models, value):
            em.params = v

    def permute(self, perm):
        for em in self.emissions_models:
            em.permute(perm)

    def _invert(self, data, input, mask, tag):
        assert data.shape[1] == self.N
        N_offsets = np.cumsum(self.N_vec)[:-1]
        states = []
        for em, dp, mp in zip(self.emissions_models,
                            np.split(data, N_offsets, axis=1),
                            np.split(mask, N_offsets, axis=1)):
            states.append(em._invert(dp, input, mp, tag))
        return np.column_stack(states)

    def forward(self, x, input, tag):
        assert x.shape[1] == self.D
        D_offsets = np.cumsum(self.D_vec)[:-1]
        datas = []
        for em, xp in zip(self.emissions_models, np.split(x, D_offsets, axis=1)):
            datas.append(em.forward(xp, input, tag))
        return np.concatenate(datas, axis=2)

    @ensure_args_are_lists
    def _initialize_with_pca(self, datas, inputs=None, masks=None, tags=None, num_iters=20):
        for data in datas:
            assert data.shape[1] == self.N

        N_offsets = np.cumsum(self.N_vec)[:-1]
        pcas = []

        split_datas = list(zip(*[np.split(data, N_offsets, axis=1) for data in datas]))
        split_masks = list(zip(*[np.split(mask, N_offsets, axis=1) for mask in masks]))
        assert len(split_masks) == len(split_datas) == self.P

        for em, dps, mps in zip(self.emissions_models, split_datas, split_masks):
            pcas.append(em._initialize_with_pca(dps, inputs, mps, tags))

        # Combine the PCA objects
        from sklearn.decomposition import PCA
        pca = PCA(self.D)
        pca.components_ = block_diag(*[p.components_ for p in pcas])
        pca.mean_ = np.concatenate([p.mean_ for p in pcas])
        # Not super pleased with this, but it should work...
        pca.noise_variance_ = np.concatenate([p.noise_variance_ * np.ones(n)
                                              for p,n in zip(pcas, self.N_vec)])
        return pca


class _CompoundOrthogonalLinearEmissions(_Emissions):
    def __init__(self, N, K, D, M=0, single_subspace=True,
                 N_vec=None, D_vec=None, **kwargs):
        """
        N_vec, D_vec are the sizes of the constituent emission models.
        Assume N_vec and D_vec are lists/tuples/arrays of length G and

        N_vec = [N_1, ..., N_P] indicates that the first group of neurons
        is size N_1, the P-th populations is size N_P.  Likewise for D_vec.
        We will assume that the data is grouped in the same way.

        We require sum(N_vec) == N and sum(D_vec) == D.
        """
        super(_CompoundOrthogonalLinearEmissions, self).__init__(N, K, D, M=M, single_subspace=single_subspace)

        assert isinstance(N_vec, (np.ndarray, list, tuple))
        N_vec = np.array(N_vec, dtype=int)
        assert np.sum(N_vec) == N

        assert isinstance(D_vec, (np.ndarray, list, tuple)) and len(D_vec) == len(N_vec)
        D_vec = np.array(D_vec, dtype=int)
        assert np.sum(D_vec) == D

        self.N_vec, self.D_vec = N_vec, D_vec

        # Save the number of subpopulations
        self.P = len(self.N_vec)

        # The main purpose of this class is to wrap a bunch of emissions instances
        self.emissions_models = [_OrthogonalLinearEmissions(n, K, d) for n, d in zip(N_vec, D_vec)]

    @property
    def Cs(self):
        if self.single_subspace:
            return np.array([block_diag(*[em.Cs[0] for em in self.emissions_models])])
        else:
            return np.array([block_diag(*[em.Cs[k] for em in self.emissions_models])
                             for k in range(self.K)])

    @property
    def ds(self):
        return np.concatenate([em.ds for em in self.emissions_models], axis=1)

    @property
    def Fs(self):
        return np.concatenate([em.Fs for em in self.emissions_models], axis=1)

    @property
    def params(self):
        return [em.params for em in self.emissions_models]

    @params.setter
    def params(self, value):
        assert len(value) == self.P
        for em, v in zip(self.emissions_models, value):
            em.params = v

    def permute(self, perm):
        for em in self.emissions_models:
            em.permute(perm)

    def _invert(self, data, input, mask, tag):
        assert data.shape[1] == self.N
        N_offsets = np.cumsum(self.N_vec)[:-1]
        states = []
        for em, dp, mp in zip(self.emissions_models,
                            np.split(data, N_offsets, axis=1),
                            np.split(mask, N_offsets, axis=1)):
            states.append(em._invert(dp, input, mp, tag))
        return np.column_stack(states)

    def forward(self, x, input, tag):
        assert x.shape[1] == self.D
        D_offsets = np.cumsum(self.D_vec)[:-1]
        datas = []
        for em, xp in zip(self.emissions_models, np.split(x, D_offsets, axis=1)):
            datas.append(em.forward(xp, input, tag))
        return np.concatenate(datas, axis=2)

    @ensure_args_are_lists
    def _initialize_with_pca(self, datas, inputs=None, masks=None, tags=None, num_iters=20):
        for data in datas:
            assert data.shape[1] == self.N

        N_offsets = np.cumsum(self.N_vec)[:-1]
        pcas = []

        split_datas = list(zip(*[np.split(data, N_offsets, axis=1) for data in datas]))
        split_masks = list(zip(*[np.split(mask, N_offsets, axis=1) for mask in masks]))
        assert len(split_masks) == len(split_datas) == self.P

        for em, dps, mps in zip(self.emissions_models, split_datas, split_masks):
            pcas.append(em._initialize_with_pca(dps, inputs, mps, tags))

        # Combine the PCA objects
        from sklearn.decomposition import PCA
        pca = PCA(self.D)
        pca.components_ = block_diag(*[p.components_ for p in pcas])
        pca.mean_ = np.concatenate([p.mean_ for p in pcas])
        # Not super pleased with this, but it should work...
        pca.noise_variance_ = np.concatenate([p.noise_variance_ * np.ones(n)
                                              for p,n in zip(pcas, self.N_vec)])
        return pca


# Allow general nonlinear emission models with neural networks
class _CompoundNeuralNetworkEmissions(_Emissions):
    def __init__(self, N, K, D, M=0, hidden_layer_sizes=(50,), single_subspace=True, N_vec=None, D_vec=None, **kwargs):
        assert single_subspace, "_NeuralNetworkEmissions only supports `single_subspace=True`"
        super(_CompoundNeuralNetworkEmissions, self).__init__(N, K, D, M=M, single_subspace=True)


        print(hidden_layer_sizes)
        #Make sure N_vec and D_vec are in correct form
        assert isinstance(N_vec, (np.ndarray, list, tuple))
        N_vec = np.array(N_vec, dtype=int)
        assert np.sum(N_vec) == N

        assert isinstance(D_vec, (np.ndarray, list, tuple)) and len(D_vec) == len(N_vec)
        D_vec = np.array(D_vec, dtype=int)
        assert np.sum(D_vec) == D

        self.N_vec, self.D_vec = N_vec, D_vec

        # Save the number of subpopulations
        self.P = len(self.N_vec)

        # The main purpose of this class is to wrap a bunch of emissions instances
        self.emissions_models = [_NeuralNetworkEmissions(n, K, d, hidden_layer_sizes=hidden_layer_sizes) for n, d in zip(N_vec, D_vec)]


    @property
    def params(self):
        return [em.params for em in self.emissions_models]

    @params.setter
    def params(self, value):
        assert len(value) == self.P
        for em, v in zip(self.emissions_models, value):
            em.params = v

    def permute(self, perm):
        pass

    def forward(self, x, input, tag):
        assert x.shape[1] == self.D
        D_offsets = np.cumsum(self.D_vec)[:-1]
        datas = []
        for em, xp in zip(self.emissions_models, np.split(x, D_offsets, axis=1)):
            datas.append(em.forward(xp, input, tag))
        return np.concatenate(datas, axis=2)


    def _invert(self, data, input, mask, tag):
        """
        Inverse is... who knows!
        """
        # print(data.shape)
        assert data.shape[1] == self.N
        N_offsets = np.cumsum(self.N_vec)[:-1]
        states = []
        for em, dp, mp in zip(self.emissions_models,
                            np.split(data, N_offsets, axis=1),
                            np.split(mask, N_offsets, axis=1)):
            states.append(em._invert(dp, input, mp, tag))
        return np.column_stack(states)


        # return npr.randn(data.shape[0], self.D)


    def log_prior(self):
        alpha=1
        ssq_all=[]
        for em in self.emissions_models:
            ssq_w=[np.sum(i**2) for i in em.weights]
            # ssq_b=[np.sum(i**2) for i in em.biases]
            # ssq_all.append(-np.sum(alpha*ssq_w+alpha*ssq_b))
            ssq_all.append(-np.sum(alpha*ssq_w))
        return np.sum(ssq_all)


class GaussianCompoundEmissions(_GaussianEmissionsMixin, _CompoundLinearEmissions):

    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None):
        datas = [interpolate_data(data, mask) for data, mask in zip(datas, masks)]
        pca = self._initialize_with_pca(datas, inputs=inputs, masks=masks, tags=tags)
        self.inv_etas[:,...] = np.log(pca.noise_variance_)


class GaussianOrthogonalCompoundEmissions(_GaussianEmissionsMixin, _CompoundOrthogonalLinearEmissions):

    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None):
        datas = [interpolate_data(data, mask) for data, mask in zip(datas, masks)]
        pca = self._initialize_with_pca(datas, inputs=inputs, masks=masks, tags=tags)
        self.inv_etas[:,...] = np.log(pca.noise_variance_)



class GaussianCompoundNeuralNetworkEmissions(_GaussianEmissionsMixin, _CompoundNeuralNetworkEmissions):
    pass


class PoissonOrthogonalCompoundEmissions(_PoissonEmissionsMixin, _CompoundOrthogonalLinearEmissions):
    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None):
        datas = [interpolate_data(data, mask) for data, mask in zip(datas, masks)]
        yhats = [self.link(np.clip(d, .1, np.inf)) for d in datas]
        self._initialize_with_pca(yhats, inputs=inputs, masks=masks, tags=tags)


class PoissonCompoundNeuralNetworkEmissions(_PoissonEmissionsMixin, _CompoundNeuralNetworkEmissions):
    pass

