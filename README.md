# SSM: Bayesian learning and inference for state space models _with multiple neural populations_ 
**This adaptation of the main ssm package is designed for fitting models with multiple, known, populations of neurons.**

# Examples

Let's say we want to fit a Poisson switching linear dynamical systems model to neurons coming from 2 known populations. First, let's import the necessary functions.

```
from ssm.variational import SLDSTriDiagVariationalPosterior
from ssm.models import SLDS
from extensions.compound_emissions.compound_emissions import PoissonOrthogonalCompoundEmissions
```

We need to set the number of latent variables (D) for each population and make a vector of the number of neurons in each population. We also need to set the number of discrete states (K) in the switching model.
```
D_pop1=3
D_pop2=3
D_vec=[D_pop1,D_pop2]

num_units_pop1=20
num_units_pop2=20
N_vec=[num_units_pop1,num_units_pop2]

K=2
```

Specify the emissions model.
```
poiss_comp_emissions=PoissonOrthogonalCompoundEmissions(N=np.sum(N_vec),K=K,D=np.sum(D_vec),D_vec=D_vec,N_vec=N_vec,link='softplus')
```

Declare the SLDS model.
```
slds = SLDS(N=np.sum(N_vec),K=K,D=np.sum(D_vec),emissions=poiss_comp_emissions)
```

Let "ys" be your data matrix of size T (number of time points) x N (number of neurons). The neurons must be ordered in terms of which population they come from. In this example, the first 20 columns of "ys" would be for the first population. <br><br>
Fit the model. 
```
slds.initialize(ys)
q = SLDSTriDiagVariationalPosterior(slds, ys)
elbos = slds.fit(q, ys, num_iters=7000, initialize=False)
```

After fitting the model, determine the continuous latent variables (the mean of the posterior) and the most likely discrete latent variables.
```
slds_x = q.mean[0]
slds_z = slds.most_likely_states(slds_x,ys)
```


Additional examples plus example notebooks will be coming shortly.


# Installation

To install this forked repo:
```
git clone https://github.com/jglaser2/ssm
cd ssm
pip install numpy cython
pip install -e .
```

For more information, see the ReadMe of the original [repo](https://github.com/slinderman/ssm). We are currently a few commits behind the main repo, but hope to merge soon.
