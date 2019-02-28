# SSM: Bayesian learning and inference for state space models _with multiple neural populations_ 
**This adaptation of the main ssm package is designed for fitting models with multiple, known, populations of neurons.**


# Examples

Let's say we want to fit a Poisson linear dynamical systems (PLDS) model to neurons coming from 2 known populations.
We need to first set the number of latent variables (D) for each population and make a vector of the number of neurons in each population.
```
D_pop1=3
D_pop2=3
D_vec=[D_pop1,D_pop2]

num_units_pop1=20
num_units_pop2=20
N_vec=[num_units_pop1,num_units_pop2]
```

Declare the model.
```
plds = LDS(N=np.sum(N_vec),D=np.sum(D_vec),emissions="poisson_compound", emission_kwargs=dict(link="softplus",N_vec=N_vec,D_vec=D_vec))
```

Let "ys" be your data matrix of size T (number of time points) x N (number of neurons). The neurons must be ordered in terms of which population they come from. In this example, the first 20 columns of "ys" would be for the first population. <br><br>
Fit the model. 
```
plds.initialize(ys)
q = SLDSTriDiagVariationalPosterior(plds, ys)
elbos = plds.fit(q, ys, num_iters=7000, initialize=False)
slds_x = q.mean[0]
```

Additional examples plus example notebooks will be coming shortly.


# Installation

To install this forked repo:
```
git clone https://github.com/jglaser2/ssm
cd ssm
pip install -e .
```


The ReadMe for the original repo (at the time of forking) is below. We are currently many commits behind the main repo, but hope to merge soon.





--------------------------------

# SSM: Bayesian learning and inference for state space models 
[![Test status](https://travis-ci.org/slinderman/ssm.svg?branch=master)](https://travis-ci.org/slinderman/ssm)

This package has fast and flexible code for simulating, learning, and performing inference in a variety of state space models. 
Currently, it supports:

- Hidden Markov Models (HMM)
- Auto-regressive HMMs (ARHMM)
- Input-output HMMs (IOHMM)
- Linear Dynamical Systems (LDS)
- Switching Linear Dynamical Systems (SLDS)
- Recurrent SLDS (rSLDS)
- Hierarchical extensions of the above
- Partial observations and missing data

We support the following observation models:

- Gaussian
- Student's t
- Bernoulli
- Poisson

HMM inference is done with either expectation maximization (EM) or stochastic gradient descent (SGD).  For SLDS, we use stochastic variational inference (SVI). 

# Examples
Here's a snippet to illustrate how we simulate from an HMM.
```
from ssm.models import HMM
T = 100  # number of time bins
K = 5    # number of discrete states
D = 2    # dimension of the observations

# make an hmm and sample from it
hmm = HMM(K, D, observations="gaussian")
z, y = hmm.sample(T)
```

Fitting an HMM is simple. 
```
test_hmm = HMM(K, D, observations="gaussian")
test_hmm.fit(y)
zhat = test_hmm.most_likely_states(y)
```

The notebooks folder has more thorough, complete examples of HMMs, SLDS, and recurrent SLDS.  

# Installation
```
git clone git@github.com:slinderman/ssm.git
cd ssm
pip install -e .
```
This will install "from source" and compile the Cython code for fast message passing and gradients.

To install with some parallel support via OpenMP, first make sure that your compiler supports it.  OS X's default Clang compiler does not, but you can install GNU gcc and g++ with conda.  Once you've set these as your default, you can install with OpenMP support using
```
USE_OPENMP=True pip install -e .
```
