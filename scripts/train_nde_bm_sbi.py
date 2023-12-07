import argparse
import csv
import os
import pickle
from functools import partial

import haiku as hk
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import torch
from haiku._src.nets.resnet import ResNet18
from sbi.inference import SNLE, SNPE_C, SNRE, prepare_for_sbi, simulate_for_sbi
from sbi_lens.metrics.c2st import c2st
from sbi_lens.simulator.config import config_lsst_y_10
from sbi_lens.simulator.LogNormal_field import lensingLogNormal
from torch_lensing_simulator import (
    PytorchCompressedSimulator,
    PytorchPrior,
)
from utils import make_plot

# script arguments
parser = argparse.ArgumentParser()
parser.add_argument("--path_to_access_sbi_lens", type=str, default=".")

parser.add_argument("--sbi_method", type=str, default="snle")
parser.add_argument("--nb_round", type=int, default=1)
parser.add_argument("--seed", type=int, default=1)

parser.add_argument("--exp_id", type=str, default="job_0")

parser.add_argument("--device", type=str, default="cpu")

args = parser.parse_args()


######## PARAMS ########
print("PARAMS---------------")
print("---------------------")

tmp = [200, 500, 1000, 1500, 2000]
nb_simulations_allow = tmp[int(args.exp_id[4:])]


print("simulation budget:", nb_simulations_allow)
print("sbi method:", args.sbi_method)
print("nb_round:", args.nb_round)

print("---------------------")
print("---------------------")

PATH = "_{}_{}_{}_{}".format(
    args.sbi_method, args.nb_round, nb_simulations_allow, args.seed
)

os.makedirs(f"./results/experiments_sbi/exp{PATH}/save_params")
os.makedirs(f"./results/experiments_sbi/exp{PATH}/fig")


nb_posterior_sample = 40_000
num_chains = 20

nb_simulations_allow_per_round = nb_simulations_allow // args.nb_round

######## CONFIG LSST Y 10 ########
print("... prepare config lsst year 10")

# load lsst year 10 settings
N = config_lsst_y_10.N
map_size = config_lsst_y_10.map_size
sigma_e = config_lsst_y_10.sigma_e
gals_per_arcmin2 = config_lsst_y_10.gals_per_arcmin2
nbins = config_lsst_y_10.nbins
a = config_lsst_y_10.a
b = config_lsst_y_10.b
z0 = config_lsst_y_10.z0
truth = config_lsst_y_10.truth
params_name = config_lsst_y_10.params_name_latex


# define lsst year 10 log normal model
model = partial(
    lensingLogNormal,
    N=N,
    map_size=map_size,
    gal_per_arcmin2=gals_per_arcmin2,
    sigma_e=sigma_e,
    nbins=nbins,
    a=a,
    b=b,
    z0=z0,
    model_type="lognormal",
    lognormal_shifts="LSSTY10",
    with_noise=True,
)


# compressor
compressor = hk.transform_with_state(lambda y: ResNet18(6)(y, is_training=False))

a_file = open(
    f"{args.path_to_access_sbi_lens}/sbi_lens/sbi_lens/data/params_compressor/opt_state_resnet_vmim.pkl",
    "rb",
)
opt_state_resnet = pickle.load(a_file)

a_file = open(
    f"{args.path_to_access_sbi_lens}/sbi_lens/sbi_lens/data/params_compressor/params_nd_compressor_vmim.pkl",
    "rb",
)
parameters_compressor = pickle.load(a_file)


# observation
m_data = np.load(
    f"{args.path_to_access_sbi_lens}/sbi_lens/sbi_lens/data/m_data__256N_10ms_27gpa_0.26se.npy"
)

m_data_comressed, _ = compressor.apply(
    parameters_compressor, opt_state_resnet, None, m_data.reshape([1, N, N, nbins])
)

observation = torch.tensor(np.array(m_data_comressed.squeeze()), device=args.device)


# full field hmc contours (ground truth)
sample_ff = np.load(
    f"{args.path_to_access_sbi_lens}/sbi_lens/sbi_lens/data/posterior_full_field__256N_10ms_27gpa_0.26se.npy"
)


prior = PytorchPrior(device=args.device)
simulator = PytorchCompressedSimulator(
    model=model,
    compressor=compressor,
    params_compressor=parameters_compressor,
    opt_state=opt_state_resnet,
    device=args.device,
).simulator

simulator, prior = prepare_for_sbi(simulator, prior)

proposal = prior


if args.sbi_method == "snle":
    inference = SNLE(prior=prior, device=args.device)

    for _ in range(args.nb_round):
        theta, x = simulate_for_sbi(
            simulator, proposal, num_simulations=nb_simulations_allow_per_round + 100
        )

        # remove potential NaNs
        inds = np.unique(np.where(torch.isnan(x).numpy())[0])
        x = torch.tensor(np.delete(x.numpy(), inds, axis=0))
        theta = torch.tensor(np.delete(theta.numpy(), inds, axis=0))

        # select randomly nb_simulations_allow_per_round simulations
        inds = np.random.randint(
            low=0, high=x.shape[0], size=nb_simulations_allow_per_round
        )
        x = x[inds]
        theta = theta[inds]

        density_estimator = inference.append_simulations(theta, x).train()
        posterior = inference.build_posterior(
            density_estimator,
            mcmc_method="slice",
            mcmc_parameters={"warmup_steps": 250},
        )
        proposal = posterior.set_default_x(observation)

    posterior_sample = posterior.sample(
        (nb_posterior_sample,),
        x=observation,
        method="slice_np_vectorized",
        num_chains=num_chains,
        mcmc_parameters={"warmup_steps": 250},
    )


elif args.sbi_method == "snpe":
    inference = SNPE_C(prior=prior, device=args.device)

    for _ in range(args.nb_round):
        theta, x = simulate_for_sbi(
            simulator, proposal, num_simulations=nb_simulations_allow_per_round + 100
        )

        # remove potential NaNs
        inds = np.unique(np.where(torch.isnan(x).numpy())[0])
        x = torch.tensor(np.delete(x.numpy(), inds, axis=0))
        theta = torch.tensor(np.delete(theta.numpy(), inds, axis=0))

        # select randomly nb_simulations_allow_per_round simulations
        inds = np.random.randint(
            low=0, high=x.shape[0], size=nb_simulations_allow_per_round
        )
        x = x[inds]
        theta = theta[inds]

        density_estimator = inference.append_simulations(
            theta, x, proposal=proposal
        ).train(num_atoms=10)
        posterior = inference.build_posterior(density_estimator)
        proposal = posterior.set_default_x(observation)

    posterior_sample = posterior.sample((nb_posterior_sample,), x=observation)


elif args.sbi_method == "snre":
    inference = SNRE(prior=prior, device=args.device)

    for _ in range(args.nb_round):
        theta, x = simulate_for_sbi(
            simulator, proposal, num_simulations=nb_simulations_allow_per_round + 100
        )

        # remove potential NaNs
        inds = np.unique(np.where(torch.isnan(x).numpy())[0])
        x = torch.tensor(np.delete(x.numpy(), inds, axis=0))
        theta = torch.tensor(np.delete(theta.numpy(), inds, axis=0))

        # select randomly nb_simulations_allow_per_round simulations
        inds = np.random.randint(
            low=0, high=x.shape[0], size=nb_simulations_allow_per_round
        )
        x = x[inds]
        theta = theta[inds]

        density_estimator = inference.append_simulations(theta, x).train(num_atoms=10)
        posterior = inference.build_posterior(
            density_estimator,
            mcmc_method="slice",
            mcmc_parameters={"warmup_steps": 250},
        )
        proposal = posterior.set_default_x(observation)

    posterior_sample = posterior.sample(
        (nb_posterior_sample,),
        x=observation,
        method="slice_np_vectorized",
        num_chains=num_chains,
        mcmc_parameters={"warmup_steps": 250},
    )

# check the number of simulations used by the algorithm
# (should be equal to nb_simulations_allow_per_round * nb_round)
theta, x, prior_masks = inference.get_simulations(starting_round=0)
print("number of simulations used: ", theta.shape[0])

posterior_sample = jnp.array(posterior_sample)
jnp.save(
    f"./results/experiments_sbi/exp{PATH}/posteriors_sample",
    posterior_sample,
)

make_plot(
    [sample_ff, posterior_sample],
    ["Ground truth", "Approx"],
    params_name,
    truth,
)

plt.savefig(f"./results/experiments_sbi/exp{PATH}/fig/contour_plot_round{0}")

print("done ✓")

print("... compute metric")

if len(posterior_sample) > len(sample_ff):
    nb_posterior_sample = len(sample_ff)
else:
    nb_posterior_sample = len(posterior_sample)

inds = np.random.randint(0, nb_posterior_sample, 10_000)
c2st_metric = c2st(sample_ff[inds], posterior_sample[inds], seed=0, n_folds=5)

print("done ✓")


print("... save info experiment")

field_names = ["experiment_id", "sbi_method", "nb_round", "nb_simulations", "c2st"]

dict = {
    "experiment_id": f"exp{PATH}",
    "sbi_method": args.sbi_method,
    "nb_round": args.nb_round,
    "nb_simulations": nb_simulations_allow,
    "c2st": c2st_metric,
}

with open(
    "./results/store_experiments_sbi.csv",
    "a",
) as csv_file:
    dict_object = csv.DictWriter(csv_file, fieldnames=field_names)
    dict_object.writerow(dict)

print("done ✓")
