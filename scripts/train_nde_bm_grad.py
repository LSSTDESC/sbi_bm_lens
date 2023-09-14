import argparse
import csv
import os
import pickle
from functools import partial

import haiku as hk
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro.distributions as dist
import optax
import tensorflow as tf
import tensorflow_probability as tfp
from haiku._src.nets.resnet import ResNet18
from sbi_lens.config import config_lsst_y_10
from sbi_lens.normflow.models import (
    AffineSigmoidCoupling,
    AffineCoupling,
    ConditionalRealNVP,
)

from sbibmlens.nn import MomentNetwork
from sbibmlens.inference.snle import SNLE
from sbibmlens.utils import make_plot
from sbibmlens.simulator.lensing_simulator import CompressedSimulator

from sbi_lens.simulator.LogNormal_field import lensingLogNormal
from sbi_lens.metrics.c2st import c2st

tfp = tfp.experimental.substrates.jax
tfb = tfp.bijectors
tfd = tfp.distributions

gpus = tf.config.experimental.list_physical_devices(device_type="GPU")

for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


# script arguments
parser = argparse.ArgumentParser()
parser.add_argument("--path_to_access_sbi_lens", type=str, default=".")

parser.add_argument("--sbi_method", type=str, default="nle")
parser.add_argument("--nb_round", type=int, default=1)


parser.add_argument("--n_flow_layers", type=int, default=4)
parser.add_argument("--n_bijector_layers", type=int, default=2)
parser.add_argument("--activ_fun", type=str, default="silu")
parser.add_argument("--nf", type=str, default="affine")

parser.add_argument("--seed", type=int, default=0)


parser.add_argument("--exp_id", type=str, default="job_0")
parser.add_argument("--bacth_size", type=int, default=80)
parser.add_argument("--total_steps", type=int, default=70_000)

parser.add_argument("--score_weight", type=float, default=0)

parser.add_argument("--score", type=str, default="unmarginal")
parser.add_argument("--score_noise", type=float, default=0)

args = parser.parse_args()

######## PARAMS ########
print("PARAMS---------------")
print("---------------------")

batch_size = args.bacth_size
tmp = [100, 200, 300, 400, 600, 800, 1000, 1500, 2000]  # range(100,1100,100)
nb_simulations_allow = tmp[int(args.exp_id[4:])]


print("simulation budget:", nb_simulations_allow)
print("score_weight:", args.score_weight)
print("exp_id:", args.exp_id[4:])
print("seed:", args.seed)
print("n_flow_layers:", args.n_flow_layers)
print("n_bijector_layers:", args.n_bijector_layers)
print("activ_fun:", args.activ_fun)
print("sbi method:", args.sbi_method)
print("nf type:", args.nf)
print("batch size:", args.bacth_size)
print("score type:", args.score)
print("score noise:", args.score_noise)
print("total_steps:", args.total_steps)

print("---------------------")
print("---------------------")

PATH = "_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(
    args.sbi_method,
    args.total_steps,
    nb_simulations_allow,
    args.seed,
    args.n_flow_layers,
    args.n_bijector_layers,
    args.activ_fun,
    args.nf,
    args.bacth_size,
    args.score_weight,
    args.score,
    args.score_noise,
)

os.makedirs(f"./results/experiments/exp{PATH}/save_params")
os.makedirs(f"./results/experiments/exp{PATH}/fig")


master_seed = jax.random.PRNGKey(args.seed)

######## CONFIG LSST Y 10 ########
print("... prepare config lsst year 10")
dim = 6

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

print("done ✓")


######## LOAD OBSERVATION AND REFERENCES POSTERIOR ########
print("... load observation and reference posterior")


# load reference posterior
sample_ff = jnp.load(
    f"{args.path_to_access_sbi_lens}/sbi_lens/sbi_lens/data/posterior_full_field__256N_10ms_27gpa_0.26se.npy"
)

# load observed mass map
m_data = jnp.load(
    f"{args.path_to_access_sbi_lens}/sbi_lens/sbi_lens/data/m_data__256N_10ms_27gpa_0.26se.npy"
)

print("done ✓")


######## COMPRESSOR ########
print("... prepare compressor")

compressor = hk.transform_with_state(lambda y: ResNet18(dim)(y, is_training=False))

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

m_data_comressed, _ = compressor.apply(
    parameters_compressor, opt_state_resnet, None, m_data.reshape([1, N, N, nbins])
)

print("done ✓")


######## SET UP SIMULATOR ########
print("... set up simulator")

if args.sbi_method == "nle":
    score_type = "conditional"
elif args.sbi_method == "npe":
    score_type = "density"

compressed_simulator = CompressedSimulator(
    model=model,
    score_type=score_type,
    compressor=compressor,
    params_compressor=parameters_compressor,
    opt_state=opt_state_resnet,
)

print("done ✓")


######## CREATE NDE ########
print("... build nde")

if args.activ_fun == "silu":
    activ_fun = jax.nn.silu
elif args.activ_fun == "sin":
    activ_fun = jnp.sin

if args.nf == "smooth":
    if args.sbi_method == "npe":
        key = jax.random.PRNGKey(0)
        omega_c = dist.TruncatedNormal(0.2664, 0.2, low=0).sample(key, (1000,))
        omega_b = dist.Normal(0.0492, 0.006).sample(key, (1000,))
        sigma_8 = dist.Normal(0.831, 0.14).sample(key, (1000,))
        h_0 = dist.Normal(0.6727, 0.063).sample(key, (1000,))
        n_s = dist.Normal(0.9645, 0.08).sample(key, (1000,))
        w_0 = dist.TruncatedNormal(-1.0, 0.9, low=-2.0, high=-0.3).sample(key, (1000,))

        theta = jnp.stack([omega_c, omega_b, sigma_8, h_0, n_s, w_0], axis=-1)

        scale = jnp.std(theta, axis=0) / 0.07
        shift = jnp.mean(theta / scale, axis=0) - 0.5

    elif args.sbi_method == "nle":
        # how these qantities are comute (with dataset form prior)
        # scale_y = jnp.std(dataset_y, axis=0) / 0.07
        # shift_y = jnp.mean(dataset_y / scale_y, axis=0) - 0.5

        scale = jnp.array(
            [1.2179337, 1.9040986, 2.070386, 1.9527259, 1.0068599, 0.38351834]
        )
        shift = jnp.array(
            [-0.51705444, -0.39985067, -0.53731424, -0.3843186, -0.5539374, -0.33893403]
        )

    bijector_layers = [128] * args.n_bijector_layers

    bijector = partial(
        AffineSigmoidCoupling,
        layers=bijector_layers,
        activation=activ_fun,
        n_components=16,
    )

    NF = partial(ConditionalRealNVP, n_layers=args.n_flow_layers, bijector_fn=bijector)

    class NDE(hk.Module):
        def __call__(self, y):
            nvp = NF(dim)(y)
            return tfd.TransformedDistribution(
                nvp, tfb.Chain([tfb.Scale(scale), tfb.Shift(shift)])
            )

elif args.nf == "affine":
    bijector_layers = [128] * args.n_bijector_layers

    bijector = partial(AffineCoupling, layers=bijector_layers, activation=activ_fun)

    NF = partial(ConditionalRealNVP, n_layers=args.n_flow_layers, bijector_fn=bijector)

    class NDE(hk.Module):
        def __call__(self, y):
            return NF(dim)(y)


if args.nf == "affine" and args.sbi_method == "npe" and args.score_weight > 0:
    raise ValueError("NDE has to be smooth")


if args.sbi_method == "npe":
    nf_log_prob = hk.without_apply_rng(
        hk.transform(lambda theta, y: NDE()(y).log_prob(theta).squeeze())
    )
    nf_get_posterior_sample = hk.transform(
        lambda y: NDE()(y).sample(len(sample_ff), seed=hk.next_rng_key())
    )
elif args.sbi_method == "nle":
    nf_log_prob = hk.without_apply_rng(
        hk.transform(lambda theta, y: NDE()(theta).log_prob(y).squeeze())
    )


print("done ✓")


######## INFERENCE ########
print("... inference")

params_init = nf_log_prob.init(
    master_seed, 0.5 * jnp.ones([1, dim]), 0.5 * jnp.ones([1, dim])
)

prior_mean = jnp.mean(compressed_simulator.prior_sample((1000,), master_seed), axis=0)


nb_steps = args.total_steps - args.total_steps * 0.2

lr_scheduler = optax.exponential_decay(
    init_value=0.001,
    transition_steps=nb_steps // 50,
    decay_rate=0.9,
    end_value=1e-5,
)


if args.sbi_method == "nle":
    inference = SNLE(NDE=nf_log_prob, init_params_nde=params_init, dim=6)
elif args.sbi_method == "npe":
    print("not implemented")

data = compressed_simulator.build_dataset(
    batch_size=nb_simulations_allow,
    seed=master_seed,
    use_prior=True,
    proposal_sample=None,
)


if args.score == "marginal":
    nb_layer = 2
    get_moments_fixed = hk.without_apply_rng(
        hk.transform_with_state(
            lambda theta, y: MomentNetwork(
                layers=[256] * nb_layer,
                batch_norm=[hk.BatchNorm(True, True, 0.999) for i in range(nb_layer)],
            )(theta, y, is_training=False)
        )
    )

    a_file = open("./data/SNR_study/params_esp.pkl", "rb")
    params_esperance = pickle.load(a_file)
    a_file = open("./data/SNR_study/state_bn_esp.pkl", "rb")
    state_bn = pickle.load(a_file)

    learned_marginal_score, _ = get_moments_fixed.apply(
        params_esperance, state_bn, data["theta"], data["y"]
    )

    learned_marginal_score += (
        learned_marginal_score
        * args.score_noise
        * np.random.normal(0.0, 1.0, size=learned_marginal_score.shape)
    )

    data = {
        "theta": data["theta"],
        "score": learned_marginal_score,
        "y": data["y"],
    }


inference.train(
    data,
    total_steps=args.total_steps,
    batch_size=args.bacth_size,
    score_weight=args.score_weight,
    learning_rate=lr_scheduler,
)


posterior_sample = inference.sample(
    log_prob_prior=compressed_simulator.prior_log_prob,
    observation=m_data_comressed,
    init_point=prior_mean,
    key=master_seed,
)

jnp.save(
    f"./results/experiments/exp{PATH}/posteriors_sample",
    posterior_sample,
)

make_plot(
    [sample_ff, posterior_sample],
    [
        "Ground truth",
        "Approx w/ {} sim, {} score weight, {} score type, {} noise score".format(
            nb_simulations_allow, args.score_weight, args.score, args.score_noise
        ),
    ],
    params_name,
    truth,
)

plt.savefig(f"./results/experiments/exp{PATH}/fig/contour_plot_round{0}")

print("done ✓")

print("... compute metric")

if len(posterior_sample) > len(sample_ff):
    nb_posterior_sample = len(sample_ff)
else:
    nb_posterior_sample = len(posterior_sample)

inds = np.random.randint(0, nb_posterior_sample, 10_000)
c2st_metric = c2st(sample_ff[inds], posterior_sample[inds], seed=0, n_folds=5)

print("done ✓")

print("... save info, params, etc.")

# save params

params_nde = inference.params
with open(
    f"./results/experiments/exp{PATH}/save_params/params_flow.pkl",
    "wb",
) as fp:
    pickle.dump(params_nde, fp)


print("done ✓")

print("... save info experiment")

field_names = [
    "experiment_id",
    "sbi_method",
    "total_steps",
    "activ_fun",
    "nb_simulations",
    "score_weight",
    "n_flow_layers",
    "n_bijector_layers",
    "seed",
    "nf type",
    "batch size",
    "score type",
    "score noise",
    "c2st",
]

dict = {
    "experiment_id": f"exp{PATH}",
    "sbi_method": args.sbi_method,
    "total_steps": args.total_steps,
    "activ_fun": args.activ_fun,
    "nb_simulations": nb_simulations_allow,
    "score_weight": args.score_weight,
    "n_flow_layers": args.n_flow_layers,
    "n_bijector_layers": args.n_bijector_layers,
    "seed": args.seed,
    "nf type": args.nf,
    "batch size": args.bacth_size,
    "score type": args.score,
    "score noise": args.score_noise,
    "c2st": c2st_metric,
}

with open(
    "./results/store_experiments.csv",
    "a",
) as csv_file:
    dict_object = csv.DictWriter(csv_file, fieldnames=field_names)
    dict_object.writerow(dict)

print("done ✓")
