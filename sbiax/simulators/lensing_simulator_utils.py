import numpyro.distributions as dist
import jax
from sbi_lens.simulator.utils import get_samples_and_scores
import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
from functools import partial


class CompressedSimulator:
    def __init__(self, model, score_type, compressor, params_compressor, opt_state):
        self.model = model
        self.score_type = score_type

        self.compressor = compressor
        self.params_compressor = params_compressor
        self.opt_state = opt_state

        self.omega_c = dist.TruncatedNormal(0.2664, 0.2, low=0)
        self.omega_b = dist.Normal(0.0492, 0.006)
        self.sigma_8 = dist.Normal(0.831, 0.14)
        self.h_0 = dist.Normal(0.6727, 0.063)
        self.n_s = dist.Normal(0.9645, 0.08)
        self.w_0 = dist.TruncatedNormal(-1.0, 0.9, low=-2.0, high=-0.3)

        self.stack = [
            self.omega_c,
            self.omega_b,
            self.sigma_8,
            self.h_0,
            self.n_s,
            self.w_0,
        ]

    def prior_sample(self, sample_shape, master_key):
        samples = []

        keys = jax.random.split(master_key, 6)

        for i, distribution in enumerate(self.stack):
            samples.append(distribution.sample(keys[i], sample_shape))

        return jnp.stack(samples).T

    def prior_log_prob(self, values):
        logp = 0

        for i, distribution in enumerate(self.stack):
            logp += distribution.log_prob(values[..., i])

        return logp

    @partial(jax.jit, static_argnums=(0,))
    def simulator(self, theta, seed):
        (_, simulation), score = get_samples_and_scores(
            self.model,
            seed,
            batch_size=1,
            score_type=self.score_type,
            thetas=theta,
            with_noise=True,
        )

        compressed_sim, _ = self.compressor.apply(
            self.params_compressor, self.opt_state, None, simulation["y"]
        )

        simulation["y"] = compressed_sim
        simulation["score"] = score

        return simulation

    def build_dataset(self, batch_size, seed, use_prior=True, proposal_sample=None):
        print("... building dataset")

        new_batch_size = batch_size + batch_size // 10
        keys = jax.random.split(seed, new_batch_size + 1)

        if use_prior:
            theta_sample = self.prior_sample((new_batch_size,), keys[-1])
        else:
            inds = np.random.randint(0, len(proposal_sample), new_batch_size)
            theta_sample = proposal_sample[inds]

        data = {"theta": [], "y": [], "score": []}

        for i, theta in enumerate(tqdm(theta_sample)):
            data_tmp = self.simulator(theta.reshape([1, 6]), keys[i])
            data["theta"].append(data_tmp["theta"].squeeze())
            data["y"].append(data_tmp["y"].squeeze())
            data["score"].append(data_tmp["score"].squeeze())

        data["theta"] = jnp.stack(data["theta"])
        data["y"] = jnp.stack(data["y"])
        data["score"] = jnp.stack(data["score"])

        inds = jnp.unique(jnp.where(jnp.isnan(data["score"]))[0])
        data["y"] = jnp.delete(data["y"], inds, axis=0)[:batch_size]
        data["score"] = jnp.delete(data["score"], inds, axis=0)[:batch_size]
        data["theta"] = jnp.delete(data["theta"], inds, axis=0)[:batch_size]

        print("... done ✓")

        return data
