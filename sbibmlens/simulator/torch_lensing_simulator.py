from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
import torch
from torch.distributions import Normal
from sbibmlens.simulator.torch_distributions import TruncatedNormal
from sbi_lens.simulator.utils import get_samples_and_scores


class PytorchCompressedSimulator:
    def __init__(self, model, compressor, params_compressor, opt_state, device="cpu"):
        self.model = model

        self.compressor = compressor
        self.params_compressor = params_compressor
        self.opt_state = opt_state

        self.device = device

    @partial(jax.jit, static_argnums=(0,))
    def _get_simulation(self, theta, key):
        (log_prob, samples), gradients = get_samples_and_scores(
            self.model,
            key,
            batch_size=1,
            score_type="density",
            thetas=theta,
            with_noise=True,
        )
        return samples["y"]

    def simulator(self, theta):
        if self.device != "cpu":
            theta = theta.cpu()  # super super slow

        theta = jnp.array(theta).reshape([-1, 6])

        random = int(np.random.randint(low=0, high=1e5, size=1))
        key = jax.random.PRNGKey(random)
        simulation = self._get_simulation(theta, key)

        compressed_sim, _ = self.compressor.apply(
            self.params_compressor, self.opt_state, None, simulation
        )

        return torch.tensor(np.array(compressed_sim).squeeze())


class PytorchPrior:
    def __init__(self, device="cpu"):
        self.device = device
        self.omega_c = TruncatedNormal(
            torch.tensor(0.2664, device=self.device), 0.2, low=0, device=self.device
        )
        self.omega_b = Normal(torch.tensor(0.0492, device=self.device), 0.006)
        self.sigma_8 = Normal(torch.tensor(0.831, device=self.device), 0.14)
        self.h_0 = Normal(torch.tensor(0.6727, device=self.device), 0.063)
        self.n_s = Normal(torch.tensor(0.9645, device=self.device), 0.08)
        self.w_0 = TruncatedNormal(
            torch.tensor(-1.0, device=self.device),
            0.9,
            low=-2.0,
            high=-0.3,
            device=self.device,
        )

        self.stack = [
            self.omega_c,
            self.omega_b,
            self.sigma_8,
            self.h_0,
            self.n_s,
            self.w_0,
        ]

    def sample(self, sample_shape):
        samples = []

        for dist in self.stack:
            samples.append(dist.sample(sample_shape))

        return torch.stack(samples).T

    def log_prob(self, values):
        logp = 0

        for i, dist in enumerate(self.stack):
            logp += dist.log_prob(values[..., i])

        return logp
