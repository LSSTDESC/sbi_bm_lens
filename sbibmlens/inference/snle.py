import jax
import jax.numpy as jnp
import optax
import tensorflow_probability as tfp
import numpy as np
from tqdm import tqdm

tfp = tfp.experimental.substrates.jax


class SNLE:
    def __init__(self, NDE, init_params_nde, dim):
        self.NDE = NDE
        self.params = init_params_nde
        self.dim = dim

    def log_prob_fn(self, params, theta, y):
        return self.NDE.apply(params, theta, y)

    def loss_nll_and_score(self, params, mu, batch, score, weight_score):
        lp, out = jax.vmap(
            jax.value_and_grad(
                lambda theta, x: self.log_prob_fn(
                    params, theta.reshape([1, self.dim]), x.reshape([1, self.dim])
                ).squeeze()
            )
        )(mu, batch)

        return (
            -jnp.mean(lp) + weight_score * jnp.sum((out - score) ** 2, axis=-1).mean()
        )

    def loss_nll(self, params, mu, batch, score, weight_score):
        lp = self.log_prob_fn(params, mu, batch)

        return -jnp.mean(lp)

    def train(
        self, data, learning_rate, total_steps=30_000, batch_size=128, score_weight=0
    ):
        dataset_theta = data["theta"]
        dataset_y = data["y"]
        if score_weight != 0:
            dataset_score = data["score"]
            loss_fn = self.loss_nll_and_score
        else:
            loss_fn = self.loss_nll

        nb_simu = len(dataset_theta)

        print("nb of simulations used for training: ", nb_simu)

        params = self.params
        optimizer = optax.adam(learning_rate)
        opt_state = optimizer.init(params)

        @jax.jit
        def update(params, opt_state, mu, batch, score, weight_score):
            """Single SGD update step."""
            loss, grads = jax.value_and_grad(loss_fn)(
                params, mu, batch, score, weight_score
            )
            updates, new_opt_state = optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)

            return loss, new_params, new_opt_state

        print("... start training")

        batch_loss = []
        lr_scheduler_store = []
        pbar = tqdm(range(total_steps))

        for batch in pbar:
            inds = np.random.randint(0, nb_simu, batch_size)
            ex_theta = dataset_theta[inds]
            ex_y = dataset_y[inds]
            if score_weight != 0:
                ex_score = dataset_score[inds]
            else:
                ex_score = None

            if not jnp.isnan(ex_y).any():
                l, params, opt_state = update(
                    params, opt_state, ex_theta, ex_y, ex_score, score_weight
                )

                batch_loss.append(l)
                pbar.set_description(f"loss {l:.3f}")

                if jnp.isnan(l):
                    break

        self.params = params
        self.loss = batch_loss

        print("done ✓")

    def sample(
        self,
        log_prob_prior,
        observation,
        init_point,
        key,
        num_results=3e4,
        num_burnin_steps=5e2,
        num_chains=12,
    ):
        print("... running hmc")

        @jax.vmap
        def unnormalized_log_prob(theta):
            prior = log_prob_prior(theta)

            likelihood = self.log_prob_fn(
                self.params,
                theta.reshape([1, self.dim]),
                jnp.array(observation).reshape([1, self.dim]),
            )

            return likelihood + prior

        # Initialize the HMC transition kernel.
        adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(
            tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn=unnormalized_log_prob,
                num_leapfrog_steps=3,
                step_size=1e-2,
            ),
            num_adaptation_steps=int(num_burnin_steps * 0.8),
        )

        # Run the chain (with burn-in).
        # @jax.jit
        def run_chain():
            # Run the chain (with burn-in).
            samples, is_accepted = tfp.mcmc.sample_chain(
                num_results=num_results,
                num_burnin_steps=num_burnin_steps,
                current_state=jnp.array(init_point) * jnp.ones([num_chains, self.dim]),
                kernel=adaptive_hmc,
                trace_fn=lambda _, pkr: pkr.inner_results.is_accepted,
                seed=key,
            )

            return samples, is_accepted

        samples_hmc, is_accepted_hmc = run_chain()
        sample_nd = samples_hmc[is_accepted_hmc]

        print("done ✓")

        return sample_nd
