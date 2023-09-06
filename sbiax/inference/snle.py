import jax
import jax.numpy as jnp
import optax
import tensorflow_probability as tfp
import numpy as np

tfp = tfp.experimental.substrates.jax


class SNLE:
    def __init__(self, NDE, init_params_nde, dim):
        self.NDE = NDE
        self.data = {}
        self.params_nde = init_params_nde
        self.dim = dim
        self.loss_all_round = []

    def log_prob_fn(self, params, theta, y):
        return self.NDE.apply(params, theta, y)

    def _loss_nll_and_score(self, params, mu, batch, score, score_weight):
        lp, out = jax.vmap(
            jax.value_and_grad(
                lambda theta, x: self.log_prob_fn(
                    params, theta.reshape([1, self.dim]), x.reshape([1, self.dim])
                ).squeeze()
            )
        )(mu, batch)

        return (
            -jnp.mean(lp) + score_weight * jnp.sum((out - score) ** 2, axis=-1).mean()
        )

    def _loss_nll(self, params, mu, batch, score, score_weight):
        lp = self.log_prob_fn(params, mu, batch)
        return -jnp.mean(lp)

    def _check_cvg(self, epoch, val_log_prob):
        # (this function is from https://github.com/mackelab/sbi)
        converged = False

        if epoch == 0 or val_log_prob < self._best_val_log_prob:
            self._best_val_log_prob = val_log_prob
            self._epochs_since_last_improvement = 0
        else:
            self._epochs_since_last_improvement += 1

        # If no validation improvement over many epochs, stop training.
        if self._epochs_since_last_improvement > 200:
            converged = True

        return converged

    def train(
        self,
        dataset,
        score_weight=0,
        round_number=0,
        learning_rate=1e-3,
        batch_size=128,
        max_iter=1e4,
    ):
        self.round_number = round_number

        mu = dataset["theta"]
        batch = dataset["y"]
        if score_weight != 0:
            score = dataset["score"]
            loss_fn = self._loss_nll_and_score
        else:
            score = None
            loss_fn = self._loss_nll

        if round_number > 0:
            self.data["theta"] = jnp.concatenate([self.data["theta"], mu], axis=0)
            self.data["y"] = jnp.concatenate([self.data["y"], batch], axis=0)
            if score_weight != 0:
                self.data["score"] = jnp.concatenate(
                    [self.data["score"], score], axis=0
                )
        else:
            self.data["theta"] = mu
            self.data["y"] = batch
            if score_weight != 0:
                self.data["score"] = score

        nb_simu = len(self.data["theta"])

        print("nb of simulations used for training: ", nb_simu)

        optimizer = optax.adam(learning_rate)

        @jax.jit
        def update(params, opt_state, mu, batch, score, score_weight):
            """Single SGD update step."""
            loss, grads = jax.value_and_grad(loss_fn)(
                params, mu, batch, score, score_weight
            )
            updates, new_opt_state = optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)

            return loss, new_params, new_opt_state

        params_nde = self.params_nde
        opt_state = optimizer.init(params_nde)

        batch_loss = []

        keep_going = True
        epoch = 0
        batch = 0

        print("... training nde")
        while keep_going and epoch < max_iter:
            inds = np.random.randint(0, nb_simu, batch_size)
            ex_theta = self.data["theta"][inds]
            ex_y = self.data["y"][inds]
            if score_weight != 0:
                ex_score = self.data["score"][inds]
            else:
                ex_score = None

            batch += 1

            l, params_nde, opt_state = update(
                params_nde, opt_state, ex_theta, ex_y, ex_score, score_weight
            )

            if jnp.isnan(l):
                print("/!\ NaN in training")
                break

            if batch % (nb_simu // batch_size) == 0:
                batch_loss.append(l)

                if self._check_cvg(epoch, batch_loss[-1]):
                    keep_going = False
                else:
                    keep_going = True

                epoch += 1

        print("nb of epoch: ", epoch)
        print("... done ✓")

        self.loss = jnp.array(batch_loss)

        if self.round_number == 0:
            self.loss_all_round = self.loss
        else:
            self.loss_all_round = jnp.concatenate([self.loss_all_round, self.loss])

        self.params_nde = params_nde

    def sample(
        self,
        log_prob_prior,
        init,
        observation,
        seed,
        num_chains=5,
        num_results=1e4,
        num_burnin_steps=5e2,
    ):
        print("... running hmc")

        @jax.vmap
        def unnormalized_log_prob(theta):
            prior = log_prob_prior(theta)

            likelihood = self.log_prob_fn(
                self.params_nde,
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
                current_state=init * jnp.ones([num_chains, self.dim]),
                kernel=adaptive_hmc,
                trace_fn=lambda _, pkr: pkr.inner_results.is_accepted,
                seed=seed,
            )

            return samples, is_accepted

        samples_hmc, is_accepted_hmc = run_chain()
        sample_nd = samples_hmc[is_accepted_hmc]

        print("... done ✓")

        self.posterior_samples = sample_nd

        return sample_nd
