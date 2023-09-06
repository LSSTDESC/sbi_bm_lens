import jax
import jax.numpy as jnp
import haiku as hk


class MomentNetwork(hk.Module):
    def __init__(
        self,
        *args,
        layers=[128, 128],
        batch_norm=[None, None],
        activation=jax.nn.leaky_relu,
        **kwargs
    ):
        self.layers = layers
        self.bn = batch_norm
        self.activation = activation
        super().__init__(*args, **kwargs)

    def __call__(self, theta, y, is_training):
        net = jnp.concatenate([theta, y], axis=-1)
        for i, layer_size in enumerate(self.layers):
            net = self.activation(hk.Linear(layer_size, name="layer%d" % i)(net))
            if self.bn[i] != None:
                net = self.bn[i](net, is_training)

        net = hk.Linear(6)(net)

        return net.squeeze()
