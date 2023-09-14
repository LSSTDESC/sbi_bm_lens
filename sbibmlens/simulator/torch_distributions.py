from torch.distributions import Normal

# stuffs to re write the prior in pytorch
# these functions are from numpyro:
# https://github.com/pyro-ppl/numpyro/blob/master/numpyro/distributions/truncated.py


class LeftTruncatedDistribution:
    def __init__(self, base_dist, low=0.0, device="cpu", *, validate_args):
        self.base_dist = base_dist
        self.low = low
        self.device = device

    def _tail_prob_at_low(self):
        # if low < loc, returns cdf(low); otherwise returns 1 - cdf(low)
        loc = self.base_dist.loc
        sign = torch.where(loc >= self.low, 1.0, -1.0)
        return self.base_dist.cdf(loc - sign * (loc - self.low))

    def _tail_prob_at_high(self):
        # if low < loc, returns cdf(high) = 1; otherwise returns 1 - cdf(high) = 0
        return torch.where(self.low <= self.base_dist.loc, 1.0, 0.0)

    def sample(self, sample_shape=()):
        u = torch.rand(sample_shape, device=self.device)
        loc = self.base_dist.loc
        sign = torch.where(loc >= self.low, 1.0, -1.0)
        return (1 - sign) * loc + sign * self.base_dist.icdf(
            (1 - u) * self._tail_prob_at_low() + u * self._tail_prob_at_high()
        )

    def log_prob(self, value):
        sign = torch.where(self.base_dist.loc >= self.low, 1.0, -1.0)
        return self.base_dist.log_prob(value) - torch.log(
            sign * (self._tail_prob_at_high() - self._tail_prob_at_low())
        )


class RightTruncatedDistribution:
    def __init__(self, base_dist, high=0.0, device="cpu", *, validate_args):
        self.base_dist = base_dist
        self.high = torch.tensor(high)
        self.device = device

    def _cdf_at_high(self):
        return self.base_dist.cdf(self.high)

    def sample(self, sample_shape=()):
        u = torch.rand(sample_shape, device=self.device)
        return self.base_dist.icdf(u * self._cdf_at_high())

    def log_prob(self, value):
        return self.base_dist.log_prob(value) - torch.log(self._cdf_at_high())


class TwoSidedTruncatedDistribution:
    def __init__(self, base_dist, low=0.0, high=1.0, device="cpu", *, validate_args):
        self.base_dist = base_dist
        self.high = torch.tensor(high)
        self.low = torch.tensor(low)
        self.device = device

    def _tail_prob_at_low(self):
        # if low < loc, returns cdf(low); otherwise returns 1 - cdf(low)
        loc = self.base_dist.loc
        sign = torch.where(loc >= self.low, 1.0, -1.0)
        return self.base_dist.cdf(loc - sign * (loc - self.low))

    def _tail_prob_at_high(self):
        # if low < loc, returns cdf(high); otherwise returns 1 - cdf(high)
        loc = self.base_dist.loc
        sign = torch.where(loc >= self.low, 1.0, -1.0)
        return self.base_dist.cdf(loc - sign * (loc - self.high))

    def _log_diff_tail_probs(self):
        loc = self.base_dist.loc
        sign = torch.where(loc >= self.low, 1.0, -1.0)
        return torch.log(sign * (self._tail_prob_at_high() - self._tail_prob_at_low()))

    def sample(self, sample_shape=()):
        u = torch.rand(sample_shape, device=self.device)
        loc = self.base_dist.loc
        sign = torch.where(loc >= self.low, 1.0, -1.0)
        return (1 - sign) * loc + sign * self.base_dist.icdf(
            (1 - u) * self._tail_prob_at_low() + u * self._tail_prob_at_high()
        )

    def log_prob(self, value):
        return self.base_dist.log_prob(value) - self._log_diff_tail_probs()


def TruncatedDistribution(
    base_dist, low=None, high=None, device="cpu", *, validate_args=None
):
    """
    A function to generate a truncated distribution.

    :param base_dist: The base distribution to be truncated. This should be a univariate
        distribution. Currently, only the following distributions are supported:
        Cauchy, Laplace, Logistic, Normal, and StudentT.
    :param low: the value which is used to truncate the base distribution from below.
        Setting this parameter to None to not truncate from below.
    :param high: the value which is used to truncate the base distribution from above.
        Setting this parameter to None to not truncate from above.
    """
    if high is None:
        if low is None:
            return base_dist
        else:
            return LeftTruncatedDistribution(
                base_dist, low=low, device=device, validate_args=validate_args
            )
    elif low is None:
        return RightTruncatedDistribution(
            base_dist, high=high, device=device, validate_args=validate_args
        )
    else:
        return TwoSidedTruncatedDistribution(
            base_dist, low=low, high=high, device=device, validate_args=validate_args
        )


def TruncatedNormal(
    loc=0.0, scale=1.0, *, low=None, high=None, device="cpu", validate_args=None
):
    return TruncatedDistribution(
        Normal(loc, scale),
        low=low,
        high=high,
        device=device,
        validate_args=validate_args,
    )
