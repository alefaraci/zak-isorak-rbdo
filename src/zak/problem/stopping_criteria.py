"""Error-based Stopping Criterion (ESC) for the adaptive enrichment of Kriging surrogates.

The equation and corollary numbers quoted in this module refer to:

    Z. Wang and A. Shafieezadeh (2019). ESC: an efficient error-based stopping criterion for
    kriging-based reliability analysis methods. Structural and Multidisciplinary Optimization.
"""

from scipy import stats
from torch import Tensor
import torch


class ESCStoppingCriterion:
    """Error-based Stopping Criterion (ESC) of Wang and Shafieezadeh (2019).

    A Kriging surrogate classifies each Monte Carlo sample as safe or failed from the sign of its
    predicted mean. ESC bounds, at a given confidence level, the number of samples whose sign is
    wrongly predicted in each of the two classes, and derives from these bounds the maximum
    relative error ``eps_max`` of the estimated failure probability.

    In this class ``S_s`` denotes the number of samples predicted safe that actually fail, and
    ``S_f`` the number of samples predicted failed that are actually safe.
    """

    def __init__(self, alpha=0.05) -> None:
        """Initialize the ESC stopping criterion.

        Args:
            alpha: Significance level of the confidence intervals (0.05 gives 95 % intervals).
        """
        self.alpha = alpha

    def compute_pwse(self, y_hat, sigma_hat) -> Tensor:
        """Compute the probability of wrong sign estimation of each sample (Eq. 18).

        Args:
            y_hat: Predicted means of the Kriging model.
            sigma_hat: Predicted standard deviations of the Kriging model.

        Returns:
            The probabilities ``Phi(-|y_hat| / sigma_hat)``, with ``Phi`` the standard normal
            CDF.
        """
        # The predicted standard deviation vanishes at the training points: clamp it to avoid a
        # division by zero.
        eps = 1e-10
        sigma_hat_stable = torch.clamp(sigma_hat, min=eps)

        ratio = torch.abs(y_hat) / sigma_hat_stable

        return torch.distributions.Normal(0, 1).cdf(-ratio)

    def compute_confidence_intervals(
        self,
        y_hat,
        sigma_hat,
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """Compute the confidence intervals of the numbers of misclassified samples.

        Args:
            y_hat: Predicted means of the Kriging model for one limit state, shape ``(n,)``.
            sigma_hat: Predicted standard deviations of the Kriging model, shape ``(n,)``.

        Returns:
            The confidence intervals ``(lower, upper)`` of ``S_s`` and of ``S_f``.
        """
        # Split the samples into predicted safe and predicted failed.
        safe_indices = (y_hat >= 0).nonzero(as_tuple=True)[0]
        failure_indices = (y_hat < 0).nonzero(as_tuple=True)[0]

        y_hat_s = y_hat[safe_indices]
        sigma_hat_s = sigma_hat[safe_indices]

        y_hat_f = y_hat[failure_indices]
        sigma_hat_f = sigma_hat[failure_indices]

        pwse_s = self.compute_pwse(y_hat_s, sigma_hat_s)
        pwse_f = self.compute_pwse(y_hat_f, sigma_hat_f)

        # Mean and standard deviation of S_s (Eq. 21).
        mu_s = torch.sum(pwse_s)
        sigma_s_squared = torch.sum(pwse_s * (1 - pwse_s))
        sigma_s = torch.sqrt(sigma_s_squared)

        # Mean of S_f (Eq. 22).
        mu_f = torch.sum(pwse_f)

        # S_s is approximated by a normal distribution (Corollary 1) and S_f by a Poisson
        # distribution (Corollary 2).
        ci_s = self._compute_ci_s_normal(mu_s, sigma_s)
        ci_f = self._compute_ci_f_poisson(mu_f)

        return ci_s, ci_f

    def _compute_ci_s_normal(self, mu_s, sigma_s) -> tuple[float, float]:
        """Compute the confidence interval of ``S_s`` from its normal approximation.

        Args:
            mu_s: Mean of ``S_s``.
            sigma_s: Standard deviation of ``S_s``.

        Returns:
            The confidence interval ``(lower, upper)`` of ``S_s`` (Corollary 1, Eq. 27).
        """
        mu_s_val = mu_s.item() if isinstance(mu_s, torch.Tensor) else mu_s
        sigma_s_val = sigma_s.item() if isinstance(sigma_s, torch.Tensor) else sigma_s

        # Standard normal quantile of the two-sided interval (about 1.96 for alpha = 0.05).
        gamma_ci = stats.norm.ppf(1 - self.alpha / 2)

        # A number of samples cannot be negative.
        lower_bound = max(0, mu_s_val - gamma_ci * sigma_s_val)
        upper_bound = mu_s_val + gamma_ci * sigma_s_val

        return (lower_bound, upper_bound)

    def _compute_ci_f_poisson(self, mu_f) -> tuple[float, float]:
        """Compute the confidence interval of ``S_f`` from its Poisson approximation.

        Args:
            mu_f: Mean of ``S_f``.

        Returns:
            The confidence interval ``(lower, upper)`` of ``S_f`` (Corollary 2, Eq. 30).
        """
        mu_f_val = mu_f.item() if isinstance(mu_f, torch.Tensor) else mu_f

        # No wrong sign is expected among the predicted failures: the interval is reduced to
        # zero.
        if mu_f_val == 0:
            return (0, 0)

        lower_bound = stats.poisson.ppf(self.alpha / 2, mu_f_val)
        upper_bound = stats.poisson.ppf(1 - self.alpha / 2, mu_f_val)

        return (lower_bound, upper_bound)

    def check_stopping_criterion(
        self,
        ci_s,
        ci_f,
        Nf,
        max_error_threshold,
    ) -> tuple[bool | Tensor, float | Tensor]:
        """Check whether the maximum relative error of the failure estimate is acceptable.

        Args:
            ci_s: Confidence interval ``(lower, upper)`` of ``S_s``.
            ci_f: Confidence interval ``(lower, upper)`` of ``S_f``.
            Nf: Number of samples predicted as failed by the Kriging model.
            max_error_threshold: Largest acceptable value of the maximum relative error.

        Returns:
            A tuple ``(stop, eps_max)``: whether ``eps_max <= max_error_threshold``, and the
            maximum relative error ``eps_max`` itself.
        """
        # With the upper bounds of the two intervals, the true number of failures lies between
        # Nf - ci_f[1] and Nf + ci_s[1]: eps_max is the largest relative error of Nf with respect
        # to these two extremes.
        eps_max_1 = abs(Nf / (Nf - ci_f[1]) - 1)
        # No predicted failure and none that could be wrong: this side has no error (and the
        # ratio above is 0 / 0).
        if Nf == 0 and ci_f[1] == 0:
            eps_max_1 = 0.0
        eps_max_2 = abs(Nf / (Nf + ci_s[1]) - 1)
        eps_max = max(eps_max_1, eps_max_2)
        return eps_max <= max_error_threshold, eps_max


def esc(mean, var, eps_thr) -> tuple[bool | Tensor, Tensor, int, str, float | Tensor]:
    """Apply ESC to the most uncertain limit state and select its next training point.

    The active limit state is the one holding the sample with the smallest value of the U
    learning function ``U = |mean| / sqrt(var)``. ESC is evaluated for this limit state only,
    and the sample proposed for enrichment is the one with the smallest U.

    Args:
        mean: Surrogate means at the Monte Carlo samples, shape ``(n_samples, n_hard)``.
        var: Surrogate variances at the Monte Carlo samples, shape ``(n_samples, n_hard)``.
        eps_thr: Threshold on the maximum relative error ``eps_max``.

    Returns:
        A tuple ``(stop, pos, activated_gk, log_str, eps_max)``: whether
        ``eps_max <= eps_thr``, the index of the sample proposed for enrichment, the index of
        the active limit state, a message for the console and the maximum relative error.
    """
    U = mean.abs() / var.sqrt()
    activated_gk = int((U.min(dim=0).values).argmin().detach().item())
    criterion = ESCStoppingCriterion(alpha=0.05)
    ci_s, ci_f = criterion.compute_confidence_intervals(
        mean[..., activated_gk],
        var[..., activated_gk].sqrt(),
    )
    Nf = (mean[..., activated_gk] <= 0).sum()
    stopping_criteria, eps_max = criterion.check_stopping_criterion(ci_s, ci_f, Nf, eps_thr)
    pos = U[..., activated_gk].argmin()
    if not stopping_criteria:
        log_str = "📢 LOCALLY: New point can be added w/"
        log_str += f" eps_max = {eps_max:.3e} > {eps_thr} for DoE{activated_gk + 1}"
    else:
        log_str = f"🛑 STOP LEARNING: eps_max = {eps_max:.3e} <= {eps_thr} found!"
    return stopping_criteria, pos, activated_gk, log_str, eps_max
