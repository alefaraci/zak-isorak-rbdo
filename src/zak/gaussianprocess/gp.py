"""Kriging (Gaussian-process) surrogates of the limit-state functions, built with BoTorch.

One surrogate is trained per limit state, in the augmented space of the design variables and of
the random variables.
"""

from functools import partial

from attrs import field, define
from torch import Tensor
from botorch.fit import fit_gpytorch_mll, fit_gpytorch_mll_torch
from torch.optim import Adam
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models import SingleTaskGP
from torch._prims_common import DeviceLikeType
from botorch.models.transforms import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.models.utils.gpytorch_modules import get_covar_module_with_dim_scaled_prior
import torch
import gpytorch

from ..utils import update_live_simple


@define(slots=True, weakref_slot=False, hash=False, cache_hash=False, kw_only=True, eq=False)
class GPRBoTorch:
    """Kriging surrogate of one limit-state function.

    The model is a BoTorch `SingleTaskGP` with a constant mean, inputs normalised to the unit
    cube, standardised outputs and a fixed, very small noise variance, so that it interpolates
    the training points.

    The kernel is selected by name:

    * "RBFKernel" and "MaternKernel": Gaussian and Matern-5/2 kernels built by BoTorch with its
      dimension-scaled log-normal prior on the lengthscales (no output scale);
    * "Gaussian", "MaternKernel32" and "MaternKernel52": plain GPyTorch Gaussian, Matern-3/2 and
      Matern-5/2 kernels wrapped in a `ScaleKernel`, without priors;
    * "MaternKernel52GammaPrior": Matern-5/2 kernel wrapped in a `ScaleKernel`, with Gamma priors
      on the lengthscales and on the output scale.

    All kernels are anisotropic (one lengthscale per input dimension).

    Attributes:
        covar_module: Name of the kernel.
        meta_bounds: Lower and upper bounds of the inputs, one row per input dimension, used to
            normalise them.
        device: Device the model is trained and evaluated on.
        model: The BoTorch model, available once `fit` has been called.
    """

    covar_module: str
    meta_bounds: Tensor
    device: DeviceLikeType = field(default="cpu")
    model: SingleTaskGP = field(init=False)

    def __attrs_post_init__(self) -> None:
        """Convert the device to a `torch.device`."""
        self.device = torch.device(self.device)

    def fit(self, train_x: Tensor, train_y: Tensor, device: DeviceLikeType | None = None) -> None:
        """Build the model on the training points and fit its hyperparameters.

        The hyperparameters maximise the exact marginal log-likelihood.

        Args:
            train_x: Training inputs, of shape (n, d).
            train_y: Values of the limit state at the training inputs, n values.
            device: Device to train on; when given, it replaces the current device.

        Raises:
            ValueError: If the name of the kernel is unknown.
        """
        if device is not None:
            self.device = torch.device(device)

        dim = train_x.shape[-1]
        match self.covar_module:
            # The first two options use BoTorch's dimension-scaled log-normal lengthscale prior;
            # the others are plain GPyTorch kernels wrapped in a ScaleKernel.
            case "RBFKernel":
                covar_module = get_covar_module_with_dim_scaled_prior(
                    ard_num_dims=dim,
                    batch_shape=torch.Size(),
                    use_rbf_kernel=True,
                )
            case "MaternKernel":
                covar_module = get_covar_module_with_dim_scaled_prior(
                    ard_num_dims=dim,
                    batch_shape=torch.Size(),
                    use_rbf_kernel=False,
                )
            case "Gaussian":
                covar_module = gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.RBFKernel(ard_num_dims=dim),
                )
            case "MaternKernel32":
                covar_module = gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=dim),
                )
            case "MaternKernel52":
                covar_module = gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=dim),
                )
            case "MaternKernel52GammaPrior":
                covar_module = gpytorch.kernels.ScaleKernel(
                    base_kernel=gpytorch.kernels.MaternKernel(
                        nu=2.5,
                        ard_num_dims=dim,
                        lengthscale_prior=gpytorch.priors.torch_priors.GammaPrior(3.0, 6.0),
                    ),
                    outputscale_prior=gpytorch.priors.torch_priors.GammaPrior(2.0, 0.15),
                )
            case _:
                msg = f"Unknown covariance module: {self.covar_module}"
                raise ValueError(msg)

        # The limit states are deterministic: the noise variance is fixed to a small value that
        # only acts as a numerical jitter.
        self.model = SingleTaskGP(
            train_X=train_x.clone().to(self.device),
            train_Y=train_y.clone().reshape(-1, 1).to(self.device),
            train_Yvar=torch.full_like(train_y.clone().reshape(-1, 1).to(self.device), 1e-7),
            input_transform=Normalize(
                d=train_x.shape[-1],
                bounds=self.meta_bounds.t().to(self.device),
            ),
            outcome_transform=Standardize(m=1, min_stdv=1e-12),
            covar_module=covar_module,
            mean_module=gpytorch.means.ConstantMean(),
        )
        self.model.to(self.device)
        mll = ExactMarginalLogLikelihood(likelihood=self.model.likelihood, model=self.model)
        mll = mll.to(self.device)
        # BoTorch's default fit (L-BFGS-B) can fail, e.g. on an ill-conditioned covariance matrix.
        # In that case the hyperparameters are fitted with Adam instead, whatever the error.
        try:
            fit_gpytorch_mll(
                mll,
                optimizer_kwargs={
                    "options": {
                        "maxiter": 500,
                        "ftol": 1e-6,
                    },
                },
            )
        except:
            update_live_simple([{"content": "Failed to fit model w/ botorch, using torch"}])
            fit_gpytorch_mll_torch(mll, step_limit=500, optimizer=partial(Adam, lr=0.001))

    def predict(
        self,
        test_x: Tensor,
        device: DeviceLikeType | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Evaluate the posterior mean and variance of the surrogate.

        Args:
            test_x: Points to evaluate, of shape (n, d).
            device: Device to predict on; when given, the model is moved to it.

        Returns:
            The posterior mean and the posterior variance, n values each.
        """
        if device is not None:
            self.device = torch.device(device)
            self.model.to(self.device)
            self.model.train().to(self.device)
        self.model.eval()
        # The enrichment criteria depend on the predictive variance: the fast (Lanczos-based)
        # approximations are disabled so that it is computed exactly, with a Cholesky
        # factorisation and a minimal jitter.
        with (
            torch.no_grad(),
            gpytorch.settings.fast_pred_var(state=False),
            gpytorch.settings.fast_computations(covar_root_decomposition=False),
            gpytorch.settings.cholesky_jitter(1e-10),
        ):
            posterior = self.model.posterior(test_x.clone().to(self.device))
            mean, variance = posterior.mean.squeeze(-1), posterior.variance.squeeze(-1)
            return mean, variance

    def fn(self, test_x: Tensor, device: DeviceLikeType | None = None) -> Tensor:
        """Evaluate the posterior mean, so that the surrogate can stand in for the limit state."""
        return self.predict(test_x, device)[0]


@define(slots=True, weakref_slot=False, hash=False, cache_hash=False, kw_only=True, eq=False)
class ListGPRBoTorch:
    """Kriging surrogates of all the limit-state functions of a problem, one per limit state.

    Attributes:
        covar_module: Names of the kernels, one per limit state (see `GPRBoTorch`).
        meta_bounds: Lower and upper bounds of the inputs, one row per input dimension.
        ng: Number of limit states.
        device: Device the models are trained and evaluated on.
        hard: The surrogates, in the order of the limit states.
    """

    covar_module: list[str]
    meta_bounds: Tensor
    ng: int
    device: DeviceLikeType = field(default="cpu")
    hard: list[GPRBoTorch] = field(init=False)

    def __attrs_post_init__(self) -> None:
        """Create one untrained surrogate per limit state."""
        self.hard = [
            GPRBoTorch(
                covar_module=self.covar_module[k],
                meta_bounds=self.meta_bounds,
                device=self.device,
            )
            for k in range(self.ng)
        ]

    def train(
        self,
        DoE: list[Tensor],
        train_y: list[Tensor],
        device: DeviceLikeType | None = None,
    ) -> None:
        """Fit every surrogate on its own design of experiments.

        Args:
            DoE: Training inputs, one tensor of shape (n_k, d) per limit state.
            train_y: Values of the limit states at the training inputs, one tensor per limit
                state.
            device: Device to train on; when given, it replaces the current device.
        """
        for k, gpr_k in enumerate(self.hard):
            gpr_k.fit(train_x=DoE[k], train_y=train_y[k], device=device)

    def evaluate_surrogate(
        self,
        test_x: Tensor,
        device: DeviceLikeType | None = None,
    ) -> list[Tensor]:
        """Evaluate the posterior mean and variance of every surrogate.

        Args:
            test_x: Points to evaluate, of shape (..., d).
            device: Device to predict on; when given, the models are moved to it.

        Returns:
            A list with the posterior means and the posterior variances, each of shape
            (..., ng): the last dimension runs over the limit states.
        """
        # Each surrogate predicts on the points flattened to (n, d). The `zip` regroups the
        # (mean, variance) pairs into all the means and all the variances, which are stacked
        # along a new last dimension and reshaped to the batch shape of `test_x`.
        return list(  # noqa: C417
            map(
                lambda x: torch.stack(x, dim=-1).view(torch.Size((*test_x.shape[:-1], self.ng))),
                zip(
                    *[
                        gpr_k.predict(test_x=test_x.view(-1, test_x.shape[-1]), device=device)
                        for gpr_k in self.hard
                    ],
                    strict=False,
                ),
            ),
        )
