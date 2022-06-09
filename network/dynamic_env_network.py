import abc
import torch
from torch import nn
import numpy as np
from torch.nn import functional as F
from typing import Optional, Sequence, Tuple, Union, cast, List


class Model(nn.Module, abc.ABC):
    def __init__(self):
        super().__init__()

    def forward(self, x, **kwargs):
        pass

    def sample(self, x, deterministic: bool = False, **kwargs):
        return self.forward(x)[0],

    def reset(self, x, **kwargs):
        return cast(torch.Tensor, x)

    @abc.abstractmethod
    def loss(self, model_in, target: Optional[torch.Tensor] = None, ) -> torch.Tensor:
        """Computes a loss that can be used to update the model using backpropagation.
        Args:
            model_in (tensor or batch of transitions): the inputs to the model.
            target (tensor, optional): the expected output for the given inputs, if it
                cannot be computed from ``model_in``.
        Returns:
            (tensor): a loss tensor.
        """
    # training process
    def update(self, model_in, optimizer: torch.optim.Optimizer, target: Optional[torch.Tensor] = None, ) -> float:
        """Updates the model using backpropagation with given input and target tensors.
        Provides a basic update function, following the steps below:
        .. code-block:: python
           optimizer.zero_grad()
           loss = self.loss(model_in, target)
           loss.backward()
           optimizer.step()
        Args:
            model_in (tensor or batch of transitions): the inputs to the model.
            optimizer (torch.optimizer): the optimizer to use for the model.
            target (tensor or sequence of tensors): the expected output for the given inputs, if it
                cannot be computed from ``model_in``.
        Returns:
             (float): the numeric value of the computed loss.
        """
        optimizer = cast(torch.optim.Optimizer, optimizer)
        self.train()
        optimizer.zero_grad()
        loss = self.loss(model_in, target)
        loss.backward()
        optimizer.step()
        return loss.item()

    def __len__(self):
        return 1

    def save(self, path):
        """Saves the model to the given path."""
        torch.save(self.state_dict(), path)

    def load(self, path):
        """Loads the model from the given path."""
        self.load_state_dict(torch.load(path))


class Ensemble(Model, abc.ABC):
    def __init__(self, num_members: int, device: Union[str, torch.device], propagation_method: str,
                 deterministic: bool = False, *args, **kwargs, ):
        super().__init__()
        self.num_members = num_members
        self.propagation_method = propagation_method
        self.device = torch.device(device)
        self.deterministic = deterministic
        self.to(device)

    def forward(self, x, **kwargs) -> Tuple[torch.Tensor, ...]:
        pass

    @abc.abstractmethod
    def loss(self, model_in, target: Optional[torch.Tensor] = None, ) -> torch.Tensor:
        """Computes a loss that can be used to update the model using backpropagation."""

    def __len__(self):
        return self.num_members

    def set_elite(self, elite_models: Sequence[int]):
        """For ensemble models, indicates if some models should be considered elite."""
        pass

    def set_propagation_method(self, propagation_method: Optional[str] = None):
        self.propagation_method = propagation_method

    def sample(self, x, deterministic: bool = False, rng: Optional[torch.Generator] = None, **kwargs, ):
        """Samples an output of the dynamics model from the modeled Gaussian."""
        if deterministic or self.deterministic:
            return self.forward(x, rng=rng)[0],

        # assert rng is not None
        means, logvars = self.forward(x, rng=rng)
        variances = logvars.exp()
        stds = torch.sqrt(variances)
        return torch.normal(means, stds, generator=rng),


class EnsembleLinearLayer(nn.Module):
    """Efficient linear layer for ensemble models."""
    def __init__(self, num_members: int, in_size: int, out_size: int, bias: bool = True):
        super().__init__()
        self.num_members = num_members
        self.in_size = in_size
        self.out_size = out_size

        self.weight = nn.Parameter(torch.rand(self.num_members, self.in_size, self.out_size))
        if bias:
            self.bias = nn.Parameter(torch.rand(self.num_members, 1, self.out_size))
            self.use_bias = True
        else:
            self.use_bias = False

        # ???
        self.elite_models: List[int] = None
        self.use_only_elite = False

    # weight * x + b
    def forward(self, x):
        if self.use_only_elite:
            xw = x.matmul(self.weight[self.elite_models, ...])
            if self.use_bias:
                return xw + self.bias[self.elite_models, ...]
            else:
                return xw
        else:
            xw = x.matmul(self.weight)
            if self.use_bias:
                return xw + self.bias
            else:
                return xw

    def extra_repr(self) -> str:
        return ("num_members={self.num_members}, in_size={self.in_size}, "
                f"out_size={self.out_size}, bias={self.use_bias}")

    def set_elite(self, elite_models: Sequence[int]):
        self.elite_models = list(elite_models)

    def toggle_use_only_elite(self):
        self.use_only_elite = not self.use_only_elite


class GaussianMLP(Ensemble):
    _ELITE_FNAME = "elite_models.pkl"

    def __init__(self, in_size: int, out_size: int, device: Union[str, torch.device], num_layers: int = 4,
                 ensemble_size: int = 1, hid_size: int = 200, use_silu: bool = False, deterministic: bool = False,
                 propagation_method: Optional[str] = None, learn_logvar_bounds: bool = False, ):

        super().__init__(ensemble_size, device, propagation_method, deterministic=deterministic)

        # the input/output size of the dynamic model
        self.in_size = in_size
        self.out_size = out_size

        # definition of activation function
        activation_cls = nn.SiLU if use_silu else nn.ReLU

        # creat ensemble linear layer, return a nn.Module class
        def create_linear_layer(l_in, l_out):
            return EnsembleLinearLayer(ensemble_size, l_in, l_out)

        # definition of the first hidden layer
        hidden_layers = [nn.Sequential(create_linear_layer(in_size, hid_size), activation_cls())]
        # definition of other hidden layers
        for i in range(num_layers - 1):
            hidden_layers.append(nn.Sequential(create_linear_layer(hid_size, hid_size), activation_cls(), ))
        self.hidden_layers = nn.Sequential(*hidden_layers)

        # mean_and_logvar is the output layer
        if deterministic:
            self.mean_and_logvar = create_linear_layer(hid_size, out_size)
        else:
            self.mean_and_logvar = create_linear_layer(hid_size, 2 * out_size)
            self.min_logvar = nn.Parameter(-10 * torch.ones(1, out_size), requires_grad=learn_logvar_bounds)
            self.max_logvar = nn.Parameter(0.5 * torch.ones(1, out_size), requires_grad=learn_logvar_bounds)

        # set initial weight parameters
        self.apply(truncated_normal_init)
        self.to(self.device)

        # ???
        self.elite_models: List[int] = None
        self._propagation_indices: torch.Tensor = None

    # ???
    def _maybe_toggle_layers_use_only_elite(self, only_elite: bool):
        if self.elite_models is None:
            return
        if self.num_members > 1 and only_elite:
            for layer in self.hidden_layers:
                # each layer is (linear layer, activation_func)
                layer[0].set_elite(self.elite_models)
                layer[0].toggle_use_only_elite()
            self.mean_and_logvar.set_elite(self.elite_models)
            self.mean_and_logvar.toggle_use_only_elite()

    # forward function, return mean and logvar (not deterministic)
    def _default_forward(self, x: torch.Tensor, only_elite: bool = False, **_kwargs):
        self._maybe_toggle_layers_use_only_elite(only_elite)
        x = self.hidden_layers(x)
        mean_and_logvar = self.mean_and_logvar(x)
        self._maybe_toggle_layers_use_only_elite(only_elite)
        if self.deterministic:
            return mean_and_logvar, None
        else:
            mean = mean_and_logvar[..., : self.out_size]
            logvar = mean_and_logvar[..., self.out_size:]
            logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
            logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)
            return mean, logvar

    # forward function with indices
    def _forward_from_indices(self, x: torch.Tensor, model_shuffle_indices: torch.Tensor) -> Tuple[
        torch.Tensor, Optional[torch.Tensor]]:
        _, batch_size, _ = x.shape
        num_models = (len(self.elite_models) if self.elite_models is not None else len(self))

        shuffled_x = x[:, model_shuffle_indices, ...].view(num_models, batch_size // num_models, -1)
        # the input of forward is shuffled_x
        mean, logvar = self._default_forward(shuffled_x, only_elite=True)

        # note that mean and logvar are shuffled
        mean = mean.view(batch_size, -1)
        mean[model_shuffle_indices] = mean.clone()  # invert the shuffle

        if logvar is not None:
            logvar = logvar.view(batch_size, -1)
            logvar[model_shuffle_indices] = logvar.clone()  # invert the shuffle

        return mean, logvar

    # forward function for ensemble (when using propagation method)
    def _forward_ensemble(self, x: torch.Tensor):
        if self.propagation_method is None:
            # call default forward function
            mean, logvar = self._default_forward(x, only_elite=False)
            if self.num_members == 1:
                mean = mean[0]
                logvar = logvar[0] if logvar is not None else None
            return mean, logvar
        # using propagation method
        assert x.ndim == 2
        model_len = (len(self.elite_models) if self.elite_models is not None else len(self))
        if x.shape[0] % model_len != 0:
            raise ValueError(f"GaussianMLP ensemble requires batch size to be a multiple of the number of models. "
                             f"Current batch size is {x.shape[0]} for {model_len} models.")

        # input process
        x = x.unsqueeze(0)
        if self.propagation_method == "random_model":
            # random sample
            model_indices = torch.randperm(x.shape[1], device=self.device)
            return self._forward_from_indices(x, model_indices)
        if self.propagation_method == "fixed_model":
            # indices have specified in advance
            return self._forward_from_indices(x, self._propagation_indices)
        if self.propagation_method == "expectation":
            # return the expected mean and logvar
            mean, logvar = self._default_forward(x, only_elite=True)
            return mean.mean(dim=0), logvar.mean(dim=0)
        raise ValueError(f"Invalid propagation method {self.propagation_method}.")

    def forward(self, x: torch.Tensor, rng: Optional[torch.Generator] = None, use_propagation: bool = True,) -> Tuple[torch.Tensor, torch.Tensor]:
        if use_propagation:
            return self._forward_ensemble(x)
        return self._default_forward(x)

    # compute MSE loss
    def _mse_loss(self, model_in: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert model_in.ndim == target.ndim
        if model_in.ndim == 2:  # add model dimension
            model_in = model_in.unsqueeze(0)
            target = target.unsqueeze(0)
        pred_mean, _ = self.forward(model_in, use_propagation=False)
        return F.mse_loss(pred_mean, target, reduction="none").sum((1, 2)).sum()

    # compute NLL loss
    def _nll_loss(self, model_in: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert model_in.ndim == target.ndim
        if model_in.ndim == 2:  # add ensemble dimension
            model_in = model_in.unsqueeze(0)
            target = target.unsqueeze(0)
        pred_mean, pred_logvar = self.forward(model_in, use_propagation=False)

        if target.shape[0] != self.num_members:
            target = target.repeat(self.num_members, 1, 1)

        # sum over ensemble dimension
        nll = (gaussian_nll(pred_mean, pred_logvar, target, reduce=False).mean((1, 2)).sum())
        nll += 0.01 * (self.max_logvar.sum() - self.min_logvar.sum())
        return nll

    # compute loss
    def loss(self, model_in: torch.Tensor, target: Optional[torch.Tensor] = None,) -> torch.Tensor:
        if self.deterministic:
            return self._mse_loss(model_in, target)
        else:
            return self._nll_loss(model_in, target)

    def reset(self, x: torch.Tensor, rng: Optional[torch.Generator] = None) -> torch.Tensor:
        assert rng is not None
        self._propagation_indices = self._sample_propagation_indices(x.shape[0], rng)
        return x

    def _sample_propagation_indices(self, batch_size: int, _rng: torch.Generator) -> torch.Tensor:
        """Returns a random permutation of integers in [0, ``batch_size``)."""
        model_len = (len(self.elite_models) if self.elite_models is not None else len(self))
        if batch_size % model_len != 0:
            raise ValueError(
                "To use GaussianMLP's ensemble propagation, the batch size must "
                "be a multiple of the number of models in the ensemble.")
        return torch.randperm(batch_size, device=self.device)

    def set_elite(self, elite_indices: Sequence[int]):
        if len(elite_indices) != self.num_members:
            self.elite_models = list(elite_indices)


def truncated_normal_init(m: nn.Module):
    """Initializes the weights of the given module using a truncated normal distribution."""
    if isinstance(m, nn.Linear):
        input_dim = m.weight.data.shape[0]
        stddev = 1 / (2 * np.sqrt(input_dim))
        truncated_normal(m.weight.data, std=stddev)
        m.bias.data.fill_(0.0)
    if isinstance(m, EnsembleLinearLayer):
        num_members, input_dim, _ = m.weight.data.shape
        stddev = 1 / (2 * np.sqrt(input_dim))
        for i in range(num_members):
            truncated_normal(m.weight.data[i], std=stddev)
        m.bias.data.fill_(0.0)


def truncated_normal(tensor: torch.Tensor, mean: float = 0, std: float = 1):
    torch.nn.init.normal_(tensor, mean=mean, std=std)
    while True:
        cond = torch.logical_or(tensor < mean - 2 * std, tensor > mean + 2 * std)
        if not torch.sum(cond):
            break
        tensor = torch.where(cond, torch.nn.init.normal_(torch.ones(tensor.shape, device=tensor.device), mean=mean, std=std), tensor, )
    return tensor


def gaussian_nll(pred_mean: torch.Tensor, pred_logvar: torch.Tensor, target: torch.Tensor, reduce: bool = True,) \
        -> torch.Tensor:

    l2 = F.mse_loss(pred_mean, target, reduction="none")
    inv_var = (-pred_logvar).exp()
    losses = l2 * inv_var + pred_logvar
    if reduce:
        return losses.sum(dim=1).mean()
    return losses
