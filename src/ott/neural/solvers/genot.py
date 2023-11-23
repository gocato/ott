import functools
import types
from functools import partial
from typing import (
    Any,
    Callable,
    Dict,
    Literal,
    Mapping,
    Optional,
    Tuple,
    Type,
    Union,
)

import diffrax
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from flax.training.train_state import TrainState
from jax import random
from tqdm import tqdm

from ott.geometry import costs
from ott.neural.models.models import BaseNeuralVectorField
from ott.neural.solvers.base_solver import (
    BaseNeuralSolver,
    ResampleMixin,
    UnbalancednessMixin,
)
from ott.neural.solvers.flows import BaseFlow, ConstantNoiseFlow
from ott.solvers import was_solver
from ott.solvers.linear import sinkhorn
from ott.solvers.quadratic import gromov_wasserstein

Match_fn_T = Callable[[jax.random.PRNGKeyArray, jnp.array, jnp.array],
                      Tuple[jnp.array, jnp.array, jnp.array, jnp.array]]
Match_latent_fn_T = Callable[[jax.random.PRNGKeyArray, jnp.array, jnp.array],
                             Tuple[jnp.array, jnp.array]]


class GENOT(UnbalancednessMixin, ResampleMixin, BaseNeuralSolver):

  def __init__(
      self,
      neural_vector_field: Type[BaseNeuralVectorField],
      input_dim: int,
      output_dim: int,
      cond_dim: int,
      iterations: int,
      valid_freq: int,
      ot_solver: Type[was_solver.WassersteinSolver],
      optimizer: Type[optax.GradientTransformation],
      flow: Type[BaseFlow] = ConstantNoiseFlow(0.0),
      k_noise_per_x: int = 1,
      t_offset: float = 1e-5,
      epsilon: float = 1e-2,
      cost_fn: Union[costs.CostFn, Literal["graph"]] = costs.SqEuclidean(),
      solver_latent_to_data: Optional[Type[was_solver.WassersteinSolver]
                                     ] = None,
      kwargs_solver_latent_to_data: Dict[str, Any] = types.MappingProxyType({}),
      scale_cost: Union[Any, Mapping[str, Any]] = 1.0,
      graph_kwargs: Dict[str, Any] = types.MappingProxyType({}),
      fused_penalty: float = 0.0,
      tau_a: float = 1.0,
      tau_b: float = 1.0,
      mlp_eta: Callable[[jnp.ndarray], float] = None,
      mlp_xi: Callable[[jnp.ndarray], float] = None,
      unbalanced_kwargs: Dict[str, Any] = {},
      callback: Optional[Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray],
                                  Any]] = None,
      callback_kwargs: Dict[str, Any] = {},
      callback_iters: int = 10,
      rng: random.PRNGKeyArray = random.PRNGKey(0),
      **kwargs: Any,
  ) -> None:
    """The GENOT training class.

    Parameters
    ----------
    neural_vector_field
    Neural vector field
    input_dim
    Dimension of the source distribution
    output_dim
    Dimension of the target distribution
    cond_dim
    Dimension of the condition
    iterations
    Number of iterations to train
    valid_freq
    Number of iterations after which to perform a validation step
    ot_solver
    Solver to match samples from the source to the target distribution
    optimizer
    Optimizer for the neural vector field
    flow
    Flow to use in the target space from noise to data. Should be of type
    `ConstantNoiseFlow` to recover the setup in the paper TODO.
    k_noise_per_x
    Number of samples to draw from the conditional distribution
    t_offset
    Offset for sampling from the time t
    epsilon
    Entropy regularization parameter for the discrete solver
    cost_fn
    Cost function to use for the discrete OT solver
    solver_latent_to_data
    Linear OT solver to match samples from the noise to the conditional distribution
    latent_to_data_epsilon
    Entropy regularization term for `solver_latent_to_data`
    latent_to_data_scale_cost
    How to scale the cost matrix for the `solver_latent_to_data` solver
    scale_cost
    How to scale the cost matrix in each discrete OT problem
    graph_kwargs
    Keyword arguments for the graph cost computation in case `cost="graph"`
    fused_penalty
    Penalisation term for the linear term in a Fused GW setting
    split_dim
    Dimension to split the data into fused term and purely quadratic term in the FGW setting
    mlp_eta
    Neural network to learn the left rescaling function
    mlp_xi
    Neural network to learn the right rescaling function
    tau_a
    Left unbalancedness parameter
    tau_b
    Right unbalancedness parameter
    callback
    Callback function
    callback_kwargs
    Keyword arguments to the callback function
    callback_iters
    Number of iterations after which to evaluate callback function
    seed
    Random seed
    kwargs
    Keyword arguments passed to `setup`, e.g. custom choice of optimizers for learning rescaling functions
    """
    BaseNeuralSolver.__init__(
        self, iterations=iterations, valid_freq=valid_freq
    )
    ResampleMixin.__init__(self)
    UnbalancednessMixin.__init__(
        self,
        source_dim=input_dim,
        target_dim=input_dim,
        cond_dim=cond_dim,
        tau_a=tau_a,
        tau_b=tau_b,
        mlp_eta=mlp_eta,
        mlp_xi=mlp_xi,
        unbalanced_kwargs=unbalanced_kwargs,
    )

    if isinstance(
        ot_solver, gromov_wasserstein.GromovWasserstein
    ) and epsilon is not None:
      raise ValueError(
          "If `ot_solver` is `GromovWasserstein`, `epsilon` must be `None`. This check is performed "
          "to ensure that in the (fused) Gromov case the `epsilon` parameter is passed via the `ot_solver`."
      )

    # setup parameters
    self.rng = rng
    self.metrics = {"loss": [], "loss_eta": [], "loss_xi": []}

    # neural parameters
    self.neural_vector_field = neural_vector_field
    self.state_neural_vector_field: Optional[TrainState] = None
    self.optimizer = optimizer
    self.noise_fn = jax.tree_util.Partial(
        jax.random.multivariate_normal,
        mean=jnp.zeros((output_dim,)),
        cov=jnp.diag(jnp.ones((output_dim,)))
    )
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.cond_dim = cond_dim
    self.k_noise_per_x = k_noise_per_x

    # OT data-data matching parameters
    self.ot_solver = ot_solver
    self.epsilon = epsilon
    self.cost_fn = cost_fn
    self.scale_cost = scale_cost
    self.graph_kwargs = graph_kwargs  # "k_neighbors", kwargs for graph.Graph.from_graph()
    self.fused_penalty = fused_penalty

    # OT latent-data matching parameters
    self.solver_latent_to_data = solver_latent_to_data
    self.kwargs_solver_latent_to_data = kwargs_solver_latent_to_data

    # callback parameteres
    self.callback = callback
    self.callback_kwargs = callback_kwargs
    self.callback_iters = callback_iters

    #TODO: check how to handle this
    self.t_offset = t_offset

    self.setup(**kwargs)

  def setup(self) -> None:
    """Set up the model.

    Parameters
    ----------
    kwargs
    Keyword arguments for the setup function
    """
    self.state_neural_vector_field = self.neural_vector_field.create_train_state(
        self.rng, self.optimizer, self.input_dim
    )
    self.step_fn = self._get_step_fn()
    if self.solver_latent_to_data is not None:
      self.match_latent_to_data_fn = self._get_sinkhorn_match_fn(
          self.solver_latent_to_data, **self.kwargs_solver_latent_to_data
      )
    else:
      self.match_latent_to_data_fn = lambda key, x, y, **_: (x, y)

    # TODO: add graph construction function
    if isinstance(self.ot_solver, sinkhorn.Sinkhorn):
      self.match_fn = self._get_sinkhorn_match_fn(
          self.ot_solver, self.epsilon, self.cost_fn, self.tau_a, self.tau_b,
          self.scale_cost
      )
    else:
      self._get_gromov_match_fn(
          self.ot_solver, self.cost_fn, self.tau_a, self.tau_b, self.scale_cost,
          self.fused_penalty
      )

  def __call__(self, train_loader, valid_loader) -> None:
    """Train GENOT."""
    batch: Dict[str, jnp.array] = {}
    for step in tqdm(range(self.iterations)):
      batch["source"], batch["source_q"], batch["target"], batch[
          "target_q"], batch["condition"] = next(train_loader)

      self.rng, rng_time, rng_match, rng_resample, rng_noise, rng_latent_data_match, rng_step_fn = jax.random.split(
          self.rng, 7
      )
      n_samples = len(batch["source"]) * self.k_noise_per_k
      batch["time"] = self.sample_t(key, n_samples)
      batch["noise"] = self.noise_fn(
          rng_noise, shape=(batch["source"], self.k_noise_per_x)
      )

      tmat = self.match_fn(rng_match, batch["source"], batch["target"])
      (batch["source"], batch["source_q"], batch["condition"]
      ), (batch["target"],
          batch["target_q"]) = self._sample_conditional_indices_from_tmap(
              rng_resample, tmat, self.k_noise_per_x,
              (batch["source"], batch["source_q"], batch["condition"]),
              (batch["target"], batch["target_q"])
          )
      rng_noise = jax.random.split(rng_noise, (len(batch["target"])))

      tmat_latent_data = jax.vmap(self.match_latent_to_data_fn, 0, 0)(
          key=rng_noise, x=batch["noise"], y=batch["target"]
      )
      (batch["source"], batch["source_q"], batch["condition"]
      ), (batch["target"], batch["target_q"]) = self._resample_data(
          rng_latent_data_match, tmat_latent_data,
          (batch["source"], batch["source_q"], batch["condition"]),
          (batch["target"], batch["target_q"])
      )

      batch = {
          key: jnp.reshape(arr, (len(batch["source"]), -1))
          for key, arr in batch.items()
      }

      self.state_neural_vector_field, loss = self.step_fn(
          rng_step_fn, self.state_neural_vector_field, batch
      )
      if self.learn_rescaling:
        self.state_eta, self.state_xi, eta_predictions, xi_predictions, loss_a, loss_b = self.unbalancedness_step_fn(
            batch, tmat.sum(axis=1), tmat.sum(axis=0)
        )
      if iter % self.valid_freq == 0:
        self._valid_step(valid_loader, iter)
        if self.checkpoint_manager is not None:
          states_to_save = {
              "state_neural_vector_field": self.state_neural_vector_field
          }
          if self.state_mlp is not None:
            states_to_save["state_eta"] = self.state_mlp
          if self.state_xi is not None:
            states_to_save["state_xi"] = self.state_xi
          self.checkpoint_manager.save(iter, states_to_save)

  def _get_step_fn(self) -> Callable:

    @jax.jit
    def step_fn(
        key: jax.random.PRNGKeyArray,
        state_neural_vector_field: train_state.TrainState,
        batch: Dict[str, jnp.array],
    ):

      def loss_fn(
          params: jax.Array, t: jax.Array, noise: jax.Array,
          batch: Dict[str, jnp.array], keys_model: random.PRNGKeyArray
      ):

        x_t = self.flow.compute_xt(noise, t, batch["latent"], batch["target"])
        apply_fn = functools.partial(
            state_neural_vector_field.apply_fn, {"params": params}
        )
        cond_input = jnp.concatenate([batch["source"], batch["condition"]],
                                     axis=-1)
        v_t = jax.vmap(apply_fn)(
            t=t, x=x_t, condition=cond_input, keys_model=keys_model
        )
        u_t = self.flow.compute_ut(t, batch["latent"], batch["target"])
        return jnp.mean((v_t - u_t) ** 2)

      grad_fn = jax.value_and_grad(loss_fn, has_aux=False)
      loss, grads = grad_fn(
          state_neural_vector_field.params,
          state_neural_vector_field.apply_fn,
          batch,
      )

      return state_neural_vector_field.apply_gradients(grads=grads), loss

    return step_fn

  def transport(
      self,
      source: jnp.array,
      seed: int = 0,
      diffeqsolve_kwargs: Dict[str, Any] = types.MappingProxyType({})
  ) -> Union[jnp.array, diffrax.Solution, Optional[jnp.ndarray]]:
    """Transport the distribution.

    Parameters
    ----------
    source
    Source distribution to transport
    seed
    Random seed for sampling from the latent distribution
    diffeqsolve_kwargs
    Keyword arguments for the ODE solver.

    Returns:
    -------
    The transported samples, the solution of the neural ODE, and the rescaling factor.
    """
    diffeqsolve_kwargs = dict(diffeqsolve_kwargs)
    rng = jax.random.PRNGKey(seed)
    latent_shape = (len(source),)
    latent_batch = self.noise_fn(rng, shape=latent_shape)
    apply_fn_partial = partial(
        self.state_neural_vector_field.apply_fn, condition=source
    )
    solution = diffrax.diffeqsolve(
        diffrax.ODETerm(
            lambda t, y, *args:
            apply_fn_partial({"params": self.state_neural_vector_field.params},
                             t=t,
                             latent=y)
        ),
        diffeqsolve_kwargs.pop("solver", diffrax.Tsit5()),
        t0=0,
        t1=1,
        dt0=diffeqsolve_kwargs.pop("dt0", None),
        y0=latent_batch,
        stepsize_controller=diffeqsolve_kwargs.pop(
            "stepsize_controller", diffrax.PIDController(rtol=1e-3, atol=1e-6)
        ),
        **diffeqsolve_kwargs,
    )
    if self.state_eta is not None:
      weight_factors = self.state_eta.apply_fn({
          "params": self.state_eta.params
      },
                                               x=source)
    else:
      weight_factors = jnp.ones(source.shape)
    return solution.ys, solution, weight_factors

  def _valid_step(self, valid_loader, iter) -> None:
    next(valid_loader)

  # TODO: add callback and logging

  @property
  def learn_rescaling(self) -> bool:
    return self.mlp_eta is not None or self.mlp_xi is not None

  def save(self, path: str) -> None:
    raise NotImplementedError

  def load(self, path: str) -> "GENOT":
    raise NotImplementedError

  def training_logs(self) -> Dict[str, Any]:
    raise NotImplementedError

  def sample_t( #TODO: make more general
      self, key: random.PRNGKey, batch_size: int
  ) -> jnp.ndarray:  #TODO: make more general
    return random.uniform(key, [batch_size, 1])
