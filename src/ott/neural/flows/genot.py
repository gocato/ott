# Copyright OTT-JAX
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import functools
import types
from typing import Any, Callable, Dict, Literal, Optional, Type, Union
import wandb
import jax
import jax.numpy as jnp
import diffrax
import optax
from flax.training import train_state
from flax.training import orbax_utils
import orbax
from pathlib import Path
from flax.training.early_stopping import EarlyStopping
from flax.training.train_state import TrainState
from orbax import checkpoint
from ott import utils
from ott.geometry import costs
from ott.neural.flows.flows import BaseFlow, ConstantNoiseFlow
from ott.neural.flows.samplers import sample_uniformly
from ott.neural.models.base_solver import (
    BaseNeuralSolver,
    ResampleMixin,
    UnbalancednessMixin,
)
from ott.solvers import was_solver
from ott.solvers.linear import sinkhorn
from ott.solvers.quadratic import gromov_wasserstein

import numpy as np
from tqdm import trange

__all__ = ["GENOT"]


class GENOT(UnbalancednessMixin, ResampleMixin, BaseNeuralSolver):
  """The GENOT training class as introduced in :cite:`klein_uscidda:23`.

  Args:
    velocity_field: Neural vector field parameterized by a neural network.
    input_dim: Dimension of the data in the source distribution.
    output_dim: Dimension of the data in the target distribution.
    cond_dim: Dimension of the conditioning variable.
    iterations: Number of iterations.
    valid_freq: Frequency of validation.
    ot_solver: OT solver to match samples from the source and the target
      distribution.
    epsilon: Entropy regularization term of the OT problem solved by
      `ot_solver`.
    cost_fn: Cost function for the OT problem solved by the `ot_solver`.
      In the linear case, this is always expected to be of type `str`.
      If the problem is of quadratic type and `cost_fn` is a string,
      the `cost_fn` is used for all terms, i.e. both quadratic terms and,
      if applicable, the linear temr. If of type :class:`dict`, the keys
      are expected to be `cost_fn_xx`, `cost_fn_yy`, and if applicable,
      `cost_fn_xy`.
    scale_cost: How to scale the cost matrix for the OT problem solved by
      the `ot_solver`. In the linear case, this is always expected to be
      not a :class:`dict`. If the problem is of quadratic type and
      `scale_cost` is a string, the `scale_cost` argument is used for all
      terms, i.e. both quadratic terms and, if applicable, the linear temr.
      If of type :class:`dict`, the keys are expected to be `scale_cost_xx`,
      `scale_cost_yy`, and if applicable, `scale_cost_xy`.
    optimizer: Optimizer for `velocity_field`.
    flow: Flow between latent distribution and target distribution.
    time_sampler: Sampler for the time.
    checkpoint_manager: Checkpoint manager.
    k_samples_per_x: Number of samples drawn from the conditional distribution
      of an input sample, see algorithm TODO.
    solver_latent_to_data: Linear OT solver to match the latent distribution
      with the conditional distribution.
    kwargs_solver_latent_to_data: Keyword arguments for `solver_latent_to_data`.
      #TODO: adapt
    fused_penalty: Fused penalty of the linear/fused term in the Fused
      Gromov-Wasserstein problem.
    tau_a: If :math:`<1`, defines how much unbalanced the problem is
    on the first marginal.
    tau_b: If :math:`< 1`, defines how much unbalanced the problem is
    on the second marginal.
    rescaling_a: Neural network to learn the left rescaling function. If
      :obj:`None`, the left rescaling factor is not learnt.
    rescaling_b: Neural network to learn the right rescaling function. If
      :obj:`None`, the right rescaling factor is not learnt.
    unbalanced_kwargs: Keyword arguments for the unbalancedness solver.
   callback_fn: Callback function.
    rng: Random number generator.
  """

  def __init__(
      self,
      velocity_field: Callable[[
          jnp.ndarray, jnp.ndarray, Optional[jnp.ndarray], Optional[jnp.ndarray]
      ], jnp.ndarray],
      input_dim: int,
      output_dim: int,
      cond_dim: int,
      iterations: int,
      valid_freq: int,
      ot_solver: Type[was_solver.WassersteinSolver],
      epsilon: float,
      cost_fn: Union[costs.CostFn, Dict[str, costs.CostFn]],
      scale_cost: Union[Union[bool, int, float,
                              Literal["mean", "max_norm", "max_bound",
                                      "max_cost", "median"]],
                        Dict[str, Union[bool, int, float,
                                        Literal["mean", "max_norm", "max_bound",
                                                "max_cost", "median"]]]],
      optimizer: Type[optax.GradientTransformation],
      flow: Type[BaseFlow] = ConstantNoiseFlow(0.0),
      time_sampler: Callable[[jax.Array, int], jnp.ndarray] = sample_uniformly,
      checkpoint_manager: Type[checkpoint.CheckpointManager] = None,
      k_samples_per_x: int = 1,
      solver_latent_to_data: Optional[Type[was_solver.WassersteinSolver]
                                     ] = None,
      kwargs_solver_latent_to_data: Dict[str, Any] = types.MappingProxyType({}),
      fused_penalty: float = 0.0,
      tau_a: float = 1.0,
      tau_b: float = 1.0,
      rescaling_a: Callable[[jnp.ndarray], float] = None,
      rescaling_b: Callable[[jnp.ndarray], float] = None,
      unbalanced_kwargs: Dict[str, Any] = types.MappingProxyType({}),
      callback_fn: Optional[Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray],
                                     Any]] = None,
      rng: Optional[jax.Array] = None,
      metrics_callback: Optional[Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], Any]] = None,
      metrics_callback_kwargs: Optional[Dict[str, Any]] = types.MappingProxyType({}),
      plot_callback: Optional[Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], Any]] = None,
      plot_callback_kwargs: Optional[Dict[str, Any]] = types.MappingProxyType({}),
      early_stopping_kwargs: Optional[EarlyStopping] = None,
      log_training: bool = False,
  ):
    rng = utils.default_prng_key(rng)
    rng, rng_unbalanced = jax.random.split(rng)
    BaseNeuralSolver.__init__(
        self, iterations=iterations, valid_freq=valid_freq
    )
    ResampleMixin.__init__(self)
    UnbalancednessMixin.__init__(
        self,
        rng=rng_unbalanced,
        source_dim=input_dim,
        target_dim=input_dim,
        cond_dim=cond_dim,
        tau_a=tau_a,
        tau_b=tau_b,
        rescaling_a=rescaling_a,
        rescaling_b=rescaling_b,
        unbalanced_kwargs=unbalanced_kwargs,
    )
    if isinstance(
        ot_solver, gromov_wasserstein.GromovWasserstein
    ) and epsilon is not None:
      raise ValueError(
          "If `ot_solver` is `GromovWasserstein`, `epsilon` must be `None`. " +
          "This check is performed to ensure that in the (fused) Gromov case " +
          "the `epsilon` parameter is passed via the `ot_solver`."
      )

    self.rng = utils.default_prng_key(rng)
    self.velocity_field = velocity_field
    self.state_velocity_field: Optional[TrainState] = None
    self.flow = flow
    self.time_sampler = time_sampler
    self.optimizer = optimizer
    self.checkpoint_manager = checkpoint_manager
    self.latent_noise_fn = jax.tree_util.Partial(
        jax.random.multivariate_normal,
        mean=jnp.zeros((output_dim,)),
        cov=jnp.diag(jnp.ones((output_dim,)))
    )
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.cond_dim = cond_dim
    self.k_samples_per_x = k_samples_per_x

    # OT data-data matching parameters
    self.ot_solver = ot_solver
    self.epsilon = epsilon
    self.cost_fn = cost_fn
    self.scale_cost = scale_cost
    self.fused_penalty = fused_penalty

    # OT latent-data matching parameters
    self.solver_latent_to_data = solver_latent_to_data
    self.kwargs_solver_latent_to_data = kwargs_solver_latent_to_data

    # callback parameteres
    self.callback_fn = callback_fn
    self.setup()

    self.metrics_callback = metrics_callback
    self._metrics_callback_kwargs = metrics_callback_kwargs
    self.plot_callback = plot_callback
    self._plot_callback_kwargs = plot_callback_kwargs
    self.early_stopping_kwargs = early_stopping_kwargs
    if self.early_stopping_kwargs is not None:
      self.early_stopping = EarlyStopping(
          **self.early_stopping_kwargs
      )
    self.log_training = log_training
  def setup(self):
    """Set up the model.

    Parameters
    ----------
    kwargs
    Keyword arguments for the setup function.
    """
    self.state_velocity_field = (
        self.velocity_field.create_train_state(
            self.rng, self.optimizer, self.output_dim
        )
    )
    self.step_fn = self._get_step_fn()
    if self.solver_latent_to_data is not None:
      self.match_latent_to_data_fn = self._get_sinkhorn_match_fn(
          ot_solver=self.solver_latent_to_data,
          **self.kwargs_solver_latent_to_data
      )
    else:
      self.match_latent_to_data_fn = lambda key, x, y, **_: (x, y)

    # TODO: add graph construction function
    if isinstance(self.ot_solver, sinkhorn.Sinkhorn):
      self.match_fn = self._get_sinkhorn_match_fn(
          ot_solver=self.ot_solver,
          epsilon=self.epsilon,
          cost_fn=self.cost_fn,
          scale_cost=self.scale_cost,
          tau_a=self.tau_a,
          tau_b=self.tau_b,
          filter_input=True
      )
    else:
      self.match_fn = self._get_gromov_match_fn(
          ot_solver=self.ot_solver,
          cost_fn=self.cost_fn,
          scale_cost=self.scale_cost,
          tau_a=self.tau_a,
          tau_b=self.tau_b,
          fused_penalty=self.fused_penalty
      )

  def __call__(self, train_loader, valid_loader):
    """Train GENOT.

    Args:
      train_loader: Data loader for the training data.
      valid_loader: Data loader for the validation data.
    """
    batch: Dict[str, jnp.array] = {}
    tbar = trange(self.iterations, leave=True)
    curr_loss = 0.0
    for iteration in tbar:
      batch = next(train_loader)

      (
          self.rng, rng_time, rng_resample, rng_noise, rng_latent_data_match,
          rng_step_fn
      ) = jax.random.split(self.rng, 6)
      batch_size = len(
          batch["source_lin"]
      ) if batch["source_lin"] is not None else len(batch["source_quad"])
      n_samples = batch_size * self.k_samples_per_x
      batch["time"] = self.time_sampler(rng_time, n_samples)
      batch["noise"] = self.sample_noise(rng_noise, n_samples)
      batch["latent"] = self.latent_noise_fn(
          rng_noise, shape=(self.k_samples_per_x, batch_size)
      )

      tmat = self.match_fn(
          batch["source_lin"], batch["source_quad"], batch["target_lin"],
          batch["target_quad"]
      )

      batch["source"] = jnp.concatenate([
          batch[el]
          for el in ["source_lin", "source_quad"]
          if batch[el] is not None
      ],
                                        axis=1)
      batch["target"] = jnp.concatenate([
          batch[el]
          for el in ["target_lin", "target_quad"]
          if batch[el] is not None
      ],
                                        axis=1)

      batch = {
          k: v for k, v in batch.items() if k in
          ["source", "target", "source_conditions", "time", "noise", "latent"]
      }

      (batch["source"], batch["source_conditions"]
      ), (batch["target"],) = self._sample_conditional_indices_from_tmap(
          rng_resample,
          tmat,
          self.k_samples_per_x, (batch["source"], batch["source_conditions"]),
          (batch["target"],),
          source_is_balanced=(self.tau_a == 1.0)
      )
      jax.random.split(rng_noise, batch_size * self.k_samples_per_x)

      if self.solver_latent_to_data is not None:
        tmats_latent_data = jnp.array(
            jax.vmap(self.match_latent_to_data_fn, 0,
                     0)(x=batch["latent"], y=batch["target"])
        )

        rng_latent_data_match = jax.random.split(
            rng_latent_data_match, self.k_samples_per_x
        )
        (batch["source"], batch["source_conditions"]
        ), (batch["target"],) = jax.vmap(self._resample_data, 0, 0)(
            rng_latent_data_match, tmats_latent_data,
            (batch["source"], batch["source_conditions"]), (batch["target"],)
        )
      batch = {
          key:
              jnp.reshape(arr, (batch_size * self.k_samples_per_x,
                                -1)) if arr is not None else None
          for key, arr in batch.items()
      }

      self.state_velocity_field, loss = self.step_fn(
          rng_step_fn, self.state_velocity_field, batch
      )
      tbar.set_postfix({"loss": loss}, refresh=True)
      curr_loss += loss
      
      if self.learn_rescaling:
        (
            self.state_eta, self.state_xi, eta_predictions, xi_predictions,
            loss_a, loss_b
        ) = self.unbalancedness_step_fn(
            source=batch["source"],
            target=batch["target"],
            condition=batch["source_conditions"],
            a=tmat.sum(axis=1),
            b=tmat.sum(axis=0),
            state_eta=self.state_eta,
            state_xi=self.state_xi,
        )
      if iteration % self.valid_freq == 0 and iteration != 0:
        if self.metrics_callback is not None:
          self._valid_step(valid_loader, iteration)
        if self.checkpoint_manager is not None:
          states_to_save = {"state_velocity_field": self.state_velocity_field}
          if self.state_eta is not None:
            states_to_save["state_eta"] = self.state_eta
          if self.state_xi is not None:
            states_to_save["state_xi"] = self.state_xi
          self.checkpoint_manager.save(iteration, states_to_save)
        if self.log_training:
          wandb.log(
            {
              "curr_loss": curr_loss / self.valid_freq,
              "loss": loss
            },
            iteration
          )
        curr_loss = 0.0

  def _get_step_fn(self) -> Callable:

    @jax.jit
    def step_fn(
        key: jax.random.PRNGKeyArray,
        state_velocity_field: train_state.TrainState,
        batch: Dict[str, jnp.array],
    ):

      def loss_fn(
          params: jnp.ndarray, batch: Dict[str, jnp.array],
          rng: jax.random.PRNGKeyArray
      ):
        x_t = self.flow.compute_xt(
            batch["noise"], batch["time"], batch["latent"], batch["target"]
        )
        apply_fn = functools.partial(
            state_velocity_field.apply_fn, {"params": params}
        )

        cond_input = jnp.concatenate(
          [
            batch[el]
            for el in ["source", "source_conditions"]
            if batch[el] is not None
          ],
          axis=1
        )

        v_t = jax.vmap(apply_fn
                      )(t=batch["time"], x=x_t, condition=cond_input, rng=rng)
        u_t = self.flow.compute_ut(
            batch["time"], batch["latent"], batch["target"]
        )
        return jnp.mean((v_t - u_t) ** 2)

      keys_model = jax.random.split(key, len(batch["noise"]))

      grad_fn = jax.value_and_grad(loss_fn, has_aux=False)
      loss, grads = grad_fn(state_velocity_field.params, batch, keys_model)

      return state_velocity_field.apply_gradients(grads=grads), loss

    return step_fn

  def transport(
      self,
      source: jnp.ndarray,
      condition: Optional[jnp.ndarray] = None,
      rng: Optional[jax.Array] = None,
      forward: bool = True,
      diffeqsolve_kwargs: Dict[str, Any] = types.MappingProxyType({}),
  ) -> Union[jnp.array, diffrax.Solution, Optional[jnp.ndarray]]:
    """Transport data with the learnt plan.

    This method pushes-forward the `source` to its conditional distribution by
      solving the neural ODE parameterized by the
      :attr:`~ott.neural.flows.genot.velocity_field`

    Args:
      source: Data to transport.
      condition: Condition of the input data.
      rng: random seed for sampling from the latent distribution.
      forward: If `True` integrates forward, otherwise backwards.
      diffeqsolve_kwargs: Keyword arguments for the ODE solver.

    Returns:
      The push-forward or pull-back distribution defined by the learnt
      transport plan.

    """
    rng = utils.default_prng_key(rng)
    if not forward:
      raise NotImplementedError
    diffeqsolve_kwargs = dict(diffeqsolve_kwargs)
    assert len(source) == len(condition) if condition is not None else True

    latent_batch = self.latent_noise_fn(rng, shape=(len(source),))
    cond_input = source if condition is None else jnp.concatenate([
        source, condition
    ],
                                                                  axis=-1)
    t0, t1 = (0.0, 1.0)

    def solve_ode(input: jnp.ndarray, cond: jnp.ndarray):
      return diffrax.diffeqsolve(
          diffrax.ODETerm(
              lambda t, x, args: self.state_velocity_field.
              apply_fn({"params": self.state_velocity_field.params},
                       t=t,
                       x=x,
                       condition=cond)
          ),
          diffeqsolve_kwargs.pop("solver", diffrax.Tsit5()),
          t0=t0,
          t1=t1,
          dt0=diffeqsolve_kwargs.pop("dt0", None),
          y0=input,
          stepsize_controller=diffeqsolve_kwargs.pop(
              "stepsize_controller",
              diffrax.PIDController(rtol=1e-5, atol=1e-5)
          ),
          **diffeqsolve_kwargs,
      ).ys[0]

    return jax.vmap(solve_ode)(latent_batch, cond_input)


  def transport_with_mean(
      self,
      source: jnp.ndarray,
      condition: Optional[jnp.ndarray] = None,
      rng: Optional[jax.Array] = None,
      forward: bool = True,
      diffeqsolve_kwargs: Dict[str, Any] = types.MappingProxyType({}),
      repetitions: int = 30,
  ) -> Union[jnp.array, diffrax.Solution, Optional[jnp.ndarray]]:
    """Return the prediction as a mean of predictions."""
    if jax.local_device_count() > 1:
      parallel_transport = jax.pmap(
        jax.tree_util.Partial(
          self.transport,
          source=source,
          condition=condition,
          forward=forward,
          diffeqsolve_kwargs=diffeqsolve_kwargs,
        )
      )
    else:
      parallel_transport = jax.vmap(
        jax.tree_util.Partial(
          self.transport,
          source=source,
          condition=condition,
          forward=forward,
          diffeqsolve_kwargs=diffeqsolve_kwargs,
        )
      )

    rngs = jax.random.split(utils.default_prng_key(rng), repetitions)
    predictions = parallel_transport(rng=rngs)

    return jnp.mean(predictions, axis=0)

  def _valid_step(self, valid_loader, step):
    batches = jax.tree_util.tree_map(
        lambda x: next(x),
        valid_loader.dataloaders
    )

    parallelizer = jax.pmap if jax.device_count() > 1 else jax.vmap

    sources = jnp.asarray([
        batches[condition]["source_lin"]
        for condition in batches
    ])
    targets = jnp.asarray([
        batches[condition]["target_lin"]
        for condition in batches
    ])
    conditions = jnp.asarray([
        batches[condition]["source_conditions"]
        for condition in batches
    ])
    names = list(batches.keys())
    predictions = parallelizer(
        lambda source, condition: self.transport_with_mean(
            source,
            condition,
            forward=True,
        )
    )(sources, conditions)

    metrics = parallelizer(
        lambda source, target, pred: self.metrics_callback(
            source,
            target,
            pred,
            **self._metrics_callback_kwargs,
        )
    )(sources, targets, predictions)

    for group_id, name in enumerate(batches.keys()):
        metrics_condition = jax.tree_util.tree_map(
            lambda x: x[group_id],
            metrics
        )
        metrics_condition_np = {
            key: np.asarray(value) for key, value in metrics_condition.items()
            if not isinstance(value, str) and value is not None
        }
        wandb.log(
            {f"Metrics/condition_{name}": metrics_condition_np},
            step,
        )

    if self.plot_callback is not None:
        for source, target, pred, name in zip(sources, targets, predictions, names):
            fig = self.plot_callback(
                    source,
                    target,
                    pred,
                    **self._plot_callback_kwargs,
                )

            fig.set_tight_layout(True)
            wandb.log({f"Plots/condition_{name}": wandb.Image(fig)})

  @property
  def learn_rescaling(self) -> bool:
    """Whether to learn at least one rescaling factor."""
    return self.rescaling_a is not None or self.rescaling_b is not None

    raise NotImplementedError

  def save(self, path: str):
      """Save the model.

      Args:
          path: Where to save the model to.
      """
      checkpoint = {
          "velocity_field_state": self.state_velocity_field,
      }
      orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
      save_args = orbax_utils.save_args_from_target(checkpoint)
      orbax_checkpointer.save(
              Path(path, "checkpoints"),
              checkpoint,
              save_args=save_args
      )

  def load(self, path: str) -> "GENOT":
    """Load a model.

    Args:
      path: Where to load the model from.

    Returns:
      An instance of :class:`ott.neural.solvers.OTFlowMatching`.
    """
    raise NotImplementedError

  @property
  def training_logs(self) -> Dict[str, Any]:
    """Logs of the training."""
    raise NotImplementedError

  def sample_noise(
      self, key: jax.random.PRNGKey, batch_size: int
  ) -> jnp.ndarray:
    """Sample noise from a standard-normal distribution.

    Args:
      key: Random key for seeding.
      batch_size: Number of samples to draw.

    Returns:
      Samples from the standard normal distribution.
    """
    return jax.random.normal(key, shape=(batch_size, self.output_dim))
