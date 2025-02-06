from copy import deepcopy
from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial
from typing import Any
from typing import Callable
from typing import Protocol

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
import numpy as np
from tqdm import tqdm


Key = Any
"""Key for a pseudo random number generator (PRNG)."""

PyTree = Any
"""PyTrees are arbitrary nests of ``jnp.ndarrays``."""

OptState = Any
"""Arbitrary object holding the state of the optimizer."""

# DeepQNetwork func takes as input the model parameters and a batch
# of observations, and returns the state-action values ``Q(s, a)``
# for each (observation, action) pair.
DeepQNetwork = Callable[[PyTree, ArrayLike], jax.Array]

# OptimizerFn takes as input parameters, their gradients, and the
# optimizer state and returns the updated parameters and the new state.
OptimizerFn = Callable[[PyTree, PyTree, OptState], tuple[PyTree, OptState]]

# EnvironmentStepFn is a step function for an environment conforming to
# the Gymnasium environments API. See:
#   https://gymnasium.farama.org/api/env/#gymnasium.Env.step
#   https://gymnasium.farama.org/api/vector/#gymnasium.vector.VectorEnv.step
#
# Both vectorized and non-vectorized environments are supported.
# The environment must be automatically reset when a terminal
# state is reached. The info dict is not used.
EnvironmentStepFn = Callable[
	[ArrayLike],
	tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike, dict],
]
"""EnvironmentStepFn implements the OpenAI Gym env.step() API with autoreset."""

# Transitions is a tuple (o, a, r, o_next, d) of nd-arrays containing:
#   - one or more observations ``o``;
#   - the selected actions ``a`` for each observation;
#   - the obtained rewards ``r``;
#   - the next observations ``o_next``;
#   - boolean flags ``d`` indicating which ``o_next`` are terminal.
Transitions = tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike]
"""Tuple (o, a, r, o_next, d) of nd-arrays."""

# ReplayBuffer is a container for storing and sampling transitions.
class ReplayBuffer(Protocol):
	def store(self, ts: Transitions) -> None:
		"""Store a transition from stepping the environment once."""
	def sample(self, rng: Key, batch_size: int) -> Transitions:
		"""Sample a batch of transitions from the buffer."""

# DQNLearner is a callable that trains an agent to maximize
# expected return from a given environment.
@dataclass(init=True, repr=False, eq=False)
class DQNLearner:
	"""DQNLearner trains a Q-network over multiple time-steps.

	At each time-step the environment is stepped using an epsilon-
	greedy policy from the computed Q-values. The transitions are
	stored in the replay buffer. Once the agent has performed a
	fixed amount of steps,the parameters of the q-network are
	updated by drawing random batches from the buffer and
	optimizing using a one-step TD target.

	The q-network and the environment are provided to the trainer
	upon initialization. The trainer can then be called multiple
	times to update the network parameters. You can see how well
	the current q-network is performing in between calling the
	trainer:

	```python
	# Initialize the trainer.
	dqn_learner = DQNLearner(q_fn, optim_fn, env_fn, **kwargs)

	# Call the trainer to update the current params.
	init_obs, _ = env.reset()
	params, opt_state = dqn_learner(rng1, params, opt_state, init_obs, n_steps, prefill_steps)

	# Test the agent after training.
	record_demo(env_fn, q_fn, params)

	# Train the agent some more.
	init_obs, _ = env.reset()
	params, opt_state = dqn_learner(rng2, params, opt_state, init_obs, n_steps, 0)
	```
	"""

	q_fn: DeepQNetwork
	optim_fn: OptimizerFn
	env_fn: EnvironmentStepFn
	replay_buffer: ReplayBuffer
	discount: float = 1.    	# discount for future rewards
	update_freq: int = 1    	# how often to update the q-network
	updates_per_step: int = -1	# number of updates per environment step
	batch_size: int = 64    	# batch size for iterating over the replay buffer
	huber_delta: float = 1. 	# bound for the huber loss transform
	p: float = 0.995        	# factor for Polyak averaging
	eps: Callable[[int], float] = lambda x: 0.05 # schedule for epsi-greedy
	train_log: dict[str, list[float]] = field( # for logging info during training
		default_factory=lambda: defaultdict(list), init=False)

	def __call__(
		self,
		rng: Key,
		params: PyTree,
		opt_state: OptState,
		init_obs: ArrayLike,
		n_steps: int,
		prefill_steps: int,
	) -> tuple[PyTree, OptState]:
		"""Update the q-network parameters using one-step TD learning.

		Args:
			rng: Key
				PRNG key array.
			params: PyTree
				Current parameters for the q-network.
			opt_state: OptState
				Current optimizer state for the optimizer function.
			init_obs: ArrayLike
				Initial observation of the environment.
			n_steps: int
				Total number of time-steps to be performed.
			prefill_steps: int
				Number of time-steps to perform for pre-filling the buffer.

		Returns:
			PyTree
				The updated model parameters.
			OptState
				The latest state of the optimizer.
		"""

		tgt_params = deepcopy(params)   # tgt network params
		pbar = tqdm(total=n_steps)      # manual progress bar
		steps = 0                       # total number of steps performed
		ep_r, ep_l = None, None			# record episode statistics
		done = None						# track when an episode is done
		obs = init_obs

		while steps < n_steps:
			s = 0 # inner loop steps counter

			# Continue stepping the environment until it is time for an update.
			while s < self.update_freq:
				# Step the environment and store the transitions.
				rng, rng_ = jax.random.split(rng)
				eps = self.eps(steps+s)
				ts, ep_r, ep_l = step(
					rng_, self.env_fn, self.q_fn, params, obs, eps, done, ep_r, ep_l,
				)
				self.replay_buffer.store(ts)

				# Bookkeeping.
				_, _, _, obs, done = ts # advance to the next observation
				self.train_log["ep_r"].extend(np.where(done, ep_r, np.nan))
				self.train_log["ep_l"].extend(np.where(done, ep_l, np.nan))

				# Stepping the environment once actually performs `n_envs` steps.
				n_envs = done.shape[0]
				s += n_envs
				pbar.update(n_envs)

			steps += s # update the total step counter
			if steps < prefill_steps: continue

			# If ``updates_per_step`` is negative, then perform as many updates
			# as steps in the environment.
			# n_updates = s * max(self.updates_per_step, 1)
			n_updates = s * self.updates_per_step if self.updates_per_step > 0 else s

			# Update the parameters of the q-network.
			for _ in range(n_updates):
				rng, rng_ = jax.random.split(rng)
				ts = self.replay_buffer.sample(rng_, self.batch_size)

				# Compute the loss.
				loss, grads = td_error(
					ts, self.q_fn, params, tgt_params, self.discount, self.huber_delta,
				)

				# Backward pass. Update the parameters of the q-network.
				params, opt_state = self.optim_fn(params, grads, opt_state)

				# Update the target network parameters using Polyak averaging.
				tgt_params = jax.tree.map(
					lambda x, y: self.p*x+(1-self.p)*y, tgt_params, params,
				)

				# Bookkeeping.
				leaves, _ = jax.tree.flatten(grads)
				grad_norm = jnp.sqrt(sum(jnp.vdot(x, x) for x in leaves))
				self.train_log["loss"].append(loss.item())
				self.train_log["grad_norm"].append(grad_norm.item())

		pbar.close()

		return params, opt_state

# step simulates a single step from the interaction loop between
# the agent and the vectorized environment and returns the
# observed transitions.
def step(
	rng: Key,
	env_fn: EnvironmentStepFn,
	q_fn: DeepQNetwork,
	params: PyTree,
	o: ArrayLike,
	eps: float,
	prev_done: ArrayLike | None,
	ep_r: ArrayLike | None,
	ep_l: ArrayLike | None,
) -> tuple[Transitions, np.array, np.array]:
	"""Step the environment and return the observed transition.

	Args:
		rng: Key
			PRNG key array.
		env_fn: EnvironmentStepFn
			Function for stepping the environment given the actions.
		q_fn: DeepQNetwork
			Function for calculating state-action values.
		params: PyTree
			The parameters of the Q-function.
		o: ArrayLike
			Current observation of the environment state.
		eps: float
			Epsilon value for epsilon-greedy action selection.
		prev_done: ArrayLike | None
			Boolean array indicating which of the observations are done.
		ep_r: ArrayLike | None
			Array containing the current accumulated rewards.
		ep_l: ArrayLike | None
			Array containing the current episode lengths.

	Returns:
		Transitions
			Tuple (o, a, r, o_next, d) of nd-arrays.
		np.array
			The updated accumulated rewards.
		np.array
			The updated episode lengths.
	"""

	# Select the actions using eps-greedy.
	q_values = q_fn(params, o) # shape (B, acts) or (acts,)
	B, A = np.atleast_2d(q_values).shape
	rng, rng_ = jax.random.split(rng)
	if jax.random.uniform(rng_) < eps:
		rng, rng_ = jax.random.split(rng)
		acts = jax.random.randint(rng_, shape=(B,), minval=0, maxval=A, dtype=int)
		if len(q_values.shape) == 1: # non-vectorized envs accept a single value as act
			acts = acts.squeeze()
	else:
		acts = jnp.argmax(q_values, axis=-1)

	# Step the environment.
	o_next, r, t, tr, _ = env_fn(acts)

	# Bookkeeping.
	r = np.atleast_1d(r)
	done = np.atleast_1d(t | tr)
	if prev_done is None: prev_done = np.ones_like(done, dtype=bool)
	if ep_r is None: ep_r = np.zeros_like(r)
	if ep_l is None: ep_l = np.zeros_like(r, dtype=int)
	ep_r[prev_done] = 0
	ep_l[prev_done] = 0
	ep_r += r
	ep_l += 1

	# TODO:
	# If any of the sub-envs is truncated then read o_next from the info dict.
	# transitions = (o, acts, r, o_next, t)
	transitions = (o, acts, r, o_next, done)

	return transitions, ep_r, ep_l

# Differentiate the output of the function with respect to the
# third input parameter, i.e. the parameters of the q-network.
@partial(jax.jit, static_argnames="q_fn")
@partial(jax.value_and_grad, argnums=2, has_aux=False)
def td_error(
	ts: Transitions,
	q_fn: DeepQNetwork,
	params: PyTree,
	tgt_params: PyTree,
	gamma: float,
	delta: float,
) -> jax.Array:
	"""Compute the mean Huber loss based on the one-step TD error.

	``err = r_t + gamma * max_a' Q(s',a') - Q(s, a)``\n
	``loss = 0.5 * err**2 if err < delta else |err|``

	Args:
		ts: Transitions
			Tuple (o, a, r, o_next, d) of nd-arrays.
		q_fn: DeepQNetwork
			Function for calculating state-action values.
		params: PyTree
			The parameters of the Q-function.
		gamma: float
			Discount factor for future rewards.
		delta: float
			Threshold at which the Huber loss changes from L2-loss
			to delta-scaled L1-loss.

	Returns:
		jax.Array
			Array of size 1 holding the value of the loss.
	"""

	obs, acts, rewards, obs_next, done = ts
	B = obs.shape[0]

	# Compute the q-values for the current obs.
	q_values = q_fn(params, obs)            # shape (B, acts)
	q_preds = q_values[jnp.arange(B), acts] # shape (B,)

	# Compute the q-values for the next obs using double q-learning.
	# Select the maximizing actions using the online network, but compute
	# the q-values using the target network.
	acts_next = jnp.argmax(q_fn(params, obs_next), axis=1)
	q_next = q_fn(tgt_params, obs_next)     # shape (B, acts)
	q_next = q_next[jnp.arange(B), acts_next]
	q_next = jax.lax.stop_gradient(q_next)

	# Calculate the Huber loss.
	# 0.5 * err^2                   if |err| <= d
	# 0.5 * d^2 + d * (|err| - d)   if |err| > d
	errs = rewards + gamma * q_next * ~done - q_preds
	abs_errs = jnp.abs(errs)
	quadratic = jnp.minimum(abs_errs, delta)
	# Same as max(abs_errs - delta, 0) but avoids potentially doubling gradient.
	linear = abs_errs - quadratic
	return jnp.mean(0.5 * quadratic**2 + delta * linear)

#