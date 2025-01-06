from copy import deepcopy
from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial
from typing import Any
from typing import Callable
from typing import Protocol
import warnings

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
import numpy as np
from tqdm import tqdm


Key = Any       # Key for a pseudo random number generator (PRNG)
PyTree = Any    # PyTrees are arbitrary nests of ``jnp.ndarrays``
OptState = Any  # Arbitrary object holding the state of the optimizer.

# DeepQNetwork func takes the model parameters and the observations as input and
# returns the state-action values ``Q(s, a)`` for each (observation, action) pair.
DeepQNetwork = Callable[[PyTree, ArrayLike], jax.Array]

# OptimizerFn takes parameters, their gradients, and the
# optimizer state as input and returns the updated parameters and
# the new state.
OptimizerFn = Callable[[PyTree, PyTree, OptState], tuple[PyTree, OptState]]

# EnvironmentStepFn is a step function for a vectorized environment
# conforming to the Gymnasium environments API. See:
#   https://gymnasium.farama.org/api/env/#gymnasium.Env.step
#   https://gymnasium.farama.org/api/vector/#gymnasium.vector.VectorEnv.step
#
# The function takes as input a batch of actions to update the environment state,
# and returns the next observations and the rewards resulting from the actions.
# The function also returns boolean arrays indicating whether any of the
# sub-environments were terminated or truncated and an info dict.
#
# If ``None`` is given as input, then the function returns the
# observations for the current state of the environment.
EnvironmentStepFn = Callable[
    [ArrayLike | None],
    tuple[jax.Array, jax.Array, jax.Array, jax.Array, dict],
]

# Transitions is a tuple (o, a, r, o_next, d) of nd-arrays containing:
#   - a batch of the current observations ``o``;
#   - the selected actions ``a`` for each observation;
#   - the obtained rewards ``r``;
#   - the next observations ``o_next``;
#   - a boolean flag ``d`` indicating if ``o_next`` was terminal.
Transitions = tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike]

# ReplayBuffer is a container for storing and sampling transitions.
# Implementations might include strategies for prioritized sampling of
# transitions or smart eviction of old experiences and others.
class ReplayBuffer(Protocol):
    def store(self, ts: Transitions) -> None:
        """Store a batch of transitions in the buffer possibly overwriting old ones."""

    def sample(self, rng: Key, batch_size: int) -> Transitions:
        """Sample a batch of transitions from the buffer."""

    def __len__(self) -> int:
        """Return the number of transitions currently stored in the buffer."""

@dataclass
class DQNTrainer:
    """
        ```dqn_trainer = DQNTrainer(**kwargs)```
        ```params, opt_state = dqn_trainer(rng, params, opt_state) ```
    """
    q_fn: DeepQNetwork
    optim_fn: OptimizerFn
    env_fn: EnvironmentStepFn
    replay_buffer: ReplayBuffer
    discount: float = 1.
    n_steps: int = 1
    n_updates: int = 1
    update_tgt: int = 1
    batch_size: int = 64
    huber_delta: float = 1.
    eps: Callable[[int], float] = lambda x: 0.05
    train_log: dict[str, list[float]] = field( # for logging info during training
        default_factory=lambda: defaultdict(list), init=False)

    def __call__(
        self,
        rng: Key,
        params: PyTree,
        opt_state: OptState,
        n_iters: int
    ) -> tuple[PyTree, OptState]:
        """Run the DQN trainer for multiple iterations to update the q-network
        parameters. Each iteration consists of two stages:\n
          1. data collection stage, where we step the environment (possibly
          multiple times) and store the transitions in the replay buffer;
          2. parameter optimization stage, where we update the q-network
          parameters (possibly multiple times) using bootstrapped TD target.

        Args:
            rng: Key
                A PRNG key array.
            params: PyTree
                Current model parameters for the agent function.
            opt_state: OptState
                Current optimizer state for the optimizer function.
            n_iters: int

        Returns:
            PyTree
                The updated model parameters.
            OptState
                The latest state of the optimizer.
        """
        # Prefill the replay buffer.
        while len(self.replay_buffer) < max(5*self.batch_size*self.n_updates, 1000):
            # Step the environment and store the transition.
            rng, rng_ = jax.random.split(rng, num=2)
            ts, info = step(rng_, self.env_fn, self.q_fn, params, eps=1.)
            self.replay_buffer.store(ts)

        # Run the training procedure.
        for i in tqdm(range(n_iters)):
            # Maybe update the parameters of the target q-network.
            if i % self.update_tgt == 0:
                tgt_params = deepcopy(params)

            # Step the environment and store the transitions.
            eps = self.eps(i)
            for _ in range(self.n_steps):
                rng, rng_ = jax.random.split(rng, num=2)
                ts, info = step(rng_, self.env_fn, self.q_fn, params, eps)
                self.replay_buffer.store(ts)

                # Bookkeeping.
                with warnings.catch_warnings():
                    # We might make a step without completing any episodes. In
                    # this case we want to store NaN in the history. Taking the
                    # mean an empty slice throws a runtime warning and returns a
                    # NaN, which is exactly what we want.
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    self.train_log["Episode Returns"].append(np.mean(info["ep_r"]))
                    self.train_log["Episode Lengths"].append(np.mean(info["ep_l"]))

            # Update the parameters of the q-network.
            rng, rng_ = jax.random.split(rng, num=2)
            for _ in range(self.n_updates):
                # Sample transitions from the replay buffer.
                rng, rng_ = jax.random.split(rng, num=2)
                batch = self.replay_buffer.sample(rng_, self.batch_size)

                # Compute the loss and perform the backward pass.
                loss, grads = td_loss(
                    batch, self.q_fn, params, tgt_params, self.discount, self.huber_delta,
                )
                params, opt_state = self.optim_fn(params, grads, opt_state)

                # Bookkeeping.
                leaves, _ = jax.tree.flatten(grads)
                grad_norm = jnp.sqrt(sum(jnp.vdot(x, x) for x in leaves))
                self.train_log["Total Grad Norm"].append(grad_norm.item())
                self.train_log["Total Loss"].append(loss.item())

        return params, opt_state

def step(
    rng: Key,
    env_fn: EnvironmentStepFn,
    q_fn: DeepQNetwork,
    params: PyTree,
    eps: float,
) -> tuple[Transitions, dict]:
    """Step the environment and return the observed transition.

    Args:
        rng: Key
            A PRNG key array.
        env_fn: EnvironmentStepFn
            Function for stepping the environment given the actions.
        q_fn: DeepQNetwork
            Function for calculating state-action values.
        params: PyTree
            The parameters of the Q-function.
        eps: float
            Epsilon value for epsilon-greedy action selection.

    Returns:
        Transitions
            A tuple (o, a, r, o_next, d) of nd-arrays.
        dict[str, Sequence[float]]
            Info dict.
    """
    o, *_ = env_fn(None)

    # Select the actions using eps-greedy.
    q_values = q_fn(params, o) # shape (B, acts)
    B, A = q_values.shape
    rng, rng_ = jax.random.split(rng, num=2)
    if jax.random.uniform(rng_) < eps:
        rng, rng_ = jax.random.split(rng, num=2)
        acts = jax.random.randint(rng_, shape=(B,), minval=0, maxval=A, dtype=int)
    else:
        acts = jnp.argmax(q_values, axis=-1)

    # Step the environment.
    o_next, r, t, tr, infos = env_fn(acts)

    # If any of the sub-envs is truncated then read o_next from the info dict.
    # Transitions in **truncated** environments are stored as **not done**.
    if tr.any():
        pass #>

    # transitions = (o, acts, r, o_next, t)
    transitions = (o, acts, r, o_next, (t | tr))

    info = {
        "ep_r": [infos["episode"]["r"][k] for k in range(B) if (t | tr)[k]],
        "ep_l": [infos["episode"]["l"][k] for k in range(B) if (t | tr)[k]],
    }

    return transitions, info

# Differentiate the output of the function with respect to the
# fourth input parameter, i.e. the parameters of the q-network.
@partial(jax.jit, static_argnames="q_fn")
@partial(jax.value_and_grad, argnums=2, has_aux=False)
def td_loss(
    batch: Transitions,
    q_fn: DeepQNetwork,
    params: PyTree,
    tgt_params: PyTree,
    discount: float,
    delta: float,
) -> jax.Array:
    obs, acts, rewards, obs_next, done = batch
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
    errs = rewards + discount * q_next * ~done - q_preds
    abs_errs = jnp.abs(errs)
    quadratic = jnp.minimum(abs_errs, delta)
    # Same as max(abs_errs - delta, 0) but avoids potentially doubling gradient.
    linear = abs_errs - quadratic
    return jnp.mean(0.5 * quadratic**2 + delta * linear)

#