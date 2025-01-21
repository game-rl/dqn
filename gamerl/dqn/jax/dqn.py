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

# EnvironmentStepFn is a step function for a vectorized
# environment conforming to the Gymnasium environments API. See:
#   https://gymnasium.farama.org/api/env/#gymnasium.Env.step
#   https://gymnasium.farama.org/api/vector/#gymnasium.vector.VectorEnv.step
#
# The function takes as input a batch of actions to update the
# environment state, and returns the next observations and the
# rewards resulting from the actions. The function also returns
# boolean arrays indicating whether any of the sub-environments
# were terminated or truncated, as well as an info dict.
#
# If ``None`` is given as input, then the function returns the
# current observations without stepping the environment.
EnvironmentStepFn = Callable[
    [ArrayLike | None],
    tuple[jax.Array, jax.Array, jax.Array, jax.Array, dict],
]

# Transitions is a tuple (o, a, r, o_next, d) of nd-arrays containing:
#   - batch of observations ``o``;
#   - the selected actions ``a`` for each observation;
#   - the obtained rewards ``r``;
#   - the next observations ``o_next``;
#   - boolean flags ``d`` indicating which ``o_next`` are terminal.
Transitions = tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike]
"""Tuple (o, a, r, o_next, d) of nd-arrays."""

# ReplayBuffer is a container for storing and sampling transitions.
# Implementations might include strategies for prioritized sampling of
# transitions, smart eviction of old experiences, and others.
class ReplayBuffer(Protocol):
    def store(self, ts: Transitions) -> None:
        """Store a batch of transitions in the buffer possibly overwriting old ones."""

    def sample(self, rng: Key, batch_size: int) -> Transitions:
        """Sample a batch of transitions from the buffer."""

# DQNTrainer is a callable that trains an agent to maximize
# expected return from a given environment.
@dataclass
class DQNTrainer:
    """DQNTrainer trains a Q-network over multiple time-steps.

    At each time-step the environment is stepped and the transitions
    are stored in the replay buffer. Once the agent has performed a
    fixed amount of steps, the parameters of the q-network are updated
    by drawing random batches from the buffer and optimizing using a
    one-step TD target.

    The q-network and the environment are provided to the trainer upon
    initialization. The trainer can then be called multiple times to
    update the network parameters. You can see how well the current
    q-network is performing in between calling the trainer:

    ```python
    # Initialize the trainer.
    dqn_trainer = DQNTrainer(q_fn, optim_fn, env_fn, **kwargs)

    # Call the trainer to update the current params.
    params, opt_state = dqn_trainer(rng1, params, opt_state, n_steps, prefill_steps)

    # Test the agent after training.
    record_demo(env_fn, q_fn, params)

    # Train the agent some more.
    params, opt_state = dqn_trainer(rng2, params, opt_state, n_steps, 0)
    ```
    """

    q_fn: DeepQNetwork
    optim_fn: OptimizerFn
    env_fn: EnvironmentStepFn
    replay_buffer: ReplayBuffer
    discount: float = 1.    # discount for future rewards
    update_freq: int = 1    # how often to update the q-network
    n_updates: int = 1      # number of updates per iteration
    batch_size: int = 64    # batch size for iterating over the replay buffer
    huber_delta: float = 1. # bound for the huber loss transform
    p: float = 0.99         # factor for Polyak averaging
    eps: Callable[[int], float] = lambda x: 0.05 # schedule for epsi-greedy
    train_log: dict[str, list[float]] = field( # for logging info during training
        default_factory=lambda: defaultdict(list), init=False)

    def __call__(
        self,
        rng: Key,
        params: PyTree,
        opt_state: OptState,
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

        n_envs = self.env_fn(None)[0].shape[0]
        self.train_log["hyperparams"].append({
            "n_steps": n_steps,
            "n_envs": n_envs,
            "n_updates": self.n_updates,
        })

        tgt_params = deepcopy(params)   # tgt network params for double q-learning
        ep_r, ep_l = [], []             # lists for storing episode stats
        pbar = tqdm(total=n_steps)      # manual progress bar
        steps = 0                       # total number of steps performed

        while steps < n_steps:
            s = 0               # inner loop steps counter
            ep_r, ep_l = [], [] # lists for storing episode stats

            # Continue stepping the environment until it is time for update.
            while s < self.update_freq:
                # Step the environment and store the transitions.
                rng, rng_ = jax.random.split(rng, num=2)
                ts, info = step(rng_, self.env_fn, self.q_fn, params, eps=self.eps(steps+s))
                self.replay_buffer.store(ts)

                # Stepping the environment once actually performs `n_envs` steps.
                s += n_envs
                pbar.update(n_envs)

                # Bookkeeping.
                ep_r.extend(info["ep_r"])
                ep_l.extend(info["ep_l"])

            steps += s              # update the total step counter
            losses, norms = [], []  # lists for storing training stats

            if steps < prefill_steps: continue

            # Update the parameters of the q-network.
            for _ in range(self.n_updates):
                rng, rng_ = jax.random.split(rng, num=2)
                ts = self.replay_buffer.sample(rng_, self.batch_size)

                # Compute the loss.
                loss, grads = td_error(
                    ts, self.q_fn, params, tgt_params, self.discount, self.huber_delta,
                )

                # Bookkeeping.
                leaves, _ = jax.tree.flatten(grads)
                grad_norm = jnp.sqrt(sum(jnp.vdot(x, x) for x in leaves))
                losses.append(loss.item())
                norms.append(grad_norm.item())

                # Backward pass. Update the parameters of the q-network.
                params, opt_state = self.optim_fn(params, grads, opt_state)

                # Update the target network parameters using Polyak averaging.
                tgt_params = jax.tree.map(
                    lambda x, y: self.p*x+(1-self.p)*y, tgt_params, params,
                )

            # Bookkeeping. Store training stats averaged over the number of
            # updates. Store episode stats averaged over the time-steps.
            self.train_log["loss"].append((np.mean(losses), np.std(losses)))
            self.train_log["grad_norm"].append((np.mean(norms), np.std(norms)))
            with warnings.catch_warnings():
                # We might have not completed any episodes this iteration.
                # Just store NaNs and ignore the warning.
                warnings.simplefilter("ignore", category=RuntimeWarning)
                self.train_log["ep_r"].append((np.mean(ep_r), np.std(ep_r)))
                self.train_log["ep_l"].append((np.mean(ep_l), np.std(ep_l)))

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
    eps: float,
) -> tuple[Transitions, dict]:
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
        eps: float
            Epsilon value for epsilon-greedy action selection.

    Returns:
        Transitions
            Tuple (o, a, r, o_next, d) of nd-arrays.
        dict[str, Sequence[float]]
            Info dict.
    """
    o, *_ = env_fn(None) # shape (B, *)

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