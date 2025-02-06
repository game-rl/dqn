from collections import deque
import os
import sys
sys.path.append("..")

import gymnasium as gym
import jax
import jax.numpy as jnp
from jax.nn.initializers import zeros, uniform
import jax.example_libraries.stax as stax
import jax.example_libraries.optimizers as optim
import numpy as np
import matplotlib.pyplot as plt

from dqn import DQNLearner

gym.logger.min_level = gym.logger.ERROR


class VanillaReplayBuffer:
	"""Really slow. Just for illustration purposes."""
	def __init__(self, capacity):
		self.buffer = deque([], maxlen=capacity)
	def store(self, ts):
		self.buffer.append(ts)
	def sample(self, rng, batch_size):
		idxs = jax.random.randint(rng, shape=(batch_size,), minval=0, maxval=len(self.buffer))
		return [ np.stack(x) for x in zip(*(self.buffer[i] for i in idxs)) ]

# DeepQNetwork is a Callable that computes Q-values.
class DeepQNetwork:
	def __init__(self, hidden_sizes, out_size):
		init_fn, apply_fn = stax.serial(
			*[
				stax.serial(
					stax.Dense(out_dim=h, W_init=uniform(1e-3), b_init=zeros),
					stax.Tanh,
				)
				for h in hidden_sizes
			],
			stax.Dense(out_dim=out_size, W_init=uniform(1e-3), b_init=zeros),
		)
		self.init_fn = init_fn
		self.apply_fn = jax.jit(apply_fn)
	def __call__(self, params, x):
		return self.apply_fn(params, x)
	def init_params(self, rng, in_shape):
		_, params = self.init_fn(rng, (-1,) + in_shape)
		return params

# EnvironmentStepFn is a Callable that steps the environment.
class EnvironmentStepFn:
	def __init__(self, rng, env):
		seed = jax.random.randint(rng, (), 0, jnp.iinfo(jnp.int32).max).item()
		self.env = env
		self.o, _ = env.reset(seed=seed)
	def __call__(self, acts):
		if acts is None:
			return self.o, None, None, None, None
		acts = np.asarray(acts)
		res = self.env.step(acts)
		self.o = res[0]
		return res

# OptimizerFn is a Callable that updates the parameters.
class OptimizerFn:
	def __init__(self, opt_update, get_params):
		self.opt_update = opt_update
		self.get_params = get_params
		self.step = 0
	def __call__(self, params, grads, opt_state):
		opt_state = self.opt_update(self.step, grads, opt_state)
		params = self.get_params(opt_state)
		self.step += 1
		return params, opt_state


if __name__ == "__main__":

	np.random.seed(seed=1234)
	rng = jax.random.PRNGKey(seed=1234)

	# Hyperparameters.
	steps_limit = 500
	hidden_sizes = [64]
	lr = 1e-4
	buffer_capacity = 20000
	discount = 0.99
	batch_size = 128
	n_steps = int(3e5)
	prefill_steps = 20000
	eps = lambda i: (n_steps-i)/n_steps * 0.12 + 0.04 # 0.16 --> 0.04

	# Define the EnvironmentStepFn.
	rng, rng_ = jax.random.split(rng)
	env = gym.wrappers.Autoreset(
		gym.make("CartPole-v1", max_episode_steps=steps_limit),
	)
	env_fn = EnvironmentStepFn(rng_, env)

	# Define the DQN function.
	in_shape = env.observation_space.shape
	out_size = env.action_space.n
	q_fn = DeepQNetwork(hidden_sizes, out_size)
	params = q_fn.init_params(rng_, (-1,) + in_shape)

	# Define the OptimizerFn.
	opt_init, opt_update, get_params = optim.adam(step_size=lr)
	opt_state = opt_init(params)
	optim_fn = OptimizerFn(jax.jit(opt_update), get_params)

	# Define the ReplayBuffer.
	buffer = VanillaReplayBuffer(capacity=buffer_capacity)

	# Define and run the DQN Trainer.
	dqn_learner = DQNLearner(
		q_fn, optim_fn, env_fn, buffer,
		discount=discount, batch_size=batch_size, eps=eps,
	)
	rng, rng_ = jax.random.split(rng)
	params, _ = dqn_learner(rng_, params, opt_state, n_steps, prefill_steps)

	env.close()
	log_dir = "CartPole-v1"
	os.makedirs(log_dir, exist_ok=True)

	# Record a demo with the trained agent.
	env = gym.wrappers.RecordVideo(
		gym.wrappers.Autoreset(
			gym.make("CartPole-v1", render_mode="rgb_array"),
		),
		video_folder=log_dir,
		video_length=1000, # around 20 sec, depends on fps (usually 50fps)
	)
	seed = jax.random.randint(rng, (), 0, jnp.iinfo(jnp.int32).max).item()
	o, _ = env.reset(seed=seed)
	while env.recording:
		o = jnp.asarray(np.expand_dims(o, axis=0))
		q_values = q_fn(params, o)
		acts = jnp.argmax(q_values, axis=-1)
		acts = np.asarray(acts)[0]
		o, r, t, tr, info = env.step(acts)
	env.close()

	# Generate plots.
	plt.style.use("seaborn-v0_8") # ggplot
	for k in dqn_learner.train_log.keys():
		fig, ax = plt.subplots()
		ys = np.array(dqn_learner.train_log[k])
		xs = np.arange(len(ys))
		xs = xs[~(ys != ys)]
		ys = ys[~(ys != ys)] # remove NaNs
		ax.plot(xs, ys, label=k)
		ax.set_xlabel("steps")
		ax.set_ylabel(k)
		ax.ticklabel_format(style="sci", axis="x", scilimits=(0,0))
		fig.savefig(os.path.join(log_dir, k +".png"))

#