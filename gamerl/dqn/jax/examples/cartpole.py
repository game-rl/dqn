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
from replay_buffers import VanillaReplayBuffer

gym.logger.min_level = gym.logger.ERROR


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
		max_norm = 1.
		leaves, _ = jax.tree.flatten(grads)
		total_norm = jnp.sqrt(sum(jnp.vdot(x, x) for x in leaves))
		clip_coef = max_norm / (total_norm + 1e-6)
		clip_coef = jnp.minimum(clip_coef, 1.0)
		grads = jax.tree.map(lambda g: clip_coef * g, grads) # clip grads
		opt_state = self.opt_update(self.step, grads, opt_state)
		params = self.get_params(opt_state)
		self.step += 1
		return params, opt_state


if __name__ == "__main__":

	np.random.seed(seed=0)
	rng = jax.random.PRNGKey(seed=0)

	# Hyperparameters.
	n_envs = 32
	steps_limit = 500
	hidden_sizes = [64]
	lr = 5e-4
	buffer_capacity = 50000
	discount = 0.8
	n_updates = 100
	batch_size = 128
	n_steps = int(1e6)
	prefill_steps = 50000
	eps = lambda i: max((3e5-i)/3e5 * 0.14 + 0.02, 0.02) # 0.16 --> 0.02 --> 0.02

	# Define the EnvironmentStepFn.
	rng, rng_ = jax.random.split(rng)
	env = gym.wrappers.vector.RecordEpisodeStatistics(
		gym.make_vec("CartPole-v1", num_envs=n_envs, vectorization_mode="sync"),
	)
	env_fn = EnvironmentStepFn(rng_, env)

	# Define the DQN function.
	in_shape = env.single_observation_space.shape
	out_size = env.single_action_space.n
	rng, rng_ = jax.random.split(rng)
	q_fn = DeepQNetwork(hidden_sizes, out_size)
	params = q_fn.init_params(rng_, (-1,) + in_shape)

	# Define the OptimizerFn.
	opt_init, opt_update, get_params = optim.adam(step_size=lr)
	opt_state = opt_init(params)
	optim_fn = OptimizerFn(jax.jit(opt_update), get_params)

	# Define the ReplayBuffer.
	buffer = VanillaReplayBuffer(capacity=buffer_capacity, obs_shape=in_shape)

	# Define the DQN Trainer.
	dqn_learner = DQNLearner(
		q_fn, optim_fn, env_fn, buffer,
		discount=discount, batch_size=batch_size, #n_updates=n_updates,
		eps=eps,
	)

	# Run the trainer.
	rng, rng_ = jax.random.split(rng)
	params, _ = dqn_learner(rng_, params, opt_state, n_steps, prefill_steps)

	env.close()
	log_dir = "CartPole-v1"
	os.makedirs(log_dir, exist_ok=True)

	# Record demo with the trained agent.
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
		if k == "hyperparams": continue # skip this

		fig, ax = plt.subplots()

		total = len(dqn_learner.train_log[k])
		avg_every = total // 100
		xs = np.arange(total)[::avg_every]
		ys = dqn_learner.train_log[k]

		if k in ["ep_r", "ep_l"]:
			total_steps = dqn_learner.train_log["hyperparams"][0]["n_steps"]
			xs = np.linspace(0, total_steps, len(xs))

		# xs has `ceil(total / avg_every)` elements.
		# We may need to pad the ys.
		avg = np.nanmean(np.pad(
			ys,
			(0, -len(ys)%avg_every),
			constant_values=np.nan,
		).reshape(-1, avg_every), axis=1)
		std = np.nanstd(np.pad(
			ys,
			(0, -len(ys)%avg_every),
			constant_values=np.nan,
		).reshape(-1, avg_every), axis=1)

		# Remove NaNs, if any.
		xs = xs[~(avg != avg)]
		avg = avg[~(avg != avg)]
		std = std[~(std != std)]

		# Plot the avg and the std.
		ax.plot(xs, avg, label=k)
		ax.fill_between(xs, avg-0.5*std, avg+0.5*std, color="k", alpha=0.25)

		x_label = "Total time-steps" if k in {"ep_r", "ep_l"} else "Gradient updates"
		ax.set_xlabel(x_label)
		ax.set_ylabel(k)
		ax.ticklabel_format(style="sci", axis="x", scilimits=(0,0))
		ax.legend()
		fig.savefig(os.path.join(log_dir, k +".png"))

#