import os
import sys
sys.path.append("..")

import gymnasium as gym
import jax
import jax.numpy as jnp
from jax.nn.initializers import uniform, glorot_uniform
import jax.example_libraries.stax as stax
import jax.example_libraries.optimizers as optim
import numpy as np
import matplotlib.pyplot as plt

from dqn import DQNLearner
from replay_buffers import VanillaReplayBuffer

gym.logger.min_level = gym.logger.ERROR


# DuelingQNetwork is a Callable that computes Q-values.
class DuelingQNetwork:
	def __init__(self, hidden_sizes, out_size):
		v_init_fn, v_apply_fn = stax.serial(
			*[
				stax.serial(
					stax.Dense(out_dim=h, W_init=glorot_uniform(), b_init=uniform(1e-2)),
					stax.Tanh,
				)
				for h in hidden_sizes
			],
			stax.Dense(out_dim=1, W_init=glorot_uniform(), b_init=uniform(1e-2)),
		)
		self.v_init_fn = v_init_fn
		self.v_apply_fn = jax.jit(v_apply_fn)

		a_init_fn, a_apply_fn = stax.serial(
			*[
				stax.serial(
					stax.Dense(out_dim=h, W_init=glorot_uniform(), b_init=uniform(1e-2)),
					stax.Tanh,
				)
				for h in hidden_sizes
			],
			stax.Dense(out_dim=out_size, W_init=glorot_uniform(), b_init=uniform(1e-2)),
		)
		self.a_init_fn = a_init_fn
		self.a_apply_fn = jax.jit(a_apply_fn)

	def __call__(self, params, x):
		v_params, a_params = params
		values = self.v_apply_fn(v_params, x)
		adv = self.a_apply_fn(a_params, x)
		return values + (adv - adv.mean(axis=-1, keepdims=True))

	def init_params(self, rng, in_shape):
		r1, r2 = jax.random.split(rng)
		_, v_params = self.v_init_fn(r1, (-1,) + in_shape)
		_, a_params = self.a_init_fn(r2, (-1,) + in_shape)
		return (v_params, a_params)

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

	np.random.seed(seed=1234)
	rng = jax.random.PRNGKey(seed=1234)

	# Hyperparameters.
	n_envs = 32
	steps_limit = 500
	hidden_sizes = [256, 128]
	buffer_capacity = 100000
	discount = 0.99
	batch_size = 128
	n_steps = int(1e6)
	prefill_steps = 50000
	lr = lambda i: (n_steps-i)/n_steps * 5e-4 + 5e-5 # 5e-4 --> 5e-5
	eps = lambda i: (n_steps-i)/n_steps * 0.14 + 0.02 # 0.16 --> 0.02

	# Define the EnvironmentStepFn.
	# Each sub-environment has a different set of random parameters,
	# that are clipped to stay in the recommended parameter space.
	rng, rng_ = jax.random.split(rng)
	env = gym.wrappers.vector.RecordEpisodeStatistics(
		gym.vector.SyncVectorEnv([
			lambda: gym.make(
				"LunarLander-v3",
				gravity=np.clip(np.random.normal(-10.0, 1.0), -11.99, -0.01),
				enable_wind=np.random.choice([True, False]),
				wind_power=np.clip(np.random.normal(15.0, 1.0), 0.01, 19.99),
				turbulence_power=np.clip(np.random.normal(1.5, 0.5), 0.01, 1.99),
				max_episode_steps=steps_limit,
			)
			for _ in range(n_envs)
		]),
	)
	env_fn = EnvironmentStepFn(rng_, env)

	# Define the DQN function.
	in_shape = env.single_observation_space.shape
	out_size = env.single_action_space.n
	rng, rng_ = jax.random.split(rng)
	q_fn = DuelingQNetwork(hidden_sizes, out_size)
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
		discount=discount, batch_size=batch_size,
		# n_updates=n_updates,
		eps=eps,
	)

	# Run the trainer.
	rng, rng_ = jax.random.split(rng)
	params, _ = dqn_learner(rng_, params, opt_state, n_steps, prefill_steps)

	env.close()
	log_dir = "LunarLander-v3"
	os.makedirs(log_dir, exist_ok=True)

	# Record demo with the trained agent.
	env = gym.wrappers.RecordVideo(
		gym.wrappers.Autoreset(
			gym.make("LunarLander-v3", render_mode="rgb_array"),
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