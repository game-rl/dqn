import os

import gymnasium as gym
import jax
import jax.numpy as jnp
from jax.nn.initializers import uniform, kaiming_uniform, zeros, glorot_normal
import jax.example_libraries.stax as stax
import jax.example_libraries.optimizers as optim
import numpy as np
import matplotlib.pyplot as plt

from dqn import DQNTrainer
from replay_buffers import VanillaReplayBuffer


# EnvironmentStepFn is a Callable that steps the environment.
class EnvironmentStepFn:
    def __init__(self, env):
        self.env = env
        self.o, _ = env.reset(seed=0)
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
        max_norm = 10.
        leaves, _ = jax.tree.flatten(grads)
        total_norm = jnp.sqrt(sum(jnp.vdot(x, x) for x in leaves))
        clip_coef = max_norm / (total_norm + 1e-6)
        clip_coef = jnp.maximum(clip_coef, 1.0)
        grads = jax.tree.map(lambda g: clip_coef * g, grads) # clip grads
        opt_state = self.opt_update(self.step, grads, opt_state)
        params = self.get_params(opt_state)
        self.step += 1
        return params, opt_state


# CartPole trains a DQN Agent on the CartPole-v1 gym environment.
def CartPole():
    np.random.seed(seed=42)
    rng = jax.random.PRNGKey(seed=42)

    # Define the EnvironmentStepFn.
    n_envs = 8
    steps_limit = 500
    env = gym.wrappers.vector.RecordEpisodeStatistics(
        gym.vector.SyncVectorEnv([
            lambda: gym.make("CartPole-v1", max_episode_steps=steps_limit)
            for _ in range(n_envs)
        ]),
    )
    env_fn = EnvironmentStepFn(env)

    # Define the DQN function.
    in_shape = env.single_observation_space.shape
    out_size = env.single_action_space.n
    init_fn, apply_fn = stax.serial(  # deep q-network [in, hid=[64], out]
        stax.Dense(out_dim=64, W_init=uniform(1e-3), b_init=zeros),
        stax.Tanh,
        stax.Dense(out_dim=out_size, W_init=uniform(1e-3), b_init=zeros),
    )
    rng, rng_ = jax.random.split(rng, num=2)
    _, params = init_fn(rng_, (-1,) + in_shape)
    q_fn = jax.jit(apply_fn) # a callable that computes state-action values

    # Define the OptimizerFn.
    opt_init, opt_update, get_params = optim.adam(step_size=1e-3)
    opt_state = opt_init(params)
    optim_fn = OptimizerFn(jax.jit(opt_update), get_params)

    # Define the ReplayBuffer.
    buffer = VanillaReplayBuffer(capacity=50000, obs_shape=in_shape)

    # Define the DQN Trainer.
    n_steps, n_updates = 32, 128
    dqn_trainer = DQNTrainer(
        q_fn, optim_fn, env_fn, buffer,
        n_steps=n_steps, discount=0.99, n_updates=n_updates, update_tgt=20, batch_size=64,
        huber_delta=1.,
    )

    log_dir = os.path.join("logs", "CartPole-v1")
    os.makedirs(log_dir, exist_ok=True)

    # Run the trainer and plot the results.
    # The total number of time steps is ``n_iters x n_steps x n_envs``.
    #                              1e5 =    400   x    32   x   8
    n_iters = 1000
    params, _ = dqn_trainer(rng, params, opt_state, n_iters=n_iters)
    generate_plots(log_dir, dqn_trainer.train_log, n_steps, n_updates)
    record_demo(rng, log_dir, "CartPole-v1", q_fn, params)
    env.close()

# LunarLander trains a DQN Agent on the LunarLander-v3 gym environment.
def LunarLander():
    np.random.seed(seed=0)
    rng = jax.random.PRNGKey(seed=0)

    # Define the EnvironmentStepFn.
    n_envs = 16
    steps_limit = 500
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
    env_fn = EnvironmentStepFn(env)

    # Define the DQN function.
    in_shape = env.single_observation_space.shape
    out_size = env.single_action_space.n
    init_fn, apply_fn = stax.serial(  # deep q-network [in, hid=[64], out]
        stax.Dense(out_dim=64, W_init=glorot_normal(), b_init=zeros),
        stax.Tanh,
        stax.Dense(out_dim=64, W_init=glorot_normal(), b_init=zeros),
        stax.Tanh,
        stax.Dense(out_dim=out_size, W_init=glorot_normal(), b_init=zeros),
    )
    rng, rng_ = jax.random.split(rng, num=2)
    _, params = init_fn(rng_, (-1,) + in_shape)
    q_fn = jax.jit(apply_fn) # a callable that computes state-action values

    # Define the OptimizerFn.
    opt_init, opt_update, get_params = optim.adam(step_size=1e-3)
    opt_state = opt_init(params)
    optim_fn = OptimizerFn(jax.jit(opt_update), get_params)

    # Define the ReplayBuffer.
    buffer = VanillaReplayBuffer(capacity=50000, obs_shape=in_shape)

    # Define the DQN Trainer.
    n_steps, n_updates = 32, 128
    dqn_trainer = DQNTrainer(
        q_fn, optim_fn, env_fn, buffer,
        n_steps=n_steps, discount=0.99, n_updates=n_updates, update_tgt=10, batch_size=64,
        huber_delta=1.,
    )

    log_dir = os.path.join("logs", "LunarLander-v3")
    os.makedirs(log_dir, exist_ok=True)

    # Run the trainer and plot the results.
    # The total number of time steps is ``n_iters x n_steps x n_envs``.
    #                             2.5e6 =  5000   x    32   x   16
    n_iters = 5000 # 2000 is fine
    params, _ = dqn_trainer(rng, params, opt_state, n_iters=n_iters)
    generate_plots(log_dir, dqn_trainer.train_log, n_steps, n_updates)
    record_demo(rng, log_dir, "LunarLander-v3", q_fn, params)
    env.close()


def record_demo(rng, log_dir, env_name, q_fn, params):
    env = gym.wrappers.RecordVideo(
        gym.wrappers.Autoreset(
            gym.make(env_name, render_mode="rgb_array"),
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

def generate_plots(log_dir, train_log, n_steps, n_updates):
    plt.style.use("ggplot")

    keys = ("Total Loss", "Total Grad Norm")
    for k in keys:
        assert isinstance(train_log[k][0], float), "non-numerical log"
        assert len(train_log[k]) % n_updates == 0, "total not divisible"
        total = len(train_log[k])
        xs = np.linspace(0, total-n_updates, total//n_updates)
        ys = np.array(train_log[k]).reshape(-1, n_updates)
        avg, std = ys.mean(axis=1), ys.std(axis=1)
        fig, ax = plt.subplots()
        ax.plot(xs, avg, label=k)
        ax.fill_between(xs, avg-0.5*std, avg+0.5*std, color="k", alpha=0.25)

        xs_ = np.linspace(0, total-n_updates, total//n_updates//20)
        ys_ = np.array(train_log[k]).reshape(-1, 20*n_updates)
        ax.plot(xs_, np.nanmean(ys_, axis=1))

        ax.set_xlabel("Gradient updates")
        ax.set_ylabel(k)
        ax.ticklabel_format(style="sci", axis="x", scilimits=(0,0))
        ax.legend()
        fig.savefig(os.path.join(log_dir, k.replace(" ", "_")+".png"))

    keys = ("Episode Returns", "Episode Lengths")
    for k in keys:
        assert isinstance(train_log[k][0], float), "non-numerical log"
        assert len(train_log[k]) % n_steps == 0, "total not divisible"
        total = len(train_log[k])
        xs = np.linspace(0, total-n_steps, total//n_steps)
        ys = np.array(train_log[k]).reshape(-1, n_steps)
        avg, std = np.nanmean(ys, axis=1), np.nanstd(ys, axis=1)
        fig, ax = plt.subplots()
        ax.plot(xs, avg, label=k)
        ax.fill_between(xs, avg-0.5*std, avg+0.5*std, color="k", alpha=0.25)

        xs_ = np.linspace(0, total-n_steps, total//n_steps//20)
        ys_ = np.array(train_log[k]).reshape(-1, 20*n_steps)
        ax.plot(xs_, np.nanmean(ys_, axis=1))

        ax.set_xlabel("Number of steps")
        ax.set_ylabel(k)
        ax.ticklabel_format(style="sci", axis="x", scilimits=(0,0))
        ax.legend()
        fig.savefig(os.path.join(log_dir, k.replace(" ", "_")+".png"))

if __name__ == "__main__":
    CartPole()
    LunarLander()

#