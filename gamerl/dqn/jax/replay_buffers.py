from typing import Any

import jax
from jax.typing import ArrayLike
import numpy as np


# Key for a pseudo random number generator (PRNG).
Key = Any

# Transitions is a tuple (o, a, r, o_next, d) of nd-arrays.
Transitions = tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike]

class VanillaReplayBuffer:

    def __init__(self, capacity: int, obs_shape: tuple[int]) -> None:
        # Pre-allocate the buffers for storing the (o, a, r, o_next, d) transitions.
        self.obs = np.empty((capacity,) + obs_shape, dtype=np.float32)
        self.acts = np.empty((capacity,), dtype=int)
        self.rewards = np.empty((capacity,), dtype=np.float32)
        self.obs_next = np.empty_like(self.obs)
        self.done = np.empty((capacity,), dtype=bool)

        # next_idx points to the location that should be filled next.
        self.next_idx = 0

        # Keep track of the total capacity, as well as the number of items
        # currently stored in the buffer.
        self.capacity = capacity
        self.size = 0

    def store(self, ts: Transitions) -> None:
        """Store a batch of transitions (o, a, r, o_next, d) in the buffer
        starting at the next available index, overwriting old transitions if necessary.
        """
        i = self.next_idx
        o, a, r, o_next, d = ts

        B = o.shape[0] # number of transitions in the batch

        if i+B <= self.capacity:
            self.obs[i:i+B] = o
            self.acts[i:i+B] = a
            self.rewards[i:i+B] = r
            self.obs_next[i:i+B] = o_next
            self.done[i:i+B] = d
        else:
            lim = self.capacity - i # number of transitions to get to the end

            self.obs[i:] = o[:lim]
            self.obs[:B-lim] = o[lim:]
            self.acts[i:] = a[:lim]
            self.acts[:B-lim] = a[lim:]
            self.rewards[i:] = r[:lim]
            self.rewards[:B-lim] = r[lim:]
            self.obs_next[i:] = o_next[:lim]
            self.obs_next[:B-lim] = o_next[lim:]
            self.done[i:] = d[:lim]
            self.done[:B-lim] = d[lim:]

        # Set the next free location. Rotate to the beginning when full.
        self.next_idx = (i+B) % self.capacity

        # Increment the size.
        self.size = min(self.size+B, self.capacity)

    def sample(self, rng: Key, batch_size: int) -> Transitions:
        idxs = jax.random.randint(rng, shape=(batch_size,), minval=0, maxval=self.size)

        return (self.obs[idxs],
            self.acts[idxs],
            self.rewards[idxs],
            self.obs_next[idxs],
            self.done[idxs])

    def __len__(self) -> int:
        return self.size

#