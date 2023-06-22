from dataclasses import dataclass
import math
from operator import indexOf

from pysages.typing import JaxArray, NamedTuple, Scalar, Callable

def step_schedule(
    initial_lr,
    step_size,
    gamma
):
    def update(i):
        lr = initial_lr
        def _loop(j, state):
            i, v = state
            coeff = jnp.maximum(0, jnp.sign(step_size*j - i))
            v = v * coeff + (1-coeff) * gamma * v
            return (i, v)
        _, lr = jax.lax.fori_loop(0, jnp.floor(i/step_size).astype(int),_loop,(i,lr))
        return lr
    return update

def multistep_schedule(
    initial_lr,
    milestones,
    gamma
):
    def update(i):
        lr = initial_lr
        for milestone in milestones:
            coeff = jnp.maximum(0, jnp.sign(milestone - i))
            lr = lr * coeff + (1-coeff) * gamma * lr
        return lr
    return update


def cosine_annealing_schedule(
    initial_lr,
    T_max,
    eta_min=0
):
    def update(i):
        lr = initial_lr
        i = jnp.minimum(T_max, i)
        coeff = (1+jnp.cos(jnp.pi * i / T_max)) / 2
        lr = (1-eta_min) * lr * coeff + eta_min
        return lr
    return update

def constant_schedule(
    initial_lr,
    factor,
    T_max 
):
    def update(i):
        lr = initial_lr
        i = jnp.minimum(T_max, i)
        lr = lr * (factor ** i)
        return lr
    return update

def exponential_schedule(
    initial_lr,
    gamma,
    T_max
):
    def update(i):
        lr = initial_lr
        i = jnp.minimum(T_max, i)
        def _loop(j, v):
            v = v * (gamma ** j)
            return v
        lr = jax.lax.fori_loop(0, i,_loop,lr)
        return lr
    return update
