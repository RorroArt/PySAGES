from dataclasses import dataclass
import math
from operator import indexOf

from pysages.typing import JaxArray, NamedTuple, Scalar, Callable

@dataclass
class LRSchedule:
    """
    Base class for all learning rate schedulers. Contains information about the last
    interation and a function that updates the learning rate for the next optimizer 
    operation. 
    """
    learning_rate: Scalar
    update: Callable
    step_count: int = 0


@dataclass
class StepLR(LRSchedule):
    def __init__(self, learning_rate, step_size, gamma, last_iter=-1):
        self.step_size = step_size
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.last_iter = last_iter

        
    def update(self, i):
        #learning_rate = self.learning_rate
        last_iter = i-1

        lr = jax.lax.cond(
            (last_iter == self.last_iter) | ((last_iter == 0) | (last_iter % self.step_size != 0)), 
            lambda _: self.learning_rate * self.gamma ** jnp.floor(last_iter / self.step_size), 
            lambda _: self.learning_rate * self.gamma ** (last_iter / self.step_size),
            1)
        
        self.last_iter = last_iter

        return lr

# Broken
@dataclass 
class MultiStepLR(LRSchedule):
    def __init__(self, learning_rate, milestones, gamma):
        self.learning_rate = learning_rate
        self.milestones = milestones
        self.gamma = gamma

    def update(self, i):
        learning_rate = self.learning_rate
        last_iter = i - 1

        if last_iter == 0 or last_iter in self.milestones:
            self.learning_rate = learning_rate * self.gamma
            return learning_rate * self.gamma 
        
        self.step_count += 1

        return learning_rate

# Not Tested
@dataclass
class CosineAnnealingLR(LRSchedule):
    def __init__(self, learning_rate, T_max, eta_min=0):
        self.T_max = T_max
        self.eta_min = eta_min
        self.learning_rate = learning_rate

    def update(self, i):
        learning_rate = self.learning_rate
        last_iter = i-1
    
        if last_iter == 0:
            return learning_rate
        elif self.step_count and last_iter > 0:
            learning_rate = self.eta_min + (learning_rate - self.eta_min) * (1-math.cos(math.pi / self.T_max)) / 2
        elif (last_iter - 1 - self.T_max) % (2 * self.T_max):
            learning_rate = + (base_lr - self.eta_min) * (1 - math.cos(math.pi / self.T_max)) / 2
        else:
            learning_rate = (1 + math.cos(math.pi * last_iter / self.T_max)) / (1 + math.cos(math.pi * (last_iter - 1) / self.T_max)) * (learning_rate - self.eta_min) + self.eta_min
        
        self.learning_rate = learning_rate
        self.step_count += 1

        return learning_rate

# Not tested
@dataclass
class ConstantLR(LRSchedule):
    def __init__(self, factor=1/3, max_iters=5):
        self.factor = factor
        self.max_iters = max_iters

        assert factor <= 1 and factor >= 0, 'Factor must be between 0 and 1' 

    def update(self, i):
        learning_rate = self.learning_rate
        last_iter = i-1
    
        if last_iter == 0 or learning_rate < self.max_iters:
            learning_rate = learning_rate + self.factor

        self.learning_rate = learning_rate
        self.step_count += 1

        return learning_rate

# Not tested
@dataclass
class ConstantLR(LRSchedule):
    def __init__(self, learning_rate, factor=1/3, max_iters=5):
        self.factor = factor
        self.max_iters = max_iters
        self.learning_rate = learning_rate

        assert factor <= 1 and factor >= 0, 'Factor must be between 0 and 1' 

    def update(self, i):
        learning_rate = self.learning_rate
        last_iter = i - 1
    
        if last_iter == 0 or learning_rate < self.max_iters:
            learning_rate = learning_rate * self.factor

        self.step_count += 1

        return learning_rate

# Not tested
@dataclass
class LinearLR(LRSchedule):
    def __init__(self, learning_rate, inital_factor=1/3, final_factor=1, max_iters=5):
        self.initial_factor = initial_factor
        self.final_factor = final_factor
        self.max_iters = max_iters
        self.learning_rate = learning_rate

        assert initial_factor <= 1 and factor >= 0, 'Initial factor must be between 0 and 1'
        assert final_factor <= 1 and factor >= 0, 'Final factor must be between 0 and 1'  

    def update(self, i):
        learning_rate = self.learning_rate
        last_iter = i - 1
    
        if last_iter == 0 or learning_rate < self.max_iters:
            learning_rate = learning_rate * self.factor

        self.learning_rate = learning_rate
        self.step_count += 1

        return learning_rate