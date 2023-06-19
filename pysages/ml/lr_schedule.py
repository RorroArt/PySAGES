from dataclasses import dataclass
import math

from pysages.typing import JaxArray, NamedTuple, Scalar

@dataclass
class LRSchedule:
    """
    Base class for all learning rate schedulers. Contains information about the last
    interation and a function that updates the learning rate for the next optimizer 
    operation. 
    """
    last_epoch: int
    learning_rate: int
    step_count: int = 0
    update: Callable


@dataclass
class StepLR(LRSchedule):
    def __init__(self, learning_rate, step_size, gamma, last_epoch=-1):
        self.learning_rate = learning_rate
        self.step_size = step_size
        self.gamma = gamma
        self.last_epoch = last_epoch

        
    def update(self):
        learning_rate = self.learning_rate

        if self.last_epoch == 0 or self.last_epoch % self.step_size == 0:
            self.learning_rate = learning_rate * gamma
            return learning_rate * gamma 

        self.last_epoch += 1           
        self.step_count += 1

        return learning_rate

@dataclass 
class MultiStepLR(LRSchedule)
def __init__(self, learning_rate, milestones, gamma, last_epoch=-1):
        self.learning_rate = learning_rate
        self.milestones = milestones
        self.gamma = gamma
        self.last_epoch = last_epoch

    def update(self):
        learning_rate = self.learning_rate
        
        if self.last_epoch == 0 or self.last_epoch is in self.milestones:
            self.learning_rate = learning_rate * gamma
            return learning_rate * gamma 
        
        self.step_count += 1
        self.last_epoch += 1

        return learning_rate

@dataclass
class CosineAnnealingLR(LRScheduler):
    def __init__(self, T_max, eta_min=0, last_epoch=-1):
        self.T_max = T_max
        self.eta_min = eta_min
        self.last_epoch = last_epoch

    def update(self):
        learning_rate = self.learning_rate
    
        if self.last_epoch == 0:
            return learning_rate
        elif self.step_count and self.last_epoch > 0:
            learning_rate = self.eta_min + (learning_rate - self.eta_min) * (1-math.cos(math.pi / self.T_max)) / 2
        elif (self.last_epoch - 1 - self.T_max) % (2 * self.T_max):
            learning_rate = + (base_lr - self.eta_min) * (1 - math.cos(math.pi / self.T_max)) / 2
        else:
            learning_rate = (1 + math.cos(math.pi * self.last_epoch / self.T_max)) /
                (1 + math.cos(math.pi * (self.last_epoch - 1) / self.T_max)) *
                (learning_rate - self.eta_min) + self.eta_min
        
        self.learning_rate = learning_rate
        self.step_count += 1
        self.last_epoch += 1

        return learning_rate

@dataclass
class ConstantLR(LRScheduler):
    def __init__(self, factor=1/3, max_iters=5, last_epoch=-1):
        self.factor = factor
        self.max_iters = max_iters
        self.last_epoch = last_epoch

        assert factor <= 1 and factor >= 0, 'Factor must be between 0 and 1' 

    def update(self):
        learning_rate = self.learning_rate
    
        if self.last_epoch == 0 or learning_rate < self.max_iters:
            learning_rate = learning_rate + self.factor

        self.learning_rate = learning_rate
        self.step_count += 1
        self.last_epoch += 1

        return learning_rate

@dataclass
class ConstantLR(LRScheduler):
    def __init__(self, factor=1/3, max_iters=5, last_epoch=-1):
        self.factor = factor
        self.max_iters = max_iters
        self.last_epoch = last_epoch

        assert factor <= 1 and factor >= 0, 'Factor must be between 0 and 1' 

    def update(self):
        learning_rate = self.learning_rate
    
        if self.last_epoch == 0 or learning_rate < self.max_iters:
            learning_rate = learning_rate * self.factor

        self.step_count += 1
        self.last_epoch += 1

        return learning_rate

@dataclass
class LinearLR(LRScheduler):
    def __init__(self, inital_factor=1/3, final_factor=1, max_iters=5, last_epoch=-1):
        self.initial_factor = initial_factor
        self.final_factor = final_factor
        self.max_iters = max_iters
        self.last_epoch = last_epoch

        assert initial_factor <= 1 and factor >= 0, 'Initial factor must be between 0 and 1'
        assert final_factor <= 1 and factor >= 0, 'Final factor must be between 0 and 1'  

    def update(self):
        learning_rate = self.learning_rate
    
        if self.last_epoch == 0 or learning_rate < self.max_iters:
            learning_rate = learning_rate * self.factor

        self.learning_rate = learning_rate
        self.step_count += 1
        self.last_epoch += 1

        return learning_rate