

@dataclass
class LRSchedule:
    """
    Base class for all learning rate schedulers. Contains information about the last
    interation and a function that updates the learning rate for the next optimizer 
    operation. 
    """
    last_epoch: Scalar
    learning_rate: Scalar
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
        self.last_epoch += 1

        if self.last_epoch == 0 or self.last_epoch % self.step_size == 0:
            self.learning_rate = learning_rate * gamma
            return learning_rate * gamma 

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
        self.last_epoch += 1

        if self.last_epoch == 0 or self.last_epoch is in self.milestones:
            self.learning_rate = learning_rate * gamma
            return learning_rate * gamma 

        return learning_rate