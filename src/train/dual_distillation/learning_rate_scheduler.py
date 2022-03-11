from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import CosineAnnealingLR


class LearningRateScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier_warmup, total_epoch_warmup, T_max, eta_min=0, last_epoch=-1):
        self.multiplier_warmup = multiplier_warmup
        self.total_epoch_warmup = total_epoch_warmup
        self.last_epoch = last_epoch
        self.after_scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=T_max, eta_min=eta_min)
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch_warmup:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier_warmup for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier_warmup for base_lr in self.base_lrs]

        if self.multiplier_warmup == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch_warmup) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier_warmup - 1.) * self.last_epoch / self.total_epoch_warmup + 1.) for base_lr in self.base_lrs]

    def step(self, epoch=None, metrics=None):
        if self.finished and self.after_scheduler:
            if epoch is None:
                self.after_scheduler.step(None)
            else:
                self.after_scheduler.step(epoch - self.total_epoch_warmup)
            self._last_lr = self.after_scheduler.get_last_lr()
        else:
            return super().step(epoch)
