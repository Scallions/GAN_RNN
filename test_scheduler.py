from cProfile import label
from collections import defaultdict
from sched import scheduler
import paddle
import paddle.optimizer.lr as lr


schedulers = [
    # lr.PolynomialDecay(2,1000,0.1, cycle=True),
    lr.PolynomialDecay(2,100,0.1, cycle=True),
    lr.CosineAnnealingDecay(2,100,0.1),
    lr.ExponentialDecay(2,0.9),
    lr.InverseTimeDecay(2,1),
    lr.MultiStepDecay(2,list(range(0,1000,100)),0.1),
    lr.NaturalExpDecay(2,0.9),
    lr.NoamDecay(32, 10, 2),
    lr.LinearWarmup(lr.CosineAnnealingDecay(2,100,0.1), 100, 0.1, 2),

]

lrs = defaultdict(list)

for i in range(1000):
    for sch in schedulers:
        sch.step()
        lrs[type(sch).__name__].append(sch.get_lr())

import matplotlib.pyplot as plt

for sch in schedulers:
    plt.plot(lrs[type(sch).__name__], label=type(sch).__name__)
plt.legend()
plt.show()
