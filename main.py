import paddle
from paddle.nn import Linear
import paddle.nn.functional as F
import os
import numpy as np
import matplotlib.pyplot as plt

train_dataset = paddle.vision.datasets.MNIST(mode="train")
