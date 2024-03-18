from DR_algorithms.DR_algorithm import DR_algorithm
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

class myTest(DR_algorithm):
    def __init__(self, algo_name):
        super(myTest, self).__init__(algo_name)
        self.add_int_hyperparameter('max epoch', 20, 10000, 10, 500)
        self.add_int_hyperparameter('batch size', 20, 10000, 10, 300)
        self.embedding = None

    def fit(self, progress_listener, X, Y):
        pass

    def transform(self, X, Y):
        pass
