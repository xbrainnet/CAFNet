import torch
import torch.nn as nn
from torch import Tensor

class MySpl_loss(nn.CrossEntropyLoss):
	def __init__(self, *args, n_samples=0, batch_size, alpha, beta, spl_lambda, spl_gamma, **kwargs):
		super(MySpl_loss, self).__init__(*args, **kwargs)
		self.spl_lambda = spl_lambda
		self.spl_gamma = spl_gamma
		self.alpha = alpha
		self.beta = beta
		self.v = torch.zeros(n_samples, batch_size).int()

	def forward(self, input: Tensor, target: Tensor, index: Tensor, c_loss: Tensor, kd_loss: Tensor) -> Tensor:
		super_loss = nn.functional.cross_entropy(input, target, reduction='none') + self.alpha * c_loss + self.beta * kd_loss
		v = self.spl_loss(super_loss)
		self.v[index] = v
		return (super_loss * v).mean()

	def increase_threshold(self):
		self.spl_lambda *= self.spl_gamma

	def spl_loss(self, super_loss):
		v = super_loss < self.spl_lambda
		return v.int()
