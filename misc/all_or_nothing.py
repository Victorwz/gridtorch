import torch
import numpy as np
from torch.autograd import Variable

class AllOrNothing(torch.autograd.Function):
    """
    Custom threshold neuron function
    """

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return (input > 0.).float() * 1.

    @staticmethod
    def backward(ctx, grad_output):
        """
        use tanh pseudo derivative
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        return (1 - torch.nn.functional.tanh(input) ** 2) * grad_input

if __name__ == '__main__':

    real_tanh = torch.nn.Tanh()
    tanh = AllOrNothing.apply


    a = torch.nn.Parameter(torch.Tensor(np.arange(-100, 100)), requires_grad=True)
    a2 = torch.nn.Parameter(torch.Tensor(np.arange(-100, 100)), requires_grad=True)


    b = Variable(a + 1, requires_grad=True)
    b2 = Variable(a2 + 1, requires_grad=True)


    r = tanh(a)
    r2 = real_tanh(a2)
    print(r)

    l = torch.mean(b * r)
    l2 = torch.mean(b2 * r2)

    l.backward()
    l2.backward()
    print(a.grad)
    print(a2.grad)
