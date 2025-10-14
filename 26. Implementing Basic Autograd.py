import torch

class Value:
    """A tiny scalar wrapper that uses PyTorch autograd to track gradients."""

    def __init__(self, data):
        self._t = torch.tensor(float(data), requires_grad=True)
        self._t.retain_grad()  # keep grads for printing

    @property
    def data(self):
        return self._t.item()

    @property
    def grad(self):
        g = self._t.grad
        return 0 if g is None else g.item()

    def __repr__(self):
        # print as int if whole number
        def fmt(x):
            return int(x) if float(x).is_integer() else round(x, 4)
        return f"Value(data={fmt(self.data)}, grad={fmt(self.grad)})"

    def _wrap(self, other):
        return other if isinstance(other, Value) else Value(other)

    def __add__(self, other):
        other = self._wrap(other)
        return Value(0)._replace(self._t + other._t)

    __radd__ = __add__

    def __mul__(self, other):
        other = self._wrap(other)
        return Value(0)._replace(self._t * other._t)

    __rmul__ = __mul__

    def relu(self):
        return Value(0)._replace(torch.relu(self._t))

    def _replace(self, tensor):
        v = Value(0)
        v._t = tensor
        v._t.retain_grad()
        return v

    def backward(self):
        self._t.backward()
