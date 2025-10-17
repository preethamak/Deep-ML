import numpy as np

# DO NOT CHANGE SEED
np.random.seed(42)

# Base Layer class
class Layer:
    def set_input_shape(self, shape):
        self.input_shape = shape

    def layer_name(self):
        return self.__class__.__name__

    def parameters(self):
        return 0

    def forward_pass(self, X, training=True):
        raise NotImplementedError()

    def backward_pass(self, accum_grad):
        raise NotImplementedError()

    def output_shape(self):
        raise NotImplementedError()

# Dense layer
class Dense(Layer):
    def __init__(self, n_units, input_shape=None):
        self.layer_input = None
        self.input_shape = input_shape
        self.n_units = n_units
        self.trainable = True
        self.W = None
        self.w0 = None
        self.optimizer_W = None
        self.optimizer_b = None

    def initialize(self, optimizer):
        """Initialize weights, bias, and optimizer."""
        limit = 1.0 / np.sqrt(self.input_shape[0])
        self.W = np.random.uniform(-limit, limit, (self.input_shape[0], self.n_units))
        self.w0 = np.zeros(self.n_units)
        self.optimizer_W = optimizer
        self.optimizer_b = optimizer

    def forward_pass(self, X, training=True):
        self.layer_input = X
        return np.dot(X, self.W) + self.w0

    def backward_pass(self, accum_grad):
        grad_input = np.dot(accum_grad, self.W.T)

        if self.trainable:
            grad_W = np.dot(self.layer_input.T, accum_grad)
            grad_b = np.sum(accum_grad, axis=0)

            self.W = self.optimizer_W.update(self.W, grad_W)
            self.w0 = self.optimizer_b.update(self.w0, grad_b)

        return grad_input

    def number_of_parameters(self):
        return np.prod(self.W.shape) + np.prod(self.w0.shape)

    def output_shape(self):
        return (self.n_units,)
