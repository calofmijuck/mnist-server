from .util import im2col, col2im
import numpy as np


def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))


def cross_entropy(x, t):
    if x.ndim == 1:
        t = t.reshape(1, t.size)
        x = x.reshape(1, x.size)

    if t.size == x.size:
        t = t.argmax(axis=1)

    ep = 1e-7
    batch_size = x.shape[0]
    return -np.sum(np.log(x[np.arange(batch_size), t] + ep)) / batch_size


class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = x <= 0
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        return dout


class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        return dout * (1 - self.out) * self.out


class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.x_orig_shape = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x
        return np.dot(x, self.W) + self.b

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        dx = dx.reshape(*self.original_x_shape)  # For tensors
        return dx


class SoftmaxLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size:  # when answer labels are one-hot encoded
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size
        return dx


class BatchNormalization:
    def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None):
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.input_shape = None

        self.running_mean = running_mean
        self.running_var = running_var

        # For backprop
        self.batch_size = None
        self.xc = None
        self.std = None
        self.dgamma = None
        self.dbeta = None

    def forward(self, x, train_flg=True):
        self.input_shape = x.shape
        if x.ndim != 2:
            N = x.shape[0]
            x = x.reshape(N, -1)
        out = self.__forward(x, train_flg)
        return out.reshape(*self.input_shape)

    def __forward(self, x, train_flg):
        if self.running_mean is None:
            D = x.shape[1]
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)

        if train_flg:
            mu = x.mean(axis=0)
            xc = x - mu
            var = np.mean(xc ** 2, axis=0)
            std = np.sqrt(var + 10e-7)
            xn = xc / std
            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            self.running_mean = (
                self.momentum * self.running_mean + (1 - self.momentum) * mu
            )
            self.running_var = (
                self.momentum * self.running_var + (1 - self.momentum) * var
            )
        else:
            xc = x - self.running_mean
            xn = xc / (np.sqrt(self.running_var + 10e-7))

        out = self.gamma * xn + self.beta
        return out

    def backward(self, dout):
        if dout.ndim != 2:
            N = dout.shape[0]
            dout = dout.reshape(N, -1)

        dx = self.__backward(dout)
        return dx.reshape(*self.input_shape)

    def __backward(self, dout):
        dbeta = dout.sum(axis=0)
        dgamma = np.sum(self.xn * dout, axis=0)
        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size

        self.dgamma = dgamma
        self.dbeta = dbeta
        return dx


class Dropout:
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flg=True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x

    def backward(self, dout):
        return dout * self.mask


class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad

        # Data itself is unnecessary
        # self.x = None
        self.shape = None
        self.col = None
        self.col_W = None

        self.dW = None
        self.db = None

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        self.shape = x.shape
        # self.x = x

        assert (H + 2 * self.pad - FH) % self.stride == 0
        assert (W + 2 * self.pad - FW) % self.stride == 0

        out_h = 1 + (H + 2 * self.pad - FH) // self.stride
        out_w = 1 + (W + 2 * self.pad - FW) // self.stride

        # Flatten input data, dimension will be (N * out_h * out_w, C * FH * FW)
        col = im2col(x, FH, FW, self.stride, self.pad)

        # Reshape the filter to have dimension (C * FH * FW, FN)
        col_W = self.W.reshape(FN, -1).T

        self.col = col
        self.col_W = col_W

        # Convolution and add bias Y = X * W + B
        out = np.dot(col, col_W) + self.b

        # Axis order should be (N, C, out_h, out_w)
        return out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

    def backward(self, dout):
        # Backprop is similar to that of Affine layer

        FN, C, FH, FW = self.W.shape

        # Axis order should be (N, out_h, out_w, C) and reshape it
        # dL/dY should have dimension (N * out_h * out_w, FN)
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)

        self.db = np.sum(dout, axis=0)

        # dL/dW = X^T * dL/dY
        self.dW = np.dot(self.col.T, dout)

        # In forward, `col_W = self.W.reshape(FN, -1).T` was done
        # So transpose dW and reshape it to the original filter size
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        # dL/dX = dL/dY * W^T
        dcol = np.dot(dout, self.col_W.T)

        # Pack the 2d array into image and return
        return col2im(dcol, self.shape, FH, FW, self.stride, self.pad)


class Pooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

        # self.x = None
        self.shape = None
        self.arg_max = None

    def forward(self, x):
        self.shape = x.shape

        N, C, H, W = x.shape
        out_h = 1 + (H - self.pool_h) // self.stride
        out_w = 1 + (W - self.pool_w) // self.stride

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h * self.pool_w)

        # Store argmax and calculate max
        self.arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)

        # Shape into (N, C, out_h, out_w)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        return out

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)

        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size),
             self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,))

        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.shape, self.pool_h,
                    self.pool_w, self.stride, self.pad)

        return dx
