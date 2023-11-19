import paddle
import paddle.nn as nn
import numpy as np
import paddle.nn.functional as F
from math import exp


def gaussian(window_size, sigma):
    gauss = paddle.to_tensor(np.array([
        exp(-(x - window_size // 2)**2 / float(2 * sigma**2))
        for x in range(window_size)
    ]))
    return gauss / gauss.sum()



def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).unsqueeze(0).unsqueeze(0)
    window = paddle.to_tensor(paddle.expand(
        _2D_window, (channel, 1, window_size, window_size)),
        stop_gradient=False)
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(
        img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(
        img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(
        img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) *
                (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                       (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(paddle.nn.Layer):

    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.shape

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            tt = img1.dtype
            window = paddle.to_tensor(window, dtype=tt)
            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel,
                     self.size_average)


class PSNRLoss(nn.Layer):

    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = paddle.to_tensor(np.array([65.481, 128.553,
                                               24.966])).reshape([1, 3, 1, 1])

    def forward(self, pred, target):
        if self.toY:
            pred = (pred * self.coef).sum(axis=1).unsqueeze(axis=1) + 16.
            target = (target * self.coef).sum(axis=1).unsqueeze(axis=1) + 16.

            pred, target = pred / 255., target / 255.
            pass

        return self.loss_weight * self.scale * paddle.log(((pred - target)**2).mean(axis=[1, 2, 3]) + 1e-8).mean()


class CharbonnierLoss():
    """Charbonnier Loss (L1).

    Args:
        eps (float): Default: 1e-12.

    """

    def __init__(self, eps=1e-12, reduction='sum'):
        self.eps = eps
        self.reduction = reduction

    def __call__(self, pred, target, **kwargs):
        """Forward Function.

        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
        """
        if self.reduction == 'sum':
            out = paddle.sum(paddle.sqrt((pred - target)**2 + self.eps))
        elif self.reduction == 'mean':
            out = paddle.mean(paddle.sqrt((pred - target)**2 + self.eps))
        else:
            raise NotImplementedError('CharbonnierLoss %s not implemented' %
                                      self.reduction)
        return out


class EdgeLoss():

    def __init__(self):
        k = paddle.to_tensor(np.array([[.05, .25, .4, .25, .05]]), dtype='float32')
        self.kernel = paddle.matmul(k.t(), k).unsqueeze(0).tile([3, 1, 1, 1])
        self.loss = CharbonnierLoss()

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, [kw // 2, kh // 2, kw // 2, kh // 2], mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered = self.conv_gauss(current)  # filter
        down = filtered[:, :, ::2, ::2]  # downsample
        new_filter = paddle.zeros_like(filtered)
        new_filter.stop_gradient = True
        new_filter[:, :, ::2, ::2] = down * 4  # upsample
        filtered = self.conv_gauss(new_filter)  # filter
        diff = current - filtered
        return diff

    def __call__(self, x, y):
        y.stop_gradient = True
        loss = self.loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss




