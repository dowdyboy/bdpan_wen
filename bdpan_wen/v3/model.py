# Copyright (c) 2022 megvii-model. All Rights Reserved.
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors

import paddle
from paddle import nn as nn
import paddle.nn.functional as F
from paddle.autograd import PyLayer

class LayerNormFunction(PyLayer):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.shape
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.reshape([1, C, 1, 1]) * y + bias.reshape([1, C, 1, 1])
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.shape
        y, var, weight = ctx.saved_tensor()
        g = grad_output * weight.reshape([1, C, 1, 1])
        mean_g = g.mean(axis=1, keepdim=True)

        mean_gy = (g * y).mean(axis=1, keepdim=True)
        gx = 1. / paddle.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(axis=3).sum(axis=2).sum(
            axis=0), grad_output.sum(axis=3).sum(axis=2).sum(axis=0)


class LayerNorm2D(nn.Layer):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2D, self).__init__()
        self.add_parameter(
            'weight',
            self.create_parameter(
                [channels],
                default_initializer=paddle.nn.initializer.Constant(value=1.0)))
        self.add_parameter(
            'bias',
            self.create_parameter(
                [channels],
                default_initializer=paddle.nn.initializer.Constant(value=0.0)))
        self.eps = eps

    def forward(self, x):
        if self.training:
            y = LayerNormFunction.apply(x, self.weight, self.bias, self.eps)
        else:
            N, C, H, W = x.shape
            mu = x.mean(1, keepdim=True)
            var = (x - mu).pow(2).mean(1, keepdim=True)
            y = (x - mu) / (var + self.eps).sqrt()
            y = self.weight.reshape([1, C, 1, 1]) * y + self.bias.reshape(
                [1, C, 1, 1])

        return y


class AvgPool2D(nn.Layer):

    def __init__(self,
                 kernel_size=None,
                 base_size=None,
                 auto_pad=True,
                 fast_imp=False,
                 train_size=None):
        super().__init__()
        self.kernel_size = kernel_size
        self.base_size = base_size
        self.auto_pad = auto_pad

        # only used for fast implementation
        self.fast_imp = fast_imp
        self.rs = [5, 4, 3, 2, 1]
        self.max_r1 = self.rs[0]
        self.max_r2 = self.rs[0]
        self.train_size = train_size

    def extra_repr(self) -> str:
        return 'kernel_size={}, base_size={}, stride={}, fast_imp={}'.format(
            self.kernel_size, self.base_size, self.kernel_size, self.fast_imp)

    def forward(self, x):
        if self.kernel_size is None and self.base_size:
            train_size = self.train_size
            if isinstance(self.base_size, int):
                self.base_size = (self.base_size, self.base_size)
            self.kernel_size = list(self.base_size)
            self.kernel_size[
                0] = x.shape[2] * self.base_size[0] // train_size[-2]
            self.kernel_size[
                1] = x.shape[3] * self.base_size[1] // train_size[-1]

            # only used for fast implementation
            self.max_r1 = max(1, self.rs[0] * x.shape[2] // train_size[-2])
            self.max_r2 = max(1, self.rs[0] * x.shape[3] // train_size[-1])

        if self.kernel_size[0] >= x.shape[-2] and self.kernel_size[
            1] >= x.shape[-1]:
            return F.adaptive_avg_pool2d(x, 1)

        if self.fast_imp:  # Non-equivalent implementation but faster
            h, w = x.shape[2:]
            if self.kernel_size[0] >= h and self.kernel_size[1] >= w:
                out = F.adaptive_avg_pool2d(x, 1)
            else:
                r1 = [r for r in self.rs if h % r == 0][0]
                r2 = [r for r in self.rs if w % r == 0][0]
                # reduction_constraint
                r1 = min(self.max_r1, r1)
                r2 = min(self.max_r2, r2)
                s = x[:, :, ::r1, ::r2].cumsum(axis=-1).cumsum(axis=-2)
                n, c, h, w = s.shape
                k1, k2 = min(h - 1, self.kernel_size[0] // r1), min(
                    w - 1, self.kernel_size[1] // r2)
                out = (s[:, :, :-k1, :-k2] - s[:, :, :-k1, k2:] -
                       s[:, :, k1:, :-k2] + s[:, :, k1:, k2:]) / (k1 * k2)
                out = paddle.nn.functional.interpolate(out,
                                                       scale_factor=(r1, r2))
        else:
            n, c, h, w = x.shape
            s = x.cumsum(axis=-1).cumsum(axis=-2)
            s = paddle.nn.functional.pad(s,
                                         [1, 0, 1, 0])  # pad 0 for convenience
            k1, k2 = min(h, self.kernel_size[0]), min(w, self.kernel_size[1])
            s1, s2, s3, s4 = s[:, :, :-k1, :-k2], s[:, :, :-k1,
                                                  k2:], s[:, :,
                                                        k1:, :-k2], s[:, :,
                                                                    k1:,
                                                                    k2:]
            out = s4 + s1 - s2 - s3
            out = out / (k1 * k2)

        if self.auto_pad:
            n, c, h, w = x.shape
            _h, _w = out.shape[2:]
            pad2d = [(w - _w) // 2, (w - _w + 1) // 2, (h - _h) // 2,
                     (h - _h + 1) // 2]
            out = paddle.nn.functional.pad(out, pad2d, mode='replicate')

        return out


def replace_layers(model, base_size, train_size, fast_imp, **kwargs):
    for n, m in model.named_children():
        if len(list(m.children())) > 0:
            ## compound module, go inside it
            replace_layers(m, base_size, train_size, fast_imp, **kwargs)

        if isinstance(m, nn.AdaptiveAvgPool2D):
            pool = AvgPool2D(base_size=base_size,
                             fast_imp=fast_imp,
                             train_size=train_size)
            assert m._output_size == 1
            setattr(model, n, pool)


'''
ref.
@article{chu2021tlsc,
  title={Revisiting Global Statistics Aggregation for Improving Image Restoration},
  author={Chu, Xiaojie and Chen, Liangyu and and Chen, Chengpeng and Lu, Xin},
  journal={arXiv preprint arXiv:2112.04491},
  year={2021}
}
'''


class Local_Base():

    def convert(self, *args, train_size, **kwargs):
        replace_layers(self, *args, train_size=train_size, **kwargs)
        imgs = paddle.rand(train_size)
        with paddle.no_grad():
            self.forward(imgs)


class SimpleGate(nn.Layer):

    def forward(self, x):
        x1, x2 = x.chunk(2, axis=1)
        return x1 * x2


class NAFBlock(nn.Layer):

    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2D(in_channels=c,
                               out_channels=dw_channel,
                               kernel_size=1,
                               padding=0,
                               stride=1,
                               groups=1,
                               bias_attr=True)
        self.conv2 = nn.Conv2D(in_channels=dw_channel,
                               out_channels=dw_channel,
                               kernel_size=3,
                               padding=1,
                               stride=1,
                               groups=dw_channel,
                               bias_attr=True)
        self.conv3 = nn.Conv2D(in_channels=dw_channel // 2,
                               out_channels=c,
                               kernel_size=1,
                               padding=0,
                               stride=1,
                               groups=1,
                               bias_attr=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2D(1),
            nn.Conv2D(in_channels=dw_channel // 2,
                      out_channels=dw_channel // 2,
                      kernel_size=1,
                      padding=0,
                      stride=1,
                      groups=1,
                      bias_attr=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2D(in_channels=c,
                               out_channels=ffn_channel,
                               kernel_size=1,
                               padding=0,
                               stride=1,
                               groups=1,
                               bias_attr=True)
        self.conv5 = nn.Conv2D(in_channels=ffn_channel // 2,
                               out_channels=c,
                               kernel_size=1,
                               padding=0,
                               stride=1,
                               groups=1,
                               bias_attr=True)

        self.norm1 = LayerNorm2D(c)
        self.norm2 = LayerNorm2D(c)

        self.drop_out_rate = drop_out_rate

        self.dropout1 = nn.Dropout(
            drop_out_rate) if drop_out_rate > 0. else None
        self.dropout2 = nn.Dropout(
            drop_out_rate) if drop_out_rate > 0. else None

        self.add_parameter(
            "beta",
            self.create_parameter(
                [1, c, 1, 1],
                default_initializer=paddle.nn.initializer.Constant(value=0.0)))
        self.add_parameter(
            "gamma",
            self.create_parameter(
                [1, c, 1, 1],
                default_initializer=paddle.nn.initializer.Constant(value=0.0)))

    def forward(self, inp):
        x = inp

        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        if self.drop_out_rate > 0:
            x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        if self.drop_out_rate > 0:
            x = self.dropout2(x)

        return y + x * self.gamma


from paddle.nn.initializer import TruncatedNormal, Constant
trunc_normal_ = TruncatedNormal(std=.02)
zeros_ = Constant(value=0.)
ones_ = Constant(value=1.)


def drop_path(x, drop_prob=0., training=False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ...
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = paddle.to_tensor(1 - drop_prob, dtype=x.dtype)
    shape = (paddle.shape(x)[0], ) + (1, ) * (x.ndim - 1)
    random_tensor = keep_prob + paddle.rand(shape, dtype=x.dtype)
    random_tensor = paddle.floor(random_tensor)  # binarize
    output = x.divide(keep_prob) * random_tensor
    return output


class DropPath(nn.Layer):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class DWConv(nn.Layer):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2D(dim, dim, 3, 1, 1, bias_attr=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x

class Mlp(nn.Layer):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2D(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2D(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LKA(nn.Layer):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2D(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2D(
            dim, dim, 5, stride=1, padding=6, groups=dim, dilation=3)
        self.conv1 = nn.Conv2D(dim, dim, 1)

    def forward(self, x):
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
        return x * attn


class Attention(nn.Layer):
    def __init__(self, d_model):
        super().__init__()
        self.proj_1 = nn.Conv2D(d_model, int(d_model * 4/3), 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = LKA(int(d_model * 4/3))
        self.proj_2 = nn.Conv2D(int(d_model * 4/3), d_model, 1)
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2D(1),
            nn.Conv2D(in_channels=d_model,
                      out_channels=d_model,
                      kernel_size=1,
                      padding=0,
                      stride=1,
                      groups=1,
                      bias_attr=True),
        )

    def forward(self, x):
        shorcut = x
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = self.sca(x) * x
        x = x + shorcut
        return x


class Block(nn.Layer):
    def __init__(self,
                 dim,
                 mlp_ratio=2.,
                 drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 use_mlp=False):
        super().__init__()
        self.norm1 = LayerNorm2D(dim)
        self.attn = Attention(dim)
        self.norm2 = LayerNorm2D(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop)
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = self.create_parameter(
            shape=[dim, 1, 1],
            default_initializer=Constant(value=layer_scale_init_value))
        self.layer_scale_2 = self.create_parameter(
            shape=[dim, 1, 1],
            default_initializer=Constant(value=layer_scale_init_value))

    def forward(self, x):
        x = x + self.layer_scale_1 * self.attn(self.norm1(x))
        x = x + self.layer_scale_2 * self.mlp(self.norm2(x))
        return x

class NAFNet(nn.Layer):
    def __init__(self,
                 img_channel=3,
                 width=16,
                 middle_blk_num=1,
                 enc_blk_nums=[],
                 dec_blk_nums=[]):
        super().__init__()
        self.intro = nn.Conv2D(in_channels=img_channel,
                               out_channels=width,
                               kernel_size=3,
                               padding=1,
                               stride=1,
                               groups=1,
                               bias_attr=True)
        self.ending = nn.Conv2D(in_channels=width,
                                out_channels=img_channel,
                                kernel_size=3,
                                padding=1,
                                stride=1,
                                groups=1,
                                bias_attr=True)

        self.encoders = nn.LayerList()
        self.decoders = nn.LayerList()
        self.middle_blks = nn.LayerList()
        self.ups = nn.LayerList()
        self.downs = nn.LayerList()

        chan = width
        for num in enc_blk_nums:
            self.downs.append(nn.Conv2D(chan, 2 * chan, 2, 2))
            chan = chan * 2
            self.encoders.append(
                nn.Sequential(*[Block(chan) for _ in range(num)]))


        # self.middle_blks = \
        #     nn.Sequential(
        #         *[Block(chan) for _ in range(middle_blk_num)]
        #     )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(nn.Conv2D(chan, chan * 2, 1, bias_attr=False),
                              nn.PixelShuffle(2)))
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(*[Block(chan) for _ in range(num)]))

        self.padder_size = 2**len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            encs.append(x)
            x = down(x)
            x = encoder(x)

        # x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + inp

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.shape
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, [0, mod_pad_w, 0, mod_pad_h])
        return x


class NAFNetLocal(Local_Base, NAFNet):
    def __init__(self,
                 *args,
                 train_size=(1, 3, 256, 256),
                 fast_imp=False,
                 **kwargs):
        Local_Base.__init__(self)
        NAFNet.__init__(self, *args, **kwargs)

        N, C, H, W = train_size
        base_size = (int(H * 1.5), int(W * 1.5))

        self.eval()
        with paddle.no_grad():
            self.convert(base_size=base_size,
                         train_size=train_size,
                         fast_imp=fast_imp)
