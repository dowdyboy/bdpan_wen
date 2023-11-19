import paddle
from paddle import nn as nn
import paddle.nn.functional as F
from paddle.autograd import PyLayer
from paddle.nn.initializer import Constant


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
    def __init__(self, d_model, expand=4/3):
        super().__init__()
        self.proj_1 = nn.Conv2D(d_model, int(d_model * expand), 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = LKA(int(d_model * expand))
        self.proj_2 = nn.Conv2D(int(d_model * expand), d_model, 1)
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
                 drop=0.2,
                 act_layer=nn.GELU,):
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


class PlusNAFNetV2(nn.Layer):
    def __init__(self,
                 img_channel=3,
                 width=16,
                 middle_blk_num=0,
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
            if num > 0:
                self.encoders.append(
                    nn.Sequential(*[Block(chan) for _ in range(num)]))
            else:
                self.encoders.append(nn.Sequential(
                    LayerNorm2D(chan),
                    nn.GELU(),
                ))


        self.middle_blks = \
            nn.Sequential(
                *[Block(chan) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(nn.Conv2D(chan, chan * 2, 1, bias_attr=False),
                              nn.PixelShuffle(2)))
            chan = chan // 2
            if num > 0:
                self.decoders.append(
                    nn.Sequential(*[Block(chan) for _ in range(num)]))
            else:
                self.decoders.append(nn.Sequential(
                    LayerNorm2D(chan),
                    nn.GELU(),
                ))

        self.padder_size = 2**len(self.encoders)

    def forward(self, inp):
        if self.training:
            B, C, H, W = inp.shape
            inp = self.check_image_size(inp)

        x = self.intro(inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            encs.append(x)
            x = down(x)
            x = encoder(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + inp

        if self.training:
            return x[:, :, :H, :W]
        else:
            return x

    def check_image_size(self, x):
        _, _, h, w = x.shape
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, [0, mod_pad_w, 0, mod_pad_h])
        return x
