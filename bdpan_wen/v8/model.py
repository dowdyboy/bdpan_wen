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


class SimpleGate(nn.Layer):

    def forward(self, x):
        x1, x2 = x.chunk(2, axis=1)
        return x1 * x2


class PixelNorm(nn.Layer):

    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        return inputs * paddle.rsqrt(
            paddle.mean(inputs * inputs, 1, keepdim=True) + 1e-8)


class DWConv(nn.Layer):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 padding_mode='zeros',
                 weight_attr=None,
                 bias_attr=None,
                 data_format="NCHW",
                 channel_first=False):
        super().__init__()
        self.channel_first = channel_first
        s_channel = out_channels if channel_first else in_channels
        self.space_conv = nn.Conv2D(in_channels=s_channel,
                                    out_channels=s_channel,
                                    kernel_size=kernel_size,
                                    padding=padding,
                                    stride=stride,
                                    dilation=dilation,
                                    groups=s_channel,
                                    padding_mode=padding_mode,
                                    weight_attr=weight_attr,
                                    bias_attr=bias_attr,
                                    data_format=data_format, )
        self.channel_conv = nn.Conv2D(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=1,
                                      padding=0,
                                      stride=1,
                                      dilation=1,
                                      groups=1,
                                      padding_mode=padding_mode,
                                      weight_attr=weight_attr,
                                      bias_attr=bias_attr,
                                      data_format=data_format,
                                      )

    def forward(self, x):
        if self.channel_first:
            x = self.channel_conv(x)
            x = self.space_conv(x)
            return x
        else:
            x = self.space_conv(x)
            x = self.channel_conv(x)
            return x


class Conv2D(nn.Layer):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 padding_mode='zeros',
                 weight_attr=None,
                 bias_attr=None,
                 data_format="NCHW"):
        super().__init__()
        self.conv = nn.Conv2D(in_channels,
                              out_channels,
                              kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              groups=groups,
                              padding_mode=padding_mode,
                              weight_attr=weight_attr,
                              bias_attr=bias_attr,
                              data_format=data_format)

    def forward(self, x):
        return self.conv(x)


class PA(nn.Layer):
    '''PA is pixel attention'''
    def __init__(self, nf):

        super(PA, self).__init__()
        self.conv = nn.Conv2D(nf, nf, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        y = self.conv(x)
        y = self.sigmoid(y)
        out = x * y

        return out


class ChannelAttention(nn.Layer):

    def __init__(self, channel, scale=2, alpha=1., ):
        super(ChannelAttention, self).__init__()
        self.alpha = alpha
        self.gap = nn.AdaptiveAvgPool2D(1)
        self.conv1 = Conv2D(channel, channel // scale, 1, 1, 0, )
        self.conv2 = Conv2D(channel // scale, channel, 1, 1, 0, )
        self.lrelu = nn.GELU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.gap(x)
        y = self.conv1(y)
        y = self.lrelu(y)
        y = self.conv2(y)
        y = self.sigmoid(y)
        x = self.alpha * x
        out = x * y
        return out


class DownBlock(nn.Layer):

    def __init__(self, in_channel, out_channel):
        super().__init__()
        # self.conv = Conv2D(in_channel, out_channel, 2, 2)
        self.conv = Conv2D(in_channel, out_channel, 3, 2, 1)
        # self.conv = DWConv(in_channel, out_channel, 3, 2, 1)
        # self.conv = Conv2D(in_channel, out_channel, 5, 2, 2)

    def forward(self, x):
        x = self.conv(x)
        return x


class UpBlock(nn.Layer):

    def __init__(self, channel):
        super().__init__()
        # self.conv = Conv2D(channel, channel * 2, 1, bias_attr=False)
        self.conv = Conv2D(channel, channel * 2, 3, 1, 1, bias_attr=False)
        self.up = nn.PixelShuffle(2)

    def forward(self, x):
        x = self.conv(x)
        x = self.up(x)
        return x
        # return F.interpolate(x, scale_factor=2, mode='nearest')


# class Block(nn.Layer):
#     def __init__(self,
#                  dim, ):
#         super().__init__()
#         self.norm1 = LayerNorm2D(dim)
#         self.act = nn.GELU()
#
#     def forward(self, x):
#         x = self.norm1(x)
#         x = self.act(x)
#         return x

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
        # self.fc1 = nn.Conv2D(in_features, hidden_features, 1)
        # self.dwconv = DWConv(in_features, hidden_features, 3, 1, 1, channel_first=True)
        self.dwconv = DWConv(in_features, out_features, 3, 1, 1, channel_first=True)
        # self.dwconv = Conv2D(in_features, out_features, 3, 1, 1, )
        self.act = act_layer()
        self.fc2 = nn.Conv2D(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

        self.pa = PA(out_features)

    def forward(self, x):
        # x = self.fc1(x)
        # x = self.dwconv(x)
        # x = self.act(x)
        # x = self.drop(x)
        # x = self.fc2(x)
        # x = self.drop(x)

        x = self.dwconv(x)
        x = self.act(x) + self.pa(x)
        return x


class LKA(nn.Layer):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2D(dim, dim, 7, padding=3, groups=dim)
        self.conv_spatial = nn.Conv2D(
            dim, dim, 7, stride=1, padding=15, groups=dim, dilation=5)
        # self.conv0 = nn.Conv2D(dim, dim, 5, padding=2, groups=dim)
        # self.conv_spatial = nn.Conv2D(
        #     dim, dim, 5, stride=1, padding=6, groups=dim, dilation=3)
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

        # layer_scale_init_value = 0.01
        # self.pa_scale = self.create_parameter(
        #     shape=[d_model, 1, 1],
        #     default_initializer=Constant(value=layer_scale_init_value))
        # self.pa = PA(d_model)

    def forward(self, x):
        shorcut = x
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = self.sca(x) * x
        x = x + shorcut
        # x = x + self.pa(shorcut) * self.pa_scale
        return x

# class Block(nn.Layer):
#     def __init__(self,
#                  dim,
#                  mlp_ratio=2.,
#                  drop=0.,
#                  drop_path=0.,
#                  act_layer=nn.GELU,
#                  use_mlp=False):
#         super().__init__()
#         layer_scale_init_value = 1e-2
#         self.norm1 = LayerNorm2D(dim)
#         self.attn = Attention(dim)
#         self.layer_scale_1 = self.create_parameter(
#             shape=[dim, 1, 1],
#             default_initializer=Constant(value=layer_scale_init_value))
#
#         self.norm2 = LayerNorm2D(dim)
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.mlp = Mlp(in_features=dim,
#                        hidden_features=mlp_hidden_dim,
#                        act_layer=act_layer,
#                        drop=drop)
#         self.layer_scale_2 = self.create_parameter(
#             shape=[dim, 1, 1],
#             default_initializer=Constant(value=layer_scale_init_value))
#
#     def forward(self, x):
#         x = x + self.layer_scale_1 * self.attn(self.norm1(x))
#         x = x + self.layer_scale_2 * self.mlp(self.norm2(x))
#         return x


class EncBlock(nn.Layer):
    def __init__(self,
                 dim,
                 mlp_ratio=2.,
                 drop=0.,
                 act_layer=nn.GELU,):
        super().__init__()
        layer_scale_init_value = 0.01
        self.norm1 = LayerNorm2D(dim)
        self.attn = Attention(dim)
        self.layer_scale_1 = self.create_parameter(
            shape=[dim, 1, 1],
            default_initializer=Constant(value=layer_scale_init_value))

        self.norm2 = LayerNorm2D(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop)
        self.layer_scale_2 = self.create_parameter(
            shape=[dim, 1, 1],
            default_initializer=Constant(value=layer_scale_init_value))

    def forward(self, x):
        x = x + self.layer_scale_1 * self.attn(self.norm1(x))
        x = x + self.layer_scale_2 * self.mlp(self.norm2(x))
        # x = x + self.attn(self.norm1(x))
        # x = x + self.mlp(self.norm2(x))
        return x


class DecBlock(nn.Layer):
    def __init__(self,
                 dim,
                 mlp_ratio=2.,
                 drop=0.,
                 act_layer=nn.GELU,):
        super().__init__()
        layer_scale_init_value = 0.01
        self.norm1 = LayerNorm2D(dim)
        self.attn = Attention(dim)
        self.layer_scale_1 = self.create_parameter(
            shape=[dim, 1, 1],
            default_initializer=Constant(value=layer_scale_init_value))

        self.norm2 = LayerNorm2D(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop)
        self.layer_scale_2 = self.create_parameter(
            shape=[dim, 1, 1],
            default_initializer=Constant(value=layer_scale_init_value))

    def forward(self, x):
        x = x + self.layer_scale_1 * self.attn(self.norm1(x))
        x = x + self.layer_scale_2 * self.mlp(self.norm2(x))
        # x = x + self.attn(self.norm1(x))
        # x = x + self.mlp(self.norm2(x))
        return x


class MiddleBlock(nn.Layer):

    def __init__(self, dim):
        super().__init__()
        layer_scale_init_value = 0.01
        self.pn = LayerNorm2D(dim)
        self.ca = LKA(dim)
        self.layer_scale = self.create_parameter(
            shape=[dim, 1, 1],
            default_initializer=Constant(value=layer_scale_init_value))

    def forward(self, x):
        x = x + self.ca(self.pn(x)) * self.layer_scale
        return x


class FaceUNETV2(nn.Layer):
    def __init__(self,
                 img_channel=3,
                 width=12,
                 middle_blk_num=0,
                 enc_blk_nums=[1, 1, 1, 1],
                 dec_blk_nums=[1, 1, 1, 1],
                 ):
        super().__init__()
        self.intro = Conv2D(in_channels=img_channel,
                           out_channels=width,
                           kernel_size=3,
                           padding=1,
                           stride=1,
                           groups=1,
                           bias_attr=True)
        self.ending = Conv2D(in_channels=width,
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
            self.downs.append(DownBlock(chan, chan * 2))
            chan = chan * 2
            if num > 0:
                self.encoders.append(
                    nn.Sequential(*[EncBlock(chan) for _ in range(num)]))
            else:
                self.encoders.append(nn.Identity())

        # self.middle_blks = \
        #     nn.Sequential(
        #         *[Block(chan) for _ in range(middle_blk_num)]
        #     )
        for _ in range(middle_blk_num):
            self.middle_blks.append(MiddleBlock(chan))

        for num in dec_blk_nums:
            if num > 0:
                self.decoders.append(
                    nn.Sequential(*[DecBlock(chan) for _ in range(num)]))
            else:
                self.decoders.append(nn.Identity())
            self.ups.append(UpBlock(chan))
            chan = chan // 2

        self.padder_size = 2**len(self.encoders)

    def forward(self, inp):
        if self.training:
            B, C, H, W = inp.shape
            inp = self.check_image_size(inp)

        x = self.intro(inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = down(x)
            x = encoder(x)
            encs.append(x)

        # x = self.middle_blks(x)
        for middle_blk in self.middle_blks:
            x = middle_blk(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = x + enc_skip
            x = decoder(x)
            x = up(x)

        x = self.ending(x)
        x = x + inp

        if self.training:
            return x[:, :, :H, :W]
        else:
            return x

    def forward_feat(self, inp):
        if self.training:
            B, C, H, W = inp.shape
            inp = self.check_image_size(inp)

        x = self.intro(inp)

        encs = []
        decs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = down(x)
            x = encoder(x)
            encs.append(x)

        # x = self.middle_blks(x)
        for middle_blk in self.middle_blks:
            x = middle_blk(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = x + enc_skip
            x = decoder(x)
            x = up(x)
            decs.append(x)

        x = self.ending(x)
        x = x + inp

        if self.training:
            return x[:, :, :H, :W], decs
        else:
            return x, decs

    def check_image_size(self, x):
        _, _, h, w = x.shape
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, [0, mod_pad_w, 0, mod_pad_h])
        return x


class AuxHead(nn.Layer):

    def __init__(self, img_channel, base_dim, scale):
        super().__init__()
        dim = base_dim * scale
        self.ups = nn.LayerList()
        self.ending = Conv2D(in_channels=base_dim,
                             out_channels=img_channel,
                             kernel_size=3,
                             padding=1,
                             stride=1,
                             groups=1,
                             bias_attr=True)
        while scale != 1:
            self.ups.append(UpBlock(dim))
            dim = dim // 2
            scale = scale // 2

    def forward(self, x):
        for up in self.ups:
            x = up(x)
        x = self.ending(x)
        return x


class FaceUNETV2Wrapper(nn.Layer):

    def __init__(self):
        super().__init__()
        self.model = FaceUNETV2(img_channel=3, width=8, middle_blk_num=0, enc_blk_nums=[1, 2, 1], dec_blk_nums=[1, 1, 1])
        self.aux_x2 = AuxHead(3, 8, 2)
        self.aux_x4 = AuxHead(3, 8, 4)

    def inner_forward(self, x):
        return self.model(x)

    def forward(self, x):
        out, decs = self.model.forward_feat(x)
        feat_x4, feat_x2 = decs[0], decs[1]
        out_x4, out_x2 = self.aux_x4(feat_x4), self.aux_x2(feat_x2)
        return out, out_x2, out_x4
