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


class SimpleGate(nn.Layer):

    def forward(self, x):
        x1, x2 = x.chunk(2, axis=1)
        return x1 * x2


class DWConv(nn.Layer):

    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1, channel_first=False):
        super().__init__()
        self.channel_first = channel_first
        s_channel = out_channel if channel_first else in_channel
        self.space_conv = nn.Conv2D(in_channels=s_channel,
                               out_channels=s_channel,
                               kernel_size=kernel_size,
                               padding=padding,
                               stride=stride,
                               groups=s_channel,
                               bias_attr=True)
        self.channel_conv = nn.Conv2D(in_channels=in_channel,
                                      out_channels=out_channel,
                                      kernel_size=1,
                                      padding=0,
                                      stride=1,
                                      groups=1,
                                      bias_attr=True)

    def forward(self, x):
        if self.channel_first:
            x = self.channel_conv(x)
            x = self.space_conv(x)
            return x
        else:
            x = self.space_conv(x)
            x = self.channel_conv(x)
            return x


class LKA(nn.Layer):
    def __init__(self, dim, layer_scale_init_value=1.0):
        super().__init__()
        self.conv0 = nn.Conv2D(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2D(
            dim, dim, 5, stride=1, padding=6, groups=dim, dilation=3)
        self.conv1 = nn.Conv2D(dim, dim, 1)
        self.scale_param = self.create_parameter(
            shape=[dim, 1, 1],
            default_initializer=nn.initializer.Constant(value=layer_scale_init_value))

    def forward(self, x):
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
        return x * attn * self.scale_param


class DownBlock(nn.Layer):
    
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.pool = nn.MaxPool2D(2, 2)
        self.conv = nn.Conv2D(in_channel, out_channel, 1, 1, 0)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x


class LKABlock(nn.Layer):

    def __init__(self, channel, gap_scale=1.0):
        super().__init__()
        self.conv1 = DWConv(channel, channel, channel_first=False)
        self.sg = SimpleGate()
        self.lka = LKA(channel // 2, )
        self.conv2 = nn.Conv2D(in_channels=channel // 2, out_channels=channel, kernel_size=1, padding=0, stride=1, groups=1, bias_attr=True)
        self.gap = nn.AdaptiveAvgPool2D(1)
        self.gap_scale = self.create_parameter(
            shape=[1, 1, 1],
            default_initializer=nn.initializer.Constant(value=gap_scale))

    def forward(self, x):
        ori_x = x
        x = self.conv1(x)
        x = self.sg(x)
        x = self.lka(x)
        x = self.conv2(x)
        x = x * self.gap(x) * self.gap_scale
        return x + ori_x


class FFNBlock(nn.Layer):
    
    def __init__(self, channel):
        super().__init__()
        self.norm1 = LayerNorm2D(channel)
        self.conv1 = DWConv(channel, 2 * channel, channel_first=False)
        self.sg = SimpleGate()
        self.conv2 = nn.Conv2D(in_channels=channel, out_channels=channel, kernel_size=1, padding=0, stride=1, groups=1, bias_attr=True)

    def forward(self, x):
        ori_x = x
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.sg(x)
        x = self.conv2(x + ori_x)
        return x


class EncoderBlock(nn.Layer):

    def __init__(self, channel, scale_type=None, layer_scale_init_value=1e-2, ):
        super().__init__()
        self.scale_type = scale_type
        self.att = LKABlock(channel, )
        self.ffn = FFNBlock(channel, )
        if scale_type == 'channel':
            self.scale_param = self.create_parameter(
                shape=[channel, 1, 1],
                default_initializer=nn.initializer.Constant(value=layer_scale_init_value))
        elif scale_type == 'global':
            self.scale_param = self.create_parameter(
                shape=[1, 1, 1],
                default_initializer=nn.initializer.Constant(value=layer_scale_init_value))
        else:
            self.scale_param = 1.0

    def forward(self, x):
        ori_x = x
        x = self.att(x) + x
        x = self.ffn(x) + x
        x = x + ori_x * self.scale_param
        return x


class UpBlock(nn.Layer):
    
    def __init__(self, channel):
        super().__init__()
        self.conv = nn.Conv2D(channel, channel * 2, 1, bias_attr=False)
        self.up = nn.PixelShuffle(2)

    def forward(self, x):
        x = self.conv(x)
        x = self.up(x)
        return x


class DecoderBlock(nn.Layer):

    def __init__(self, channel, ):
        super().__init__()
        self.conv = DWConv(channel, 2 * channel, channel_first=False)
        self.sg = SimpleGate()

    def forward(self, x):
        return self.sg(self.conv(x))


class IdentityEraseNet(nn.Layer):

    def __init__(self, img_channel, base_channel, enc_blk_nums=[], dec_blk_nums=[], ):
        super().__init__()
        self.entry_conv = nn.Conv2D(in_channels=img_channel, out_channels=base_channel,
                                    kernel_size=3, padding=1, stride=1, groups=1, bias_attr=True)
        self.out_conv = nn.Conv2D(in_channels=base_channel, out_channels=img_channel,
                                  kernel_size=3, padding=1, stride=1, groups=1, bias_attr=True)
        self.encoders = nn.LayerList()
        self.decoders = nn.LayerList()
        self.ups = nn.LayerList()
        self.downs = nn.LayerList()

        chan = base_channel
        for num in enc_blk_nums:
            self.downs.append(DownBlock(chan, chan * 2))
            chan = chan * 2
            self.encoders.append(
                nn.Sequential(*[EncoderBlock(chan, scale_type='channel') for _ in range(num)]))

        for num in dec_blk_nums:
            self.ups.append(UpBlock(chan))
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(*[DecoderBlock(chan) for _ in range(num)]))

        self.padder_size = 2**len(self.encoders)

    def forward(self, x):
        if self.training:
            B, C, H, W = x.shape
            x = self.check_image_size(x)
        ori_x = x

        x = self.entry_conv(x)

        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            encs.append(x)
            x = down(x)
            x = encoder(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.out_conv(x)
        x = x + ori_x

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

