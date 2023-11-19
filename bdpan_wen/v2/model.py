import paddle.nn as nn
import numpy as np
import paddle.nn.functional as F
import paddle


class ConvWithActivation(nn.Layer):
    '''
    SN convolution for spetral normalization conv
    '''

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 activation=nn.GELU()):
        super(ConvWithActivation, self).__init__()
        self.conv2d = nn.Conv2D(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                dilation=dilation, groups=groups, bias_attr=bias)
        self.conv2d = nn.utils.spectral_norm(self.conv2d)
        self.activation = activation

    def forward(self, input):
        x = self.conv2d(input)
        if self.activation is not None:
            return self.activation(x)
        else:
            return x


class Residual(nn.Layer):

    def __init__(self, in_channels, out_channels, same_shape=True, same_channel=True, activation=nn.GELU(), ):
        super(Residual, self).__init__()
        self.activation = activation
        self.same_shape = same_shape
        self.same_channel = same_channel
        strides = 1 if same_shape else 2
        self.conv1 = nn.Conv2D(in_channels, in_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2D(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv1 = nn.utils.spectral_norm(self.conv1)
        self.conv2 = nn.utils.spectral_norm(self.conv2)
        if not same_shape or not same_channel:
            self.conv3 = nn.Conv2D(in_channels, out_channels, kernel_size=1, padding=0, stride=strides)
            self.conv3 = nn.utils.spectral_norm(self.conv3)
        self.batch_norm2d = nn.BatchNorm2D(out_channels)

    def forward(self, x):
        out = self.activation(self.conv1(x))
        out = self.conv2(out)
        if not self.same_shape or not self.same_channel:
            x = self.conv3(x)
        out = self.batch_norm2d(out + x)
        return self.activation(out)


class Lateral(nn.Layer):

    def __init__(self, in_out_channel, hidden_channel):
        super(Lateral, self).__init__()
        self.conv1 = nn.Conv2D(in_out_channel, hidden_channel, kernel_size=1, padding=0, stride=1, )
        self.conv2 = nn.Conv2D(hidden_channel, hidden_channel, kernel_size=3, padding=1, stride=1, )
        self.conv3 = nn.Conv2D(hidden_channel, in_out_channel, kernel_size=1, padding=0, stride=1, )
        self.conv1 = nn.utils.spectral_norm(self.conv1)
        self.conv2 = nn.utils.spectral_norm(self.conv2)
        self.conv3 = nn.utils.spectral_norm(self.conv3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class DeConvWithActivation(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 output_padding=1, bias=True, activation=nn.GELU()):
        super(DeConvWithActivation, self).__init__()
        self.conv2d = nn.Conv2DTranspose(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                         padding=padding, dilation=dilation, groups=groups,
                                         output_padding=output_padding, bias_attr=bias)
        self.conv2d = nn.utils.spectral_norm(self.conv2d)
        self.activation = activation

    def forward(self, x):
        x = self.conv2d(x)
        if self.activation is not None:
            return self.activation(x)
        else:
            return x


# class ShallowUNET(nn.Layer):
#     def __init__(self, n_in_channel=3, n_out_channel=3, unet_num_c=[32, 64, 128], ):
#         super(ShallowUNET, self).__init__()
#         # downsample
#         self.down_conv1 = ConvWithActivation(n_in_channel, unet_num_c[0], kernel_size=7, stride=2, padding=3)
#         self.down_conv2 = ConvWithActivation(unet_num_c[0], unet_num_c[1], kernel_size=5, stride=2, padding=2)
#         # self.down_conv1 = ConvWithActivation(n_in_channel, unet_num_c[0], kernel_size=3, stride=2, padding=1)
#         # self.down_conv2 = ConvWithActivation(unet_num_c[0], unet_num_c[1], kernel_size=3, stride=2, padding=1)
#         self.down_conv3 = ConvWithActivation(unet_num_c[1], unet_num_c[2], kernel_size=3, stride=2, padding=1)
#         # res
#         self.res1_1 = Residual(unet_num_c[0], unet_num_c[0])
#         self.res2_1 = Residual(unet_num_c[1], unet_num_c[1])
#         self.res2_2 = Residual(unet_num_c[1], unet_num_c[1])
#         self.res2_3 = Residual(unet_num_c[1], unet_num_c[1])
#         self.res3_1 = Residual(unet_num_c[2], unet_num_c[2])
#         self.res3_2 = Residual(unet_num_c[2], unet_num_c[2])
#         # fine bottom
#         self.fine_bottom = nn.Sequential(
#             ConvWithActivation(unet_num_c[2], unet_num_c[2], kernel_size=1, ),
#             Residual(unet_num_c[2], unet_num_c[2]),
#         )
#         # lateral
#         self.lateral1 = Lateral(unet_num_c[0], 2 * unet_num_c[0], )
#         self.lateral2 = Lateral(unet_num_c[1], 2 * unet_num_c[1], )
#         self.lateral3 = Lateral(unet_num_c[2], 2 * unet_num_c[2], )
#         # upsample
#         self.up_conv1 = DeConvWithActivation(unet_num_c[0] * 2, n_out_channel, kernel_size=7, stride=2, padding=3, )
#         self.up_conv2 = DeConvWithActivation(unet_num_c[1] * 2, unet_num_c[0], kernel_size=5, stride=2, padding=2, )
#         # self.up_conv1 = DeConvWithActivation(unet_num_c[0] * 2, n_out_channel, kernel_size=3, stride=2, padding=1, )
#         # self.up_conv2 = DeConvWithActivation(unet_num_c[1] * 2, unet_num_c[0], kernel_size=3, stride=2, padding=1, )
#         self.up_conv3 = DeConvWithActivation(unet_num_c[2] * 2, unet_num_c[1], kernel_size=3, stride=2, padding=1, )
#
#     def forward(self, x):
#         x = self.down_conv1(x)
#         x = self.res1_1(x)
#         d_x1 = x
#         x = self.down_conv2(x)
#         x = self.res2_1(x)
#         x = self.res2_2(x)
#         x = self.res2_3(x)
#         d_x2 = x
#         x = self.down_conv3(x)
#         x = self.res3_1(x)
#         x = self.res3_2(x)
#         d_x3 = x
#         x = self.fine_bottom(x)
#
#         x = paddle.concat([self.lateral3(d_x3), x], axis=1)
#         x = self.up_conv3(x)
#         x = paddle.concat([self.lateral2(d_x2), x], axis=1)
#         x = self.up_conv2(x)
#         x = paddle.concat([self.lateral1(d_x1), x], axis=1)
#         x = self.up_conv1(x)
#
#         return x

class ShallowUNET(nn.Layer):
    def __init__(self, n_in_channel=3, n_out_channel=3, unet_num_c=[16, 32, 64], ):
        super(ShallowUNET, self).__init__()
        # downsample
        self.down_conv1 = ConvWithActivation(n_in_channel, unet_num_c[0], kernel_size=7, stride=2, padding=3)
        self.down_conv2 = ConvWithActivation(unet_num_c[0], unet_num_c[1], kernel_size=5, stride=2, padding=2)
        self.down_conv3 = ConvWithActivation(unet_num_c[1], unet_num_c[2], kernel_size=3, stride=2, padding=1)
        # res
        self.res1_1 = Residual(unet_num_c[0], unet_num_c[0])
        self.res2_1 = Residual(unet_num_c[1], unet_num_c[1])
        self.res2_2 = Residual(unet_num_c[1], unet_num_c[1])
        self.res2_3 = Residual(unet_num_c[1], unet_num_c[1])
        self.res3_1 = Residual(unet_num_c[2], unet_num_c[2])
        self.res3_2 = Residual(unet_num_c[2], unet_num_c[2])
        # fine bottom
        self.fine_bottom = nn.Sequential(
            ConvWithActivation(unet_num_c[2], unet_num_c[2], kernel_size=1, ),
        )
        # lateral
        self.lateral1 = Lateral(unet_num_c[0], 2 * unet_num_c[0], )
        self.lateral2 = Lateral(unet_num_c[1], 2 * unet_num_c[1], )
        self.lateral3 = Lateral(unet_num_c[2], 2 * unet_num_c[2], )
        # upsample
        self.up_conv1 = DeConvWithActivation(unet_num_c[0] * 1, n_out_channel, kernel_size=7, stride=2, padding=3, activation=None, )
        self.up_res1 = Residual(unet_num_c[0] * 2, unet_num_c[0], same_channel=False)
        self.up_conv2 = DeConvWithActivation(unet_num_c[1] * 1, unet_num_c[0], kernel_size=5, stride=2, padding=2, )
        self.up_res2 = Residual(unet_num_c[1] * 2, unet_num_c[1], same_channel=False)
        self.up_conv3 = DeConvWithActivation(unet_num_c[2] * 1, unet_num_c[1], kernel_size=3, stride=2, padding=1, )
        self.up_res3 = Residual(unet_num_c[2] * 2, unet_num_c[2], same_channel=False)

    def forward(self, x):
        x = self.down_conv1(x)
        x = self.res1_1(x)
        d_x1 = x
        x = self.down_conv2(x)
        x = self.res2_1(x)
        x = self.res2_2(x)
        x = self.res2_3(x)
        d_x2 = x
        x = self.down_conv3(x)
        x = self.res3_1(x)
        x = self.res3_2(x)
        d_x3 = x
        x = self.fine_bottom(x)

        x = self.up_res3(paddle.concat([self.lateral3(d_x3), x], axis=1))
        x = self.up_conv3(x)
        x = self.up_res2(paddle.concat([self.lateral2(d_x2), x], axis=1))
        x = self.up_conv2(x)
        x = self.up_res1(paddle.concat([self.lateral1(d_x1), x], axis=1))
        x = self.up_conv1(x)

        return x







