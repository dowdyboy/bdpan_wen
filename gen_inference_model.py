import paddle
import cv2
import os
from paddle.io import Dataset, DataLoader
import paddle.vision.transforms as T
# v1
# from bdpan_wen.v1.model import STRAIDR
# v3
# from bdpan_wen.v3.model import NAFNet
# v4
# from bdpan_wen.v4.model import IdentityEraseNet
# v7
# from bdpan_wen.v7.model import FaceUNET
# v8
from bdpan_wen.v8.model import FaceUNETV2
import sys
import numpy as np


chk_path = 'checkpoints/v8/chk_best_step_1484000/model_0.pdparams'


def build_model():
    # v1
    # model = STRAIDR(unet_num_c=[8, 16, 32, 32, 64],
    #                 fine_num_c=[16],)
    # v3
    # model = NAFNet(img_channel=3, width=8, enc_blk_nums=[2, 2, 1, 0], dec_blk_nums=[1, 1, 1, 0])
    # v4
    # model = IdentityEraseNet(3, 12, enc_blk_nums=[1, 2, 2, 1], dec_blk_nums=[1, 1, 1, 1])
    # v7
    # model = FaceUNET(img_channel=3, width=8, middle_blk_num=0, enc_blk_nums=[1, 1, 1, 0], dec_blk_nums=[1, 1, 1, 0])
    # v8
    model = FaceUNETV2(img_channel=3, width=8, middle_blk_num=0, enc_blk_nums=[1, 2, 1], dec_blk_nums=[1, 1, 1])

    model.load_dict(paddle.load(chk_path))
    model.eval()
    return model


model = build_model()
paddle.jit.save(model, 'model/model', input_spec=[paddle.static.InputSpec(shape=[None, 3, 1024, 1024], dtype=paddle.float32)])
