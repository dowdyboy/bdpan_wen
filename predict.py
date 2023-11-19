import os
import sys
import cv2
import time
import glob
import paddle
import paddle.vision.transforms as T
import numpy as np


is_test_aug = False
model_path = 'model/model'
model = paddle.jit.load(model_path)
model.eval()


def to_tensor(img):
    img = paddle.to_tensor(img, dtype=paddle.float32)
    img = paddle.transpose(img, [2, 0, 1])
    img = paddle.reshape(img, [1, ] + img.shape)
    img = img / 255.
    return img


def to_img_arr(x):
    x = paddle.transpose(paddle.clip(x * 255., 0., 255., ), [1, 2, 0])
    x = paddle.round(x)
    x = paddle.cast(x, paddle.uint8)
    y = x.numpy()
    return y


def process(src_image_dir, save_dir, infer_time):
    image_paths = glob.glob(os.path.join(src_image_dir, "*.png"))
    infer_time = float(infer_time)
    with paddle.no_grad():
        for image_path in image_paths:
            filename = os.path.basename(image_path)
            img = cv2.imread(image_path)
            before_time = time.time()

            # pre
            img = to_tensor(img)
            # infer
            # before_time = time.time()
            if is_test_aug:
                result = (model(img) + paddle.flip(model(paddle.flip(img, axis=3)), axis=3)) / 2.
            else:
                result = model(img)
            # infer_time += (time.time() - before_time)
            # post
            result = to_img_arr(result[0], )

            infer_time += (time.time() - before_time)
            cv2.imwrite(os.path.join(save_dir, filename), result)
    return infer_time


if __name__ == "__main__":
    assert len(sys.argv) == 4

    src_image_dir = sys.argv[1]
    save_dir = sys.argv[2]
    infer_time = sys.argv[3]

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    all_infer_time = process(src_image_dir, save_dir, infer_time)
    print('all_infer_time:', all_infer_time)
