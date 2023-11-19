import paddle.vision.transforms
from paddle.io import Dataset
import paddle.vision.transforms as T
import paddle.vision.transforms.functional as F
import os
import cv2
import random
import numpy as np


def identity_tensor(img):
    img = paddle.to_tensor(img, dtype=paddle.float32)
    img = paddle.transpose(img, [2, 0, 1])
    return img


def normalize_tensor(img):
    img = paddle.to_tensor(img, dtype=paddle.float32)
    img = paddle.transpose(img, [2, 0, 1])
    img = img / 255.
    return img


def load_image(filepath, is_rgb=True, ):
    img = cv2.imread(filepath)
    if is_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def save_image(x, save_path):
    x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, x)


class PairedRandomHorizontalFlip(T.RandomHorizontalFlip):

    def __init__(self, prob=0.5, keys=None):
        super().__init__(prob, keys=keys)

    def _get_params(self, inputs):
        params = {}
        params['flip'] = random.random() < self.prob
        return params

    def _apply_image(self, image):
        if self.params['flip']:
            if isinstance(image, list):
                image = [F.hflip(v) for v in image]
            else:
                return F.hflip(image)
        return image


class PairedResize():

    def __init__(self, prob, resize_range=[512, 1024], alg_list=['nearest', 'bilinear', 'bicubic', 'lanczos'], ):
        super().__init__()
        self.prob = prob
        self.resize_range = resize_range
        self.alg_list = alg_list
        self.params = dict()

    def _get_params(self, inputs):
        params = {}
        params['resize'] = random.random() < self.prob
        if params['resize']:
            params['resize_size'] = random.randint(self.resize_range[0], self.resize_range[1])
            params['resize_alg'] = random.choice(self.alg_list)
        return params

    def _apply_image(self, image, image_gt):
        if self.params['resize']:
            image = F.resize(image, size=self.params['resize_size'], interpolation=self.params['resize_alg'])
            image_gt = F.resize(image_gt, size=self.params['resize_size'], interpolation=self.params['resize_alg'])
        return image, image_gt

    def __call__(self, img_pair):
        image, image_gt = img_pair
        self.params = self._get_params(img_pair)
        image, image_gt = self._apply_image(image, image_gt)
        return image, image_gt


class PairedRotate():

    def __init__(self, prob, ):
        super().__init__()
        self.prob = prob
        self.params = dict()

    def _get_params(self, inputs):
        params = {}
        params['rotate'] = random.random() < self.prob
        if params['rotate']:
            params['rotate_type'] = random.choice([-1, 1])
        return params

    def _apply_image(self, image, image_gt):
        if self.params['rotate']:
            image = np.rot90(image, k=self.params['rotate_type'])
            image_gt = np.rot90(image_gt, k=self.params['rotate_type'])
        return image, image_gt

    def __call__(self, img_pair):
        image, image_gt = img_pair
        self.params = self._get_params(img_pair)
        image, image_gt = self._apply_image(image, image_gt)
        return image, image_gt


class PairedMosaic():

    def __init__(self, prob=0.0, split_range=[0.25, 0.75], image_path_list=[], is_rgb=True, ):
        super().__init__()
        self.prob = prob
        self.split_range = split_range
        self.image_path_list = image_path_list
        self.is_rgb = is_rgb
        self.image_index_list = list(range(len(self.image_path_list)))
        self.params = dict()

    def _get_params(self, inputs):
        params = {}
        params['mosic'] = random.random() < self.prob
        if params['mosic']:
            params['split_x'] = random.uniform(self.split_range[0], self.split_range[1])
            params['split_y'] = random.uniform(self.split_range[0], self.split_range[1])
            params['select_mosic_image_path'] = list(np.random.choice(self.image_index_list, 3, ))
        return params

    def _apply_image(self, image, image_gt):
        if self.params['mosic']:
            h, w, _ = image.shape
            split_x = int(w * self.params['split_x'])
            split_y = int(h * self.params['split_y'])
            image_x_list = [load_image(self.image_path_list[idx][0], is_rgb=self.is_rgb) for idx in self.params['select_mosic_image_path']]
            image_gt_list = [load_image(self.image_path_list[idx][1], is_rgb=self.is_rgb) for idx in self.params['select_mosic_image_path']]
            res_image = np.zeros((h, w, 3), dtype=np.uint8)
            res_gt = np.zeros((h, w, 3), dtype=np.uint8)
            res_image[:split_y, :split_x, :] = image[:split_y, :split_x, :]
            res_gt[:split_y, :split_x, :] = image_gt[:split_y, :split_x, :]
            res_image[:split_y, split_x:, :] = image_x_list[0][:split_y, split_x:, :]
            res_gt[:split_y, split_x:, :] = image_gt_list[0][:split_y, split_x:, :]
            res_image[split_y:, :split_x, :] = image_x_list[1][split_y:, :split_x, :]
            res_gt[split_y:, :split_x, :] = image_gt_list[1][split_y:, :split_x, :]
            res_image[split_y:, split_x:, :] = image_x_list[2][split_y:, split_x:, :]
            res_gt[split_y:, split_x:, :] = image_gt_list[2][split_y:, split_x:, :]
            return res_image, res_gt
        return image, image_gt

    def __call__(self, img_pair):
        image, image_gt = img_pair
        self.params = self._get_params(img_pair)
        image, image_gt = self._apply_image(image, image_gt)
        return image, image_gt


class FaceDataset(Dataset):

    def __init__(self, root_dir,
                 is_to_tensor=True,
                 use_cache=False,
                 is_train=True,
                 is_rgb=True,
                 is_normalize=True,
                 rate=1.0,
                 h_flip_p=0.5,
                 mosic_p=0.0,
                 rotate_p=0.0, ):
        super(FaceDataset, self).__init__()
        self.root_dir = root_dir
        self.is_to_tensor = is_to_tensor
        self.use_cache = use_cache
        self.is_train = is_train
        self.is_rgb = is_rgb
        self.is_normalize = is_normalize
        self.rate = rate
        self.image_path_list = []
        self.image_cache = dict()
        self.to_tensor = self._init_to_tensor()

        self._init_image_path()

        self.random_hflip = PairedRandomHorizontalFlip(prob=h_flip_p, keys=['image', 'image'], )
        self.mosaic = PairedMosaic(prob=mosic_p, image_path_list=self.image_path_list, is_rgb=is_rgb, )
        # self.resize = PairedResize(prob=resize_p, resize_range=[512, 1256], )
        self.rotate = PairedRotate(prob=rotate_p, )

    def _init_to_tensor(self):
        if self.is_normalize:
            return normalize_tensor
        else:
            return identity_tensor

    def _init_image_path(self):
        x_dir = os.path.join(self.root_dir, 'image')
        gt_dir = os.path.join(self.root_dir, 'groundtruth')
        for file_name in os.listdir(x_dir):
            self.image_path_list.append([
                os.path.join(x_dir, file_name),
                os.path.join(gt_dir, file_name)
            ])
        if self.is_train:
            self.image_path_list = self.image_path_list[:int(len(self.image_path_list) * self.rate)]
        else:
            self.image_path_list = self.image_path_list[int(len(self.image_path_list) * self.rate):]

    def _load_image(self, filepath):
        if self.use_cache:
            raise NotImplementedError
        else:
            return load_image(filepath, is_rgb=self.is_rgb)

    def _apply_aug(self, x, gt):
        x, gt = self.mosaic((x, gt))
        x, gt = self.random_hflip((x, gt))
        # x, gt = self.resize((x, gt))
        x, gt = self.rotate((x, gt))
        return x, gt

    def __getitem__(self, idx):
        x = self._load_image(self.image_path_list[idx][0])
        gt = self._load_image(self.image_path_list[idx][1])

        x, gt = self._apply_aug(x, gt)

        if self.is_to_tensor:
            x = self.to_tensor(x)
            gt = self.to_tensor(gt)
        return x, gt

    def __len__(self):
        return len(self.image_path_list)
