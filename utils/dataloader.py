import os
import glob
import random
import cv2
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image, ImageEnhance
from .utils import cvtColor, preprocess_input
from .utils_aug import CenterCrop, ImageNetPolicy, RandomResizedCrop, Resize
from .degradations.dfd_degradation import DFDNet_degradation


def read_lut_file(path):
    with open(path) as fd:
        lines = [x.rstrip() for x in fd.readlines()]
    LUT_SIZE = -1
    for line in lines:
        if line.startswith('LUT_3D_SIZE'):
            LUT_SIZE = int(line.split(' ')[-1])
    if LUT_SIZE == -1:
        raise Exception('Invalid lut file, no LUT_3D_SIZE defined')
    points = []
    for line in lines[-LUT_SIZE**3:]:
        r, g, b = line.split(' ')
        p = np.array([np.float(r), np.float(g), np.float(b)])
        points.append(p)
    lut = np.array(points)
    return LUT_SIZE, lut


def init_luts(pathes):
    luts_dict = dict()
    for lut_path in pathes:
        LUT_SIZE, lut = read_lut_file(lut_path)
        lut_key = os.path.basename(lut_path)
        luts_dict[lut_key] = (LUT_SIZE, lut)
    return luts_dict

def glob_lut_pathes(lut_dir):
    pathes = glob.glob(lut_dir + '/*.*')
    lut_pathes = []
    for path in pathes:
        if path.endswith('.cube') or path.endswith('.CUBE'):
            lut_pathes.append(path)
    return lut_pathes

class LUTConverter:
    def __init__(self, lut_dir='cubes'):
        lut_pathes = glob_lut_pathes(lut_dir)
        self.lut_dict = init_luts(lut_pathes)
        self.lut_list = list(self.lut_dict.items())

    def convert(self, index, img):
        key, value = self.lut_list[index]
        lut_size, lut = value
        img = np.floor(img / 256 * lut_size)
        pixels = img.reshape(-1, 3)
        r, g, b = pixels[:, 0], pixels[:, 1], pixels[:, 2]
        idx = r + g * lut_size + b * (lut_size ** 2)
        idx = idx.astype(np.int64)
        new_pixels = lut[idx]
        new_img = np.array(new_pixels).reshape(img.shape)
        new_img = (new_img * 255).astype(np.uint8)
        return new_img

    def convert_random(self, img):
        index = random.randint(0, len(self.lut_dict.items())-1)
        return self.convert(index, img)
    
def random_crop(image, max_crop_percent=0.3):
    """
    对图像进行上下左右随机缩减0~30%的概率裁剪。

    参数:
        image (PIL.Image): 输入图像。
        max_crop_percent (float): 最大裁剪比例，默认为0.3（30%）。

    返回:
        PIL.Image: 裁剪后的图像。
    """
    width, height = image.size
    
    # 随机生成四个边的裁剪比例
    left_crop_percent = random.uniform(0, max_crop_percent)
    top_crop_percent = random.uniform(0, max_crop_percent)
    right_crop_percent = random.uniform(0, max_crop_percent)
    bottom_crop_percent = random.uniform(0, max_crop_percent)
    
    # 计算裁剪后的边界
    left = int(left_crop_percent * width)
    top = int(top_crop_percent * height)
    right = width - int(right_crop_percent * width)
    bottom = height - int(bottom_crop_percent * height)
    
    # 裁剪图像
    cropped_image = image.crop((left, top, right, bottom))
    
    return cropped_image
    
class DataGenerator(data.Dataset):
    def __init__(self, annotation_lines, input_shape, random=True, autoaugment_flag=True):
        self.annotation_lines   = annotation_lines
        self.input_shape        = input_shape
        self.random             = random
        # if random:
        #     self.lut_converter = LUTConverter()
        self.degradator = DFDNet_degradation(resolution=input_shape[0])

        self.autoaugment_flag   = autoaugment_flag
        if self.autoaugment_flag:
            self.resize_crop = RandomResizedCrop(input_shape)
            self.policy      = ImageNetPolicy()
            
            self.resize      = Resize(input_shape[0] if input_shape[0] == input_shape[1] else input_shape)
            self.center_crop = CenterCrop(input_shape)

    def __len__(self):
        return len(self.annotation_lines)

    def __getitem__(self, index):
        annotation_path = self.annotation_lines[index].split(';')[1].split()[0]
        image = Image.open(annotation_path)
        #------------------------------#
        #   读取图像并转换成RGB图像
        #------------------------------#
        image = cvtColor(image)
        # width, height = image.size
        if self.autoaugment_flag:
            import random
            index = random.randint(0, 10000)
            if "/face/" in annotation_path:
                image.save(os.path.join("result", "xx_"+str(index)+".jpg"))
            
            crop_prob = random.random()
            if 0<= crop_prob <= 0.3:
                image = self.resize(image)
                image = self.center_crop(image)
            elif 0.3 < crop_prob < 0.7:
                image = image.resize(self.input_shape, Image.BILINEAR)
            elif 0.7 <= crop_prob <= 1:
                image = random_crop(image)
                image = image.resize(self.input_shape, Image.BILINEAR)
            
            # image = random_crop(image)
            # image = image.resize(self.input_shape, Image.BILINEAR)
            
            # if "/face/" in annotation_path:
            #     image.save(os.path.join("result", "xx_"+str(index)+".png"))
            
            # image = self.AutoAugment(image, random=self.random)
            # image = cv2.resize(image, (256, 256))
            image = np.array(image).astype(np.float32)
            
            
            if random.random() > 0.5 and "pornpics" in annotation_path:
                image = image[:, :, ::-1]
                _, image = self.degradator.degrade_process(image)
            
            if random.random() > 0.8:
                h, w = image.shape[:2]
                center = (w//2, h//2)
                angle = random.uniform(0, 360)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                image = cv2.warpAffine(image, M, (w, h))
            
            if "/face/" in annotation_path:
                image.save(os.path.join("result", "xx_"+str(index)+".png"))
                
        image = np.transpose(preprocess_input(np.array(image).astype(np.float32)), [2, 0, 1])

        y = int(self.annotation_lines[index].split(';')[0])
        return image, y

    def rand(self, a=0, b=1):
        return np.random.rand()*(b-a) + a
    
    def randomColor(self, image, saturation=1, brightness=1, contrast=1, sharpness=1):
        if self.rand() < saturation:
            random_factor = np.random.randint(0, 31) / 10.  # 随机因子
            image = ImageEnhance.Color(image).enhance(random_factor)  # 调整图像的饱和度
        if self.rand() < brightness:
            random_factor = np.random.randint(10, 21) / 10.  # 随机因子
            image = ImageEnhance.Brightness(image).enhance(random_factor)  # 调整图像的亮度
        if self.rand() < contrast:
            random_factor = np.random.randint(10, 21) / 10.  # 随机因1子
            image = ImageEnhance.Contrast(image).enhance(random_factor)  # 调整图像对比度
        if self.rand() < sharpness:
            random_factor = np.random.randint(0, 31) / 10.  # 随机因子
            ImageEnhance.Sharpness(image).enhance(random_factor)  # 调整图像锐度
        
        return image
    
    def AutoAugment(self, image, random=True, hue=.1, sat=1.5, val=1.5):
        if not random:
            return np.array(image, np.float32)
        # image.save('1.jpg')

        # 翻转图像
        if self.rand() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        # # 颜色抖动
        # if self.rand() < 0.3:
        #    image = self.randomColor(image)
        # # 滤镜
        # if self.rand() < 0.3:
        #     image = np.array(image, np.uint8)
        #     image = self.lut_converter.convert_random(image)
        
        return np.array(image, np.float32)
            
def detection_collate(batch):
    images = []
    targets = []
    for image, y in batch:
        images.append(image)
        targets.append(y)
    images  = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    targets = torch.from_numpy(np.array(targets)).type(torch.FloatTensor).long()
    return images, targets
