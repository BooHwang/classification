import os
import numpy as np
import cv2
import math
import torch
import random
from PIL import Image
from scipy.io import loadmat

from utils.degradations import degradations

current_root = os.path.dirname(os.path.abspath("__file__"))

class DFDNet_degradation(object):
    def __init__(self, resolution=512):
        self.kernel_list = ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso']
        self.kernel_prob = [0.45, 0.25, 0.03, 0.03, 0.12, 0.12]
        self.kernel_range = [2 * v + 1 for v in range(1, 11)]  # 21
        self.blur_sigma = [1, 10]  # /10                       # 30
        self.downsample_range = [10, 60]                       # 200
        self.noise_range = [0, 12]
        self.jpeg_range = [30, 70]
        self.blur_prob = 0.8
        self.gray_prob = 0.1
        self.color_jitter_prob = 0.0
        self.color_jitter_pt_prob = 0.0
        self.shift = 20/255.
        self.sigma_color = 5
        self.sigma_spatial = 3
        self.resolution = resolution

    def AddNoise(self,img, gray_noise=False): # noise
        if random.random() > 0.8: #
            return img
        self.sigma = np.random.randint(self.noise_range[0], self.noise_range[1])
        img_tensor = torch.from_numpy(np.array(img)).float()
        if gray_noise:
            noise = torch.randn(img_tensor.size()[0:2]).mul_(self.sigma/1.0)
            noise = torch.unsqueeze(noise, axis=2).repeat(1,1,3)
        else:
            noise = torch.randn(img_tensor.size()).mul_(self.sigma/1.0)

        noiseimg = torch.clamp(noise+img_tensor,0,255)
        return Image.fromarray(np.uint8(noiseimg.numpy()))

    def AddBlur(self,img): # gaussian blur or motion blur
        if random.random() > 0.8: #
            return img
        img = np.array(img)
        if random.random() > 0.35: ##gaussian blur
            kernel_size = random.choice(self.kernel_range)
            kernel = degradations.random_mixed_kernels(
                self.kernel_list,
                self.kernel_prob,
                kernel_size,
                self.blur_sigma,
                self.blur_sigma,
                [-math.pi, math.pi],
                noise_range=None)
            img = cv2.filter2D(img, -1, kernel)
        else: #motion blur
            M = random.randint(1,32)
            KName = os.path.join(current_root, 'utils/degradations/MotionBlurKernel/m_%02d.mat' % M)
            k = loadmat(KName)['kernel']
            k = k.astype(np.float32)
            k /= np.sum(k)
            img = cv2.filter2D(img,-1,k)
        return Image.fromarray(img)

    def AddDownSample(self,img): # downsampling
        if random.random() > 0.85: #
            return img
        sampler = random.randint(self.downsample_range[0], self.downsample_range[1])*1.0
        img = img.resize((int(self.resolution/sampler*10.0), int(self.resolution/sampler*10.0)), Image.BICUBIC)
        return img

    def AddJPEG(self,img): # JPEG compression
        if random.random() > 0.6: #
            return img
        imQ = random.randint(self.jpeg_range[0], self.jpeg_range[1])
        img = np.array(img)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY),imQ] # (0,100),higher is better,default is 95
        _, encA = cv2.imencode('.jpg',img,encode_param)
        img = cv2.imdecode(encA,1)
        return Image.fromarray(img)

    def AddUpSample(self,img):
        return img.resize((self.resolution, self.resolution), Image.BICUBIC)

    def degrade_process(self, img_gt):
        if random.random() > 0.5:
            img_gt = cv2.flip(img_gt, 1)

        is_gray = False
        if np.random.uniform() < self.gray_prob:
            img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2GRAY)
            img_gt = np.tile(img_gt[:, :, None], [1, 1, 3])
            is_gray = True
        
        img_gt_hq = img_gt/255.
        
        img_gt_hq = img_gt_hq.astype(np.float32)
            
        img_gt_copy = img_gt.copy()

        img_gt_copy = np.clip(img_gt_copy, 0, 255).astype(np.uint8)
        img_gt_copy = cv2.cvtColor(img_gt_copy, cv2.COLOR_BGR2RGB)
        A = Image.fromarray(img_gt_copy)

        # A = transforms.ColorJitter(0.3, 0.3, 0.3, 0)(A)
        A = self.AddUpSample(self.AddJPEG(self.AddNoise(self.AddDownSample(self.AddBlur(A)), gray_noise=is_gray)))
   
        img_lq = np.array(A)
        # img_lq = cv2.cvtColor(img_lq, cv2.COLOR_RGB2BGR).astype(np.float32)  / 255.

        return img_gt_hq, img_lq
    
if __name__ == "__main__":
    import os
    import numpy as np
    from glob import glob
    from tqdm import tqdm
    
    resolution = 256
    degradator = DFDNet_degradation(resolution=resolution)
    
    save_dir = "result"
    os.makedirs(save_dir, exist_ok=True)

    # face_root = "/data4/face_parsing_task/val_test/face_detect/face_classification/datasets/train/face"
    # face_root = "/data4/face_parsing_task/val_test/face_detect/face_classification/datasets_face/pornpics"
    face_root = "/data4/face_parsing_task/val_test/face_detect/face_classification/datasets/train/online_image/face"
    face_paths = glob(os.path.join(face_root, "*.jpg"))
    
    for face_path in tqdm(face_paths):
        img_name = os.path.basename(face_path)
        img = cv2.imread(face_path)
        img_gt_hq, img_lq = degradator.degrade_process(img)
        img_lq = img_lq[:, :, ::-1]

        # 是否需要旋转
        h, w = img_lq.shape[:2]
        center = (w//2, h//2)
        angle = random.uniform(0, 360)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        img_lq = cv2.warpAffine(img_lq, M, (w, h))
        
        contrast_img = np.concatenate([cv2.resize(img, (resolution, resolution)), img_lq], axis=1)
        cv2.imwrite(os.path.join(save_dir, img_name), contrast_img)