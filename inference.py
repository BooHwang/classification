#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File      : inference.py
@Time      : 2024/05/21 10:52:49
@Author    : Huang Bo
@Contact   : cenahwang0304@gmail.com
@Desc      : None
'''

import torch
import torch.nn as nn
from torchvision import transforms

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

# Net
class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.0, inverted_residual_setting=None, round_nearest=8):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280

        inverted_residual_setting = [
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)

        features = [ConvBNReLU(3, input_channel, stride=2)]

        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel

        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        self.features = nn.Sequential(*features)

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])
        x = self.classifier(x)
        return x
    
    def freeze_backbone(self):
        for param in self.features.parameters():
            param.requires_grad = False

    def Unfreeze_backbone(self):
        for param in self.features.parameters():
            param.requires_grad = True

def mobilenet_v2(num_classes=1000):
    model = MobileNetV2()
    model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(model.last_channel, num_classes),
        )
    return model

class Face_Predict(object):
    def __init__(self, model_path: str="", gpu_id: int=0):
        self.model = mobilenet_v2(num_classes=2)
        self.device = torch.device(f"cuda:{str(gpu_id)}" if torch.cuda.is_available() else "cpu")
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"), strict=True)
        self.model.eval()
        self.model = self.model.to(self.device)
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.unsqueeze(0))])

    def preprocess(self, img):
        img = img.resize((256, 256), Image.BILINEAR)
        img_tensor = self.transform(img).to(self.device)
        img_tensor = self.normalize(img_tensor)
        return img_tensor
        
    @torch.no_grad()
    def __call__(self, img):
        img_tensor = self.preprocess(img)
        pred_score = torch.softmax(self.model(img_tensor)[0], dim=-1).cpu().numpy()
        pred = pred_score.argmax(0)
        
        return pred, pred_score[0]
    
if __name__ == "__main__":
    import os
    from PIL import Image
    from glob import glob
    from tqdm import tqdm
    
    gpu_id = 0
    best_checkpoint = "./logs/face_0523/best_epoch_weights.pth"
    
    face_predict = Face_Predict(best_checkpoint, gpu_id)
    
    img_root = "face"
    img_paths = [y for x in os.walk(img_root) for y in glob(os.path.join(x[0], "*.*g"))]
    
    predict_root = "face_predict"
    face_root = os.path.join(predict_root, "face")
    notface_root = os.path.join(predict_root, "notface")
    os.makedirs(face_root, exist_ok=True)
    os.makedirs(notface_root, exist_ok=True)

    all_scores = list()
    for img_path in tqdm(img_paths):
        img_name = os.path.basename(img_path)
        img = Image.open(img_path)
        pred, pred_score = face_predict(img)
        all_scores.append(pred_score)
        assert pred in [0, 1], "pred get error value"
        if pred == 0:
            img.save(os.path.join(face_root, img_name))
        elif pred == 1:
            img.save(os.path.join(notface_root, img_name))
            
    