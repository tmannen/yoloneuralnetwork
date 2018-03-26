"""

Utility for loading the VOC2007 and VOC2012 data.

Originally from: https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py
"""

import torch.utils.data as data
from PIL import Image, ImageDraw
import os
import os.path
import sys
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET
import numpy as np
from torchvision import transforms
import utils
from configparser import ConfigParser

config = ConfigParser()
config.read("../yoloconfig.cfg")

INPUT_SIZE = int(config['DEFAULT']['input_size'])
classes = config['DEFAULT']['classes'].split(",")
S = int(config['DEFAULT']['num_cells'])
B = int(config['DEFAULT']['num_bboxes'])
C = len(classes)
class_dict = {cls: i for i, cls in enumerate(classes)}
datadir_train = config['DEFAULT']['datadir_train']
datadir_test = config['DEFAULT']['datadir_test']

class TransformYOLOTrainVal(object):
    def __init__(self, keep_difficult=False):
        self.keep_difficult = keep_difficult

    def __call__(self, og_img_size, target):
        res = []
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj[0].text.lower().strip()
            bbox = obj.find('bndbox')
            bndbox = [int(bb.text)-1 for bb in bbox]
            bndbox = utils.scale_bbox_to_square(og_img_size, bndbox)

            res += [bndbox + [name]]

        target = utils.VOC_to_YOLO_full(res)
        return target

class TransformYOLOTest(object):
    def __init__(self, keep_difficult=False):
        self.keep_difficult = keep_difficult

    def __call__(self, og_img_size, target):
        res = []
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj[0].text.lower().strip()
            bbox = obj.find('bndbox')
            
            bndbox = [int(bb.text)-1 for bb in bbox]
            bndbox = utils.scale_bbox_to_square(og_img_size, bndbox)

            res += [bndbox + [name]]

        target = res
        return target

class VOCDetection(data.Dataset):
    def __init__(self, root, image_set, transform=None, target_transform=None):
        self.root = root
        self.image_set = image_set
        self.transform = transform
        self.target_transform = target_transform

        dataset_name = 'VOC2007'
        self._annopath = os.path.join(self.root, dataset_name, 'Annotations', '%s.xml')
        self._imgpath = os.path.join(self.root, dataset_name, 'JPEGImages', '%s.jpg')
        self._imgsetpath = os.path.join(self.root, dataset_name, 'ImageSets', 'Main', '%s.txt')
 
        with open(self._imgsetpath % self.image_set) as f:
            self.ids = f.readlines()
        if "_" in image_set: #if only one class wanted
            self.ids = [x.split()[0] for x in self.ids if int(x.split()[1]) == 1]
        self.ids = [x.strip('\n') for x in self.ids]

    def __getitem__(self, index):
        img_id = self.ids[index]

        target = ET.parse(self._annopath % img_id).getroot()

        img = Image.open(self._imgpath % img_id).convert('RGB')
        og_img_size = img.size
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(og_img_size, target)

        if 'test' in self.image_set: #we need image id when testing for MAP calculation
            return img, target, img_id

        return img, target

    def __len__(self):
        return len(self.ids)

    def show(self, index):
        img, target = self.__getitem__(index)
        draw = ImageDraw.Draw(img)
        for obj in target:
            draw.rectangle(obj[0:4], outline=(255,0,0))
            draw.text(obj[0:2], obj[4], fill=(0,255,0))
        img.show()

if __name__ == "__main__":
    ds = VOCDetection('/home/tman/koulu/Deep Learning/project/datasets/VOC2007/VOCdevkit/',
        'dog_train',
        transform=transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        target_transform=TransformYOLOVOCDetectionAnnotation(False))
    img, target = ds[0]
    print(target)
    #ds.show(0)
    img.show()
    #print(target_transform(target))