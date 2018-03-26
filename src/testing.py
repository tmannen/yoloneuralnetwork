import torch
from torch.autograd import Variable
import numpy as np
import utils

output = torch.zeros(4, 7, 7, 30)
output = Variable(torch.zeros(4, 7, 7, 30))
labels = Variable(torch.zeros(4, 7, 7, 29))

bboxes_pred_numpy = output[:,:,:,22:].data.numpy().reshape(-1, 7, 7, 2, 4)
bboxes_target_numpy = labels[:,:,:,21:].data.numpy().reshape(-1, 7, 7, 2, 4)

ious = utils.IOUs(bboxes_target_numpy, bboxes_pred_numpy)

mask = labels[:,:,:,20] == 1 #only take elements with an object in it
mask = mask.unsqueeze(-1)
mask[1,2,3,0] = 1
mask[3,5,5,0] = 1

npmask = mask.data.numpy()==1
npmask = npmask.repeat(2, axis=3)

100x100 image

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

img = np.zeros((100, 100, 3))
bbox_pred = np.array((30, 20, 75, 80))
bbox_true = np.array((35, 5, 60, 95))

def IOU_single(box_a, box_b):
    max_xy = np.minimum(box_a[2:], box_b[2:])
    min_xy = np.maximum(box_a[:2], box_b[:2])
    inter = np.clip(max_xy - min_xy, 0, None)
    inter = inter[0] * inter[1]

    area_a = (box_a[2]-box_a[0]) * (box_a[3]-box_a[1])
    area_b = (box_b[2]-box_b[0]) * (box_b[3]-box_b[1])
    union = area_a + area_b - inter
    return inter / union

def IOU_yolo(box_a, box_b):
    max_xy = np.minimum(box_a[2:], box_b[2:])
    min_xy = np.maximum(box_a[:2], box_b[:2])
    inter = np.clip(max_xy - min_xy, 0, None)
    inter = inter[0] * inter[1]

    area_a = (box_a[2]-box_a[0]) * (box_a[3]-box_a[1])
    area_b = (box_b[2]-box_b[0]) * (box_b[3]-box_b[1])
    union = area_a + area_b - inter
    return inter / union

print(IOU_single(bbox_pred, bbox_true))
fig,ax = plt.subplots()

inter = ()

ax.imshow(img)

for bbox, color in [(bbox_pred, 'b'), (bbox_true, 'r')]:
    xmin, ymin, xmax, ymax = bbox
    w = xmax - xmin
    h = ymax - ymin
    rect = Rectangle((xmin, ymin), w, h, linewidth=1, edgecolor=color, facecolor='none', fill=False)
    ax.add_patch(rect)

plt.show()

pred_min_x = 0.4*33 - 0.5*0.4*100
pred_min_x
0.4*0.7
0.4*0.7*10000
area_pred = 0.4*0.7
area_true = 0.2*0.8
pred_x_min = 0.4*0.33 - 0.5*0.4
pred_x_min
true_x_min = 0.3*0.33 - 0.5*0.2
true_x_min
pred_x_max = 0.4*0.33 + 0.5*0.4
true_x_max = 0.3*0.33 + 0.5*0.2
pred_y_min = 0.5*0.33 - 0.5*0.7
pred_y_max = 0.5*0.33 + 0.5*0.7
true_y_min = 0.4*0.33 - 0.5*0.8
true_y_max = 0.4*0.33 + 0.5*0.8
max(pred_x_min, true_x_min)
max(pred_x_min, true_x_min) * min(pred_x_max, true_x_max)
min(pred_x_max, true_x_max) - max(pred_x_min, true_x_min)
interx = min(pred_x_max, true_x_max) - max(pred_x_min, true_x_min)
intery = min(pred_y_max, true_y_max) - max(pred_y_min, true_y_min)
interx*intery
inter = interx*intery
inter/(area_pred + area_true - inter)