import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from PIL import Image
import numpy as np
import matplotlib.patches as patches
import torch
from torch.autograd import Variable

from configparser import ConfigParser

config = ConfigParser()
config.read("../yoloconfig.cfg")

INPUT_SIZE = int(config['DEFAULT']['input_size'])
classes = config['DEFAULT']['classes'].split(",")
S = int(config['DEFAULT']['num_cells'])
B = int(config['DEFAULT']['num_bboxes'])
C = len(classes)
class_dict = {cls: i for i, cls in enumerate(classes)}
CELL_LEN = int(INPUT_SIZE / S)

#intersect and jaccard functions from: https://github.com/amdegroot/ssd.pytorch/blob/master/layers/box_utils.py#L48
def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]
    

def scale_bbox_to_square(img_size, bbox):
    xmin, ymin, xmax, ymax = bbox
    w, h = img_size
    scale_x = INPUT_SIZE / w
    scale_y = INPUT_SIZE / h

    return [int(xmin * scale_x), int(ymin * scale_y) , int(xmax * scale_x), int(ymax * scale_y)]

def IOUs(box_a, box_b):
    """
    Takes in box_a: (BS, S, S, B, 4)
    box_b = (BS, S, S, B, 4)
    """
    max_xy = np.minimum(box_a[:, :, :, :, 2:], box_b[:, :, :, :, 2:])
    min_xy = np.maximum(box_a[:, :, :, :, :2], box_b[:, :, :, :, :2])
    inter = np.clip(max_xy - min_xy, 0, None)
    inter = inter[:,:,:,:,0] * inter[:,:,:,:,1]

    area_a = (box_a[:, :, :,:, 2]-box_a[:, :, :,:, 0]) * (box_a[:, :, :,:, 3]-box_a[:, :, :,:, 1])
    area_b = (box_b[:, :, :, :, 2]-box_b[:, :, :, :, 0]) * (box_b[:, :, :, :, 3]-box_b[:, :, :, :, 1])
    union = area_a + area_b - inter + 1e-11 #eps so it's not zero

    return inter / union

def IOUs_torch(box_a, box_b):
    """
    Takes in box_a: (BS, S, S, B, 4)
    box_b = (BS, S, S, B, 4)
    """
    max_xy = torch.min(box_a[..., 2:], box_b[..., 2:])
    min_xy = torch.max(box_a[..., :2], box_b[..., :2])
    inter = torch.clamp(max_xy - min_xy, 0)
    inter = inter[...,0] * inter[...,1]

    area_a = (box_a[..., 2]-box_a[..., 0]) * (box_a[..., 3]-box_a[..., 1])
    area_b = (box_b[..., 2]-box_b[..., 0]) * (box_b[..., 3]-box_b[..., 1])
    union = area_a + area_b - inter + 1e-11 #eps so it's not zero

    return inter / union

def IOU_single(box_a, box_b):
    """
    TODO: test this shit
    """
    max_xy = np.minimum(box_a[2:], box_b[2:])
    min_xy = np.maximum(box_a[:2], box_b[:2])
    inter = np.clip(max_xy - min_xy, 0, None)
    inter = inter[0] * inter[1]

    area_a = (box_a[2]-box_a[0]) * (box_a[3]-box_a[1])
    area_b = (box_b[2]-box_b[0]) * (box_b[3]-box_b[1])
    union = area_a + area_b - inter
    return inter / union

def VOC_to_YOLO_full(bboxes_classes):
    """
        
    """
    results = np.zeros((S, S, C + 1 + B*4)) #the 1 is an indicator if an object(s middle?) is in the cell

    for elem in bboxes_classes:

        bbox = elem[:4]
        cls = elem[4]
        xmin, ymin, xmax, ymax = bbox
        w = xmax - xmin
        h = ymax - ymin

        center_x = (xmin + xmax)/2
        center_y = (ymin + ymax)/2

        grid_center_idx_x = int(center_x / CELL_LEN)
        grid_center_idx_y = int(center_y / CELL_LEN)
        grid_norm_x = center_x / CELL_LEN - grid_center_idx_x
        grid_norm_y = center_y / CELL_LEN - grid_center_idx_y

        #normalize width and height
        w_normed = w / INPUT_SIZE
        h_normed = h / INPUT_SIZE

        class_onehot = np.zeros(C)
        class_onehot[class_dict[cls]] = 1
        results[grid_center_idx_y, grid_center_idx_x, :C] = class_onehot
        results[grid_center_idx_y, grid_center_idx_x, C] = 1 
        results[grid_center_idx_y, grid_center_idx_x, C + 1:] = np.tile([grid_norm_x, grid_norm_y, w_normed, h_normed], B)

    return results

def VOC_to_YOLO_full(bboxes_classes):
    """
        
    """
    results = np.zeros((S, S, C + 1 + B*4)) #the 1 is an indicator if an object(s middle?) is in the cell
    mask = torch.zeros(S, S)

    for elem in bboxes_classes:

        bbox = elem[:4]
        cls = elem[4]
        xmin, ymin, xmax, ymax = bbox
        w = xmax - xmin
        h = ymax - ymin

        center_x = (xmin + xmax)/2
        center_y = (ymin + ymax)/2

        grid_center_idx_x = int(center_x / CELL_LEN)
        grid_center_idx_y = int(center_y / CELL_LEN)
        grid_norm_x = center_x / CELL_LEN - grid_center_idx_x
        grid_norm_y = center_y / CELL_LEN - grid_center_idx_y

        #normalize width and height
        w_normed = w / INPUT_SIZE
        h_normed = h / INPUT_SIZE

        class_onehot = np.zeros(C)
        class_onehot[class_dict[cls]] = 1
        results[grid_center_idx_y, grid_center_idx_x, :C] = class_onehot
        results[grid_center_idx_y, grid_center_idx_x, C] = 1 
        results[grid_center_idx_y, grid_center_idx_x, C + 1:] = np.tile([grid_norm_x, grid_norm_y, w_normed, h_normed], B)

    return results

def YOLO_to_VOC(bboxes):
    """
    bboxes shape = (BS, S, S, B, 4) => grid_x, grid_y, w, h
    YOLO format to normal (xmin, ymin, xmax, ymax)

    """
    out = np.zeros(bboxes.shape)
    x_mults = np.reshape(np.repeat(np.arange(S), S), (S, S))
    x_mults = np.stack([x_mults for i in range(B)]).transpose()
    y_mults = np.reshape(np.tile(np.arange(S), S), (S, S))
    y_mults = np.stack([y_mults for i in range(B)]).transpose()

    x_centers = bboxes[:,:,:,:,0] * CELL_LEN + CELL_LEN * x_mults[None,...]
    y_centers = bboxes[:,:,:,:,1] * CELL_LEN + CELL_LEN * y_mults[None,...]
    widths = bboxes[:,:,:,:,2] * INPUT_SIZE
    heights = bboxes[:,:,:,:,3] * INPUT_SIZE

    out[:,:,:,:,0] = x_centers - widths/2
    out[:,:,:,:,1] = y_centers - heights/2
    out[:,:,:,:,2] = x_centers + widths/2
    out[:,:,:,:,3] = y_centers + heights/2

    out[bboxes == 0] = 0 #real values should never be zero, so we zero those elements. otherwise the yolo -> voc of (0,0,0,0) is gibberish

    return out

def YOLO_to_VOC_single(bbox, grid_y, grid_x):
    """
    bboxes shape = (BS, S, S, 4) => grid_x, grid_y, w, h (BS = batch size)
    YOLO format to normal (xmin, ymin, xmax, ymax)

    """
    w = bbox[2] * INPUT_SIZE
    h = bbox[3] * INPUT_SIZE

    xmin = bbox[0] * CELL_LEN + grid_x * CELL_LEN - w / 2
    ymin = bbox[1] * CELL_LEN + grid_y * CELL_LEN - h / 2
    xmax = xmin + w
    ymax = ymin + h

    return [xmin, ymin, xmax, ymax]

def interpret_yolo(output, threshold):
    bboxes = output[:,:,22:].reshape(S, S, B, 4)
    confidences = output[:,:,20:20+B]
    class_scores = output[:,:,:20]

    max_scores = []
    max_classes = np.argmax(class_scores, axis=2)

    for i in range(B):
        temp_scores = class_scores * confidences[:,:,i,None]
        max_scores.append(np.max(temp_scores, axis=2))

    best_scores_per_cell = np.max(np.stack(max_scores), axis=0)
    best_score_index = np.argmax(np.stack(max_scores), axis=0)

    results = []

    for idxs in np.argwhere(best_scores_per_cell > threshold):
        row = idxs[0]
        col = idxs[1]
        result = {}
        bbox_idx = best_score_index[row,col]
        bbox = bboxes[row,col,bbox_idx,:]
        bbox[2:] = np.square(bbox[2:])
        bbox = YOLO_to_VOC_single(bbox, row, col)
        result['score'] = best_scores_per_cell[row,col]
        result['confidence'] = confidences[row,col,bbox_idx]
        result['bbox'] = bbox
        result['class'] = max_classes[row,col]
        result['class_name'] = classes[result['class']]
        results.append(result)

    return results

def interpret_yolo_conf(output, threshold):
    bboxes = output[:,:,22:].reshape(S, S, B, 4)
    confidences = output[:,:,20:20+B]
    class_scores = output[:,:,:20]

    max_scores = []
    max_classes = np.argmax(class_scores, axis=2)

    for i in range(B):
        temp_scores = class_scores * confidences[:,:,i,None]
        max_scores.append(np.max(temp_scores, axis=2))

    best_scores_per_cell = np.max(np.stack(max_scores), axis=0)
    best_score_index = np.argmax(np.stack(max_scores), axis=0)
    best_conf_per_cell = np.max(confidences, axis=2)
    best_conf_index = np.argmax(confidences, axis=2)

    results = []

    for idxs in np.argwhere(best_conf_per_cell > threshold):
        row = idxs[0]
        col = idxs[1]
        result = {}
        bbox_idx = best_conf_index[row,col]
        bbox = bboxes[row,col,bbox_idx,:]
        bbox[2:] = np.square(bbox[2:])
        bbox = YOLO_to_VOC_single(bbox, row, col)
        result['score'] = best_scores_per_cell[row,col]
        result['confidence'] = confidences[row,col,bbox_idx]
        result['bbox'] = bbox
        result['class'] = max_classes[row,col]
        result['class_name'] = classes[result['class']]
        results.append(result)

    return results

def py_cpu_nms(dets, thresh):
    """
    Pure Python NMS (non maximum suppression) baseline.
    Originally written by Ross Girshick
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

def plot_img(image, results):
    image = np.array(image)
    if image.shape[0] == 3:
        #torch takes images in 3, w, h instead of w, h, 3
        image = image.transpose((1, 2, 0))

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    image = np.clip(image, 0, 1)

    fig,ax = plt.subplots(1)

    ax.imshow(image)

    for bbox, cls in results:
        xmin, ymin, xmax, ymax = bbox
        w = xmax - xmin
        h = ymax - ymin
        # Create a Rectangle patch
        rect = patches.Rectangle((xmin, ymin), w, h, linewidth=1, edgecolor='r', facecolor='none', fill=False)
        ax.text(xmin, ymin, cls)
        # Add the patch to the Axes
        ax.add_patch(rect)

    plt.show()

def save_results(results):
    path = "/home/tman/koulu/Deep Learning/project/results/"
    flat_results = [item for items in results for item in items]

    classes = set([r['class_name'] for r in flat_results])
    class_lists = {cls : [] for cls in classes}

    for res in flat_results:
        class_lists[res['class_name']].append(res)

    for cls in class_lists:
        with open(path + cls, "w") as writefile:
            for res in class_lists[cls]:
                img_id = res['img_id']
                confidence = str(res['confidence'])
                bbox = [str(int(x)) for x in res['bbox']]
                string = " ".join([img_id, confidence] + bbox + ["\n"])
                writefile.write(string)

"""
if __name__ == "__main__":
    img = Image.open("../test/auto.jpg").convert('RGB')
    img_square = img.resize((INPUT_SIZE,INPUT_SIZE))
    bbox = (90, 220, 1000, 700) #bbox = (70, 100, 400, 250)
    bbox_square = scale_bbox_to_square(img.size, bbox)
    bbox_test = (50, 300, 800, 500)
    vals = VOC_to_YOLO_full(bbox_square)
    yolo_bbox = vals['bbox']
    target_bbox = np.append(yolo_bbox[:2], np.sqrt(yolo_bbox[2:]))
    yolo_test = np.zeros((S, S, C + B * 5))
    bboxes = yolo_test[:,:,22:].reshape(S, S, B, 4)
    bboxes[4,3,1,:] = bbox_square
    ious = IOUs(bboxes, bbox_square)
    #max_idx = np.where(ious == np.max(ious))
    
    grid_x = vals['grid_x_idx']
    grid_y = vals['grid_y_idx']
    max_iou_idx = np.argmax(ious[grid_y,grid_x,:])
    max_iou = np.max(ious[grid_y,grid_x,:])
    #plot_img(img_square, back)

    torched = Variable(torch.from_numpy(yolo_test))
    t_bbox = torched[grid_y,grid_x,22 + max_iou_idx*4:26 + max_iou_idx*4]
    target_bbox = Variable(torch.from_numpy(target_bbox))
    #mse?

    pred_confidence = torched[grid_y, grid_x, 20 + max_iou_idx]
    target_confidence = max_iou #needs to be Variable?
    #mse?

    pred_noobj_conf = torched[:, :, 20:22]
    target_noobj_conf = torch.zeros_like(pred_noobj_conf)
    #make the responsible cell subtraction equal zero so it doesnt contribute to gradient in noobj
    target_noobj_conf[grid_y, grid_x, max_iou_idx] = pred_confidence.data

    #mse?

    gg = voc_eval.voc_eval("/home/tman/koulu/Deep Learning/project/results/{}",
    "/home/tman/koulu/Deep Learning/project/datasets/VOC2007Test/VOCdevkit/VOC2007/Annotations/{}",
    "/home/tman/koulu/Deep Learning/project/datasets/VOC2007Test/VOCdevkit/VOC2007/ImageSets/Main/dog_test.txt", 
    "dog", "cache", use_07_metric=True)
"""

