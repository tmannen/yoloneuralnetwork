"""
use pretrained VGG or something since yolo V1, SSD and all seem to use some pretraining?

Check transfer learning tutorial on pytorch
"""

import torchvision.models as models
import torch.nn as nn
import torch
import numpy as np
from configparser import ConfigParser
import utils
import dataset_loading
from torch.autograd import Variable
import torch.optim as optim
from torch.optim import lr_scheduler
import net
from torchvision import transforms

config = ConfigParser()
config.read("../yoloconfig.cfg")

INPUT_SIZE = int(config['DEFAULT']['input_size'])
classes = config['DEFAULT']['classes'].split(",")
S = int(config['DEFAULT']['num_cells'])
B = int(config['DEFAULT']['num_bboxes'])
C = len(classes)
class_dict = {cls: i for i, cls in enumerate(classes)}
root_path = config['DEFAULT']['root']

class YOLOModel(nn.Module):
    def __init__(self, base_model, only_fc=False):
        super(YOLOModel, self).__init__()
        #self.dropout = nn.Dropout(0.5)
        self.conv1_bn = nn.BatchNorm2d(512)
        self.conv2_bn = nn.BatchNorm2d(512)
        self.fc_bn = nn.BatchNorm1d(4096)

        self.features = nn.Sequential(
            *list(base_model.features.children())
        )

        self.conv = nn.Sequential(
            self.conv1_bn,
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            self.conv2_bn,
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2)
        )

        self.output = nn.Sequential(
            nn.Linear(25088, 4096),
            self.fc_bn,
            nn.ReLU(inplace=True),
            nn.Linear(4096, S * S * (C + B*5))
        )

        if only_fc:
            for p in self.features.parameters():
                p.requires_grad = False

    def forward(self, x):
        f = self.features(x)
        c = self.conv(f)
        y = self.output(c.view(-1, 25088))
        return y


def train_model(model, optimizer, mse, lambda_coord, lambda_noobj, batch_size, dataloaders, best_val_res,
                    epochs=1, validate_every=5, shuffle=True, use_gpu=True, scheduler=None):
    for epoch in range(epochs):
        for phase in ['train', 'val']:
            if phase == 'val' and epoch % validate_every != 0:
                break
            if phase == 'train':
                if scheduler != None:
                    scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode
                
            running_loss = 0.0
        
            for i_batch, sample_batched in enumerate(dataloaders[phase]):
                inputs, labels = sample_batched
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.float().cuda())
                else:
                    inputs = Variable(inputs)
                    labels = Variable(labels)

                optimizer.zero_grad()

                output = model(inputs)
                output = output.view(-1, S, S, C + B*5)

                mask = labels[:,:,:,20] == 1 #only take elements with an object in it
                mask = mask.unsqueeze(-1) #make the shape broadcastable with output and labels
                pred_obj = torch.masked_select(output, mask).view(-1, C + B*5) #where there is an object
                tar_obj = torch.masked_select(labels, mask).view(-1, C + 1 + B*4)

                pred_obj_bboxes = pred_obj[:,20+B:].contiguous().view(-1, B, 4)
                tar_obj_bboxes = tar_obj[:,21:].contiguous().view(-1, B, 4)
                #get data to calculate IOUs between predictions and targets

                ious = utils.IOUs(pred_obj_bboxes, tar_obj_bboxes)
                if use_gpu:
                    ious = ious.cuda()

                #get the indexes of the max ious - which bbox is responsible for this
                max_iou_idxs = torch.max(ious, dim=1)[1]

                #select only those bboxes from preds and targets

                #helper for gather function, needs to be same dim as input:
                #note: pred_obj_bboxes same size as target, so both can use this
                bbox_idxs = max_iou_idxs.view(-1, 1, 1).expand(pred_obj_bboxes.size(0), 1, pred_obj_bboxes.size(2))
                pred_resp_bboxes = pred_obj_bboxes.gather(1, bbox_idxs).squeeze()
                #probably never zero but so torch.sqrt doesnt output nan:
                target_resp_bboxes = torch.clamp(tar_obj_bboxes.gather(1, bbox_idxs).squeeze(), min=0)

                bbox_xy_loss = mse(pred_resp_bboxes[:,:2], target_resp_bboxes[:,:2])
                bbox_wh_loss = mse(pred_resp_bboxes[:,2:], torch.sqrt(target_resp_bboxes[:,2:]))
                bbox_loss = lambda_coord * (bbox_xy_loss + bbox_wh_loss)

                pred_obj_conf = pred_obj[:,20:20+B]
                pred_obj_conf = torch.gather(pred_obj_conf, dim=1, index = max_iou_idxs.view(-1, 1))
                tar_obj_conf = torch.gather(objious, dim=1, index = max_iou_idxs.view(-1, 1))

                conf_loss = mse(pred_obj_conf, tar_obj_conf)

                pred_conf_noobj = torch.masked_select(output[:,:,:,20:20+B], mask==0)
                target_conf_noobj = torch.zeros_like(pred_conf_noobj)

                if use_gpu:
                    target_conf_noobj = target_conf_noobj.cuda()

                conf_noobj_loss = lambda_noobj * mse(pred_conf_noobj, target_conf_noobj)

                pred_class = pred_obj[:,:20]
                target_class = tar_obj[:,:20]

                class_loss = mse(pred_class, target_class)

                loss = bbox_loss + conf_loss + conf_noobj_loss + class_loss
                                   
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.data[0]
            
            if phase == 'val' and running_loss < best_val_res:
                print("New best model found")
                print("Val loss: ", running_loss)
                best_val_res = running_loss
                #best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), root_path + 'models/bestmodel.pt')
            if phase == 'train':
                print(optimizer.param_groups[0]['lr'])
                print("Running epoch loss (training): ", running_loss)

    ret = {}
    ret['best_val_res'] = best_val_res
    ret['optimizer'] = optimizer
    ret['scheduler'] = scheduler
    return ret

if __name__ == "__main__":
    mse = nn.MSELoss()
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, yolo.parameters()), lr=0.001, momentum=0.9)
    lambda_coord = 5
    lambda_noobj = 0.5
    batch_size = 4
    epochs = 30

    train_model(yolo, optimizer, mse, lambda_coord, lambda_noobj, batch_size, epochs, ds)