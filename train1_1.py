import torch
from torch import nn
from torchvision import models, transforms
from torchvision.utils import make_grid
import torch.optim as optim
import numpy as np
from tensorboardX import SummaryWriter

from PIL import Image
# import json
from depth_decoder import *
from encoder_layer import *
from dataloader_ver1_0 import *
class Args(object):
    def __init__(self):
        self.batch_size = 2
        self.filenames_file = './train_with_gt.txt'
        self.do_kb_crop = True
        self.do_random_rotate = True
        self.input_height = 352
        self.input_width = 1216

# class BasicEDmodel(nn.Module):
#     def __init__(self, params):
#         super(BasicEDmodel, self).__init__()
#         self.encoder = encoder('resnet50')
#         self.decoder = bts(params, self.encoder.feat_out_channels, params.bts_size)

#     def forward(self, x, focal):
#         skip_feat = self.encoder(x)
#         return self.decoder(skip_feat, focal)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print('using cuda as device')
else:
    device = torch.device("cpu")

num_epochs = 5000
learning_rate = 1e-2
adam_eps = 1e-3
scheduler_step_size = 15

writer = SummaryWriter('./logs')
parameters_to_train = []

encoder = models.resnet50(pretrained=True)
encoder.train()
encoder.to(device)
parameters_to_train += list(encoder.parameters())

decoder = DepthDecoder(np.array([64, 64*4, 128*4, 256*4, 512*4]), np.array([0, 1, 2, 3]))
decoder.train()
decoder.to(device)
parameters_to_train += list(decoder.parameters())

model_optimizer = optim.AdamW([{'params': encoder.parameters(), 'weight_decay': 1e-2},
                                   {'params': decoder.parameters(), 'weight_decay': 0}],
                                  lr=learning_rate, eps=adam_eps)
# args.gpu, learning_rate, end_learning_rate, adam_eps, num_epochs, weight_decay
args = Args()
dataloader = kittiDataLoader(args, 'train')

global epoch
epoch = 0
def train():
    global epoch
    while epoch < num_epochs:
        for step, sample_batched in enumerate(dataloader.data):
            model_optimizer.zero_grad()
            # before_op_time = time.time()
            image = torch.autograd.Variable(sample_batched['image'])
            image = image.to(device)
            depth_gt = torch.autograd.Variable(sample_batched['depth'])
            depth_gt = depth_gt.to(device)
            # tips!: don't use x.to(device), x=x.to(device) recommanded
            features = []
            # image = image
            # image = image.transpose(2,3)
            # print(image.shape)
            # x = (image - 0.45) / 0.225#have to valid
            # print(x.requires_grad)#
            image = encoder.conv1(image)
            # print(image.requires_grad)
            image = encoder.bn1(image)
            features.append(encoder.relu(image))
            features.append(encoder.layer1(encoder.maxpool(features[-1])))
            features.append(encoder.layer2(features[-1]))
            features.append(encoder.layer3(features[-1]))
            features.append(encoder.layer4(features[-1]))
        
            out = decoder.forward(features)
            depth_pred = out[("disp", 0)]
            # depth_est = model(image)
            mask = depth_gt > 1.0

            loss = {}
            silog_loss_call = silog_loss(0.85)
            loss["silog"] = silog_loss_call.forward(depth_pred, depth_gt, mask)
            # loss = silog_criterion.forward(depth_est, depth_gt, mask.to(torch.bool))
            loss["silog"].backward()
            writer.add_scalar('silog_loss', loss["silog"], epoch)
            # for param_group in optimizer.param_groups:
                # current_lr = (args.learning_rate - end_learning_rate) * (1 - global_step / num_total_steps) ** 0.9 + end_learning_rate
                # param_group['lr'] = current_lr
            model_optimizer.step()
            print('silog_loss： {:.12f}, step： {}'.format(loss["silog"], epoch))
            # print('[epoch][s/s_per_e/gs]: [{}][{}/{}/{}], lr: {:.12f}, loss: {:.12f}'.format(epoch, step, steps_per_epoch, global_step, current_lr, loss))
            epoch += 1


if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    train()    
    torch.save(encoder, './resnet_encoder.pkl')
    torch.save(decoder, './decoder.pkl')
    print('training done, models saved')    

