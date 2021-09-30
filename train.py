import torch
from torch import nn
from torchvision import models, transforms
from torchvision.utils import make_grid
import torch.optim as optim
import numpy as np
from tensorboardX import SummaryWriter

from PIL import Image
import json
from depth_decoder import *

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


learning_rate = 1e-2
adam_eps = 1e-3
scheduler_step_size = 15
writer = SummaryWriter('./logs')

parameters_to_train = []
encoder = models.resnet101(pretrained=True)
# print(encoder)
encoder.to(device)
encoder.train()
parameters_to_train += list(encoder.parameters())

decoder = DepthDecoder(np.array([64, 64*4, 128*4, 256*4, 512*4]), np.array([0, 1, 2, 3]))
decoder.train()
decoder.to(device)
parameters_to_train += list(decoder.parameters())
model_optimizer = optim.AdamW([{'params': encoder.parameters(), 'weight_decay': 1e-2},
                                   {'params': decoder.parameters(), 'weight_decay': 0}],
                                  lr=learning_rate, eps=adam_eps)
# model_optimizer = optim.Adam(parameters_to_train, learning_rate)
# model_lr_scheduler = optim.lr_scheduler.StepLR(model_optimizer, scheduler_step_size, 0.1)

# transformation
tf = transforms.Compose([transforms.Resize((1216, 352)),
                         transforms.ToTensor(),
                         #transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                         ])
tf_gt = transforms.Compose([transforms.Resize((1216, 352)),
                            transforms.ToTensor()])
global iter
iter = 0

def run_one_epoch(path_image, path_gt):
    global iter
    model_optimizer.zero_grad()

    image = Image.open(path_image)
    # print(image.size, 'image_shape')
    gt = Image.open(path_gt)
    img = tf(image)
    img = img.to(device)
    # print(img.shape)#test
    img = img.unsqueeze(0)
    # print(img.shape)#test

    ground_truth = tf_gt(gt)
    ground_truth = ground_truth.unsqueeze(0)
    ground_truth = ground_truth.to(device)
    # saveImg = tensor_to_PIL(img)
    # saveImg.save('img_src'+str(iter)+'.jpg')

    features = []
    x = (img - 0.45) / 0.225#have to valid
    # x.requires_grad_(True)#
    # print(x.requires_grad)#
    x = encoder.conv1(x)
    # print(x.requires_grad)
    x = encoder.bn1(x)
    features.append(encoder.relu(x))
    features.append(encoder.layer1(encoder.maxpool(features[-1])))
    features.append(encoder.layer2(features[-1]))
    features.append(encoder.layer3(features[-1]))
    features.append(encoder.layer4(features[-1]))

    out = decoder.forward(features)
    depth_pred = out[("disp", 0)]
    # print(depth_pred.requires_grad)#true
    # print(depth_pred.shape)
    # mean_disp = depth_pred.mean(2, True).mean(3, True)
    # print(mean_disp.shape)
    losses = {}
    # losses["loss"] = 1000 * get_smooth_loss(depth_pred, img)
    # print('smooth_loss = {}'.format(losses["loss"]), end=" ")
    # losses["loss"].backward()   # smooth_loss backward   

    # depth_pred = torch.clamp(F.interpolate(depth_pred, [1216, 352], mode="bilinear", align_corners=False), 1e-3, 80)
    # depth_pred = depth_pred.detach()
    # print(depth_pred.requires_grad, "important")
    
    depth_gt = ground_truth
    mask = depth_gt > 0
    
    # garg/eigen crop
    # crop_mask = torch.zeros_like(mask)
    # crop_mask[:, :, 153:371, 44:1197] = 1
    # mask = mask * crop_mask
    
    ## depth_gt = depth_gt[mask]
    ## depth_pred = depth_pred[mask]
    # depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)
    ## depth_pred = depth_pred * torch.median(depth_gt) / torch.median(depth_pred)
    ## depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)
    # print(depth_pred.requires_grad,"###")
    # losses["error_g"] = torch.zeros([1, 1], dtype=torch.float, requires_grad = True)
    # losses["error"] = 10*compute_depth_errors(depth_gt, depth_pred)
    silog_loss_call = silog_loss(0.85)
    losses["silog"] = silog_loss_call.forward(depth_pred, depth_gt, mask)
    # losses["loss"] = (losses["error"][1]).requires_grad_()
    # losses["loss"] = losses["loss"].clone() + losses["error"][0]
    # losses["error_g"] = losses["error"][0]
    # losses["error_g"].backward()
    # print(losses["error"].requires_grad, "@@@@")
    # losses["error"].requires_grad_()
    losses["silog"].backward()
    writer.add_scalar('silog_loss', losses["silog"], iter)

    #losses["error"][0].requires_grad_().backward()  # pc_loss backward
    model_optimizer.step()
    # model_lr_scheduler.step()
    iter += 1

    print('silog_loss = {}'.format(losses["silog"]), end=" ")
    # for idx, item in enumerate(losses["error"]):
        # print(item)

if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    data_file = open('train_with_gt_shuffled.txt')
    line = data_file.readline()
    while line :
        train_ = line.split()[0]
        groundtruth_ = line.split()[1]
        run_one_epoch(train_.strip('\n'), groundtruth_.strip('\n'))
        print('step {} done'.format(int(iter)))
        line = data_file.readline()

    data_file.close()    
    torch.save(encoder, './resnet_encoder.pkl')
    torch.save(decoder, './decoder.pkl')
    print('training done, models saved')    




# def tensor_to_PIL(tensor):
#     unloader = transforms.ToPILImage()
#     image = tensor.cpu().clone()
#     image = image.squeeze(0)
#     image = unloader(image)
#     return image

# def save_img(tensor, name):
#     tensor = tensor.permute((1, 0, 2, 3))
#     im = make_grid(tensor, normalize=True, scale_each=True, nrow=8, padding=2).permute((1, 2, 0))
#     im = (im.data.numpy() * 255.).astype(np.uint8)
#     Image.fromarray(im).save(name + '.jpg')
# if __name__ == '__main__':
#     torch.autograd.set_detect_anomaly(True)
#     train_file = open("train_shuffled.txt")
#     train_ = train_file.readline()
#     groundtruth_file = open("groundtruth.txt")
#     groundtruth_ = groundtruth_file.readline()

#     while train_ and groundtruth_:
#         run_one_epoch(train_.strip('\n'), groundtruth_.strip('\n'))
#         print('step {} done'.format(int(iter)))
#         train_ = train_file.readline()
#         groundtruth_ = groundtruth_file.readline()


#     train_file.close()
#     groundtruth_file.close() 
    
#     torch.save(model, './resnet_encoder.pkl')
#     torch.save(decoder, './decoder.pkl')
#     print('training done, models saved')    




    
    # run_one_epoch('D:\\1work\\KITTI\\data\\2011_09_26_drive_0014_sync\\image_02\\data\\0000000005.png', 'D:\\1work\\KITTI\\data\\2011_09_26_drive_0014_sync\\proj_depth\\groundtruth\\image_02\\0000000005.png')
    # print('step {} done'.format(int(iter)))
    # run_one_epoch('D:\\1work\\KITTI\\data\\2011_09_26_drive_0014_sync\\image_02\\data\\0000000006.png', 'D:\\1work\\KITTI\\data\\2011_09_26_drive_0014_sync\\proj_depth\\groundtruth\\image_02\\0000000006.png')
    # print('step {} done'.format(int(iter)))
# # image = Image.open('000003.png')
# image = Image.open('D:\\1work\\KITTI\\data\\2011_09_26_drive_0014_sync\\image_02\\data\\0000000005.png')
# # gt = Image.open('D:\\1work\\KITTI\\data\\2011_09_26_drive_0014_sync\\image_02\\data\\0000000005.png')

# gt = Image.open('D:\\1work\\KITTI\\data\\2011_09_26_drive_0014_sync\\proj_depth\\groundtruth\\image_02\\0000000005.png')
# # image = Image.open('assets/1.jpg')

# ground_truth = tf_gt(gt)
# img = tf(image)
# img = img.unsqueeze(0)
# ground_truth = ground_truth.unsqueeze(0)


# saveImg = tensor_to_PIL(img)
# saveImg.save('img_src.jpg')
# features = []
# y = img
# x = img
# #x = torch.cat([x, y])
# print(x.shape)
# x = (img - 0.45) / 0.225
# save_img(x, 'img_norm')

# x = model.conv1(x)
# x = model.bn1(x)
# features.append(model.relu(x))
# features.append(model.layer1(model.maxpool(features[-1])))
# features.append(model.layer2(features[-1]))
# features.append(model.layer3(features[-1]))
# features.append(model.layer4(features[-1]))
# print(len(features))
# print(features[0].shape)
# decoder = DepthDecoder(np.array([64, 64, 128, 256, 512]), np.array([0, 1, 2, 3]))
# out = decoder.forward(features)
# for i in range(3, -1, -1):
#     print(out[("disp", i)].shape)
#     save_img(out[("disp", i)], str(i) + '.jpg')

# depth_pred = out[("disp", 0)]
# print(depth_pred.shape)
# mean_disp = depth_pred.mean(2, True).mean(3, True)
# print(mean_disp.shape)

# losses = {}
# losses["loss"] = get_smooth_loss(depth_pred, img)
# print('loss = {}'.format(losses["loss"]))
# losses["loss"].backward()

# depth_pred = torch.clamp(F.interpolate(
# depth_pred, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80)
# depth_pred = depth_pred.detach()

# depth_gt = ground_truth
# mask = depth_gt > 0

# # garg/eigen crop
# crop_mask = torch.zeros_like(mask)
# crop_mask[:, :, 153:371, 44:1197] = 1
# mask = mask * crop_mask

# depth_gt = depth_gt[mask]
# depth_pred = depth_pred[mask]
# depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

# depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

# losses["error"] = compute_depth_errors(depth_gt, depth_pred)

# for idx, item in enumerate(losses["error"]):
#     print(item)

