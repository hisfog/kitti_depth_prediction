from PIL import Image
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import random

from image_utils import image_augment
def _is_pil_image(img):
    return isinstance(img, Image.Image)

def is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})

def preprocessing_transforms(mode):
    return transforms.Compose([ToTensor(mode = model)])##implemented below

class kittiDataLoader(object):
    def __init__(self, args, mode):
        if mode == 'train':
            self.training_samples = DataLoadPreprocess(args)

class DataLoadPreprocess(Dataset):
    def __init__(self, args):
        self.args = args
        with open(args.filenames_file, 'r') as f:
            self.filenames = filenames_file.readlines()
        self.transform = preprocessing_transforms(mode='train')
        self.to_tensor = ToTensor#implemented below


    def __getitem__(self, idx):
        sample_path = self.filenames[idx]
        focal = float(sample_path.split()[2])
        if(random.random() > 0.5):
            image_path = os.path.join(self.args.data_path, "./", sample_path.split()[3])
            depth_path = os.path.join(self.args.gt_path, "./", sample_path.split()[4])
        else:
            image_path = os.path.join(self.args.data_path, "./", sample_path.split()[0])#why 3or4
            depth_path = os.path.join(self.args.gt_path, "./", sample_path.split()[1])
        image = Image.open(image_path)
        depth_gt = Image.open(depth_path)

        if self.args.do_kb_crop is True:
            height = image.height
            width = image.width
            top_margin = int(height - 352)#352? ans:In KITTI benchmark, the provided test images have been cropped to 352 x 1216, but train image size is 375*1242 
            left_margin = int((width - 1216)/2)#why divied by 2 ans:ensured the cropped image just in the center of source image
            depth_gt = depth_gt.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))
            image = image.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))

        if self.args.do_random_rotate is True:
            image = image_augment.rotate_image(image, random_angle)
            depth_gt = image_augment.rotate_image(depth_gt, random_angle , flag=Imgae.NEAREST)
        # convert from PIL.Image to np.ndarray
        image = np.asarray(image, dtype=np.float32) / 255.0
        depth_gt = np.asarray(depth_gt, dtype=np.float32)
        depth_gt = np.expand_dims(depth_gt, axis=2)#important
            
        depth_gt = depth_gt / 256.0
        image, depth_gt = image_augment.random_crop(image, depth_gt, self.args.input_height, self.args.input_width)
        image, depth_gt = self.train_preprocess(image, depth_gt)
        sample = {'image': image, 'depth': depth_gt, 'focal': focal}

        if self.transform:
            sample = self.transform(sample)
        return sample
    def train_preprocess(self, image, depth_gt):
        # do random flipping
        image = image_augment.random_flip(image)
        depth_gt = image_augment.random_flip(depth_gt)
        #do random gamma, brightness, color agumention
        image = image_augment.gamma_augment(image)
        image = image_augment.brightness_augment(image)
        image = image_augment.color_augment(image)

        return image, depth_gt
        #np.ndarray to np.ndarray

    def __len__():
        return len(self.filenames)



class ToTensor(object):
    #function class
    def __init__(self, mode):
        self.mode = mode
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])##imagenet paramaters
        ## why these values?
    
    def __call__(self, sample):
        image, focal = sample['image'], sample['focal']
        image = self.to_tensor(image)
        image = self.normalize(image)

        if self.mode =='test':
            return {'image':image, 'focal':focal}

        depth = sample['depth']
        if self.mode == 'train':
            depth = self.to_tensor(depth)
            return {'image':image, 'depth':depth, 'focal':focal}
        else:
            has_valid_depth = sample['has_valid_depth']
            return {'image':image, 'depth':depth, 'focal':focal, 'has_valid_depth':has_valid_depth}

    def to_tensor(self, pic):
        if not (_is_pil_image(pic) or _is_numpy_image(pic)):
            raise TypeError(
                'pic should be PIL image or ndarray. Got {}'.format(type(pic)))
        ##for ndarray
        if isinstance(pic, np.ndarray):
            img = torch.from_numpy()
            return img
        ##for PIL image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.form_numpy(np.array(pic, np.int16, copy=False))##create a tensor
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))

        if pic.mode == 'YCbCr':##color space
            nchannel = 3
        elif pic.mode =='I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)

        img = img.view(pic.size[1], pic.size[0], nchannel)##change dims

        img = img.transpose(0, 1).transpose(0, 2).contiguous()#

        if isinstance(img, torch.ByteTensor):
            return img.float()
        else:
            return img
        # np.ndarray to tensor for most of the time


