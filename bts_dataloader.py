#MyDataLoader: top class for interfaces
#DataLoadPreprocess: preprocess images, including to_tensor(normalize included), flip, crop, gamma, brightness, etc.
#top of image data, has to deal with image file paths
#ToTensor: function class, implemented ToTensor operation
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.utils.data.distributed
from trochvision import transforms
from PIL import Image
import os
import random
# from distributed_sampler_no_evenly_divisible import *
def _is_pil_image(img):
    return isinstance(img, Image.Image)

def is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})

def preprocessing_transforms(mode):
    return transforms.Compose([ToTensor(mode = model)])##implemented below
class MyDataLoader(object):
    def __init__(self, args, mode):
        if mode == 'train':
            self.training_samples = DataLoadPreprocess(args, mode, transform=preprocessing_transforms(mode))
            if args.distributed:
                self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.training_samples)
            else:
                self.train_sampler = None
    
            self.data = DataLoader(self.training_samples, args.batch_size,
                                   shuffle=(self.train_sampler is None),
                                   num_workers=args.num_threads,
                                   pin_memory=True,
                                   sampler=self.train_sampler)
        elif mode == 'online_eval':
            #nothing todo
        elif mode == 'test':
            self.testing_samples = DataLoadPreprocess(args, mode, transfrom=preprocessing_transforms(mode))
            self.data = DataLoader(self.testing_samples, shuffle=False, num_workers=1)        
        else:
            print('mode should be one of \'train, test, online_eval\'. Got {}'.format(mode))

class DataLoadPreprocess(Dataset):
    def __init__(self, args, mode, transfrom=None, is_for_online_eval=False):
        self.args = args
        if mode =='online_eval':
            print('nothing to do with online_eval')
        else
            with open(args.filenames_file, 'r') as f:
                self.filenames = filenames_file.readlines()
        self.mode = mode
        self.transform = transform
        self.to_tensor = ToTensor#implemented below
        self.is_for_online_eval = is_for_online_eval

    def __getitem__(self, idx):
        sample_path = self.filenames[idx]
        focal = float(sample_path.split()[2])
        if self.mode == 'train':
            if self.args.dataset =='kitti' and self.args.use_right is True and random.random() > 0.5:#why>0.5
                image_path = os.path.join(self.args.data_path, "./", sample_path.split()[3])#why 3or4
                depth_path = os.path.join(self.args.gt_path, "./", sample_path.split()[4])
            else:
                image_path = os.path.join(self.args.data_path, "./", sample_path.split()[0])#why 3or4
                depth_path = os.path.join(self.args.gt_path, "./", sample_path.split()[1])
            image = Image.open(image_path)
            depth_gt = Image.open(gt_path)

            if self.args.do_kb_crop is True:
                height = image.height
                width = image.width
                top_margin = int(height - 352)#352? ans:In KITTI benchmark, the provided test images have been cropped to 352 x 1216, but train image size is 375*1242 
                left_margin = int((width - 1216)/2)#why divied by 2 ans:ensured the cropped image just in the center of source image
                depth_gt = depth_gt.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))
                image = image.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))

            if self.args.do_random_rotate is True:
                random_angle = (random.random() - 0.5) * 2 * self.args.degree
                image = self.rotate_image(image, random_angle)
                depth_gt = self.rotate_image(depth_gt, random_angle , flag=Imgae.NEAREST)

            image = np.asarray(image, dtype=np.float32) / 255.0
            depth_gt = np.asarray(depth_gt, dtype=np.float32)
            depth_gt = np.expand_dims(depth_gt, axis=2)#important
            
            depth_gt = depth_gt / 256.0

            image, depth_gt = self.random_crop(image, depth_gt, self.args.input_height, self.args.input_width)
            image, depth_gt = self.train_preprocess(image, depth_gt)
            sample = {'image': image, 'depth': depth_gt, 'focal': focal}
        else:
            print('test and online_eval is not supported yet.')

        if self.transform:
            sample = self.transform(sample)
        
        return sample
        #end of __getitem__ ,path image to np.ndarray

    def random_crop(self, img, depth, height, width):
        assert img.shape[0] >= height
        assert img.shape[1] >= width
        assert img.shape[0] == depth.shape[0]
        assert img.shape[1] == depth.shape[1]
        x = random.randint(0, img.shape[1] - width)
        y = random.randint(0, img.shape[0] - height)
        img = img[y:y+height , x:x+width, :]
        depth = depth[y:y+height ,x:x+width, :]#confirmed final input size is height*width
        return img, depth
        #np.ndarray to np.ndarray

    def train_preprocess(self, image, depth_gt):
        # do random flipping
        do_flip = random.random()
        if do_flip > 0.5:
            image = (image[:, ::-1, :]).copy()
            depth_gt = (depth_gt[:, ::-1, :]).copy()#flip left and right

        #do random gamma, brightness, color agumention
        do_agument = random.random()
        if do_agument > 0.5:
            image = self.agument_image(image)

        return image, depth_gt
        #np.ndarray to np.ndarray

    def agument_image(self, image):
        #gamma agumention
        gamma = random.uniform(0.9, 1.1)
        image_aug = image ** gamma
        #brightness agumention
        if self.args.dataset == 'nyu':
            brightness = random.uniform(0.75, 1.25)
        else:
            brightness = random.uniform(0.9, 1.1)

        image_aug = image_aug * brightness
        #color agumention
        colors = np.random.uniform(0.9, 1.1, size=3)
        white = np.ones((image.shape[0], image.shape[1]))
        color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
        image_aug *= color_image
        image_aug = np.clip(image_aug, 0, 1)
        return image_aug
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
        else???
        return img
        # np.ndarray to tensor for most of the time


