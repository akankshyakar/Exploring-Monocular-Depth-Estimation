import os
import os.path
import torch
import numpy as np
import torch.utils.data as data
import h5py
import dataloader.transforms as transforms
from path import Path
import random
from dataloader import custom_transforms

IMG_EXTENSIONS = ['.h5',]

def get_intrinsics():
    fx_rgb = 5.1885790117450188e+02
    fy_rgb = 5.1946961112127485e+02
    cx_rgb = 3.2558244941119034e+02
    cy_rgb = 2.5373616633400465e+02

    return np.array([
        [fx_rgb, 0, cx_rgb],
        [0, fy_rgb, cy_rgb],
        [0, 0, 1],
    ], dtype=float)

# intrinsics_global = get_intrinsics()

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def make_dataset(dir, class_to_idx):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue
        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)
    return images

def h5_loader(path, image_only=False):
    h5f = h5py.File(path, "r")
    rgb = np.array(h5f['rgb'])
    rgb = np.transpose(rgb, (1, 2, 0))
    depth = np.array(h5f['depth'])
    if not image_only:
        return rgb, depth
    else:
        return rgb

# def rgb2grayscale(rgb):
#     return rgb[:,:,0] * 0.2989 + rgb[:,:,1] * 0.587 + rgb[:,:,2] * 0.114

to_tensor = transforms.ToTensor()

class MyDataloader(data.Dataset):
    modality_names = ['rgb', 'rgbd', 'd'] # , 'g', 'gd'
    color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4)

    def __init__(self, root, type, sparsifier=None, modality='rgb', loader=h5_loader, sequence_length=3):
        classes, class_to_idx = find_classes(root)
        imgs = make_dataset(root, class_to_idx)
        assert len(imgs)>0, "Found 0 images in subfolders of: " + root + "\n"
        print("Found {} images in {} folder.".format(len(imgs), type))
        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx

        normalize = custom_transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                        std=[0.5, 0.5, 0.5])
        train_transform = custom_transforms.Compose([
            custom_transforms.RandomHorizontalFlip(),
            custom_transforms.RandomScaleCrop(),
            custom_transforms.ArrayToTensor(),
            normalize
        ])
        valid_transform = custom_transforms.Compose([custom_transforms.ArrayToTensor(), normalize])
        
        if type == 'train':
            self.transform = train_transform
        elif type == 'val':
            self.transform = valid_transform
        else:
            raise (RuntimeError("Invalid dataset type: " + type + "\n"
                                "Supported dataset types are: train, val"))
        self.loader = loader
        self.sparsifier = sparsifier

        # assert (modality in self.modality_names), "Invalid modality type: " + modality + "\n" + \
        #                         "Supported dataset types are: " + ''.join(self.modality_names)
        # self.modality = modality

        self.scenes = os.listdir(self.root)
        self.scenes = [os.path.join(self.root, scene) for scene in self.scenes]
        self.scenes = [Path(scene) for scene in self.scenes]
        self.crawl_folders(sequence_length)
        self.get_intrinsics = get_intrinsics
        # import pdb; pdb.set_trace()


    def crawl_folders(self, sequence_length):
        sequence_set = []
        demi_length = (sequence_length-1)//2
        shifts = list(range(-demi_length, demi_length + 1))
        shifts.pop(demi_length)
        for scene in self.scenes:
            # Intrinsics added in train function itself
            # intrinsics = np.genfromtxt(scene/'cam.txt').astype(np.float32).reshape((3, 3))
            intrinsics = get_intrinsics()
            intrinsics = torch.tensor(intrinsics, dtype=torch.float32)
            imgs = sorted(scene.files('*.h5'))
            if len(imgs) < sequence_length:
                continue
            for i in range(demi_length, len(imgs)-demi_length):
                sample = {'intrinsics': intrinsics, 'tgt': imgs[i], 'ref_imgs': []}
                # sample = {'tgt': imgs[i], 'ref_imgs': []}
                for j in shifts:
                    sample['ref_imgs'].append(imgs[i+j])
                sequence_set.append(sample)
        random.shuffle(sequence_set)
        self.samples = sequence_set


    def train_transform(self, rgb, depth):
        raise (RuntimeError("train_transform() is not implemented. "))

    # def train_transform(self, rgb, ref_imgs, depth):
    #     raise (RuntimeError("train_transform() is not implemented. "))

    def val_transform(rgb, depth):
        raise (RuntimeError("val_transform() is not implemented."))

    def create_sparse_depth(self, rgb, depth):
        if self.sparsifier is None:
            return depth
        else:
            mask_keep = self.sparsifier.dense_to_sparse(rgb, depth)
            sparse_depth = np.zeros(depth.shape)
            sparse_depth[mask_keep] = depth[mask_keep]
            return sparse_depth

    def create_rgbd(self, rgb, depth):
        sparse_depth = self.create_sparse_depth(rgb, depth)
        rgbd = np.append(rgb, np.expand_dims(sparse_depth, axis=2), axis=2)
        return rgbd

    def __getitem__(self, index):
        # rgb, depth = self.__getraw__(index)
        # rgb, ref_imgs, depth = self.__getraw__(index)
        sample = self.samples[index]
        rgb, depth = self.loader(sample['tgt'])
        ref_imgs = [self.loader(ref_img, image_only=True) for ref_img in sample['ref_imgs']]
        
        if self.transform is not None:
            imgs, depth_tensor, intrinsics = self.transform([rgb] + ref_imgs, depth, np.copy(sample['intrinsics']))
            rgb_tensor = imgs[0]
            ref_imgs_tensor = imgs[1:]
        else:
            # intrinsics = np.copy(sample['intrinsics'])
            raise(RuntimeError("transform not defined"))


        return rgb_tensor, ref_imgs_tensor, depth_tensor.unsqueeze(0), intrinsics


    def __len__(self):
        return len(self.samples)
