import os
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import json
from torch import zeros, zeros_like

class UnalignedFlagDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size, use_flag=True))   # load (images, flags) from '/path/to/data/trainA', this is noisy dataset
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size, use_flag=False))   # load (images) '/path/to/data/trainB', this is clean dataset
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (tuple(str, str)) -- image paths, flag paths, tuple
            B_paths (tuple(str, str)) -- image paths, flag paths, tuple
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        A_img = Image.open(A_path[0]).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        A_flag_dict = self.get_flag_from_file(A_path[1])
        A_flag = self.dict2tensor(A_flag_dict)
        B_flag = zeros_like(A_flag) # A represents clean images without any noisy type
        # apply image transformation
        A = self.transform_A(A_img)
        B = self.transform_B(B_img)

        return {'A': A, 'B': B, 'A_paths': A_path[0], 'B_paths': B_path[0], 'A_flags': A_flag, 'B_flags': B_flag}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)

    def get_flag_from_file(self, dir):
        with open(dir) as log:
            return json.load(log)["flags"]

    def dict2tensor(self, flag_dict):
        flag_tensor = zeros(len(flag_dict))
        key2idx = {key: i for i, key in enumerate(flag_dict.keys())}
        for key, val in flag_dict.items():
            idx = key2idx[key]
            flag_tensor[idx] = 1 if val else 0
        return flag_tensor