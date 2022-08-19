"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
import torch
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Iterator
from options.test_options import TestOptions
from data import create_dataset
from torchvision.transforms import ToTensor
from models import create_model
from PIL import Image
from util.util import tensor2im
from time import time


def image_segment_extractor(img_src: Image.Image, patch_size: int, stride: int) -> Iterator[Image.Image]:
    """extract **square** patches from original image with certain size and stride
    Params:
    @@img_src: PIL.Image.Image
    @@patch_size: int
    @@stride: int
    
    Yield:
    tuple: (img_src: Image.Image, remaining_num: int)"""

    # Correct the image size so that both width and height are divisible by patch_size
    width, height = img_src.size
    correct_width = width - (width - patch_size) % stride
    correct_height = height - (height - patch_size) % stride
    img_src.resize((correct_width, correct_height))

    wid_num = (correct_width - patch_size) // stride + 1
    hei_num = (correct_height - patch_size) // stride + 1

    # Create the image cropping bounding box list
    box_list = []
    for hei_i in range(hei_num):
        for wid_i in range(wid_num):
            box=(wid_i*stride, hei_i*stride, wid_i*stride+patch_size, hei_i*stride+patch_size)
            box_list.append(box)
    num_patches = wid_num*hei_num
    print(f"\nImage has totolly {num_patches} patches...\nExtracting...\n")

    # Crop the image and yield
    for _, box in enumerate(box_list):
        yield img_src.crop(box)

def get_image_metadata(img_src: Image.Image, patch_size=256, stride=128) -> dict:
    # Correct the image size so that both width and height are divisible by patch_size
    width, height = img_src.size
    correct_width = width - (width - patch_size) % stride
    correct_height = height - (height - patch_size) % stride

    return {'width': correct_width, 'height': correct_height, 'patch_size': patch_size, 'stride': stride}

def rebuild_image(img_seg_list, width, height, patch_size, stride) -> Image.Image:
    """Reference: https://zhuanlan.zhihu.com/p/281404684"""
    assert (width - patch_size) % stride == 0 and (height - patch_size) % stride == 0
    # number of patches for x axis
    n_patches_x = (width - patch_size) // stride + 1
    # number of patches for y axis
    n_patches_y = (height - patch_size) // stride + 1
    
    assert n_patches_y*n_patches_x == len(img_seg_list)
    # initial a np.narray image
    img = np.zeros((height, width))
    img_weight = np.zeros_like(img)
    patch_index = 0
    
    for i in range(n_patches_y):
        for j in range(n_patches_x):
            img_seg = Image.fromarray(img_seg_list[patch_index]).convert('L')
            img_seg_array = np.asarray(img_seg)
            x1 = i * stride
            x2 = x1 + patch_size
            y1 = j * stride
            y2 = y1 + patch_size
            img[x1:x2, y1:y2] += img_seg_array
            img_weight[x1:x2, y1:y2] += 1

            patch_index += 1

    img /= img_weight # average the overlap parts
    return Image.fromarray(img)


def run_inference(opt, dataset):
    timestamp_model_setup = time()
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    model.eval()

    # create a result dir
    result_dir = os.path.join(opt.results_dir, opt.name) # ./results/moe_17_aug/
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    # instantiate a Image.Image -> Tensor transform class
    transform_img2tensor = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))])

    for i, data in enumerate(dataset): # each iter generate a full image. It need to be segmented.
        TIMESTAMP_img_seg = time()
        img = Image.open(data['A_paths'][0]).convert('L')
        img_seg_iter = image_segment_extractor(img, opt.load_size, opt.stride)
        metadata = get_image_metadata(img, opt.load_size, opt.stride)
        prc_img_seg_list = []
        print(f'Processing image {data["A_paths"]} ...')
        TIMESTAMP_seg_proc = time()
        seg_num = 0
        for k, img_seg in enumerate(img_seg_iter):
            model.set_input(transform_img2tensor(img_seg)[None,:,:,:])  # unpack data from data loader
            # TIMESTAMP_single_inference = time()
            result_tensor = model.forward()
            # print('Single image segment inference time', time()-TIMESTAMP_single_inference)
            result_img_seg = tensor2im(result_tensor)
            prc_img_seg_list.append(result_img_seg) # run inference
            seg_num += 1
        TIMESTAMP_img_rebuild = time()
        
        result_img = rebuild_image(prc_img_seg_list, **metadata)
        result_path = os.path.join(result_dir, os.path.split(data['A_paths'][0])[-1])
        result_img.convert('L').save(result_path)
        TIMESTAMP_END = time()
        print(f'Successfully process the image\
            \nSegmentation step {TIMESTAMP_seg_proc-TIMESTAMP_img_seg:.4f}\
            \nModel inference time {TIMESTAMP_img_rebuild-TIMESTAMP_seg_proc:.4f}\
            \nPer seg inference {(TIMESTAMP_img_rebuild-TIMESTAMP_seg_proc)/seg_num:.4f}\
            \nImage rebuild time {TIMESTAMP_END-TIMESTAMP_img_rebuild:.4f}')

def mp_datasets(opt, nprocs):
    dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
    result_dir = os.path.join(opt.results_dir, opt.name)
    # if image exists in result dir, no need to process
    exists_img = os.listdir(result_dir)
    paths = sorted([os.path.join(dir_A,img_dir) for img_dir in os.listdir(dir_A) if not img_dir in exists_img])
    print('Totally ',len(paths), ' images to be processed')
    dataset_size = len(paths)

    dataset_part_size = dataset_size // nprocs + 1
    dataset_parts = []
    for i in range(nprocs):
        if i == nprocs-1:
            dataset_parts.append(imgDataset(paths[i*dataset_part_size:]))
            break
        dataset_parts.append(imgDataset(paths[i*dataset_part_size: (i+1) * dataset_part_size]))
    return dataset_parts

class imgDataset(Dataset):
    def __init__(self, img_dir_list) -> None:
        super().__init__()

        self.A_paths = img_dir_list
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.transfrom_A = ToTensor()

    def __getitem__(self, index):
        """Return a data point and its metadata information.
        Parameters:
            index (int)      -- a random integer for data indexing
        Returns a dictionary that contains A and A_paths
            A (tensor)       -- an image in the input domain
            A_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        A_img = Image.open(A_path).convert('L')
        # no need to transform, the segment function need Image.Image for input
        A = self.transfrom_A(A_img)

        return {'A': A, 'A_paths': A_path}

    def __len__(self):
        return self.A_size


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    opt.stride = 128

    # timestamp_model_setup = time()
    # dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    # model = create_model(opt)      # create a model given opt.model and other options
    # model.setup(opt)               # regular setup: load and print networks; create schedulers
    # model.eval()

    ########################################################

    # create a result dir
    result_dir = os.path.join(opt.results_dir, opt.name) # ./results/moe_17_aug/
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    import multiprocessing as mp
    nprocs = 1
    dataset_parts = mp_datasets(opt, nprocs)  # create a dataset given opt.dataset_mode and other options
    mp.set_start_method('spawn') # torch multiprocessing acquirement
    pool=[]
    for i, dataset in enumerate(dataset_parts):
        process = mp.Process(target=run_inference, args=(opt, DataLoader(dataset, opt.batch_size, False, num_workers=opt.num_threads)))
        process.start()
        pool.append(process)
    for p in pool:
        p.join()
    ########################################################

    # # create a result dir
    # result_dir = os.path.join(opt.results_dir, opt.name) # ./results/moe_17_aug/
    # # instantiate a Image.Image -> Tensor transform class
    # transform_img2tensor = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))])

    # for i, data in enumerate(dataset): # each iter generate a full image. It need to be segmented.
    #     TIMESTAMP_img_seg = time()
    #     img = Image.open(data['A_paths'][0])
    #     img_seg_iter = image_segment_extractor(img, opt.load_size, opt.stride)
    #     metadata = get_image_metadata(img, opt.load_size, opt.stride)
    #     prc_img_seg_list = []
    #     print(f'Processing image {data["A_paths"]} ...')
    #     TIMESTAMP_seg_proc = time()
    #     seg_num = 0
    #     for k, img_seg in enumerate(img_seg_iter):
    #         model.set_input(transform_img2tensor(img_seg)[None,:,:,:])  # unpack data from data loader
    #         TIMESTAMP_single_inference = time()
    #         result_tensor = model.forward()
    #         print('Single image segment inference time', time()-TIMESTAMP_single_inference)
    #         result_img_seg = tensor2im(result_tensor)
    #         prc_img_seg_list.append(result_img_seg) # run inference
    #         seg_num += 1
    #     TIMESTAMP_img_rebuild = time()
        
    #     result_img = rebuild_image(prc_img_seg_list, **metadata)
    #     result_path = os.path.join(result_dir, os.path.split(data['A_paths'][0])[-1])
    #     result_img.convert('L').save(result_path)
    #     TIMESTAMP_END = time()
    #     print(f'Successfully process the image\
    #         \nSegmentation step {TIMESTAMP_seg_proc-TIMESTAMP_img_seg:.4f}\
    #         \nModel inference time {TIMESTAMP_img_rebuild-TIMESTAMP_seg_proc:.4f}\
    #         \nPer seg inference {(TIMESTAMP_img_rebuild-TIMESTAMP_seg_proc)/seg_num:.4f}\
    #         \nImage rebuild time {TIMESTAMP_END-TIMESTAMP_img_rebuild:.4f}')
