from easydict import EasyDict as edict
import numpy as np

__C = edict()
cfg = __C

# Other options
__C.NET = 'NET_NAME'
__C.GPU_ID = '0'
__C.NUM_CLASS = 751
__C.DEBUG = False
__C.FILE_PATH = './'   # working directory

# Train options
__C.TRAIN = edict()
__C.TRAIN.imgs_path = 'bounding_box_train'
__C.TRAIN.pose_path = 'poses'
__C.TRAIN.idx_path = 'train_idx.txt'
__C.TRAIN.LR = 0.0002
__C.TRAIN.LR_DECAY = 10
__C.TRAIN.MAX_EPOCH = 20

__C.TRAIN.DISPLAY = 100
__C.TRAIN.BATCH_SIZE = 32
__C.TRAIN.NUM_WORKERS = 32

# GAN options
__C.TRAIN.ngf = 64
__C.TRAIN.ndf = 64
__C.TRAIN.num_resblock = 9
__C.TRAIN.lambda_idt = 10
__C.TRAIN.lambda_att = 1


# Test options
__C.TEST = edict()
__C.TEST.imgs_path = 'bounding_box_test'
__C.TEST.pose_path = 'poses_test'
__C.TEST.idx_path = 'test_idx.txt'
__C.TEST.BATCH_SIZE = 1
__C.TEST.GPU_ID = '0'
