import os
import os.path as osp
import math
import time
import glob
import abc
from torch.utils.data import DataLoader
import torch.optim
import torchvision.transforms as transforms
import numpy as np
from config import cfg
from timer import Timer
from logger import colorlogger
from torch.nn.parallel.data_parallel import DataParallel
from model import get_model
import re

# dynamic dataset import
exec('from ' + cfg.dataset + ' import ' + cfg.dataset)

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

class Base(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, log_name='logs.txt'):
        
        self.cur_epoch = 0

        # timer
        self.tot_timer = Timer()
        self.gpu_timer = Timer()
        self.read_timer = Timer()

        # logger
        self.logger = colorlogger(cfg.log_dir, log_name=log_name)

    @abc.abstractmethod
    def _make_batch_generator(self):
        return

    @abc.abstractmethod
    def _make_model(self):
        return

class Trainer(Base):
    
    def __init__(self):
        super(Trainer, self).__init__(log_name = 'train_logs.txt')

    def get_optimizer(self, optimizable_params):
        optimizer = torch.optim.Adam(optimizable_params, lr=0.0, eps=1e-15)
        return optimizer
    
    def get_scheduler(self):
        scheduler = get_expon_lr_func(lr_init=cfg.position_lr_init*float(self.cam_dist['radius']),
                                                    lr_final=cfg.position_lr_final*float(self.cam_dist['radius']),
                                                    lr_delay_mult=cfg.position_lr_delay_mult,
                                                    max_steps=cfg.position_lr_max_steps)
        return scheduler

    def set_lr(self, cur_itr, tot_itr):
        for param_group in self.optimizer.param_groups:
            if param_group['name'] == 'mean_scene':
                lr = self.scheduler(cur_itr)
                param_group['lr'] = lr
            elif 'human' in param_group['name']:
                if (cur_itr > 0.75 * tot_itr) and (cur_itr <= 0.95 * tot_itr):
                    param_group['lr'] = cfg.lr / 10
                elif (cur_itr > 0.95 * tot_itr):
                    param_group['lr'] = cfg.lr / 100
            elif 'smplx' in param_group['name']:
                if (cur_itr > 0.75 * tot_itr) and (cur_itr <= 0.95 * tot_itr):
                    param_group['lr'] = cfg.smplx_param_lr / 10
                elif (cur_itr > 0.95 * tot_itr):
                    param_group['lr'] = cfg.smplx_param_lr / 100

    def _make_batch_generator(self):
        # data load and construct batch generator
        self.logger.info("Creating dataset...")
        trainset_loader = eval(cfg.dataset)(transforms.ToTensor(), 'train')
        self.itr_per_epoch = math.ceil(len(trainset_loader) / cfg.num_gpus / cfg.batch_size)
        self.batch_generator = DataLoader(dataset=trainset_loader, batch_size=cfg.num_gpus*cfg.batch_size, shuffle=True, num_workers=cfg.num_thread, pin_memory=True)
        self.scene = trainset_loader.scene
        self.cam_dist = trainset_loader.cam_dist
        self.smplx_params = trainset_loader.smplx_params

    def _make_model(self, epoch=None):
        if cfg.continue_train:
            ckpt = self.load_model()
            scene_point_num = ckpt['network']['scene_gaussian.point_num']
            model = get_model(None, None, scene_point_num, self.smplx_params)
            model = DataParallel(model).cuda()
            model.module.load_state_dict(ckpt['network'], strict=False)
        else:
            model = get_model(self.scene, self.cam_dist, None, self.smplx_params)
            model = DataParallel(model).cuda()
        optimizer = self.get_optimizer(model.module.optimizable_params)
        scheduler = self.get_scheduler()

        if cfg.continue_train:
            start_epoch = ckpt.get('epoch', 0)
            start_itr = ckpt.get('itr', 0)
            #optimizer.load_state_dict(ckpt['optimizer'], strict=False)
        else:
            start_epoch = 0
            start_itr = 0

        model.train()
        for module in model.module.eval_modules:
            module.eval()

        self.start_epoch = start_epoch
        self.start_itr = start_itr
        self.logger.info(f"Continue training from Epoch {self.start_epoch}, Itr {self.start_itr}")

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

    def save_model(self, state, epoch, itr):
        if itr == 0:
            file_path = osp.join(cfg.model_dir, f'snapshot_{epoch}.pth')
            torch.save(state, file_path)
            self.logger.info("Write snapshot into {}".format(file_path))
        else:
            file_path = osp.join(cfg.model_dir, f'snapshot_{epoch}_{itr}.pth')
            torch.save(state, file_path)
            self.logger.info("Write snapshot into {}".format(file_path))

    def load_model(self):
        # 定义损坏文件的大小阈值（50KB）
        corruption_threshold_bytes = 50 * 1024
        model_file_list = glob.glob(osp.join(cfg.model_dir,'*.pth'))
        for file_path in model_file_list:
            file_size = os.path.getsize(file_path)
            if file_size < corruption_threshold_bytes:
                self.logger.warning(f"Deleting corrupted model file: {file_path} (Size: {file_size / 1024:.2f}KB)")
                os.remove(file_path)
        model_file_list = glob.glob(osp.join(cfg.model_dir,'*.pth'))

        # 通过解析文件名中的 epoch 和 itr 来找到最新的文件
        def get_epoch_itr_from_name(filename):
            # 匹配 snapshot_{epoch}_{itr}.pth
            match = re.search(r'snapshot_(\d+)_(\d+)\.pth', os.path.basename(filename))
            if match:
                return int(match.group(1)), int(match.group(2))
            # 兼容旧格式 snapshot_{epoch}.pth
            match = re.search(r'snapshot_(\d+)\.pth', os.path.basename(filename))
            if match:
                return int(match.group(1)), 0 # 旧格式视为第0个itr
            return -1, -1 # 无法解析的文件名
    
        model_path = max(model_file_list , key=get_epoch_itr_from_name)
        self.logger.info('Load checkpoint from {}'.format(model_path))
        ckpt = torch.load(model_path, map_location='cpu')
        return ckpt

class Tester(Base):
    def __init__(self, test_epoch):
        super(Tester, self).__init__(log_name = 'test_logs.txt')
        self.test_epoch = int(test_epoch)

    def _make_batch_generator(self):
        # data load and construct batch generator
        self.logger.info("Creating dataset...")
        testset_loader = eval(cfg.dataset)(transforms.ToTensor(), 'test')
        batch_generator = DataLoader(dataset=testset_loader, batch_size=cfg.num_gpus*cfg.batch_size, shuffle=False, num_workers=cfg.num_thread, pin_memory=True)
        
        self.testset = testset_loader
        self.batch_generator = batch_generator
        self.smplx_params = testset_loader.smplx_params

    def _make_model(self):
        model_path = os.path.join(cfg.model_dir, 'snapshot_%d.pth' % self.test_epoch)
        assert os.path.exists(model_path), 'Cannot find model at ' + model_path
        self.logger.info('Load checkpoint from {}'.format(model_path))
        ckpt = torch.load(model_path)
        scene_point_num = ckpt['network']['scene_gaussian.point_num']
       
        # prepare network
        self.logger.info("Creating graph...")
        model = get_model(None, None, scene_point_num, self.smplx_params)
        model = DataParallel(model).cuda()
        model.module.load_state_dict(ckpt['network'], strict=False)
        model.eval()

        self.model = model

