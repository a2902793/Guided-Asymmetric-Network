import logging
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.optim import lr_scheduler

import models.networks as networks
from .base_model import BaseModel

logger = logging.getLogger('base')


class DualSR_pretrain(BaseModel):
    def __init__(self, opt):
        super(DualSR_pretrain, self).__init__(opt)
        train_opt = opt['train']

        # define network and load pretrained models
        self.netG = networks.define_G(opt).to(self.device)
        self.load()


        if self.is_train:
            self.netG.train()

            # loss
            loss_type = train_opt['pixel_criterion']
            if loss_type == 'l1':
                self.cri_pix = nn.L1Loss().to(self.device) #計算output和target之差的絕對值，可選返回同維度的tensor或者是一個標量。
            elif loss_type == 'l2':
                self.cri_pix = nn.MSELoss().to(self.device) #計算output和target之差的平方，可選返回同維度的tensor或者是一個標量。
            else:
                raise NotImplementedError('Loss type [{:s}] is not recognized.'.format(loss_type))
            self.l_pix_w = train_opt['pixel_weight']

            # optimizers
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            optim_params = []
                                  # named_parameters() 是給出網絡層的名字和參數的迭代器
            for k, v in self.netG.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    logger.warning('Params [{:s}] will not optimize.'.format(k))
            self.optimizer_G = torch.optim.Adam(
                optim_params, lr=train_opt['lr_G'], weight_decay=wd_G)
            self.optimizers.append(self.optimizer_G)

            # schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                logger.info(f'Optimizers => {self.optimizers}')
                for optimizer in self.optimizers:
                    self.schedulers.append(lr_scheduler.MultiStepLR(optimizer, \
                        train_opt['lr_steps'], train_opt['lr_gamma']))
            else:
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

            self.log_dict = OrderedDict()
        # print network
        self.print_network()

    def feed_data(self, data, need_HR=True):
        self.var_L = data['LR'].to(self.device)  # LR
        if need_HR:
            self.real_H = data['HR'].to(self.device)  # HR

    # @torch.jit.script
    def optimize_parameters(self, step):
        self.optimizer_G.zero_grad()
        self.fake_Low, self.fake_high, self.fake_mask = self.netG(self.var_L)
        l_pix = self.l_pix_w * self.cri_pix(self.fake_Low, self.real_H) + self.l_pix_w * self.cri_pix(self.fake_high, self.real_H) + self.l_pix_w * self.cri_pix(self.fake_mask, self.real_H) 
        l_pix.backward()
        self.optimizer_G.step()

        # set log
        self.log_dict['l_pix'] = l_pix.item()

    def test(self):
        self.netG.eval()
        with torch.no_grad():
            self.fake_L, self.fake_H, self.fake_mask = self.netG(self.var_L)
        self.netG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_HR=True):
        out_dict = OrderedDict()
        out_dict['fake_LF'] = self.fake_L.detach()[0].float().cpu()
        out_dict['fake_HF'] = self.fake_H.detach()[0].float().cpu()
        out_dict['LR'] = self.var_L.detach()[0].float().cpu()
        out_dict['SR'] = self.fake_mask.detach()[0].float().cpu()
        if need_HR:
            out_dict['HR'] = self.real_H.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)

        logger.debug('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.debug(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.debug('Loading pretrained model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG)

    def save(self, iter_step):
        self.save_network(self.netG, 'G', iter_step)