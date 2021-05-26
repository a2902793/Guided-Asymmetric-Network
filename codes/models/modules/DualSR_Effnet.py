import math
import torch
torch.manual_seed(17)
import torch.nn as nn
import torch.nn.utils.spectral_norm as SN
import torchvision
from torchvision import transforms
from . import block as B
from . import architecture

from torch.nn import functional as F
from models.modules.utils import round_filters, round_repeats, drop_connect, get_same_padding_conv2d, get_model_params, efficientnet_params, load_pretrained_weights, Swish, MemoryEfficientSwish, calculate_output_image_size



class DualSR_Effnet(nn.Module):
    #                      3       3  64     2         
    def __init__(self, in_nc, out_nc, nf, nb_e, 
            gc=32, upscale=4, norm_type=None, 
            act_type='leakyrelu', mode='CNA', upsample_mode='upconv',
            model_name=None):
        super(DualSR_Effnet, self).__init__()
        
        n_upscale = int(math.log(upscale, 2))
        if upscale == 3:
            n_upscale = 1

        fea_conv = B.conv_block(in_nc, 40, kernel_size=3, norm_type=None, act_type=None)
        blocks_concat = B.conv_block(80, 40, kernel_size=3, norm_type=None, act_type=None)

        mb_block_l0 = B.EfficientNet()
        mb_block_l1 = B.EfficientNet()
        mb_block_l2 = B.EfficientNet()
        mb_block_l3 = B.EfficientNet()
        mb_block_l4 = B.EfficientNet()
        mb_block_l5 = B.EfficientNet()

        mb_block_h0 = B.EfficientNet()
        mb_block_h1 = B.EfficientNet()
        mb_block_h2 = B.EfficientNet()
        mb_block_h3 = B.EfficientNet()
        mb_block_h4 = B.EfficientNet()
        mb_block_h5 = B.EfficientNet()
        
        mb_block_m0 = B.EfficientNet()
        mb_block_m1 = B.EfficientNet()
        mb_block_m2 = B.EfficientNet()
        mb_block_m3 = B.EfficientNet()
        mb_block_m4 = B.EfficientNet()
        mb_block_m5 = B.EfficientNet()
    
        self.first = B.sequential(fea_conv)
        self.body_l0 = B.sequential(mb_block_l0)
        self.body_l1 = B.sequential(mb_block_l1)
        self.body_l2 = B.sequential(mb_block_l2)
        self.body_l3 = B.sequential(mb_block_l3)
        self.body_l4 = B.sequential(mb_block_l4)
        self.body_l5 = B.sequential(mb_block_l5)

        self.body_h0 = B.sequential(mb_block_h0)
        self.body_h1 = B.sequential(mb_block_h1)
        self.body_h2 = B.sequential(mb_block_h2)
        self.concat = B.sequential(blocks_concat)
        self.body_h3 = B.sequential(mb_block_h3)
        self.body_h4 = B.sequential(mb_block_h4)
        self.body_h5 = B.sequential(mb_block_h5)

        self.body_m0= B.sequential(mb_block_m0)
        self.body_m1 = B.sequential(mb_block_m1)
        self.body_m2 = B.sequential(mb_block_m2)
        self.body_m3 = B.sequential(mb_block_m3)
        self.body_m4 = B.sequential(mb_block_m4)
        self.body_m5 = B.sequential(mb_block_m5)
        

        '''
        reconstruct_image 
        '''
        # self.concat = B.conv_block(152, 40, kernel_size=3, norm_type=None, act_type=None)
        
        self.LR_conv_l = B.conv_block(40, nf, kernel_size=3, norm_type=norm_type, act_type=None, mode=mode)

        self.LR_conv_h = B.conv_block(40, nf, kernel_size=3, norm_type=norm_type, act_type=None, mode=mode)

        self.LR_conv_M = B.conv_block(40, nf, kernel_size=3, norm_type=norm_type, act_type=None, mode=mode)


        if upsample_mode == 'upconv':
            upsample_block = B.upconv_blcok
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.pixelshuffle_block
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))
        if upscale == 3:
            self.upsampler_l = B.sequential(*upsample_block(nf, nf, 3, act_type=act_type))
            self.upsampler_h = B.sequential(*upsample_block(nf, nf, 3, act_type=act_type))
            self.upsampler_m = B.sequential(*upsample_block(nf, nf, 3, act_type=act_type))
        else:
            self.upsampler_l = B.sequential(*[upsample_block(nf, nf, act_type=act_type) for _ in range(n_upscale)])
            self.upsampler_h = B.sequential(*[upsample_block(nf, nf, act_type=act_type) for _ in range(n_upscale)])
            self.upsampler_m = B.sequential(*[upsample_block(nf, nf, act_type=act_type) for _ in range(n_upscale)])

        self.HR_conv0_l = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=act_type)
        self.HR_conv0_h = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=act_type)
        self.HR_conv0_m = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=act_type)

        self.HR_conv1_l = B.conv_block(nf, out_nc, kernel_size=3, norm_type=None, act_type=None)
        self.HR_conv1_h = B.conv_block(nf, out_nc, kernel_size=3, norm_type=None, act_type=None)
        self.HR_conv1_m = B.conv_block(nf, out_nc, kernel_size=3, norm_type=None, act_type=None)


    def forward(self, x): 
        x = self.first(x)
        # x = self._swish(self._bn0(self._conv_stem(x)))
        # low
        x_l_0 = self.body_l0(x)
        x_l_1 = self.body_l1(x_l_0)
        x_l_2 = self.body_l2(x_l_1)
        x_l_2 = x_l_2.mul(0.2) + x
        x_l_3 = self.body_l3(x_l_2)
        x_l_4 = self.body_l4(x_l_3)
        x_l_5 = self.body_l5(x_l_4)
        x_l_5 = x_l_5.mul(0.2) + x_l_2
        x_l = x_l_5 + x
        
        x_l = self.LR_conv_l(x_l)
        x_l = self.upsampler_l(x_l)
        x_fea_l = self.HR_conv0_l(x_l)
        x_l = self.HR_conv1_l(x_fea_l)

        # high
        x_h_0 = self.body_h0(x)
        x_h_1 = self.body_h1(x_h_0)
        x_h_2 = self.body_h2(x_h_1)
        x_h_2 = x_h_2.mul(0.2) + x
        x_h_2 = self.concat(torch.cat((x_l_5, x_h_2), 1))
        x_h_3 = self.body_h3(x_h_2)
        x_h_4 = self.body_h4(x_h_3)
        x_h_5 = self.body_h5(x_h_4)
        x_h_5 = x_h_5.mul(0.2) + x_h_2
        x_h = x_h_5 + x

        x_h = self.LR_conv_h(x_h)
        x_h = self.upsampler_h(x_h)
        x_fea_h = self.HR_conv0_h(x_h)
        x_h = self.HR_conv1_h(x_fea_h)

        # mask
        x_m_0 = self.body_m0(x)
        x_m_1 = self.body_m1(x_m_0)
        x_m_2 = self.body_m2(x_m_1)
        x_m_2 = x_m_2.mul(0.2) + x
        x_m_3 = self.body_m3(x_m_2)
        x_m_4 = self.body_m4(x_m_3)
        x_m_5 = self.body_m5(x_m_4)
        x_m_5 = x_m_5.mul(0.2) + x_m_2
        x_m = x_m_5 + x

        m = self.LR_conv_M(x_m)
        m = self.upsampler_m(m)
        m = self.HR_conv0_m(m)
        M_sigmoid = torch.sigmoid(m)
        combine = M_sigmoid.mul(x_fea_h) + (1 - M_sigmoid).mul(x_fea_l)

        combine = self.HR_conv1_m(combine)

        return x_l, x_h, combine

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'.format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))
