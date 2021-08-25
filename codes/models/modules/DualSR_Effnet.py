import math
import torch
torch.manual_seed(17)
import torch.nn as nn
from . import block as B

class DualSR_Effnet(nn.Module):      
    def __init__(self, low_layers, mask_layers, high_layers,
            in_nc=3, out_nc=3, nf=64, upscale=4, norm_type=None, act_type='leakyrelu', mode='CNA', upsample_mode='upconv'):
        super(DualSR_Effnet, self).__init__()
        
        n_upscale = int(math.log(upscale, 2))
        if upscale == 3:
            n_upscale = 1

        fea_conv = B.conv_block(in_nc, 40, kernel_size=3, norm_type=None, act_type=None)
        blocks_concat = B.conv_block(80, 40, kernel_size=3, norm_type=None, act_type=None)

        #-----------------------------------------------------------------------------
        self.shared_conv = B.sequential(fea_conv)                               
        #-----------------------------------------------------------------------------
        low_mb_front = [B.Dense_MB() for _ in range(low_layers//2)]
        self.low_dmb_front = B.sequential(*low_mb_front)
        low_mb_back = [B.Dense_MB() for _ in range(low_layers//2)]
        self.low_dmb_back = B.sequential(*low_mb_back)
        #-----------------------------------------------------------------------------
        mask_mb_front = [B.Dense_MB() for _ in range(mask_layers//2)]
        self.mask_dmb_front = B.sequential(*mask_mb_front)
        mask_mb_back = [B.Dense_MB() for _ in range(mask_layers//2)]
        self.mask_dmb_back = B.sequential(*mask_mb_back)
        #-----------------------------------------------------------------------------
        high_mb_front = [B.Dense_MB() for _ in range(high_layers//2)]
        self.high_dmb_front = B.sequential(*high_mb_front)
        self.concat = B.sequential(blocks_concat)
        high_mb_back = [B.Dense_MB() for _ in range(high_layers//2)]
        self.high_dmb_back = B.sequential(*high_mb_back)
        #-----------------------------------------------------------------------------

        self.LR_conv_l = B.conv_block(40, nf, kernel_size=3, norm_type=norm_type, act_type=None, mode=mode)
        self.LR_conv_m = B.conv_block(40, nf, kernel_size=3, norm_type=norm_type, act_type=None, mode=mode)
        self.LR_conv_h = B.conv_block(40, nf, kernel_size=3, norm_type=norm_type, act_type=None, mode=mode)

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

        self.HR_conv_l = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=act_type)
        self.HR_conv_m = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=act_type)
        self.HR_conv_h = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=act_type)

        '''
        reconstruct_image 
        '''
        self.low_freq_conv = B.conv_block(nf, out_nc, kernel_size=3, norm_type=None, act_type=None)
        self.final_conv = B.conv_block(nf, out_nc, kernel_size=3, norm_type=None, act_type=None)
        self.high_freq_conv = B.conv_block(nf, out_nc, kernel_size=3, norm_type=None, act_type=None)

    def forward(self, shared_conv): 
        shared_conv = self.shared_conv(shared_conv)    
        #----------------------------------------------------------------------
        # low
        _l = self.low_dmb_front(shared_conv).mul(0.2) + shared_conv
        _l_dmb = self.low_dmb_back(_l).mul(0.2) + _l
        
        _l_fea = self.HR_conv_l(
                    self.upsampler_l(
                        self.LR_conv_l(_l_dmb + shared_conv)))
        #----------------------------------------------------------------------
        # mask
        _m = self.mask_dmb_front(shared_conv).mul(0.2) + shared_conv
        _m_dmb = self.mask_dmb_back(_m).mul(0.2) + _m + shared_conv

        mask = self.HR_conv_m(
                self.upsampler_m(
                    self.LR_conv_m(_m_dmb)))
        #----------------------------------------------------------------------
        # high
        _h = self.concat(torch.cat((_l_dmb, (self.high_dmb_front(shared_conv).mul(0.2) + shared_conv)), 1))
        _h_dmb = self.high_dmb_back(_h).mul(0.2) + _h + shared_conv

        _h_fea = self.HR_conv_h(
                    self.upsampler_h(
                        self.LR_conv_h(_h_dmb)))
        #----------------------------------------------------------------------
        # For image construction
        low_freq = self.low_freq_conv(_l_fea)

        M_sigmoid = torch.sigmoid(mask)
        combine = M_sigmoid.mul(_h_fea) + (1 - M_sigmoid).mul(_l_fea)
        combine = self.final_conv(combine)

        high_freq = self.high_freq_conv(_h_fea)

        return low_freq, high_freq, combine

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
