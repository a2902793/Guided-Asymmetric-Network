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
# from utils import (
#     round_filters,
#     round_repeats,
#     drop_connect,
#     get_same_padding_conv2d,
#     get_model_params,
#     efficientnet_params,
#     load_pretrained_weights,
#     Swish,
#     MemoryEfficientSwish,
#     calculate_output_image_size
# )
from models.modules.utils import round_filters, round_repeats, drop_connect, get_same_padding_conv2d, get_model_params, efficientnet_params, load_pretrained_weights, Swish, MemoryEfficientSwish, calculate_output_image_size

# import models.modules.utils

class MBBlock(nn.Module):
    """Mobile Inverted Residual Bottleneck Block.
    Args:
        block_args (namedtuple): BlockArgs, defined in utils.py.
        global_params (namedtuple): GlobalParam, defined in utils.py.
        image_size (tuple or list): [image_height, image_width].
    References:
        [1] https://arxiv.org/abs/1704.04861 (MobileNet v1)
        [2] https://arxiv.org/abs/1801.04381 (MobileNet v2)
        [3] https://arxiv.org/abs/1905.02244 (MobileNet v3)
    """

    def __init__(self, block_args, global_params, image_size=None):
        super().__init__()
        self._block_args = block_args
        self._bn_mom = 1 - global_params.batch_norm_momentum # pytorch's difference from tensorflow
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip  # whether to use skip connection and drop connect

        # Expansion phase (Inverted Bottleneck)
        inp = self._block_args.input_filters  # number of input channels
        oup = self._block_args.input_filters * self._block_args.expand_ratio  # number of output channels
        if self._block_args.expand_ratio != 1:
            Conv2d = get_same_padding_conv2d(image_size=image_size)
            self._expand_conv = Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
            # image_size = calculate_output_image_size(image_size, 1) <-- this wouldn't modify image_size

        # Depthwise convolution phase
        k = self._block_args.kernel_size
        s = self._block_args.stride
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._depthwise_conv = Conv2d(
            in_channels=oup, out_channels=oup, groups=oup,  # groups makes it depthwise
            kernel_size=k, stride=s, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
        image_size = calculate_output_image_size(image_size, s)

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            Conv2d = get_same_padding_conv2d(image_size=(1, 1))
            num_squeezed_channels = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))
            self._se_reduce = Conv2d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = Conv2d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)

        # Pointwise convolution phase
        final_oup = self._block_args.output_filters
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._project_conv = Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)
        self._swish = MemoryEfficientSwish()

    def forward(self, inputs, drop_connect_rate=None):
        """MBConvBlock's forward function.
        Args:
            inputs (tensor): Input tensor.
            drop_connect_rate (bool): Drop connect rate (float, between 0 and 1).
        Returns:
            Output of this block after processing.
        """

        # Expansion and Depthwise Convolution
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = self._expand_conv(inputs)
            x = self._bn0(x)
            x = self._swish(x)

        x = self._depthwise_conv(x)
        x = self._bn1(x)
        x = self._swish(x)

        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_reduce(x_squeezed)
            x_squeezed = self._swish(x_squeezed)
            x_squeezed = self._se_expand(x_squeezed)
            x = torch.sigmoid(x_squeezed) * x

        # Pointwise Convolution
        x = self._project_conv(x)
        x = self._bn2(x)

        # Skip connection and drop connect
        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:
            # The combination of skip connection and drop connect brings about stochastic depth.
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs  # skip connection
        return x

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export).
        Args:
            memory_efficient (bool): Whether to use memory-efficient version of swish.
        """
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()

class DualSR_Effnet(nn.Module):
    #                      3       3  64     2         
    def __init__(self, in_nc, out_nc, nf, nb_e, 
            gc=32, upscale=4, norm_type=None, 
            act_type='leakyrelu', mode='CNA', upsample_mode='upconv',
            blocks_args=None, global_params=None):
        super(DualSR_Effnet, self).__init__()

        #print("DualSR_Effnet.py")

        n_upscale = int(math.log(upscale, 2))
        if upscale == 3:
            n_upscale = 1
        assert isinstance(blocks_args, list), 'blocks_args should be a list'
        assert len(blocks_args) > 0, 'block args must be greater than 0'
        
        '''
        simple_conv
        '''
        fea_conv = B.conv_block(in_nc, nf, kernel_size=3, norm_type=None, act_type=None)
        
        '''
        shared_module
        '''
        rb_blocks_e = [B.RRDB(nf, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
            norm_type=norm_type, act_type=act_type, mode='CNA') for _ in range(nb_e)] #nb_e=2
        
        self.body_ex = B.sequential(fea_conv, B.ShortcutBlock(B.sequential(*rb_blocks_e)))
        '''
        eff_net
        '''
        self._global_params = global_params
        self._blocks_args = blocks_args

        # Batch norm parameters
        bn_mom = 1 - self._global_params.batch_norm_momentum # 0.99
        bn_eps = self._global_params.batch_norm_epsilon #1e-3

        # Get stem static or dynamic convolution depending on image size
        image_size = global_params.image_size
        Conv2d = get_same_padding_conv2d(image_size=image_size)

        # Stem
        in_channels = 64  # rgb
        out_channels = round_filters(32, self._global_params)  # number of output channels
        self._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=1, bias=False)
        self._bn0 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)
        image_size = calculate_output_image_size(image_size, 2)

        # Build blocks
        self._blocks = nn.ModuleList([])
        for block_args in self._blocks_args:

            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters, self._global_params),
                output_filters=round_filters(block_args.output_filters, self._global_params),
                num_repeat=round_repeats(block_args.num_repeat, self._global_params)
            )

            # The first block needs to take care of stride and filter size increase.
            self._blocks.append(MBBlock(block_args, self._global_params, image_size=image_size))
            image_size = calculate_output_image_size(image_size, block_args.stride)
            if block_args.num_repeat > 1: # modify block_args to keep same output size
                block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(MBBlock(block_args, self._global_params, image_size=image_size))
                # image_size = calculate_output_image_size(image_size, block_args.stride)  # stride = 1

        # Head
        in_channels = block_args.output_filters  # output of final block
        out_channels = round_filters(64, self._global_params)
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._conv_head = Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        # Final linear layer
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._dropout = nn.Dropout(self._global_params.dropout_rate)
        self._fc = nn.Linear(out_channels, self._global_params.num_classes)
        self._swish = MemoryEfficientSwish()
        

        '''
        reconstruct_image 
        '''
        self.LR_conv_l = B.conv_block(nf, nf, kernel_size=3, norm_type=norm_type, act_type=None, mode=mode)

        self.LR_conv_h = B.conv_block(nf, nf, kernel_size=3, norm_type=norm_type, act_type=None, mode=mode)

        self.LR_conv_M = B.conv_block(nf, nf, kernel_size=3, norm_type=norm_type, act_type=None, mode=mode)


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

    def extract_endpoints(self, inputs):
        """Use convolution layer to extract features
        from reduction levels i in [1, 2, 3, 4, 5].
        Args:
            inputs (tensor): Input tensor.
        Returns:
            Dictionary of last intermediate features
            with reduction levels i in [1, 2, 3, 4, 5].
            Example:
                >>> import torch
                >>> from efficientnet.model import EfficientNet
                >>> inputs = torch.rand(1, 3, 224, 224)
                >>> model = EfficientNet.from_pretrained('efficientnet-b0')
                >>> endpoints = model.extract_endpoints(inputs)
                >>> print(endpoints['reduction_1'].shape)  # torch.Size([1, 16, 112, 112])
                >>> print(endpoints['reduction_2'].shape)  # torch.Size([1, 24, 56, 56])
                >>> print(endpoints['reduction_3'].shape)  # torch.Size([1, 40, 28, 28])
                >>> print(endpoints['reduction_4'].shape)  # torch.Size([1, 112, 14, 14])
                >>> print(endpoints['reduction_5'].shape)  # torch.Size([1, 1280, 7, 7])
        """
        endpoints = dict()

        # Stem
        x = self._swish(self._bn0(self._conv_stem(inputs)))
        prev_x = x

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks) # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints['reduction_{}'.format(len(endpoints)+1)] = prev_x
            prev_x = x

        # Head
        x = self._swish(self._bn1(self._conv_head(x)))
        endpoints['reduction_{}'.format(len(endpoints)+1)] = x

        return endpoints

    def extract_features(self, inputs):
        """use convolution layer to extract feature .
        Args:
            inputs (tensor): Input tensor.
        Returns:
            Output of the final convolution
            layer in the efficientnet model.
        """
        # Stem
        x = self._swish(self._bn0(self._conv_stem(inputs)))

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks) # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)

        # Head
        x = self._swish(self._bn1(self._conv_head(x)))

        return x

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export).

        Args:
            memory_efficient (bool): Whether to use memory-efficient version of swish.
        """
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()
        for block in self._blocks:
            block.set_swish(memory_efficient)

    def forward(self, x):
        # head
        x = self.body_ex(x) # shared module做2次RRDB後

        # low
        x_l = self.extract_features(x)
        x_l = self.LR_conv_l(x_l)
        x_l = self.upsampler_l(x_l)
        x_fea_l = self.HR_conv0_l(x_l)
        x_l = self.HR_conv1_l(x_fea_l)

        # high
        x_h = self.extract_features(x)
        x_h = self.LR_conv_h(x_h)
        x_h = self.upsampler_h(x_h)
        x_fea_h = self.HR_conv0_h(x_h)
        x_h = self.HR_conv1_h(x_fea_h)

        # mask
        m = self.extract_features(x)
        m = self.LR_conv_M(m)
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


