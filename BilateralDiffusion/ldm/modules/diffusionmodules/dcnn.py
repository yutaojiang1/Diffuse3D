import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import _depthconv_ext as depthconv

from torch.autograd import Function
from torch.nn.modules.utils import _pair
import cffi

def depth_conv(input,
                  depth,
                  weight,
                  bias,
                  stride=1,
                  padding=0,
                  dilation=1):

    if input is not None and input.dim() != 4:
        raise ValueError(
            "Expected 4D tensor as input, got {}D tensor instead.".format(
                input.dim()))

    f = DepthconvFunction()
    # print bias
    if isinstance(bias, torch.nn.Parameter):
        return f.apply(input, depth, weight, _pair(stride), _pair(padding), _pair(dilation), bias)
    else:
        return f.apply(input, depth, weight, _pair(stride), _pair(padding), _pair(dilation))

class DepthconvFunction(Function):
    @staticmethod
    def forward(ctx, input, depth, weight, stride, padding, dilation, bias = None):
        # print('forward')
        null = cffi.FFI().NULL
        if bias is None:
            # print bias, self.bias
            bias = null
        output_size = [int((input.size()[i + 2] + 2 * padding[i] - weight.size()[i + 2]) / stride[i] + 1)
                       for i in range(2)]

        output = input.new(*DepthconvFunction._output_size(input, weight, stride, padding, dilation))

        columns = input.new(weight.size(1) * weight.size(2) * weight.size(3),
                                  output_size[0] * output_size[1]).zero_()
        ones = input.new(output_size[0] * output_size[1]).zero_()

        ctx.save_for_backward(input, depth, weight, bias, columns, ones)
        ctx.data = stride, padding, dilation, null

        if not input.is_cuda:
            raise NotImplementedError
        else:
            if not isinstance(input, torch.cuda.FloatTensor):
                raise NotImplementedError
            depthconv.depthconv_forward_cuda(
                    input, depth, weight, bias,  output, columns,ones,
                    weight.size(3), weight.size(2), stride[1], stride[0],
                    padding[1], padding[0], dilation[1], dilation[0])

        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        # print('backward')
        input, depth, weight, bias, columns, ones = ctx.saved_tensors
        stride, padding, dilation, null = ctx.data

        grad_input = grad_weight = grad_bias = None

        if not grad_output.is_cuda:
            raise NotImplementedError
        else:
            if not isinstance(grad_output, torch.cuda.FloatTensor):
                raise NotImplementedError
            if ctx.needs_input_grad[0]:
                grad_input = input.new(*input.size()).zero_()
                depthconv.depthconv_backward_input_cuda(
                    input, depth, grad_output, grad_input,
                    weight, columns,
                    weight.size(3),
                    weight.size(2), stride[1], stride[0],
                    padding[1], padding[0], dilation[1],
                    dilation[0])

            if ctx.needs_input_grad[2]:
                grad_weight = weight.new(*weight.size()).zero_()
                if len(ctx.needs_input_grad) == 7:
                    if ctx.needs_input_grad[6]:
                        grad_bias = weight.new(*bias.size()).zero_()
                    else:
                        grad_bias = null
                else:
                    grad_bias = null

                depthconv.depthconv_backward_parameters_cuda(
                    input, depth, grad_output, grad_weight, grad_bias, columns,
                    ones,
                    weight.size(3),
                    weight.size(2), stride[1], stride[0],
                    padding[1], padding[0], dilation[1],
                    dilation[0], 1)

                if len(ctx.needs_input_grad) == 4:
                    if not ctx.needs_input_grad[3]:
                        grad_bias = None
                else:
                    grad_bias = None

        return grad_input, None, grad_weight, None, None, None, grad_bias

    @staticmethod
    def _output_size(input, weight, stride_in, padding_in, dilation_in):
        channels = weight.size(0)

        output_size = (input.size(0), channels)
        for d in range(input.dim() - 2):
            in_size = input.size(d + 2)
            pad = padding_in[d]
            kernel = dilation_in[d] * (weight.size(d + 2) - 1) + 1
            stride = stride_in[d]
            output_size += ((in_size + (2 * pad) - kernel) // stride + 1, )
        if not all(map(lambda s: s > 0, output_size)):
            raise ValueError(
                "convolution input is too small (output would be {})".format(
                    'x'.join(map(str, output_size))))
        return output_size

class DConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=True):
        super(DConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels, *self.kernel_size))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input:torch.Tensor, depth:torch.Tensor):
        return depth_conv(input, depth, self.weight, self.bias, self.stride,
                             self.padding, self.dilation)

