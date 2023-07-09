from typing import Tuple

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """
    Reshape an image tensor for 2D pooling

    Args:
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.
    """

    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0

    tile_h = height // kh
    tile_w = width // kw

    input = input.contiguous()
    input = input.view(batch, channel, tile_h, kh, tile_w, kw)
    input = input.permute(0, 1, 2, 4, 3, 5)

    input = input.contiguous()
    input = input.view(batch, channel, tile_h, tile_w, kw * kh)

    return input, tile_h, tile_w


def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """
    Tiled average pooling 2D

    Args:
        input : batch x channel x height x width
        kernel : height x width of pooling

    Returns:
        Pooled tensor
    """
    batch, channel, height, width = input.shape
    input, tile_h, tile_w = tile(input, kernel)
    input = input.mean(4)
    input = input.view(batch, channel, tile_h, tile_w)
    return input


max_reduce = FastOps.reduce(operators.max, -1e9)


def argmax(input: Tensor, dim: int) -> Tensor:
    """
    Compute the argmax as a 1-hot tensor.

    Args:
        input : input tensor
        dim : dimension to apply argmax


    Returns:
        :class:`Tensor` : tensor with 1 on highest cell in dim, 0 otherwise

    """
    out = max_reduce(input, dim)
    return out == input


class Max(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: Tensor) -> Tensor:
        "Forward of max should be max reduction"
        dimInt = int(dim[0])
        ctx.save_for_backward(input, dimInt)
        return max_reduce(input, dimInt)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        "Backward of max should be argmax (see above)"
        input, dimInt = ctx.saved_values
        return (grad_output * argmax(input, dimInt), 0.0)


def max(input: Tensor, dim: int) -> Tensor:
    return Max.apply(input, input._ensure_tensor(dim))


def softmax(input: Tensor, dim: int) -> Tensor:
    r"""
    Compute the softmax as a tensor.



    $z_i = \frac{e^{x_i}}{\sum_i e^{x_i}}$

    Args:
        input : input tensor
        dim : dimension to apply softmax

    Returns:
        softmax tensor
    """
    input = input.exp()
    sum_along_axis = input.sum(dim)
    return input / sum_along_axis


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    r"""
    Compute the log of the softmax as a tensor.

    $z_i = x_i - \log \sum_i e^{x_i}$

    See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations

    Args:
        input : input tensor
        dim : dimension to apply log-softmax

    Returns:
         log of softmax tensor
    """
    # input = input.exp()
    # sumOfDim = input.sum(dim)
    # return input - sumOfDim

    m = max(input, dim)
    t = input - m
    t = t.exp()
    t = t.sum(dim)
    t = t.log()
    input = input - t - m
    return input


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """
    Tiled max pooling 2D

    Args:
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
        Tensor : pooled tensor
    """
    batch, channel, height, width = input.shape
    input, tile_h, tile_w = tile(input, kernel)
    input = max(input, 4)
    input = input.view(batch, channel, tile_h, tile_w)
    return input


def dropout(input: Tensor, rate: float, ignore: bool = False) -> Tensor:
    """
    Dropout positions based on random noise.

    Args:
        input : input tensor
        rate : probability [0, 1) of dropping out each position
        ignore : skip dropout, i.e. do nothing at all

    Returns:
        tensor with random positions dropped out
    """
    if not ignore:
        bit_tensor = rand(input.shape, input.backend) > rate
        input = bit_tensor * input
    return input
