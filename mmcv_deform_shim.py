"""
Shim for mmcv.ops to provide deformable convolution using torchvision ops
This allows the code to run without compiled CUDA ops using torchvision's CUDA kernels
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d

class ModulatedDeformConv2d(nn.Module):
    """Modulated deformable convolution using torchvision ops."""
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                 padding=0, dilation=1, groups=1, bias=True, deform_groups=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        self.groups = groups
        self.deform_groups = deform_groups
        
        # Standard convolution weights
        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels // groups, *self.kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
            
        nn.init.kaiming_uniform_(self.weight, a=1)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x, offset, mask):
        """Forward pass with offset and mask using torchvision deform_conv2d."""
        return modulated_deform_conv2d(
            x, offset, mask, self.weight, self.bias, 
            self.stride, self.padding, self.dilation, self.groups, self.deform_groups
        )


def modulated_deform_conv2d(x, offset, mask, weight, bias, stride=1, 
                           padding=0, dilation=1, groups=1, deform_groups=1):
    """
    Modulated deformable convolution using torchvision's deform_conv2d.
    """
    # Normalize parameters
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    
    try:
        # Use torchvision's deform_conv2d
        # Note: torchvision's API requires offset to have shape [B, C*kh*kw*2, H, W]
        # and mask to have shape [B, C*kh*kw, H, W]
        out = deform_conv2d(
            x, 
            offset, 
            weight, 
            stride=stride,
            padding=padding,
            dilation=dilation,
            mask=mask
        )
        
        # Add bias if present
        if bias is not None:
            out = out + bias.view(1, -1, 1, 1)
        
        return out
    
    except Exception as e:
        # Fallback to regular convolution if deform_conv2d fails
        print(f"Warning: deform_conv2d failed ({e}), using regular convolution")
        return F.conv2d(x, weight, bias, stride, padding, dilation, groups)
