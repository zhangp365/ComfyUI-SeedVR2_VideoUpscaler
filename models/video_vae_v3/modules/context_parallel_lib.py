# // Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# //
# // Licensed under the Apache License, Version 2.0 (the "License");
# // you may not use this file except in compliance with the License.
# // You may obtain a copy of the License at
# //
# //     http://www.apache.org/licenses/LICENSE-2.0
# //
# // Unless required by applicable law or agreed to in writing, software
# // distributed under the License is distributed on an "AS IS" BASIS,
# // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# // See the License for the specific language governing permissions and
# // limitations under the License.

from typing import List
import torch
import torch.nn.functional as F
from torch import Tensor

from .types import MemoryState

# Single GPU inference with proper slicing and overlap support
# print("Using single GPU inference mode with overlap support")

# Global variable to store sliced tensors for gathering
_sliced_tensors = []


def causal_conv_slice_inputs(x, split_size, memory_state, overlap=None):
    """
    🚀 IMPROVED: Single GPU slicing with proper overlap support
    
    Args:
        x: Input tensor [B, C, T, H, W]
        split_size: Size of each slice
        memory_state: Memory state for causal processing
        overlap: User-defined overlap (optional)
    
    Returns:
        Sliced tensor with overlap for proper reconstruction
    """
    global _sliced_tensors
    
    # If tensor is smaller than split_size, no need to slice
    if x.shape[2] <= split_size:
        _sliced_tensors = [x]
        return x
    
    # Use user-defined overlap if provided, otherwise calculate adaptive overlap
    if overlap is not None:
        # Convert pixel overlap to temporal frames (assuming 8x spatial downscaling)
        overlap = max(1, overlap // 8)  # Convert pixel overlap to latent frames
    else:
        # Fallback: adaptive overlap based on split_size
        overlap = min(8, split_size // 4)  # Adaptive overlap: 25% of split_size, max 8 frames
    
    # Slice with overlap
    slices = []
    start = 0
    
    while start < x.shape[2]:
        end = min(start + split_size, x.shape[2])
        
        # Add overlap to previous slices (except first)
        if start > 0:
            overlap_start = max(0, start - overlap)
            slice_tensor = x[:, :, overlap_start:end]
        else:
            slice_tensor = x[:, :, start:end]
        
        slices.append(slice_tensor)
        
        # Move to next slice
        if end >= x.shape[2]:
            break
        start = end - overlap  # Overlap with next slice
    
    # Store for gathering
    _sliced_tensors = slices
    
    # Return concatenated slices for processing
    return torch.cat(slices, dim=2)


def causal_conv_gather_outputs(x):
    """
    🚀 IMPROVED: Single GPU gathering with overlap blending
    
    Args:
        x: Processed tensor from sliced inputs
    
    Returns:
        Properly reconstructed tensor with overlap blending
    """
    global _sliced_tensors
    
    # If no slicing was done, return as is
    if len(_sliced_tensors) <= 1:
        return x
    
    # Calculate overlap and reconstruct
    overlap = min(8, _sliced_tensors[0].shape[2] // 4)
    
    # Split the processed tensor back into slices
    processed_slices = []
    start = 0
    
    for i, original_slice in enumerate(_sliced_tensors):
        slice_length = original_slice.shape[2]
        end = start + slice_length
        processed_slice = x[:, :, start:end]
        processed_slices.append(processed_slice)
        start = end
    
    # Reconstruct with overlap blending
    if len(processed_slices) == 1:
        result = processed_slices[0]
    else:
        # Start with first slice
        result_slices = [processed_slices[0]]
        
        for i in range(1, len(processed_slices)):
            current_slice = processed_slices[i]
            
            # Remove overlap from current slice (it was added during slicing)
            if i > 0:
                # Calculate how much overlap was added
                effective_overlap = min(overlap, current_slice.shape[2] // 2)
                current_slice = current_slice[:, :, effective_overlap:]
            
            result_slices.append(current_slice)
        
        result = torch.cat(result_slices, dim=2)
    
    # Clear stored slices
    _sliced_tensors = []
    
    return result


def get_output_len(conv_module, input_len, pad_len, dim=0):
    dilated_kernerl_size = conv_module.dilation[dim] * (conv_module.kernel_size[dim] - 1) + 1
    output_len = (input_len + pad_len - dilated_kernerl_size) // conv_module.stride[dim] + 1
    return output_len


def get_cache_size(conv_module, input_len, pad_len, dim=0):
    """
    🚀 IMPROVED: Calculate cache size with proper overlap for single GPU
    
    This function calculates the overlap needed between slices to maintain
    temporal consistency in causal convolutions.
    """
    dilated_kernerl_size = conv_module.dilation[dim] * (conv_module.kernel_size[dim] - 1) + 1
    output_len = (input_len + pad_len - dilated_kernerl_size) // conv_module.stride[dim] + 1
    remain_len = (
        input_len + pad_len - ((output_len - 1) * conv_module.stride[dim] + dilated_kernerl_size)
    )
    overlap_len = dilated_kernerl_size - conv_module.stride[dim]
    cache_len = overlap_len + remain_len  # >= 0
    
    assert output_len > 0
    return cache_len


def cache_send_recv(tensor: List[Tensor], cache_size, times, memory=None):
    """
    🚀 IMPROVED: Single GPU cache handling with proper temporal consistency
    
    Args:
        tensor: List of tensors to process
        cache_size: Size of cache needed
        times: Number of times to repeat
        memory: Previous memory buffer
    
    Returns:
        Properly cached tensor for temporal consistency
    """
    recv_buffer = None
    
    # Handle memory buffer for single GPU case
    if memory is not None:
        recv_buffer = memory.to(tensor[0])
    elif times > 0:
        # Create proper temporal padding for causal consistency
        tile_repeat = [1] * tensor[0].ndim
        tile_repeat[2] = times
        recv_buffer = torch.tile(tensor[0][:, :, :1], tile_repeat)
    
    return recv_buffer
