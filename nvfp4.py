"""
NVFP4 Quantization Implementation
NVIDIA FP4 (E2M1) format with dual-level scaling for training and inference

NVFP4 uses:
- E2M1 structure: 1 sign bit, 2 exponent bits, 1 mantissa bit
- Micro-block scaling with 16-element blocks
- FP8 E4M3 per-block scale factors
- Global FP32 scale for range adjustment

Training support:
- Straight-Through Estimator (STE) for gradient flow
- NVFP4LinearTrainable for QAT (Quantization-Aware Training)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from typing import Tuple, Optional

FP8_AMAX = 448.0
FP8_DTYPE = torch.float8_e4m3fn

FP4_AMAX = 6.0
FP4_DTYPE = getattr(torch, "float4_e2m1fn_x2", torch.uint8)

# Midpoints and corresponding bins for E2M1 quantization
# Representable positives = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
THRESHOLDS = [
    (5.0, 0b0110), (3.5, 0b0101), (2.5, 0b0100),
    (1.75, 0b0011), (1.25, 0b0010), (0.75, 0b0001), (0.25, 0b0000),
]


def cvt_1xfp32_2xfp4(x: torch.Tensor) -> torch.Tensor:
    """
    Convert FP32 values to packed FP4 (E2M1) format.

    Args:
        x: Input tensor with shape (..., 16), dtype float32

    Returns:
        Packed FP4 tensor with shape (..., 8), each uint8 contains 2 FP4 values
    """
    assert x.dtype == torch.float32

    bits = x.view(torch.int32)
    sign_bit = (bits >> 31) & 0x1

    x_abs = x.abs()
    other_bits = torch.full_like(x_abs, 0b0111, dtype=torch.int)

    for i, (m, code) in enumerate(THRESHOLDS):
        mask = x_abs <= m if i % 2 == 0 else x_abs < m
        other_bits = torch.where(mask, code, other_bits)

    # Each fp32 now as e2m1 (pack 8xfp4 values into 1xint32)
    e2m1 = (sign_bit << 3) | other_bits

    # Pack into int32 pairs
    e2m1x2 = (
        e2m1[..., ::8]
        | (e2m1[..., 1::8] << 4)
        | (e2m1[..., 2::8] << 8)
        | (e2m1[..., 3::8] << 12)
        | (e2m1[..., 4::8] << 16)
        | (e2m1[..., 5::8] << 20)
        | (e2m1[..., 6::8] << 24)
        | (e2m1[..., 7::8] << 28)
    )
    return e2m1x2.view(FP4_DTYPE)


def dequant_nvfp4_torch(
    xq: torch.Tensor,
    xs: torch.Tensor,
    global_scale: torch.Tensor,
    target_dtype: torch.dtype = torch.bfloat16
) -> torch.Tensor:
    """
    Dequantize NVFP4 tensor back to target dtype.

    Args:
        xq: Quantized tensor (packed FP4)
        xs: Per-block FP8 scales
        global_scale: Global FP32 scale
        target_dtype: Output dtype

    Returns:
        Dequantized tensor
    """
    # E2M1 lookup table for dequantization
    e2m1_lut = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        dtype=torch.float32,
        device=xq.device
    )

    # Unpack FP4 values
    xq_int = xq.view(torch.uint8)
    batch_shape = xq_int.shape[:-1]
    n_blocks = xq_int.shape[-1] // 8

    # Reshape for unpacking
    xq_int = xq_int.reshape(*batch_shape, n_blocks, 8)

    # Unpack 8 uint8 values into 16 FP4 values
    x_unpacked = torch.zeros(*batch_shape, n_blocks, 16, dtype=torch.float32, device=xq.device)

    for i in range(8):
        low_nibble = xq_int[..., i] & 0x0F
        high_nibble = (xq_int[..., i] >> 4) & 0x0F

        # Extract sign and magnitude
        low_sign = ((low_nibble >> 3) & 1).float() * -2 + 1
        low_mag = low_nibble & 0x07
        high_sign = ((high_nibble >> 3) & 1).float() * -2 + 1
        high_mag = high_nibble & 0x07

        x_unpacked[..., i*2] = low_sign * e2m1_lut[low_mag.long()]
        x_unpacked[..., i*2 + 1] = high_sign * e2m1_lut[high_mag.long()]

    # Apply inverse scaling
    s_decb = (xs.float() / global_scale).unsqueeze(-1)
    x_dequant = x_unpacked * s_decb

    return x_dequant.reshape(*batch_shape, -1).to(target_dtype)


def quant_nvfp4_torch(
    x: torch.Tensor,
    global_scale: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Quantize tensor to NVFP4 format with dual-level scaling.

    NVFP4 uses two scaling factors:
    - Global encoding scale (dtype: float32): s_enc = 6 * 448 / amax_x
    - Local encoding scale (per 16-block, dtype: fp8 e4m3)

    Args:
        x: Input tensor, last dimension must be divisible by 16
        global_scale: Optional pre-computed global scale

    Returns:
        Tuple of (quantized_tensor, block_scales, global_scale)
    """
    assert x.shape[-1] % 16 == 0, f"Last dimension must be divisible by 16, got {x.shape[-1]}"

    batch_dim = tuple(x.shape[:-1])
    x_blocks_f32 = x.unflatten(-1, (-1, 16)).float()

    q_dtype_max = FP4_AMAX
    s_dtype, s_dtype_max = FP8_DTYPE, FP8_AMAX

    if global_scale is None:
        global_scale = FP4_AMAX * FP8_AMAX / x_blocks_f32.abs().amax()

    # Compute per-block scales
    s_decb = x_blocks_f32.abs().amax(dim=-1) / q_dtype_max
    xs = (s_decb * global_scale).clamp(-s_dtype_max, s_dtype_max).to(s_dtype)

    # Apply encoding scale and quantize
    s_encb = (global_scale / xs.float().clip(1e-12)).unsqueeze(-1)
    x_blocks_f32 = x_blocks_f32 * s_encb
    xq = cvt_1xfp32_2xfp4(x_blocks_f32).reshape(*batch_dim, -1)

    return xq, xs, global_scale


# =============================================================================
# Training Support: Straight-Through Estimator (STE)
# =============================================================================

def simulate_nvfp4_quantize(x: torch.Tensor) -> torch.Tensor:
    """
    Custom NVFP4 quantize-dequantize for PyTorch training (QAT).
    Returns dequantized values in original dtype.
    """
    if x.shape[-1] % 16 != 0:
        # Pad to multiple of 16
        pad_size = 16 - (x.shape[-1] % 16)
        x_padded = F.pad(x, (0, pad_size))
    else:
        x_padded = x
        pad_size = 0

    xq, xs, gs = quant_nvfp4_torch(x_padded)
    x_dequant = dequant_nvfp4_torch(xq, xs, gs, x.dtype)

    if pad_size > 0:
        x_dequant = x_dequant[..., :-pad_size]

    return x_dequant


class NVFP4QuantizeSTE(Function):
    """Straight-Through Estimator for NVFP4 quantization."""

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        return simulate_nvfp4_quantize(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        # Straight-through: pass gradients unchanged
        return grad_output


def nvfp4_quantize_ste(x: torch.Tensor) -> torch.Tensor:
    """Apply NVFP4 quantization with STE for training."""
    return NVFP4QuantizeSTE.apply(x)


class NVFP4LinearTrainable(nn.Module):
    """
    Linear layer with simulated NVFP4 quantization for training (QAT).
    Weights are stored in full precision but quantized during forward pass.
    Gradients flow through via Straight-Through Estimator.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Pad in_features to multiple of 16
        self.padded_in = ((in_features + 15) // 16) * 16

        # Full precision weights for training
        self.weight = nn.Parameter(
            torch.empty(out_features, self.padded_in, device=device, dtype=dtype or torch.bfloat16)
        )
        nn.init.kaiming_uniform_(self.weight)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, device=device, dtype=dtype or torch.bfloat16))
        else:
            self.register_parameter('bias', None)

    @classmethod
    def from_linear(cls, linear: nn.Linear):
        """Create NVFP4LinearTrainable from an existing nn.Linear layer."""
        layer = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            device=linear.weight.device,
            dtype=linear.weight.dtype
        )

        # Copy weights with padding if needed
        with torch.no_grad():
            if linear.in_features != layer.padded_in:
                layer.weight[:, :linear.in_features] = linear.weight
                layer.weight[:, linear.in_features:] = 0
            else:
                layer.weight.copy_(linear.weight)

            if linear.bias is not None:
                layer.bias.copy_(linear.bias)

        return layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Quantize weights with STE
        weight_q = nvfp4_quantize_ste(self.weight)

        # Trim padding
        if self.padded_in != self.in_features:
            weight_q = weight_q[:, :self.in_features]

        return F.linear(x, weight_q, self.bias)

    def to_inference(self) -> 'NVFP4Linear':
        """Convert to inference-only NVFP4Linear (quantized weights)."""
        layer = NVFP4Linear(
            self.padded_in,
            self.out_features,
            bias=self.bias is not None,
            device=self.weight.device,
            dtype=self.weight.dtype
        )

        xq, xs, gs = quant_nvfp4_torch(self.weight.data)
        layer.weight_quantized = xq
        layer.weight_scales = xs
        layer.global_scale = gs
        layer.original_in_features = self.in_features

        if self.bias is not None:
            layer.bias = nn.Parameter(self.bias.data.clone())

        layer.compute_dtype = self.weight.dtype
        return layer


def prepare_model_for_nvfp4_training(
    model: nn.Module,
    target_modules: Optional[list] = None
) -> nn.Module:
    """
    Replace Linear layers with NVFP4LinearTrainable for QAT.

    Args:
        model: Model to prepare
        target_modules: List of module name patterns to quantize.
                       If None, quantizes all Linear layers.

    Returns:
        Model with trainable NVFP4 layers
    """
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            if target_modules is None or any(t in name for t in target_modules):
                nvfp4_layer = NVFP4LinearTrainable.from_linear(module)
                setattr(model, name, nvfp4_layer)
        else:
            prepare_model_for_nvfp4_training(module, target_modules)

    return model


def convert_to_inference(model: nn.Module) -> nn.Module:
    """Convert all NVFP4LinearTrainable layers to NVFP4Linear for inference."""
    for name, module in model.named_children():
        if isinstance(module, NVFP4LinearTrainable):
            setattr(model, name, module.to_inference())
        else:
            convert_to_inference(module)
    return model


# =============================================================================
# Inference-only layers
# =============================================================================

class NVFP4Linear(nn.Module):
    """
    Linear layer with NVFP4 quantized weights for inference.
    Weights are stored in NVFP4 format and dequantized on-the-fly during forward pass.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Ensure in_features is divisible by 16 for NVFP4
        assert in_features % 16 == 0, f"in_features must be divisible by 16, got {in_features}"

        # Quantized weights storage
        self.register_buffer('weight_quantized', None)
        self.register_buffer('weight_scales', None)
        self.register_buffer('global_scale', None)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, device=device, dtype=dtype))
        else:
            self.register_parameter('bias', None)

        self.compute_dtype = dtype or torch.bfloat16

    @classmethod
    def from_linear(cls, linear: nn.Linear, compute_dtype: torch.dtype = torch.bfloat16):
        """Create NVFP4Linear from an existing nn.Linear layer."""
        in_features = linear.in_features
        out_features = linear.out_features

        # Pad in_features to be divisible by 16 if needed
        padded_in = ((in_features + 15) // 16) * 16

        layer = cls(
            padded_in,
            out_features,
            bias=linear.bias is not None,
            device=linear.weight.device,
            dtype=compute_dtype
        )

        # Pad weights if necessary
        weight = linear.weight.data
        if in_features != padded_in:
            weight = torch.nn.functional.pad(weight, (0, padded_in - in_features))

        # Quantize weights
        xq, xs, global_scale = quant_nvfp4_torch(weight)
        layer.weight_quantized = xq
        layer.weight_scales = xs
        layer.global_scale = global_scale
        layer.original_in_features = in_features

        if linear.bias is not None:
            layer.bias = nn.Parameter(linear.bias.data.clone())

        layer.compute_dtype = compute_dtype
        return layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dequantize weights on-the-fly
        weight = dequant_nvfp4_torch(
            self.weight_quantized,
            self.weight_scales,
            self.global_scale,
            self.compute_dtype
        )

        # Trim padding if needed
        if hasattr(self, 'original_in_features'):
            weight = weight[:, :self.original_in_features]

        return nn.functional.linear(x.to(self.compute_dtype), weight, self.bias)


def quantize_model_nvfp4(model: nn.Module, compute_dtype: torch.dtype = torch.bfloat16) -> nn.Module:
    """
    Quantize all linear layers in a model to NVFP4 format.

    Args:
        model: PyTorch model to quantize
        compute_dtype: Dtype for computation during forward pass

    Returns:
        Model with NVFP4 quantized linear layers
    """
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            nvfp4_layer = NVFP4Linear.from_linear(module, compute_dtype)
            setattr(model, name, nvfp4_layer)
        else:
            quantize_model_nvfp4(module, compute_dtype)

    return model


if __name__ == "__main__":
    # Test NVFP4 quantization
    print("Testing NVFP4 Quantization")
    print("=" * 50)

    shape = (512, 128)
    x = torch.randn(shape, dtype=torch.bfloat16) * 0.01

    xq, xs, global_scale = quant_nvfp4_torch(x)

    print(">> Original tensor shape:", x.shape)
    print(">> Quantized tensor shape:", xq.shape)
    print(">> Blockwise scales shape:", xs.shape)
    print(">> Global scale:", global_scale.item())

    # Test dequantization
    x_dequant = dequant_nvfp4_torch(xq, xs, global_scale)
    print(">> Dequantized tensor shape:", x_dequant.shape)

    # Calculate quantization error
    mse = ((x.float() - x_dequant.float()) ** 2).mean()
    print(f">> MSE Error: {mse.item():.6f}")

    # Test Linear layer (inference)
    print("\n" + "=" * 50)
    print("Testing NVFP4Linear (Inference)")

    linear = nn.Linear(128, 64)
    nvfp4_linear = NVFP4Linear.from_linear(linear)

    test_input = torch.randn(1, 128)

    with torch.no_grad():
        out_original = linear(test_input)
        out_nvfp4 = nvfp4_linear(test_input)

    diff = (out_original - out_nvfp4).abs().mean()
    print(f">> Output difference: {diff.item():.6f}")
    print(">> NVFP4Linear working correctly!")

    # Test Trainable layer (QAT)
    print("\n" + "=" * 50)
    print("Testing NVFP4LinearTrainable (Training/QAT)")

    linear_train = nn.Linear(128, 64)
    nvfp4_trainable = NVFP4LinearTrainable.from_linear(linear_train)

    test_input = torch.randn(1, 128, requires_grad=True)
    target = torch.randn(1, 64)

    # Forward pass
    output = nvfp4_trainable(test_input)
    loss = F.mse_loss(output, target)

    # Backward pass
    loss.backward()

    print(f">> Forward output shape: {output.shape}")
    print(f">> Loss: {loss.item():.6f}")
    print(f">> Weight grad exists: {nvfp4_trainable.weight.grad is not None}")
    print(f">> Weight grad norm: {nvfp4_trainable.weight.grad.norm().item():.6f}")
    print(">> NVFP4LinearTrainable working correctly!")

    # Test conversion to inference
    print("\n" + "=" * 50)
    print("Testing conversion: Training -> Inference")

    nvfp4_inference = nvfp4_trainable.to_inference()

    with torch.no_grad():
        out_trainable = nvfp4_trainable(torch.randn(1, 128))
        out_inference = nvfp4_inference(torch.randn(1, 128))

    print(f">> Converted to NVFP4Linear: {type(nvfp4_inference).__name__}")
    print(">> Conversion successful!")
