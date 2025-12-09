#models_LH.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp


class SimpleDecoder(nn.Module):
    """Simple decoder for UGF-Net to avoid API compatibility issues"""
    
    def __init__(self, encoder_channels, decoder_channels=256, num_classes=30):
        super().__init__()
        
        # ASPP module for multi-scale context
        self.aspp = ASPPModule(encoder_channels[-1], decoder_channels)
        
        # Decoder blocks with upsampling (4x upsampling total = 16x with final)
        self.decoder_block1 = nn.Sequential(
            nn.Conv2d(decoder_channels, decoder_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(decoder_channels // 2),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        
        self.decoder_block2 = nn.Sequential(
            nn.Conv2d(decoder_channels // 2, decoder_channels // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(decoder_channels // 4),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        
        self.decoder_block3 = nn.Sequential(
            nn.Conv2d(decoder_channels // 4, decoder_channels // 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(decoder_channels // 8),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        
        self.decoder_block4 = nn.Sequential(
            nn.Conv2d(decoder_channels // 8, decoder_channels // 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(decoder_channels // 8),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        
        # Segmentation head
        self.segmentation_head = nn.Conv2d(decoder_channels // 8, num_classes, kernel_size=1)
    
    def forward(self, x):
        x = self.aspp(x)
        x = self.decoder_block1(x)  # 2x
        x = self.decoder_block2(x)  # 4x
        x = self.decoder_block3(x)  # 8x
        x = self.decoder_block4(x)  # 16x (back to original resolution)
        x = self.segmentation_head(x)
        return x


class ASPPModule(nn.Module):
    """Atrous Spatial Pyramid Pooling"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # Different dilation rates for multi-scale context
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=18, dilation=18, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Global pooling branch
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        size = x.shape[2:]
        
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        x5 = F.interpolate(self.global_pool(x), size=size, mode='bilinear', align_corners=False)
        
        x = torch.cat([x1, x2, x3, x4, x5], dim=1)
        x = self.fusion(x)
        
        return x


class UncertaintyGatedFusionModel(nn.Module):
    """
    Uncertainty-Gated Fusion Network (UGF-Net)
    
    Key Innovation: Uses pixel-wise epistemic uncertainty from auxiliary heads
    to dynamically weight RGB vs NIR features at each spatial location.
    """
    
    def __init__(self, num_classes=30, encoder_name='resnet50', encoder_weights='imagenet'):
        super().__init__()
        
        # Separate encoders for RGB and NIR
        self.rgb_encoder = smp.encoders.get_encoder(
            encoder_name, 
            in_channels=3, 
            depth=5, 
            weights=encoder_weights
        )
        
        # NIR preprocessing: expand 1 channel to 3 to use pretrained weights
        self.nir_preprocess = nn.Conv2d(1, 3, kernel_size=1, bias=False)
        with torch.no_grad():
            self.nir_preprocess.weight.fill_(1.0/3.0)
        
        self.nir_encoder = smp.encoders.get_encoder(
            encoder_name, 
            in_channels=3, 
            depth=5, 
            weights=encoder_weights
        )
        
        encoder_channels = self.rgb_encoder.out_channels
        
        # === THE NOVELTY: Auxiliary Uncertainty Heads ===
        # These heads force each encoder to attempt segmentation independently
        # Their confusion (entropy) tells us which sensor to trust
        self.rgb_aux_head = nn.Sequential(
            nn.Conv2d(encoder_channels[-1], 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )
        
        self.nir_aux_head = nn.Sequential(
            nn.Conv2d(encoder_channels[-1], 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )
        
        # Fusion block after uncertainty gating
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(encoder_channels[-1] * 2, encoder_channels[-1], 
                     kernel_size=3, padding=1),
            nn.BatchNorm2d(encoder_channels[-1]),
            nn.ReLU(inplace=True)
        )
        
        # Use a simpler custom decoder instead of DeepLabV3+
        # to avoid API compatibility issues
        self.decoder = SimpleDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=256,
            num_classes=num_classes
        )
    
    def compute_entropy(self, logits):
        """
        Compute pixel-wise entropy (uncertainty) from logits
        
        High entropy = high uncertainty = model is confused
        Low entropy = low uncertainty = model is confident
        
        Returns: [B, 1, H, W] entropy map
        """
        probs = F.softmax(logits, dim=1)
        log_probs = F.log_softmax(logits, dim=1)
        entropy = -(probs * log_probs).sum(dim=1, keepdim=True)
        return entropy
    
    def forward(self, rgb, nir, weather=None):
        # 1. Extract features from both modalities
        nir_3ch = self.nir_preprocess(nir)
        rgb_features = self.rgb_encoder(rgb)
        nir_features = self.nir_encoder(nir_3ch)
        
        rgb_bottleneck = rgb_features[-1]
        nir_bottleneck = nir_features[-1]
        
        # Ensure spatial dimensions match
        if rgb_bottleneck.shape[2:] != nir_bottleneck.shape[2:]:
            nir_bottleneck = F.interpolate(
                nir_bottleneck,
                size=rgb_bottleneck.shape[2:],
                mode='bilinear',
                align_corners=False
            )
        
        # 2. Generate auxiliary predictions (the "confidence test")
        rgb_logits = self.rgb_aux_head(rgb_bottleneck)
        nir_logits = self.nir_aux_head(nir_bottleneck)
        
        # 3. Calculate pixel-wise uncertainty (entropy)
        # Detach to prevent backprop through gating logic itself
        rgb_entropy = self.compute_entropy(rgb_logits).detach()
        nir_entropy = self.compute_entropy(nir_logits).detach()
        
        # 4. Generate attention gates based on uncertainty
        # Lower entropy = higher weight (trust the confident sensor)
        entropy_stack = torch.cat([-rgb_entropy, -nir_entropy], dim=1)
        gates = F.softmax(entropy_stack, dim=1)
        
        rgb_gate = gates[:, 0:1, :, :]
        nir_gate = gates[:, 1:2, :, :]
        
        # 5. Uncertainty-weighted fusion
        fused_bottleneck = (rgb_bottleneck * rgb_gate) + (nir_bottleneck * nir_gate)
        
        # Concatenate with original RGB features for skip connections
        combined = torch.cat([rgb_bottleneck, fused_bottleneck], dim=1)
        combined = self.fusion_conv(combined)
        
        # 6. Decode 
        final_output = self.decoder(combined)
        
        # During training, return auxiliary outputs for deep supervision
        if self.training:
            # Resize auxiliary outputs to match final output size
            rgb_aux_out = F.interpolate(
                rgb_logits, 
                size=final_output.shape[2:], 
                mode='bilinear',
                align_corners=False
            )
            nir_aux_out = F.interpolate(
                nir_logits, 
                size=final_output.shape[2:], 
                mode='bilinear',
                align_corners=False
            )
            return final_output, rgb_aux_out, nir_aux_out, rgb_gate, nir_gate
        
        # During inference, optionally return gates for visualization
        return final_output, rgb_gate, nir_gate


# Keep your existing models for comparison
class BaselineRGBModel(nn.Module):
    """Baseline: RGB only"""
    
    def __init__(self, num_classes=30, encoder_name='resnet50', encoder_weights='imagenet'):
        super().__init__()
        
        self.model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=num_classes
        )
    
    def forward(self, rgb, nir=None, weather=None):
        return self.model(rgb)


class EarlyFusionModel(nn.Module):
    """Simple early fusion: concatenate RGB + NIR"""
    
    def __init__(self, num_classes=30, encoder_name='resnet50', encoder_weights='imagenet'):
        super().__init__()
        
        base_model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=num_classes
        )
        
        # Modify first conv to accept 4 channels
        original_conv = base_model.encoder.conv1
        new_conv = nn.Conv2d(
            4, original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False if original_conv.bias is None else True
        )
        
        if encoder_weights == 'imagenet':
            with torch.no_grad():
                new_conv.weight[:, :3, :, :] = original_conv.weight
                new_conv.weight[:, 3:4, :, :] = original_conv.weight.mean(dim=1, keepdim=True)
        
        base_model.encoder.conv1 = new_conv
        self.model = base_model
    
    def forward(self, rgb, nir, weather=None):
        x = torch.cat([rgb, nir], dim=1)
        return self.model(x)


def get_model(model_type='ugf', num_classes=30, encoder_name='resnet50'):
    """Factory function to get model by type"""
    
    models = {
        'baseline': BaselineRGBModel,
        'early': EarlyFusionModel,
        'ugf': UncertaintyGatedFusionModel,
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return models[model_type](
        num_classes=num_classes,
        encoder_name=encoder_name,
        encoder_weights='imagenet'
    )


if __name__ == '__main__':
    # Test the UGF model
    batch_size = 2
    rgb = torch.randn(batch_size, 3, 512, 1024)
    nir = torch.randn(batch_size, 1, 512, 1024)
    
    print("Testing UGF-Net...")
    model = UncertaintyGatedFusionModel(num_classes=30)
    model.train()
    
    output, rgb_aux, nir_aux, rgb_gate, nir_gate = model(rgb, nir)
    
    print(f"Main output shape: {output.shape}")
    print(f"RGB aux shape: {rgb_aux.shape}")
    print(f"NIR aux shape: {nir_aux.shape}")
    print(f"RGB gate shape: {rgb_gate.shape}")
    print(f"NIR gate shape: {nir_gate.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # Test output size matches input
    assert output.shape == (batch_size, 30, 512, 1024), f"Output shape mismatch: {output.shape}"
    print("\n✓ Output size correct!")