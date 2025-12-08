import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

class EarlyFusionModel(nn.Module):
    """Simple early fusion: concatenate RGB + NIR as 4-channel input"""
    
    def __init__(self, num_classes=30, encoder_name='resnet50', encoder_weights='imagenet'):
        super().__init__()
        
        # Modify first conv to accept 4 channels (RGB + NIR)
        self.model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=4,
            classes=num_classes
        )
        
        # Modify the first layer to accept 4 channels
        if encoder_weights == 'imagenet':
            # Copy pretrained RGB weights and initialize NIR channel
            original_conv = self.model.encoder.conv1
            new_conv = nn.Conv2d(4, original_conv.out_channels, 
                               kernel_size=original_conv.kernel_size,
                               stride=original_conv.stride,
                               padding=original_conv.padding,
                               bias=False if original_conv.bias is None else True)
            
            # Copy RGB weights
            with torch.no_grad():
                new_conv.weight[:, :3, :, :] = original_conv.weight
                # Initialize NIR channel as average of RGB
                new_conv.weight[:, 3:4, :, :] = original_conv.weight.mean(dim=1, keepdim=True)
            
            self.model.encoder.conv1 = new_conv
    
    def forward(self, rgb, nir, weather=None):
        # Concatenate RGB and NIR
        x = torch.cat([rgb, nir], dim=1)
        return self.model(x)


class LateFusionModel(nn.Module):
    """Late fusion: separate encoders, fuse features at decoder"""
    
    def __init__(self, num_classes=30, encoder_name='resnet50', encoder_weights='imagenet'):
        super().__init__()
        
        # RGB encoder
        self.rgb_encoder = smp.encoders.get_encoder(
            encoder_name,
            in_channels=3,
            depth=5,
            weights=encoder_weights
        )
        
        # NIR encoder (separate)
        self.nir_encoder = smp.encoders.get_encoder(
            encoder_name,
            in_channels=1,
            depth=5,
            weights=encoder_weights
        )
        
        # Fusion module
        encoder_channels = self.rgb_encoder.out_channels
        self.fusion_layers = nn.ModuleList([
            nn.Conv2d(c * 2, c, kernel_size=1) for c in encoder_channels
        ])
        
        # Decoder
        self.decoder = smp.decoders.deeplabv3plus.decoder.DeepLabV3PlusDecoder(
            encoder_channels=encoder_channels,
            out_channels=256,
            atrous_rates=(12, 24, 36),
            output_stride=16
        )
        
        # Segmentation head
        self.segmentation_head = smp.base.SegmentationHead(
            in_channels=256,
            out_channels=num_classes,
            kernel_size=1,
            upsampling=4
        )
    
    def forward(self, rgb, nir, weather=None):
        # Encode both modalities
        rgb_features = self.rgb_encoder(rgb)
        nir_features = self.nir_encoder(nir)
        
        # Fuse at each level
        fused_features = []
        for i, (rgb_feat, nir_feat, fusion_layer) in enumerate(
            zip(rgb_features, nir_features, self.fusion_layers)
        ):
            # Concatenate and reduce channels
            concat_feat = torch.cat([rgb_feat, nir_feat], dim=1)
            fused_feat = fusion_layer(concat_feat)
            fused_features.append(fused_feat)
        
        # Decode
        decoder_output = self.decoder(*fused_features)
        masks = self.segmentation_head(decoder_output)
        
        return masks


class AdaptiveFusionModel(nn.Module):
    """Weather-conditioned adaptive fusion (NOVEL CONTRIBUTION)"""
    
    def __init__(self, num_classes=30, encoder_name='resnet50', encoder_weights='imagenet'):
        super().__init__()
        
        # RGB encoder
        self.rgb_encoder = smp.encoders.get_encoder(
            encoder_name,
            in_channels=3,
            depth=5,
            weights=encoder_weights
        )
        
        # NIR encoder
        self.nir_encoder = smp.encoders.get_encoder(
            encoder_name,
            in_channels=1,
            depth=5,
            weights=encoder_weights
        )
        
        # Weather embedding
        self.num_weather = 4  # FOG, RAIN, LOWLIGHT, SNOW
        self.weather_embedding = nn.Embedding(self.num_weather, 64)
        
        # Adaptive fusion modules (weather-conditioned attention)
        encoder_channels = self.rgb_encoder.out_channels
        self.adaptive_fusion = nn.ModuleList([
            WeatherConditionedFusion(c, weather_dim=64) 
            for c in encoder_channels
        ])
        
        # Decoder
        self.decoder = smp.decoders.deeplabv3plus.decoder.DeepLabV3PlusDecoder(
            encoder_channels=encoder_channels,
            out_channels=256,
            atrous_rates=(12, 24, 36),
            output_stride=16
        )
        
        # Segmentation head
        self.segmentation_head = smp.base.SegmentationHead(
            in_channels=256,
            out_channels=num_classes,
            kernel_size=1,
            upsampling=4
        )
    
    def forward(self, rgb, nir, weather):
        # Encode both modalities
        rgb_features = self.rgb_encoder(rgb)
        nir_features = self.nir_encoder(nir)
        
        # Get weather embedding
        weather_emb = self.weather_embedding(weather)  # [B, 64]
        
        # Adaptive fusion at each level
        fused_features = []
        for rgb_feat, nir_feat, fusion_module in zip(
            rgb_features, nir_features, self.adaptive_fusion
        ):
            fused_feat = fusion_module(rgb_feat, nir_feat, weather_emb)
            fused_features.append(fused_feat)
        
        # Decode
        decoder_output = self.decoder(*fused_features)
        masks = self.segmentation_head(decoder_output)
        
        return masks


class WeatherConditionedFusion(nn.Module):
    """Weather-conditioned attention for feature fusion"""
    
    def __init__(self, channels, weather_dim=64):
        super().__init__()
        
        # Learn attention weights conditioned on weather
        self.weather_fc = nn.Sequential(
            nn.Linear(weather_dim, channels * 2),
            nn.ReLU(inplace=True),
            nn.Linear(channels * 2, 2),  # 2 attention weights (RGB, NIR)
            nn.Softmax(dim=1)
        )
        
        # Feature refinement
        self.conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, rgb_feat, nir_feat, weather_emb):
        B, C, H, W = rgb_feat.shape
        
        # Compute attention weights based on weather
        attention = self.weather_fc(weather_emb)  # [B, 2]
        rgb_weight = attention[:, 0].view(B, 1, 1, 1)
        nir_weight = attention[:, 1].view(B, 1, 1, 1)
        
        # Weighted combination
        weighted_rgb = rgb_feat * rgb_weight
        weighted_nir = nir_feat * nir_weight
        
        # Concatenate and refine
        combined = torch.cat([weighted_rgb, weighted_nir], dim=1)
        fused = self.conv(combined)
        
        return fused


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


def get_model(model_type='adaptive', num_classes=30, encoder_name='resnet50'):
    """Factory function to get model by type"""
    
    models = {
        'baseline': BaselineRGBModel,
        'early': EarlyFusionModel,
        'late': LateFusionModel,
        'adaptive': AdaptiveFusionModel
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return models[model_type](
        num_classes=num_classes,
        encoder_name=encoder_name,
        encoder_weights='imagenet'
    )


# Test models
if __name__ == '__main__':
    batch_size = 2
    rgb = torch.randn(batch_size, 3, 512, 1024)
    nir = torch.randn(batch_size, 1, 512, 1024)
    weather = torch.randint(0, 4, (batch_size,))
    
    print("Testing models...")
    
    for model_type in ['baseline', 'early', 'late', 'adaptive']:
        print(f"\n{model_type.upper()} Model:")
        model = get_model(model_type, num_classes=30)
        
        with torch.no_grad():
            output = model(rgb, nir, weather)
        
        print(f"  Output shape: {output.shape}")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
