import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

class EarlyFusionModel(nn.Module):
    """Simple early fusion: concatenate RGB + NIR as 4-channel input"""
    
    def __init__(self, num_classes=30, encoder_name='resnet50', encoder_weights='imagenet'):
        super().__init__()
        
        # Create base model with 3 channels first
        base_model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=num_classes
        )
        
        # Modify first conv to accept 4 channels (RGB + NIR)
        original_conv = base_model.encoder.conv1
        new_conv = nn.Conv2d(4, original_conv.out_channels, 
                           kernel_size=original_conv.kernel_size,
                           stride=original_conv.stride,
                           padding=original_conv.padding,
                           bias=False if original_conv.bias is None else True)
        
        # Copy RGB weights and initialize NIR channel
        if encoder_weights == 'imagenet':
            with torch.no_grad():
                new_conv.weight[:, :3, :, :] = original_conv.weight
                # Initialize NIR channel as average of RGB
                new_conv.weight[:, 3:4, :, :] = original_conv.weight.mean(dim=1, keepdim=True)
        
        base_model.encoder.conv1 = new_conv
        self.model = base_model
    
    def forward(self, rgb, nir, weather=None):
        # Concatenate RGB and NIR
        x = torch.cat([rgb, nir], dim=1)
        return self.model(x)


class LateFusionModel(nn.Module):
    """Late fusion: separate encoders, fuse features and use simple decoder"""
    
    def __init__(self, num_classes=30, encoder_name='resnet50', encoder_weights='imagenet'):
        super().__init__()
        
        # RGB encoder
        self.rgb_encoder = smp.encoders.get_encoder(
            encoder_name,
            in_channels=3,
            depth=5,
            weights=encoder_weights
        )
        
        # NIR preprocessing: expand to 3 channels
        self.nir_preprocess = nn.Conv2d(1, 3, kernel_size=1, bias=False)
        with torch.no_grad():
            self.nir_preprocess.weight.fill_(1.0/3.0)
        
        # NIR encoder
        self.nir_encoder = smp.encoders.get_encoder(
            encoder_name,
            in_channels=3,
            depth=5,
            weights=encoder_weights
        )
        
        # Get encoder output channels
        encoder_channels = self.rgb_encoder.out_channels
        
        # Fusion at bottleneck
        self.bottleneck_fusion = nn.Sequential(
            nn.Conv2d(encoder_channels[-1] * 2, encoder_channels[-1], kernel_size=1),
            nn.BatchNorm2d(encoder_channels[-1]),
            nn.ReLU(inplace=True)
        )
        
        # Simple decoder head
        self.decoder = nn.Sequential(
            nn.Conv2d(encoder_channels[-1], 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        )
        
        self.segmentation_head = nn.Conv2d(64, num_classes, kernel_size=1)
    
    def forward(self, rgb, nir, weather=None):
        # Preprocess NIR
        nir_3ch = self.nir_preprocess(nir)
        
        # Get features from both encoders
        rgb_features = self.rgb_encoder(rgb)
        nir_features = self.nir_encoder(nir_3ch)
        
        # Fuse at bottleneck
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
        
        # Concatenate and fuse
        fused = torch.cat([rgb_bottleneck, nir_bottleneck], dim=1)
        fused = self.bottleneck_fusion(fused)
        
        # Decode
        x = self.decoder(fused)
        
        # Final upsampling to match input size
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
        
        # Segmentation head
        output = self.segmentation_head(x)
        
        return output


class AdaptiveFusionModel(nn.Module):
    """Weather-conditioned adaptive fusion"""
    
    def __init__(self, num_classes=30, encoder_name='resnet50', encoder_weights='imagenet'):
        super().__init__()
        
        # RGB encoder
        self.rgb_encoder = smp.encoders.get_encoder(
            encoder_name,
            in_channels=3,
            depth=5,
            weights=encoder_weights
        )
        
        # NIR preprocessing: expand to 3 channels
        self.nir_preprocess = nn.Conv2d(1, 3, kernel_size=1, bias=False)
        with torch.no_grad():
            self.nir_preprocess.weight.fill_(1.0/3.0)
        
        # NIR encoder
        self.nir_encoder = smp.encoders.get_encoder(
            encoder_name,
            in_channels=3,
            depth=5,
            weights=encoder_weights
        )
        
        # Weather embedding
        self.num_weather = 4
        self.weather_embedding = nn.Embedding(self.num_weather, 64)
        
        # Get encoder output channels
        encoder_channels = self.rgb_encoder.out_channels
        
        # Weather-conditioned fusion
        self.adaptive_fusion = WeatherConditionedFusion(
            encoder_channels[-1], 
            weather_dim=64
        )
        
        # Simple decoder head
        self.decoder = nn.Sequential(
            nn.Conv2d(encoder_channels[-1], 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        )
        
        self.segmentation_head = nn.Conv2d(64, num_classes, kernel_size=1)
    
    def forward(self, rgb, nir, weather):
        # Preprocess NIR
        nir_3ch = self.nir_preprocess(nir)
        
        # Get features from both encoders
        rgb_features = self.rgb_encoder(rgb)
        nir_features = self.nir_encoder(nir_3ch)
        
        # Get weather embedding
        weather_emb = self.weather_embedding(weather)
        
        # Adaptive fusion at bottleneck
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
        
        fused = self.adaptive_fusion(rgb_bottleneck, nir_bottleneck, weather_emb)
        
        # Decode
        x = self.decoder(fused)
        
        # Final upsampling to match input size
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
        
        # Segmentation head
        output = self.segmentation_head(x)
        
        return output


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
        
        # Feature refinement after fusion
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
        
        # Concatenate and refine to original channels
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