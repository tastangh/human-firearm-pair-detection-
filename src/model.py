# src/model.py
import torch
import torch.nn as nn
import torchvision.models as models
import config
import numpy as np

class SaliencySingleStreamCNN(nn.Module):
    def __init__(self, num_classes_classifier=config.NUM_CLASSES_CLASSIFIER,
                 backbone_name='resnet18', pretrained=True, # Eğitimde 'resnet18' kullanıldı
                 num_attention_channels=config.NUM_ATTENTION_CHANNELS,
                 num_reconstruction_channels=config.NUM_RECONSTRUCTION_CHANNELS,
                 roi_size=config.ROI_SIZE,
                 color_space_aware=False): # YCbCr için özel bir işlem gerekirse diye
        super(SaliencySingleStreamCNN, self).__init__()
        
        self.color_space_aware = color_space_aware # Şu an kullanılmıyor ama gelecekte işe yarayabilir

        weights = None
        if pretrained:
            if backbone_name == 'resnet18': weights = models.ResNet18_Weights.IMAGENET1K_V1
            elif backbone_name == 'resnet34': weights = models.ResNet34_Weights.IMAGENET1K_V1
            elif backbone_name == 'resnet50': weights = models.ResNet50_Weights.IMAGENET1K_V1
            else: raise ValueError(f"Desteklenmeyen backbone: {backbone_name}")

        if backbone_name == 'resnet18':
            backbone = models.resnet18(weights=weights)
            encoder_feature_dim = backbone.fc.in_features
            decoder_input_channels = 256 # ResNet18 layer4 çıkışı (BN sonrası değil, conv sonrası)
            if hasattr(backbone.layer4[-1], 'conv2'): # BasicBlock
                decoder_input_channels = backbone.layer4[-1].conv2.out_channels
            elif hasattr(backbone.layer4[-1], 'conv3'): # BottleneckBlock (ResNet50+)
                decoder_input_channels = backbone.layer4[-1].conv3.out_channels

            if roi_size[0] == 224: decoder_start_spatial_dim = 7
            else: decoder_start_spatial_dim = roi_size[0] // 32
        elif backbone_name == 'resnet34':
            backbone = models.resnet34(weights=weights)
            encoder_feature_dim = backbone.fc.in_features
            decoder_input_channels = 512 # ResNet34 layer4 çıkışı
            if hasattr(backbone.layer4[-1], 'conv2'):
                decoder_input_channels = backbone.layer4[-1].conv2.out_channels
            if roi_size[0] == 224: decoder_start_spatial_dim = 7
            else: decoder_start_spatial_dim = roi_size[0] // 32
        elif backbone_name == 'resnet50':
            backbone = models.resnet50(weights=weights)
            encoder_feature_dim = backbone.fc.in_features
            decoder_input_channels = 2048 # ResNet50 layer4 çıkışı (bottleneck sonrası)
            if hasattr(backbone.layer4[-1], 'conv3'):
                decoder_input_channels = backbone.layer4[-1].conv3.out_channels
            if roi_size[0] == 224: decoder_start_spatial_dim = 7
            else: decoder_start_spatial_dim = roi_size[0] // 32
        else:
            raise ValueError(f"Desteklenmeyen backbone: {backbone_name}")

        original_conv1 = backbone.conv1
        self.modified_conv1 = nn.Conv2d(
            3 + num_attention_channels,
            original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=original_conv1.bias
        )
        if pretrained:
            self.modified_conv1.weight.data[:, :3, :, :] = original_conv1.weight.data.clone()
            if num_attention_channels > 0:
                torch.nn.init.kaiming_normal_(self.modified_conv1.weight.data[:, 3:, :, :], mode='fan_in', nonlinearity='relu')
        
        self.feb_conv1 = self.modified_conv1
        self.feb_bn1 = backbone.bn1
        self.feb_relu = backbone.relu
        self.feb_maxpool = backbone.maxpool
        self.feb_layer1 = backbone.layer1
        self.feb_layer2 = backbone.layer2
        self.feb_layer3 = backbone.layer3
        self.feb_layer4 = backbone.layer4

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(encoder_feature_dim, 512),
            nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(256, num_classes_classifier)
        )

        decoder_layers = []
        current_channels = decoder_input_channels
        # Decoder'ı roi_size'a kadar upsample edecek şekilde dinamik oluşturma
        # Her adımda spatial boyutu ikiye katlamayı hedefleriz.
        # Örn: 7 -> 14 -> 28 -> 56 -> 112 -> 224 (5 adım)
        num_upsample_steps = int(np.log2(roi_size[0] / decoder_start_spatial_dim))

        for i in range(num_upsample_steps):
            out_channels = current_channels // 2 if current_channels // 2 >= 32 else 32 # min 32 kanal
            if i == num_upsample_steps -1 : # Son upsample katmanıysa
                out_channels = num_reconstruction_channels if num_upsample_steps > 0 else current_channels # Eğer hiç upsample yoksa
            
            if current_channels == 0: # Hata durumunu engellemek için
                print(f"UYARI: Decoder input kanalı sıfır oldu! (decoder_input_channels: {decoder_input_channels}, current_channels: {current_channels})")
                break

            decoder_layers.append(nn.ConvTranspose2d(current_channels, out_channels, kernel_size=4, stride=2, padding=1))
            
            if i < num_upsample_steps - 1: # Son katman hariç ReLU ve BatchNorm
                decoder_layers.append(nn.ReLU(inplace=True))
                decoder_layers.append(nn.BatchNorm2d(out_channels))
            current_channels = out_channels
        
        # Eğer hiç upsample adımı yoksa (örn. decoder_start_spatial_dim == roi_size)
        # ve kanal sayısı farklıysa, bir 1x1 conv ekle
        if num_upsample_steps == 0 and current_channels != num_reconstruction_channels:
            decoder_layers.append(nn.Conv2d(current_channels, num_reconstruction_channels, kernel_size=1))

        decoder_layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, pbb_image_tensor, human_attention_mask, firearm_attention_mask):
        input_tensor = torch.cat((pbb_image_tensor, human_attention_mask, firearm_attention_mask), dim=1)
        
        x = self.feb_conv1(input_tensor)
        x = self.feb_bn1(x)
        x = self.feb_relu(x)
        x = self.feb_maxpool(x)
        x = self.feb_layer1(x)
        x = self.feb_layer2(x)
        x = self.feb_layer3(x)
        encoded_features = self.feb_layer4(x)

        class_feat = self.avgpool(encoded_features)
        class_feat = torch.flatten(class_feat, 1)
        classification_output = self.classifier(class_feat)

        reconstructed_masks = self.decoder(encoded_features)

        return classification_output, reconstructed_masks

if __name__ == '__main__':
    model = SaliencySingleStreamCNN(backbone_name='resnet18', roi_size=(224,224)).to(config.DEVICE)
    print(model)
    batch_size = 2
    dummy_pbb = torch.randn(batch_size, 3, config.ROI_SIZE[0], config.ROI_SIZE[1]).to(config.DEVICE)
    dummy_human_mask = torch.rand(batch_size, 1, config.ROI_SIZE[0], config.ROI_SIZE[1]).to(config.DEVICE)
    dummy_firearm_mask = torch.rand(batch_size, 1, config.ROI_SIZE[0], config.ROI_SIZE[1]).to(config.DEVICE)
    
    classification_out, reconstruction_out = model(dummy_pbb, dummy_human_mask, dummy_firearm_mask)
    print("\nSınıflandırma Çıktı Şekli:", classification_out.shape)
    print("Rekonstrüksiyon Çıktı Şekli:", reconstruction_out.shape) # Beklenen [B, num_recon_channels, roi_H, roi_W]
    assert reconstruction_out.shape[2] == config.ROI_SIZE[0] and reconstruction_out.shape[3] == config.ROI_SIZE[1]
    assert reconstruction_out.shape[1] == config.NUM_RECONSTRUCTION_CHANNELS