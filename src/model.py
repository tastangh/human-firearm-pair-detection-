# src/model.py
import torch
import torch.nn as nn
import torchvision.models as models
import config

class SaliencySingleStreamCNN(nn.Module):
    def __init__(self, num_classes_classifier=config.NUM_CLASSES_CLASSIFIER,
                 backbone_name='resnet18', pretrained=True,
                 num_attention_channels=config.NUM_ATTENTION_CHANNELS,
                 num_reconstruction_channels=config.NUM_RECONSTRUCTION_CHANNELS,
                 roi_size=config.ROI_SIZE):
        super(SaliencySingleStreamCNN, self).__init__()
        
        weights = None
        if pretrained:
            if backbone_name == 'resnet18': weights = models.ResNet18_Weights.IMAGENET1K_V1
            elif backbone_name == 'resnet34': weights = models.ResNet34_Weights.IMAGENET1K_V1
            elif backbone_name == 'resnet50': weights = models.ResNet50_Weights.IMAGENET1K_V1
            else: raise ValueError(f"Desteklenmeyen backbone: {backbone_name}")

        if backbone_name == 'resnet18':
            backbone = models.resnet18(weights=weights)
            encoder_feature_dim = backbone.fc.in_features # Genellikle 512
            # ResNet18'in layer4'ü 256 kanal çıkarır, avgpool sonrası 512 olur, biz layer4'ü alacağız.
            # layer4'ün çıktısı (örneğin, resnet18 için) [B, 256, H/32, W/32]
            # Eğer doğrudan fc.in_features alırsak, bu avgpool sonrası değerdir.
            # Decoder için layer4 çıktısının kanal sayısını bilmeliyiz.
            # Resnet18/34 için layer4 çıkışı 512 kanal, spatial 7x7 (224x224 input için)
            decoder_input_channels = 512
            if roi_size[0] == 224: # Yaygın durum
                 decoder_start_spatial_dim = 7
            else: # Daha genel bir hesaplama gerekebilir veya sabit varsayılabilir
                 decoder_start_spatial_dim = roi_size[0] // 32


        elif backbone_name == 'resnet34':
            backbone = models.resnet34(weights=weights)
            encoder_feature_dim = backbone.fc.in_features # 512
            decoder_input_channels = 512
            if roi_size[0] == 224: decoder_start_spatial_dim = 7
            else: decoder_start_spatial_dim = roi_size[0] // 32

        elif backbone_name == 'resnet50':
            backbone = models.resnet50(weights=weights)
            encoder_feature_dim = backbone.fc.in_features # 2048
            decoder_input_channels = 2048 # Bottleneck yapısında layer4 çıkışı
            if roi_size[0] == 224: decoder_start_spatial_dim = 7
            else: decoder_start_spatial_dim = roi_size[0] // 32
        else:
            raise ValueError(f"Desteklenmeyen backbone: {backbone_name}")

        # --- Encoder (Feature Encoding Block - FEB) ---
        original_conv1 = backbone.conv1
        # Yeni conv1: RGB (3) + dikkat kanalları
        self.modified_conv1 = nn.Conv2d(
            3 + num_attention_channels,
            original_conv1.out_channels, # genelde 64
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=original_conv1.bias
        )
        if pretrained: # Önceden eğitilmiş ağırlıkların RGB kısmını kopyala
            self.modified_conv1.weight.data[:, :3, :, :] = original_conv1.weight.data.clone()
            if num_attention_channels > 0: # Dikkat kanalları için ağırlıkları sıfırla veya Kaiming ile başlat
                torch.nn.init.kaiming_normal_(self.modified_conv1.weight.data[:, 3:, :, :], mode='fan_in', nonlinearity='relu')
        
        # FEB'in parçaları
        self.feb_conv1 = self.modified_conv1
        self.feb_bn1 = backbone.bn1
        self.feb_relu = backbone.relu
        self.feb_maxpool = backbone.maxpool
        self.feb_layer1 = backbone.layer1
        self.feb_layer2 = backbone.layer2
        self.feb_layer3 = backbone.layer3
        self.feb_layer4 = backbone.layer4 # Bu, decoder için kullanılacak özellik haritası

        # --- Classifier Head ---
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # Global average pooling
        self.classifier = nn.Sequential(
            nn.Linear(encoder_feature_dim, 512), # encoder_feature_dim, backbone.fc.in_features ile aynı olmalı
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes_classifier) # BCEWithLogitsLoss için 1 çıktı
        )

        # --- Decoder (Saliency Reconstruction) ---
        # decoder_input_channels'dan (örn. 512 ResNet34 için) num_reconstruction_channels'a (örn. 2)
        # ve decoder_start_spatial_dim'den (örn. 7) roi_size'a (örn. 224) upsample edecek.
        # Bu, basit bir FCN tarzı decoder örneğidir. Daha karmaşık (örn. U-Net skip connections) olabilir.
        layers = []
        current_channels = decoder_input_channels
        current_spatial_dim = decoder_start_spatial_dim

        # Hedef roi_size'a ulaşana kadar upsample et
        # Her adımda spatial boyutu ikiye katlamayı hedefliyoruz
        while current_spatial_dim < roi_size[0]:
            out_channels = current_channels // 2
            if out_channels < 32 : out_channels = 32 # Minimum kanal sayısı
            if current_spatial_dim * 2 > roi_size[0] and current_spatial_dim < roi_size[0]: # Son adımda tam boyuta getir
                target_spatial_dim = roi_size[0]
                stride = target_spatial_dim // current_spatial_dim # Tam sayı olmalı, değilse padding/kernel ayarı gerekir
                kernel_size = stride # Veya 2*stride -1 gibi
                padding = 0 # Veya (kernel_size - stride) // 2
                # Bu kısım biraz hassas, kernel, stride, padding iyi ayarlanmalı
                # Genellikle kernel=4, stride=2, padding=1 iyi çalışır ve boyutu 2 katına çıkarır
                # Eğer tam 2 katı değilse, son katmanda ConvTranspose2d yerine Interplate + Conv2d denenebilir.
                # Şimdilik standart ConvTranspose2d ile devam edelim.
                layers.append(nn.ConvTranspose2d(current_channels, out_channels, kernel_size=4, stride=2, padding=1))

            else: # Standart 2x upsample
                 layers.append(nn.ConvTranspose2d(current_channels, out_channels, kernel_size=4, stride=2, padding=1))

            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.BatchNorm2d(out_channels))
            current_channels = out_channels
            current_spatial_dim *= 2
            if current_spatial_dim >= roi_size[0]: # Son hedef boyuta ulaşıldıysa veya aşıldıysa
                break
        
        # Son katman: num_reconstruction_channels'a düşür ve Sigmoid
        layers.append(nn.Conv2d(current_channels, num_reconstruction_channels, kernel_size=1, stride=1, padding=0)) # 1x1 conv
        layers.append(nn.Sigmoid()) # Maske olasılıkları için (0-1 aralığında)
        
        self.decoder = nn.Sequential(*layers)

    def forward(self, pbb_image_tensor, human_attention_mask, firearm_attention_mask):
        # Dikkat maskeleri [B, 1, H, W] şeklinde olmalı
        # Dataset'ten gelenler zaten bu şekilde (ToTensor sonrası)
        
        # Giriş tensörünü oluştur: PBB görüntüsü + dikkat maskeleri
        # Maskeler zaten 0-1 aralığında olmalı (datasetten)
        input_tensor = torch.cat((pbb_image_tensor, human_attention_mask, firearm_attention_mask), dim=1)
        
        # FEB (Encoder)
        x = self.feb_conv1(input_tensor)
        x = self.feb_bn1(x)
        x = self.feb_relu(x)
        x = self.feb_maxpool(x) # Spatial: 224->112->56
        
        x = self.feb_layer1(x) # Spatial: 56
        x = self.feb_layer2(x) # Spatial: 28
        x = self.feb_layer3(x) # Spatial: 14
        encoded_features = self.feb_layer4(x) # Spatial: 7 (ResNet için), Kanal: decoder_input_channels

        # Classifier path
        class_feat = self.avgpool(encoded_features)
        class_feat = torch.flatten(class_feat, 1)
        classification_output = self.classifier(class_feat)

        # Decoder path
        reconstructed_masks = self.decoder(encoded_features) # Çıktı: [B, num_recon_channels, roi_H, roi_W]

        return classification_output, reconstructed_masks

if __name__ == '__main__':
    model = SaliencySingleStreamCNN(backbone_name='resnet18').to(config.DEVICE) # Test için daha küçük model
    print(model)

    # Dummy input (Dataset'ten gelen formatta)
    batch_size = 4
    dummy_pbb = torch.randn(batch_size, 3, config.ROI_SIZE[0], config.ROI_SIZE[1]).to(config.DEVICE)
    dummy_human_mask = torch.rand(batch_size, 1, config.ROI_SIZE[0], config.ROI_SIZE[1]).to(config.DEVICE) # 0-1 arası
    dummy_firearm_mask = torch.rand(batch_size, 1, config.ROI_SIZE[0], config.ROI_SIZE[1]).to(config.DEVICE) # 0-1 arası
    
    classification_out, reconstruction_out = model(dummy_pbb, dummy_human_mask, dummy_firearm_mask)
    print("\nSınıflandırma Çıktı Şekli:", classification_out.shape) # Beklenen: [batch_size, 1]
    print("Rekonstrüksiyon Çıktı Şekli:", reconstruction_out.shape) # Beklenen: [batch_size, num_recon_channels, roi_H, roi_W]