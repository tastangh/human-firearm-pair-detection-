# src/model.py
import torch
import torch.nn as nn
import torchvision.models as models
import config

class DualStreamCNN(nn.Module):
    def __init__(self, num_classes=1, backbone_name='resnet34', pretrained=True):
        super(DualStreamCNN, self).__init__()
        
        weights = None
        if pretrained:
            if backbone_name == 'resnet18': weights = models.ResNet18_Weights.IMAGENET1K_V1
            elif backbone_name == 'resnet34': weights = models.ResNet34_Weights.IMAGENET1K_V1
            elif backbone_name == 'resnet50': weights = models.ResNet50_Weights.IMAGENET1K_V1
            else: raise ValueError(f"Desteklenmeyen backbone: {backbone_name}")

        # --- İnsan Akışı ---
        if backbone_name == 'resnet18':
            human_backbone = models.resnet18(weights=weights)
            num_ftrs_human = human_backbone.fc.in_features
        elif backbone_name == 'resnet34':
            human_backbone = models.resnet34(weights=weights)
            num_ftrs_human = human_backbone.fc.in_features
        elif backbone_name == 'resnet50':
            human_backbone = models.resnet50(weights=weights)
            num_ftrs_human = human_backbone.fc.in_features
        
        self.human_features = nn.Sequential(*list(human_backbone.children())[:-1]) # Son FC katmanını kaldır

        # --- Silah Akışı ---
        if backbone_name == 'resnet18':
            firearm_backbone = models.resnet18(weights=weights) # Aynı ağırlıklarla başlatılabilir veya farklı
            num_ftrs_firearm = firearm_backbone.fc.in_features
        elif backbone_name == 'resnet34':
            firearm_backbone = models.resnet34(weights=weights)
            num_ftrs_firearm = firearm_backbone.fc.in_features
        elif backbone_name == 'resnet50':
            firearm_backbone = models.resnet50(weights=weights)
            num_ftrs_firearm = firearm_backbone.fc.in_features
            
        self.firearm_features = nn.Sequential(*list(firearm_backbone.children())[:-1]) # Son FC katmanını kaldır

        # Sınıflandırıcı
        # ResNet özelliklerinin çıktısı (avgpool sonrası) [batch, num_ftrs, 1, 1] şeklindedir. Düzleştireceğiz.
        self.classifier = nn.Sequential(
            nn.Linear(num_ftrs_human + num_ftrs_firearm, 512),
            nn.BatchNorm1d(512), # Batch Normalization eklendi
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes) # İkili sınıflandırma için 1 logit çıktısı (BCEWithLogitsLoss ile kullanılacak)
        )

    def forward(self, human_roi, firearm_roi):
        h_feat = self.human_features(human_roi)
        f_feat = self.firearm_features(firearm_roi)

        h_feat = torch.flatten(h_feat, 1)
        f_feat = torch.flatten(f_feat, 1)
        
        # Özellikleri birleştir
        combined_feat = torch.cat((h_feat, f_feat), dim=1)
        
        output = self.classifier(combined_feat)
        return output

if __name__ == '__main__':
    model = DualStreamCNN(num_classes=1, backbone_name='resnet18') # Daha hızlı test için resnet18
    print(model)

    # Dummy input
    dummy_human_roi = torch.randn(4, 3, config.ROI_SIZE[0], config.ROI_SIZE[1])
    dummy_firearm_roi = torch.randn(4, 3, config.ROI_SIZE[0], config.ROI_SIZE[1])
    
    output = model(dummy_human_roi, dummy_firearm_roi)
    print("Model çıktı şekli:", output.shape) # Beklenen: [4, 1]