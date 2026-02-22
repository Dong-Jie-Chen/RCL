"""
Model architectures for source-free domain adaptation.

The SHOT model (feature extractor + bottleneck + classifier) follows:
    Liang et al., "Do We Really Need to Access the Source Data?
    Source Hypothesis Transfer for Unsupervised Domain Adaptation", ICML 2020.
    https://github.com/tim-learn/SHOT
"""

import torch
import torch.nn as nn
import torch.nn.utils.weight_norm as weightNorm
from torchvision import models


res_dict = {
    "resnet50": models.resnet50,
    "resnet101": models.resnet101,
}


class ResBase(nn.Module):
    """ResNet feature extractor (without the final FC layer)."""

    def __init__(self, res_name):
        super(ResBase, self).__init__()
        model_resnet = res_dict[res_name](weights=None)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.in_features = model_resnet.fc.in_features

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


class feat_bottleneck(nn.Module):
    """Bottleneck layer with optional batch normalization."""

    def __init__(self, feature_dim, bottleneck_dim=256, type="ori"):
        super(feat_bottleneck, self).__init__()
        self.bn = nn.BatchNorm1d(bottleneck_dim, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)
        self.type = type

    def forward(self, x):
        x = self.bottleneck(x)
        if self.type == "bn":
            x = self.bn(x)
        return x


class feat_classifier(nn.Module):
    """Classifier head with optional weight normalization."""

    def __init__(self, class_num, bottleneck_dim=256, type="linear"):
        super(feat_classifier, self).__init__()
        self.type = type
        if type == 'wn':
            self.fc = weightNorm(nn.Linear(bottleneck_dim, class_num), name="weight")
        else:
            self.fc = nn.Linear(bottleneck_dim, class_num)

    def forward(self, x):
        return self.fc(x)


class SHOT(nn.Module):
    """SHOT model: ResNet backbone + bottleneck + classifier.

    Args:
        ckpts: dict with keys 'netF', 'netB', 'netC' pointing to checkpoint files,
               or None if loading from a full state_dict via prev_stage_ckpt.
        dataset: dataset name for config lookup.
        prev_stage_ckpt: path to a full model state_dict from a previous stage.
    """

    CONFIGS = {
        'office_home': {'arch': 'resnet50', 'class_num': 65, 'bottleneck_dim': 256},
        'domainnet': {'arch': 'resnet50', 'class_num': 126, 'bottleneck_dim': 256},
        'visda': {'arch': 'resnet101', 'class_num': 12, 'bottleneck_dim': 256},
    }

    def __init__(self, ckpts, dataset='office_home', prev_stage_ckpt=None):
        super(SHOT, self).__init__()
        cfg = self.CONFIGS[dataset]

        self.netF = ResBase(res_name=cfg['arch'])
        self.netB = feat_bottleneck(
            type='bn',
            feature_dim=self.netF.in_features,
            bottleneck_dim=cfg['bottleneck_dim'],
        )
        self.netC = feat_classifier(
            type='wn',
            class_num=cfg['class_num'],
            bottleneck_dim=cfg['bottleneck_dim'],
        )

        if prev_stage_ckpt is not None:
            self.load_state_dict(torch.load(prev_stage_ckpt))
        elif ckpts is not None:
            self.netF.load_state_dict(torch.load(ckpts['netF']))
            self.netB.load_state_dict(torch.load(ckpts['netB']))
            self.netC.load_state_dict(torch.load(ckpts['netC']))

    def forward(self, x, return_feature=False):
        if return_feature:
            return self.netB(self.netF(x))
        return self.netC(self.netB(self.netF(x)))

    def infer(self, x):
        return self.netC(self.netB(self.netF(x)))

    def get_feature(self, x):
        return self.netF(x)

    def get_output(self, x):
        return self.netC(self.netB(x))
