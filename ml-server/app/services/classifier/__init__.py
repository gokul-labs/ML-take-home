import torch.nn as nn
from transformers import ViTConfig, ViTModel

vit_config = ViTConfig()


class FineTunedVITModel(nn.Module):
    def __init__(self, config=vit_config, num_labels=3):
        super(FineTunedVITModel, self).__init__()

        self.finetunedmodel = ViTModel(vit_config)
        self.custom_classifier = (
            nn.Linear(vit_config.hidden_size, num_labels)
        )

    def forward(self, x):
        x = self.finetunedmodel(x)['last_hidden_state']
        output = self.custom_classifier(x[:, 0, :])
        return output