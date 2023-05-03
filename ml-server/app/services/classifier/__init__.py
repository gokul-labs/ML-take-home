import torch.nn as nn
from transformers import ViTConfig, ViTModel
from torch import Tensor

vit_config = ViTConfig()


class FineTunedVITModel(nn.Module):
    """
    Fine tuned ViT Model for identifying leaf
     health status
    """
    def __init__(self, config: ViTConfig=vit_config, num_labels: int = 3):
        super(FineTunedVITModel, self).__init__()

        self.finetunedmodel = ViTModel(vit_config)
        self.custom_classifier = (
            nn.Linear(vit_config.hidden_size, num_labels)
        )

    def forward(self, x: Tensor):
        """
        Inference against the classification model
        :param x: image as tensor
        :return: Classification status
        """
        x = self.finetunedmodel(x)['last_hidden_state']
        output = self.custom_classifier(x[:, 0, :])
        return output
