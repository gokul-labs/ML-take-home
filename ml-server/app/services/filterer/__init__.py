import torch.nn as nn
from torch import Tensor
from transformers import ViTConfig, ViTModel
vit_config = ViTConfig()


class FineTunedVITModel(nn.Module):
    """
    Fine tuned ViT Model for filtering images
    """
    def __init__(self, config: ViTConfig=vit_config, num_labels: int = 2):
        super(FineTunedVITModel, self).__init__()

        self.finetunedmodel = ViTModel(vit_config,
                                       add_pooling_layer=False)
        self.custom_filter = (
            nn.Linear(vit_config.hidden_size, num_labels)
        )

    def forward(self, x: Tensor):
        """
        Inference against the filter model
        :param x: image as tensor
        :return: Filter status
        """
        x = self.finetunedmodel(x)['last_hidden_state']
        output = self.custom_filter(x[:, 0, :])
        return output
