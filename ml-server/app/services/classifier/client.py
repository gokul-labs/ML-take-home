from typing import Any, List, Union
from pathlib import Path
from PIL import Image
from transformers import pipeline
from redis import connection

from app.models.schemas.classifier import ClassifierResponseSchema

imageType = Union[Image.Image, Any]

MODEL_BASE = "google/vit-base-patch16-224-in21k"
MODEL_NAME = "./mlmodels/vit-potatoes-plant-health-status/"

from torchvision.transforms import CenterCrop, Compose, Normalize, RandomHorizontalFlip, RandomResizedCrop, Resize, ToTensor
import torch
import torch
import torch.nn as nn
from transformers import ViTConfig, ViTModel

vit_config = ViTConfig()
print(vit_config)


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

import torch.nn.functional as nnf

class ImageClassifier:
    def __init__(self):
        # self.model_name = MODEL_NAME if Path(MODEL_NAME).exists() else MODEL_BASE
        # self.classifier = pipeline("image-classification", model=self.model_name)
        #
        # if not Path(MODEL_NAME).exists():
        #     self.classifier.save_pretrained(MODEL_NAME)

        FTMODEL = "./mlmodels/finetuned_model_1_cpu.pt"
        self.ftmodel = FineTunedVITModel(vit_config)
        self.ftmodel.load_state_dict(torch.load(FTMODEL))
        self.ftmodel.eval()

    def predict(self, image: imageType) -> List[ClassifierResponseSchema]:
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        size = 224
        test_val_transforms = Compose(
            [
                Resize(size),
                ToTensor(),
                Normalize(mean=mean, std=std)
            ])
        image = test_val_transforms(image)
        with torch.no_grad():
            output = self.ftmodel(image.unsqueeze(0))
        prediction = output.argmax(dim=1).item()
        print(prediction)
        return prediction
