from typing import Any, Union
from PIL import Image

from typing import List
from . import FineTunedVITModel, vit_config
from .schema import ClassifierResponseSchema
from torchvision.transforms import Compose, Normalize, Resize, \
    ToTensor
import torch
import torch.nn as nn
import numpy as np


imageType = Union[Image.Image, Any]
MODEL_BASE = "google/vit-base-patch16-224-in21k"
MODEL_NAME = "./mlmodels/vit-potatoes-plant-health-status/"
labels = ["early_blight", "healthy", "late_blight"]

class ImageClassifier:
    def __init__(self):
        # self.model_name = MODEL_NAME if Path(MODEL_NAME).exists() else MODEL_BASE
        # self.classifier = pipeline("image-classification", model=self.model_name)
        #
        # if not Path(MODEL_NAME).exists():
        #     self.classifier.save_pretrained(MODEL_NAME)
        FTMODEL = "/Users/gokul/Documents/GLABS/ML-take-home/ml-server/mlmodels/finetuned_model_2_cpu.pt"
        self.ftmodel = FineTunedVITModel(vit_config)
        self.ftmodel.load_state_dict(torch.load(FTMODEL))
        self.ftmodel.eval()

    def map_prediction_to_label(self, prediction):
        probs = nn.functional.softmax(prediction, dim=-1)
        probs = np.ndarray.tolist(probs.cpu().detach().numpy())
        ordered_probs = [i[0] for i in sorted(enumerate(probs[0]), key=lambda x: x[1])]
        result = []
        for idx in range(2, -1, -1):
            temp = {"label": "",
                    "score": ""}
            marker = ordered_probs.index(idx)
            temp["label"] = labels[marker]
            temp["score"] = probs[0][marker]
            result.append(temp)
        return result

    def classify(self, image: imageType) -> List[ClassifierResponseSchema]:
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        size = 224
        test_val_transforms = Compose(
            [
                Resize((size, size)),
                ToTensor(),
                Normalize(mean=mean, std=std)
            ])
        image = test_val_transforms(image)
        with torch.no_grad():
            output = self.ftmodel(image.unsqueeze(0))
        result = self.map_prediction_to_label(output)
        return result
