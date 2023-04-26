from typing import Any, Union
from typing import List
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms import Compose, Normalize, Resize, \
    ToTensor

from .__init__ import FineTunedVITModel, vit_config
from .schema import ClassifierResponseSchema

imageType = Union[Image.Image, Any]
MODEL_BASE = "google/vit-base-patch16-224-in21k"
MODEL_NAME = "./mlmodels/vit-potatoes-plant-health-status/"
LABELS = ["Early blight", "Healthy", "Late blight"]


class ImageClassifier:
    """
    Class implementation for classifying images uploaded to the service
    into 3 health status
    """
    def __init__(self):
        # self.model_name = MODEL_NAME if Path(MODEL_NAME).exists()\
        #     else MODEL_BASE
        # self.classifier = pipeline("image-classification",
        #                            model=self.model_name)
        #
        # if not Path(MODEL_NAME).exists():
        #     self.classifier.save_pretrained(MODEL_NAME)
        FTMODEL = "mlmodels/finetuned_model_2_cpu.pt"
        self.ftmodel = FineTunedVITModel(vit_config)
        self.ftmodel.load_state_dict(torch.load(FTMODEL))
        self.ftmodel.eval()

    def map_prediction_to_label(self, prediction):
        """
        Map the model prediction to string labels
        :param prediction: prediction with probabilities
        :return: prediction with labels and scores
        """
        probs = nn.functional.softmax(prediction, dim=-1)
        probs = np.ndarray.tolist(probs.cpu().detach().numpy())
        ordered_probs = [i[0] for i in sorted(enumerate(probs[0]),
                                              key=lambda x: x[1])]
        result = []
        for idx in range(2, -1, -1):
            temp = {"label": "",
                    "score": ""}
            marker = ordered_probs.index(idx)
            temp["label"] = LABELS[marker]
            temp["score"] = probs[0][marker]
            result.append(temp)
        return result

    def classify(self, image: imageType) -> List[ClassifierResponseSchema]:
        """
        Function to check the image against the classification model
        :param image: Input image
        :return: Classification status
        """
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
