from typing import Any, Union
from PIL import Image
imageType = Union[Image.Image, Any]

from typing import List
from .schema import FiltererResponseSchema


from torchvision.transforms import Compose, Normalize, Resize, \
    ToTensor
import torch
from . import FineTunedVITModel, vit_config

MODEL_BASE = "google/vit-base-patch16-224-in21k"
MODEL_NAME = "./mlmodels/vit-potatoes-plant-health-status/"


class ImageFilterer:
    def __init__(self):
        # self.model_name = MODEL_NAME if Path(MODEL_NAME).exists() else MODEL_BASE
        # self.classifier = pipeline("image-classification", model=self.model_name)
        #
        # if not Path(MODEL_NAME).exists():
        #     self.classifier.save_pretrained(MODEL_NAME)
        FTMODEL = "/Users/gokul/Documents/GLABS/ML-take-home/ml-server/mlmodels/finetuned_filter_model_1_cpu.pt"
        self.ftmodel = FineTunedVITModel(vit_config)
        self.ftmodel.load_state_dict(torch.load(FTMODEL))
        self.ftmodel.eval()

    def filter(self, image: imageType) -> FiltererResponseSchema:
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        size = 224
        test_val_transforms = Compose(
            [
                Resize((224,224)),
                ToTensor(),
                Normalize(mean=mean, std=std)
            ])
        image = test_val_transforms(image)
        with torch.no_grad():
            output = self.ftmodel(image.unsqueeze(0))
        prediction = output.argmax(dim=1).item()
        return prediction
