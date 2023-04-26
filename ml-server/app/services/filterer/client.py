from typing import Any, Union
import torch
from PIL import Image
from torchvision.transforms import Compose, Normalize, Resize, \
    ToTensor

from .__init__ import FineTunedVITModel, vit_config
from .schema import FiltererResponseSchema

MODEL_BASE = "google/vit-base-patch16-224-in21k"
MODEL_NAME = "./mlmodels/vit-potatoes-plant-health-status/"
IMAGE_TYPE = Union[Image.Image, Any]


class ImageFilterer:
    """
    Class implementation for filtering images uploaded to the service.
    The intent is to filter out images that are not relevant
    i.e) not a leaf image
    """
    def __init__(self):
        # self.model_name = MODEL_NAME if Path(MODEL_NAME).exists()\
        #     else MODEL_BASE
        # self.classifier = pipeline("image-classification",
        #                            model=self.model_name)
        #
        # if not Path(MODEL_NAME).exists():
        #     self.classifier.save_pretrained(MODEL_NAME)

        FTMODEL = "mlmodels/finetuned_filter_model_1_cpu.pt"
        self.ftmodel = FineTunedVITModel(vit_config)
        self.ftmodel.load_state_dict(torch.load(FTMODEL))
        self.ftmodel.eval()

    def filter(self, image: IMAGE_TYPE) -> FiltererResponseSchema:
        """
        Function to check the image against the filtering model
        :param image: Input image
        :return: Filter status
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
        prediction = output.argmax(dim=1).item()
        return prediction
