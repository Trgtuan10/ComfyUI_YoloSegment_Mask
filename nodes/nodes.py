from combine_mask import Obscuring_object_pipeline
from PIL import Image
import torch
import comfyui


class ObjectMaskNode(comfyui.Node):
    @classmethod
    def INPUT_TYPES(cls):
        # Define the required input types
        return {
            "required": {
                "image": ("IMAGE",),
                "option": (["all", "overlap face"],),
            }
        }

    RETURN_TYPES = ("MASK",) 
    FUNCTION = "create_mask"
    OUTPUT_NODE = True

    def __init__(self):
        super().__init__()
        self.pipeline = Obscuring_object_pipeline("../yolov8x-seg.pt", "jonathandinu/face-parsing")

    def create_mask(self, image: Image.Image, option: str) -> Image.Image:
        if not isinstance(image, Image.Image):
            raise ValueError("The input 'image' must be a PIL Image.")
        
        # Generate the mask using the pipeline
        mask = self.pipeline(image, option)
        
        return mask