from PIL import Image
import numpy as np
from torch import nn

from ultralytics import YOLO
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

image_processor = SegformerImageProcessor.from_pretrained("jonathandinu/face-parsing")
face_parsing_model = SegformerForSemanticSegmentation.from_pretrained("jonathandinu/face-parsing").to("cuda")    

yolo_model = YOLO("../../../models/yolo/yolov8x-seg.pt")

class Obscuring_object_pipeline:
    def __init__(self, yolo_model_path, face_parsing_model_path):
        self.yolo_model = YOLO(yolo_model_path)
        self.face_parsing_model = SegformerForSemanticSegmentation.from_pretrained(face_parsing_model_path).to("cuda")
        self.image_processor = SegformerImageProcessor.from_pretrained(face_parsing_model_path)
        self.back_ground_value = 100

    def _get_estimate_face_mask(self, image: Image.Image) -> np.ndarray:
        inputs = image_processor(images=image, return_tensors="pt").to("cuda")
        outputs = face_parsing_model(**inputs)
        logits = outputs.logits
        upsampled_logits = nn.functional.interpolate(logits, size=image.size[::-1], mode='bilinear', align_corners=False)
        labels = upsampled_logits.argmax(dim=1)[0]
        labels = labels.cpu().numpy()
        non_face_mask = (labels == 0).astype(np.uint8) +\
                        (labels == 16).astype(np.uint8) +\
                        (labels == 17).astype(np.uint8) +\
                        (labels == 18).astype(np.uint8) +\
                        (labels == 13).astype(np.uint8)
        face_mask = 1 - non_face_mask
        return face_mask

    def _yolo_segmentation_map(self, image: Image.Image) -> np.ndarray:
        results = yolo_model(image)[0]
        masks_array = results.masks.data.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy()

        labels_mask = np.zeros((image.size[1], image.size[0]), dtype=np.int32)
        labels_mask.fill(self.back_ground_value)

        for i, (mask, class_id) in enumerate(zip(masks_array, class_ids)):
            mask_image = Image.fromarray(mask * 255).convert("L")
            mask_image = mask_image.resize(image.size)
            mask_resized = np.array(mask_image) // 255
            labels_mask[mask_resized == 1] = class_id

        return labels_mask

    def _get_obscuring_area(self, face_mask: np.ndarray, yolo_mask: np.ndarray) -> np.ndarray:
        face_yolo = (yolo_mask == 0).astype(np.uint8)
        area = np.logical_and(face_mask, np.logical_not(face_yolo))

        return area

    def __call__(self, image: Image.Image, option: str) -> np.ndarray:
        face_mask = self._get_estimate_face_mask(image)
        yolo_mask = self._yolo_segmentation_map(image)

        obscuring_area =  self._get_obscuring_area(face_mask, yolo_mask)

        #unique label of yolo_mask in the obscuring area
        unique_label = np.unique(yolo_mask[obscuring_area])

        #get the mask of unique_label
        obscuring_object_mask = np.zeros_like(yolo_mask)
        for label in unique_label:
            if label != 0 and label != self.back_ground_value:
                obscuring_object_mask[yolo_mask == label] = 1

        return obscuring_object_mask

import time
if __name__ == "__main__":
    image = Image.open("image7.png")
    pipeline = Obscuring_object_pipeline("/workspace/NTT/sam/checkpoints/yolov8x-seg.pt", "jonathandinu/face-parsing")
    t = time.time()
    mask = pipeline(image)
    print("Time:", time.time() - t)
    Image.fromarray(mask.astype(np.uint8) * 255).save("mask_final.png")
