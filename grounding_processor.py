import os
import torch
import numpy as np
import cv2
from PIL import Image
# from transformers import AutoProcessor, AutoModelForObjectDetection
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from typing import List, Tuple, Optional, Union

def calculate_iou(bbx1: List[float], bbx2: List[float]) -> Tuple[float, float]:
    """Calculate IoU between two bounding boxes."""
    x1 = max(bbx1[0], bbx2[0])
    y1 = max(bbx1[1], bbx2[1])
    x2 = min(bbx1[2], bbx2[2])
    y2 = min(bbx1[3], bbx2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (bbx1[2] - bbx1[0]) * (bbx1[3] - bbx1[1])
    area2 = (bbx2[2] - bbx2[0]) * (bbx2[3] - bbx2[1])
    
    ratio1 = intersection / area1 if area1 > 0 else 0
    ratio2 = intersection / area2 if area2 > 0 else 0
    
    return ratio1, ratio2

def get_initial_bboxes(
    image: Union[str, np.ndarray], 
    text_prompt: str, 
    confidence_threshold: float = 0.2,
    model_id: str = "IDEA-Research/grounding-dino-tiny"
) -> Tuple[List[np.ndarray], List[int]]:
    """
    Generate initial bounding boxes using Grounding DINO for a given image and text prompt.
    
    Args:
        image: Path to image file (str) or numpy array of image (np.ndarray)
        text_prompt: Text description of objects to detect
        confidence_threshold: Confidence threshold for filtering detections
        model_id: Model ID from HuggingFace
        
    Returns:
        Tuple containing:
        - List of bounding boxes in format [np.array([x1, y1, x2, y2]), ...]
        - List of object IDs starting from 0
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    try:
        # Handle different input types
        if isinstance(image, str):
            # Check if image path exists
            if not os.path.exists(image):
                raise FileNotFoundError(f"Image not found: {image}")
            # Load image using PIL
            pil_image = Image.open(image)
            print(f"Image loaded successfully from path: {image}")
        else:
            # Convert numpy array to PIL Image
            if not isinstance(image, np.ndarray):
                raise ValueError("Image must be either a path string or numpy array")
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            if len(image.shape) == 2:  # Grayscale
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif len(image.shape) == 3 and image.shape[2] == 3:  # BGR
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image)
            print("Image converted from numpy array to PIL Image")
            
        print(f"Image size: {pil_image.size}")
        
        # Initialize model and processor
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
        
        # Prepare inputs
        inputs = processor(
            images=pil_image,
            text=text_prompt,
            return_tensors="pt"
        ).to(device)
        
        print(f"Running inference with text prompt: '{text_prompt}'")
        with torch.no_grad():
            outputs = model(**inputs)
            
        # Post-process outputs
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=confidence_threshold,
            text_threshold=confidence_threshold,
            target_sizes=[pil_image.size[::-1]]  # Convert (W,H) to (H,W)
        )[0]
        
        boxes = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        
        print(f"Found {len(boxes)} detections before confidence filtering")
        print(f"Scores: {scores}")
        
        # Convert boxes to numpy array format and filter by confidence
        filtered_boxes = []
        filtered_scores = []
        for box, score in zip(boxes, scores):
            if score >= confidence_threshold:
                filtered_boxes.append(np.array(box, dtype=np.float32))
                filtered_scores.append(score)
        
        print(f"Kept {len(filtered_boxes)} detections after confidence filtering (threshold={confidence_threshold})")
        if filtered_boxes:
            print(f"Final boxes: {filtered_boxes}")
            print(f"Final scores: {filtered_scores}")
        
        # Generate object IDs
        obj_ids = list(range(len(filtered_boxes)))
        
        return filtered_boxes, obj_ids
        
    except Exception as e:
        print(f"Error during object detection: {str(e)}")
        import traceback
        traceback.print_exc()
        return [], []
