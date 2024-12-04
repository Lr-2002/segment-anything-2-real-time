import os
import torch
import numpy as np
import cv2
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from typing import List, Tuple, Optional, Union

class GroundingDINOProcessor:
    def __init__(self, model_id: str = "IDEA-Research/grounding-dino-tiny"):
        """Initialize the Grounding DINO processor"""
        self.model_id = model_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")
        print(f"Initializing GroundingDINOProcessor on device: {self.device}")
        
        try:
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)
            self.model.eval()  # Set to evaluation mode
            

            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    @staticmethod
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
        self,
        image: Union[str, np.ndarray], 
        text_prompt: str, 
        confidence_threshold: float = 0.2
    ) -> Tuple[List[np.ndarray], List[int]]:
        """
        Generate initial bounding boxes using Grounding DINO for a given image and text prompt.
        """
        try:
            # Clear GPU memory before processing
            # if torch.cuda.is_available():
            #     torch.cuda.empty_cache()
            #     torch.cuda.synchronize()
            
            # Handle different input types
            if isinstance(image, str):
                if not os.path.exists(image):
                    raise FileNotFoundError(f"Image not found: {image}")
                pil_image = Image.open(image)
                print(f"Image loaded successfully from path: {image}")
            else:
                if not isinstance(image, np.ndarray):
                    raise ValueError("Image must be either a path string or numpy array")
                if image.dtype != np.uint8:
                    image = (image * 255).astype(np.uint8)
                if len(image.shape) == 2:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                elif len(image.shape) == 3 and image.shape[2] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(image)
                print("Image converted from numpy array to PIL Image")
                
            print(f"Image size: {pil_image.size}")
            
            try:
                # Prepare inputs with explicit device placement
                inputs = self.processor(
                    images=pil_image,
                    text=text_prompt,
                    return_tensors="pt"
                )
                
                # Move inputs to device one by one
                for key in inputs:
                    if torch.is_tensor(inputs[key]):
                        inputs[key] = inputs[key].to(self.device)
                
                # if torch.cuda.is_available():
                #     torch.cuda.synchronize()
                
                print(f"Running inference with text prompt: '{text_prompt}'")
                if torch.cuda.is_available():
                    print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}GB")
                    print(f"Current GPU memory usage: {torch.cuda.memory_allocated() / 1e9:.2f}GB")
                
                try:
                    with torch.no_grad():
                        # Ensure model is in eval mode
                        # self.model.eval()
                        
                        try:
                            outputs = self.model(**inputs)
                        except RuntimeError as e:
                            if "out of memory" in str(e):
                                print("GPU OOM error, clearing cache and retrying...")
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                                outputs = self.model(**inputs)
                            else:
                                print(f"Error during model inference: {str(e)}")
                                raise e
                    
                    # Post-process outputs
                    results = self.processor.post_process_grounded_object_detection(
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
                    
                    # Filter by confidence threshold
                    mask = scores >= confidence_threshold
                    boxes = boxes[mask]
                    
                    # Create object IDs
                    obj_ids = list(range(len(boxes)))
                    
                    return boxes, obj_ids
                    
                except Exception as e:
                    print(f"Error during model inference: {str(e)}")
                    if torch.cuda.is_available():
                        print(f"GPU memory after error: {torch.cuda.memory_allocated() / 1e9:.2f}GB")
                    raise
                    
            except Exception as e:
                print(f"Error in input preparation: {str(e)}")
                raise
                
        except Exception as e:
            print(f"Error in get_initial_bboxes: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Try to clean up GPU memory in case of error
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                except:
                    pass
                    
            return [], []
