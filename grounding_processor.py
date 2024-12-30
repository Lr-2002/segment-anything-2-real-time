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
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
                model_id
            ).to(self.device)
            self.model.eval()  # Set to evaluation mode

            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def calculate_iou(self, bbx1, bbx2):
        """计算两个边界框的IOU（Intersection over Union）。"""
        x1 = max(bbx1[0], bbx2[0])
        y1 = max(bbx1[1], bbx2[1])
        x2 = min(bbx1[2], bbx2[2])
        y2 = min(bbx1[3], bbx2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (bbx1[2] - bbx1[0]) * (bbx1[3] - bbx1[1])
        area2 = (bbx2[2] - bbx2[0]) * (bbx2[3] - bbx2[1])

        # 计算相交面积对两个边界框面积的比例
        ratio1 = intersection / area1 if area1 > 0 else 0
        ratio2 = intersection / area2 if area2 > 0 else 0

        return ratio1, ratio2

    def calculate_area(self, bbox):
        """计算单个边界框的面积"""
        x1, y1, x2, y2 = bbox
        return (x2 - x1) * (y2 - y1)

    def update_bboxes(self, bboxes, image_width, image_height):
        """更新和过滤边界框"""
        height_threshold = 0.01 * image_height
        width_threshold = 0.01 * image_width
        agents = []

        # 找到靠近图像边界的边界框
        for i, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = bbox
            if (
                x1 <= width_threshold  # 靠近左边界
                or x2 >= (image_width - width_threshold)  # 靠近右边界
                or y1 <= height_threshold  # 靠近上边界
                or y2 >= (image_height - height_threshold)  # 靠近下边界
            ):
                agents.append(i)

        # 基于IoU过滤agents
        agents_to_remove = set()
        for i in agents:
            if len(agents) - len(agents_to_remove) <= 1:
                break
            for j in range(len(bboxes)):
                if i == j:
                    continue
                ious = self.calculate_iou(bboxes[i], bboxes[j])
                if i != j and (ious[0] > 0.9 or ious[1] > 0.9):
                    agents_to_remove.add(i)
                    break

        # 计算非agent边界框的平均面积
        non_agent_bboxes = [
            bbox for i, bbox in enumerate(bboxes) if i not in agents_to_remove
        ]
        non_agent_areas = [self.calculate_area(bbox) for bbox in non_agent_bboxes]
        mean_area = np.mean(non_agent_areas) if non_agent_areas else 0

        # 移除过大的非agent边界框
        final_bboxes = []
        final_indices = []

        for i, bbox in enumerate(bboxes):
            if i not in agents_to_remove:
                area = self.calculate_area(bbox)
                if (i in agents and i not in agents_to_remove) or area <= 0.025 * (
                    image_width * image_height
                ):
                    final_bboxes.append(bbox)
                    final_indices.append(i)

        return np.array(final_bboxes), final_indices

    def get_initial_bboxes(
        self,
        image: Union[str, np.ndarray],
        text_prompt: str,
        confidence_threshold: float = 0.2,
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
                    raise ValueError(
                        "Image must be either a path string or numpy array"
                    )
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
                    images=pil_image, text=text_prompt, return_tensors="pt"
                )

                # Move inputs to device one by one
                for key in inputs:
                    if torch.is_tensor(inputs[key]):
                        inputs[key] = inputs[key].to(self.device)

                # if torch.cuda.is_available():
                #     torch.cuda.synchronize()

                print(f"Running inference with text prompt: '{text_prompt}'")
                if torch.cuda.is_available():
                    print(
                        f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}GB"
                    )
                    print(
                        f"Current GPU memory usage: {torch.cuda.memory_allocated() / 1e9:.2f}GB"
                    )

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
                        target_sizes=[pil_image.size[::-1]],  # Convert (W,H) to (H,W)
                    )[0]

                    boxes = results["boxes"].cpu().numpy()
                    scores = results["scores"].cpu().numpy()

                    print(f"Found {len(boxes)} detections before confidence filtering")
                    print(f"Scores: {scores}")

                    # Filter by confidence threshold
                    mask = scores >= confidence_threshold
                    boxes = boxes[mask]

                    if len(boxes) > 0:
                        filtered_boxes, filtered_indices = self.update_bboxes(
                            boxes, pil_image.size[0], pil_image.size[1]
                        )
                        obj_ids = list(range(len(filtered_boxes)))
                        return filtered_boxes, obj_ids
                    else:
                        raise Exception(
                            "No objects detected after confidence filtering"
                        )

                except Exception as e:
                    print(f"Error during model inference: {str(e)}")
                    if torch.cuda.is_available():
                        print(
                            f"GPU memory after error: {torch.cuda.memory_allocated() / 1e9:.2f}GB"
                        )
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
