import os
from tqdm import tqdm
import torch
import numpy as np
import cv2
import imageio
import sys 
import glob
import tempfile
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

from sam2.build_sam import build_sam2_camera_predictor
from grounding_processor import GroundingDINOProcessor
import time

class OnlineProcessor:
    def __init__(self, model_cfg, sam2_checkpoint):
        """Initialize OnlineProcessor without requiring initial frame or bboxes"""
        self.model_cfg = model_cfg
        self.sam2_checkpoint = sam2_checkpoint
        self.predictor = self.build_model()
        self.box_estimator = GroundingDINOProcessor()
        self.if_init = False
        self.ann_frame_idx = 0
        self.obj_ids = []
        self.text_prompt = None
        self.confidence_threshold = 0.2
        self.output_dict = {"cond_frame_outputs": {}}  # Initialize output_dict

    def build_model(self):
        return build_sam2_camera_predictor(self.model_cfg, self.sam2_checkpoint)

    def reset(self, frame, text_prompt="object", confidence_threshold=0.35):
        """Reset the tracker with a new frame and text prompt"""
        try:
            # Reset internal state
            self.if_init = False
            self.ann_frame_idx = 0
            self.obj_ids = []
            self.text_prompt = text_prompt
            self.confidence_threshold = confidence_threshold
            self.output_dict = {"cond_frame_outputs": {}}

            # Convert frame to RGB if needed
            if len(frame.shape) == 2:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            else:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Get initial bounding boxes from Grounding DINO
            boxes, obj_ids = self.box_estimator.get_initial_bboxes(
                frame_rgb, 
                text_prompt,
                confidence_threshold=confidence_threshold
            )
            
            if len(boxes) == 0:
                print(f"No objects detected with prompt '{text_prompt}'")
                return False
            
            print(f"Detected {len(boxes)} objects with prompt '{text_prompt}'")
            print(f"Object IDs: {obj_ids}")

            # Reset predictor state (safe to call now with our new check)
            self.predictor.reset_state()
            
            # Initialize first frame
            self.predictor.load_first_frame(frame_rgb)

            self.if_init = True
            self.obj_ids = obj_ids
            # Add detected objects one by one
            for box, obj_id in zip(boxes, obj_ids):
                self.add_new_prompt(bbox=box, obj_id=obj_id)
            
            return True
            
        except Exception as e:
            print(f"Critical error in reset: {str(e)}")
            raise e

    def init_state(self, bboxes):
        """Initialize state with given bounding boxes"""
        if bboxes is None:
            raise ValueError("bboxes cannot be None.")
        for bbox, obj_id in zip(bboxes, self.obj_ids):
            self.add_new_prompt(bbox=bbox, obj_id=obj_id)

    def add_new_prompt(self, points=None, labels=None, bbox=None, obj_id=None):
        """Add a new prompt for tracking"""
        if not self.if_init:
            raise ValueError("Model is not initialized. Please initialize with a frame first.")
        _, out_obj_ids, out_mask_logits = self.predictor.add_new_prompt(
            frame_idx=self.ann_frame_idx, obj_id=obj_id, points=points, labels=labels, bbox=bbox
        )
        return out_obj_ids, out_mask_logits

    def batch_add_bbox(self, bbox_list=None, obj_id_list=None):
        """Add multiple bounding boxes at once"""
        if not self.if_init:
            raise ValueError("Model is not initialized. Please initialize with a frame first.")
        for bbox, obj_id in zip(bbox_list, obj_id_list):
            self.add_new_prompt(obj_id=obj_id, bbox=bbox)

    def add_frame(self, frame):
        """Process a new frame and return masks"""
        try:
            # Convert frame to RGB if needed
            if len(frame.shape) != 3 or frame.shape[2] != 3:
                raise ValueError("Input frame must be a 3-channel color image")
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if isinstance(frame, np.ndarray) else frame
            width, height = frame_rgb.shape[:2][::-1]
            
            if not self.if_init:
                print("Initializing tracker with first frame...")
                self.predictor.load_first_frame(frame_rgb)
                self.if_init = True
                return np.array([])
            
            if len(self.obj_ids) == 0:
                print("No objects to track")
                return np.array([])
            
            # Track objects in current frame
            try:
                out_obj_ids, out_mask_logits = self.predictor.track(frame_rgb)
            except Exception as track_error:
                print(f"Error during tracking: {str(track_error)}")
                import traceback
                traceback.print_exc()
                return np.array([])
            
            if len(out_obj_ids) == 0:
                print(f"Frame {self.ann_frame_idx + 1}: No objects tracked")
                return np.array([])
            
            # Update frame index and store results
            self.ann_frame_idx += 1
            
            # Convert masks to numpy arrays with proper error handling
            masks = []
            try:
                for i, obj_id in enumerate(out_obj_ids):
                    mask = (out_mask_logits[i] > 0.0).permute(1,2,0).cpu().numpy()
                    mask = (mask * 255).astype(np.uint8)
                    masks.append(mask)
                    
                    # Store detailed tracking info in output_dict
                    if obj_id not in self.output_dict["cond_frame_outputs"].get(self.ann_frame_idx, {}):
                        self.output_dict["cond_frame_outputs"][self.ann_frame_idx] = {}
                    
                    self.output_dict["cond_frame_outputs"][self.ann_frame_idx][obj_id] = {
                        "mask": mask,
                        "confidence": float((out_mask_logits[i] > 0.0).float().mean().item()),
                        "frame_size": (width, height)
                    }
            except Exception as mask_error:
                print(f"Error processing masks: {str(mask_error)}")
                import traceback
                traceback.print_exc()
            
            # Stack masks if we have any
            masks = np.stack(masks, axis=0) if masks else np.array([])
            
            # Log tracking status
            print(f"Frame {self.ann_frame_idx} - Tracked {len(masks)} objects")
            print(f"Frame {self.ann_frame_idx} - Object IDs: {out_obj_ids}")
            
            return masks
            
        except Exception as e:
            print(f"Critical error in add_frame: {str(e)}")
            import traceback
            traceback.print_exc()
            return np.array([])

    def process_video(self, video_path, output_folder):
        """Process a video file and save masks"""
        cap = cv2.VideoCapture(video_path)
        cnt = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            processed_frame = self.add_frame(frame)
            for idx, mask in enumerate(processed_frame):
                cv2.imwrite(os.path.join(output_folder, str(cnt) + '_' + str(idx) + '.png'), mask)
            cnt += 1
        cap.release()

    def process_image_dirs(self, image_dir):
        """Process all images in a directory and save masks"""
        image_list = sorted(glob.glob(os.path.join(image_dir, '*.jpg')))
        cnt = 0
        for image_path in tqdm(image_list, desc="Processing images"):
            frame = cv2.imread(image_path)
            if frame is None:
                continue
            
            try:
                masks = self.add_frame(frame)
                if len(masks) > 0:  # Only save if we have valid masks
                    for idx, mask in enumerate(masks):
                        output_path = os.path.join(image_dir, str(cnt) + '_' + str(idx) + '.png')
                        print(f"Saving mask to: {output_path}")
                        cv2.imwrite(output_path, mask)
            except Exception as e:
                print(f"Error processing frame {cnt}: {str(e)}")
                import traceback
                traceback.print_exc()
            
            cnt += 1

if __name__=='__main__':
    # Create processor without initial bboxes
    processor = OnlineProcessor(
        model_cfg="sam2_hiera_l.yaml",
        sam2_checkpoint="checkpoints/sam2_hiera_large.pt"
    )
    
    # Initialize with text prompt to automatically detect objects
    processor.reset(
        frame=cv2.imread('/ssd/lt/processed_dataset/lt_sim_seged/val/video_EiQKGXdvcmtlcl8wMDFfZXBfMTRfMDZfMDZfMjIQIxguIAMw6AkqIGY1MjU4NTI4N2ExNTc2Yjg1ZGZiYzI5OWI0OTgxMWZj/images/00000.jpg'),
        text_prompt="object",  # Adjust this prompt based on what objects you want to detect
        confidence_threshold=0.1
    )
    
    # Process all images in directory
    processor.process_image_dirs('/ssd/lt/processed_dataset/lt_sim_seged/val/video_EiQKGXdvcmtlcl8wMDFfZXBfMTRfMDZfMDZfMjIQIxguIAMw6AkqIGY1MjU4NTI4N2ExNTc2Yjg1ZGZiYzI5OWI0OTgxMWZj/images/')
    # processor.process_video("../notebooks/videos/aquarium/aquarium.mp4", "./output_frames")
    
    
    """.git/
    
    1. init 
    2. reset 
    3. for loop
        add_frame 
    
    """
