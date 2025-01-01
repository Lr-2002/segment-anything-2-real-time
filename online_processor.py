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

# if torch.cuda.get_device_properties(0).major >= 8:
#     # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
#     torch.backends.cuda.matmul.allow_tf32 = True
#     torch.backends.cudnn.allow_tf32 = True

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
        self.confidence_threshold = 0.1
        self.output_dict = {"cond_frame_outputs": {}}  # Initialize output_dict

    def build_model(self):
        return build_sam2_camera_predictor(self.model_cfg, self.sam2_checkpoint)

    def reset_with_bbox(self, frame, bboxes, obj_ids):
        """Reset the tracker with a new frame and bounding boxes"""
        return self.reset(frame, boxes=bboxes, obj_ids=obj_ids)

    def frame_check(self, frame, is_rgb=True):
        frame_rgb = frame
        if len(frame.shape) == 2:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        if not is_rgb:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame_rgb

    def annotate_and_save_image(self, bboxes, frame, output_path="annotated_image.png"):
        """
        Annotates an RGB frame with bounding boxes and saves it as a PNG file.

        Parameters:
            bboxes (np.ndarray): An array of shape (11, 4), where each row represents a bounding box
                                in the format [x_min, y_min, x_max, y_max].
            frame (np.ndarray): An RGB image of shape (180, 320, 3).
            output_path (str): Path to save the annotated image. Default is "annotated_image.png".
        """

        # Convert the RGB frame to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Draw each bounding box on the frame
        for bbox in bboxes:
            # Convert bounding box coordinates to integers
            x_min, y_min, x_max, y_max = map(int, bbox)
            # Draw a rectangle with a random color
            color = tuple(np.random.randint(0, 256, size=3).tolist())
            cv2.rectangle(frame_bgr, (x_min, y_min), (x_max, y_max), color, thickness=2)

        # Save the annotated image as a PNG file
        cv2.imwrite(output_path, frame_bgr)

        print(f"Annotated image saved to {output_path}")

    def reset(
        self,
        frame,
        text_prompt="object.",
        confidence_threshold=0.1,
        is_rgb=True,
        boxes=None,
        obj_ids=None,
        id=0,
    ):
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
            frame_rgb = self.frame_check(frame, is_rgb)
            if boxes is None and obj_ids is None:
                torch.cuda.empty_cache()

                boxes, obj_ids = self.box_estimator.get_initial_bboxes(
                    frame_rgb, text_prompt, confidence_threshold=confidence_threshold
                )

            if len(boxes) == 0:
                print(f"No objects detected with prompt '{text_prompt}'")
                raise ValueError(f"No objects detected with prompt '{text_prompt}'")

            print(f"Detected {len(boxes)} objects with prompt '{text_prompt}'")
            print(f"Object IDs: {obj_ids}")
            import os

            os.makedirs("annotation", exist_ok=True)
            self.annotate_and_save_image(
                boxes, frame_rgb, output_path=f"annotation/annotated_image_{id}.png"
            )
            # Reset predictor state (safe to call now with our new check)
            self.predictor.reset_state()

            # Initialize first frame
            self.predictor.load_first_frame(frame_rgb)

            self.if_init = True
            self.obj_ids = obj_ids
            # Add detected objects one by one
            for box, obj_id in zip(boxes, obj_ids):
                self.add_new_prompt(bbox=box, obj_id=obj_id)

            return True, (boxes, obj_id)

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
            raise ValueError(
                "Model is not initialized. Please initialize with a frame first."
            )
        _, out_obj_ids, out_mask_logits = self.predictor.add_new_prompt(
            frame_idx=self.ann_frame_idx,
            obj_id=obj_id,
            points=points,
            labels=labels,
            bbox=bbox,
        )
        return out_obj_ids, out_mask_logits

    def batch_add_bbox(self, bbox_list=None, obj_id_list=None):
        """Add multiple bounding boxes at once"""
        if not self.if_init:
            raise ValueError(
                "Model is not initialized. Please initialize with a frame first."
            )
        for bbox, obj_id in zip(bbox_list, obj_id_list):
            self.add_new_prompt(obj_id=obj_id, bbox=bbox)

    def add_frame(self, frame):
        """
        Process a new frame and return masks
        notice: input must be rgb
        """
        # try:
        # Convert frame to RGB if needed
        if len(frame.shape) != 3 or frame.shape[2] != 3:
            raise ValueError("Input frame must be a 3-channel color image")

        frame_rgb = frame
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
        # try:
        out_obj_ids, out_mask_logits = self.predictor.track(frame_rgb)
        # except Exception as track_error:
        # print(f"Error during tracking: {str(track_error)}")
        # import traceback
        # traceback.print_exc()
        # return np.array([])

        if len(out_obj_ids) == 0:
            print(f"Frame {self.ann_frame_idx + 1}: No objects tracked")
            return np.array([])

        # Update frame index and store results
        self.ann_frame_idx += 1

        # Convert masks to numpy arrays with proper error handling
        masks = []
        try:
            for i, obj_id in enumerate(out_obj_ids):
                mask = (out_mask_logits[i] > 0.0).permute(1, 2, 0).cpu().numpy()
                mask = (mask * 255).astype(np.uint8)
                masks.append(mask)

                # Store detailed tracking info in output_dict
                if obj_id not in self.output_dict["cond_frame_outputs"].get(
                    self.ann_frame_idx, {}
                ):
                    self.output_dict["cond_frame_outputs"][self.ann_frame_idx] = {}

                self.output_dict["cond_frame_outputs"][self.ann_frame_idx][obj_id] = {
                    "mask": mask,
                    "confidence": float(
                        (out_mask_logits[i] > 0.0).float().mean().item()
                    ),
                    "frame_size": (width, height),
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

        # except Exception as e:
        #     print(f"Critical error in add_frame: {str(e)}")
        #     import traceback
        #     traceback.print_exc()
        #     return np.array([])

    def process_video(self, video_path, output_folder):
        """Process a video file and save masks"""
        cap = cv2.VideoCapture(video_path)
        cnt = 0
        os.makedirs(output_folder, exist_ok=True)
        video_masks = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            processed_frame = self.add_frame(frame)
            video_masks.append(processed_frame)
            for idx, mask in enumerate(processed_frame):
                saved = cv2.imwrite(
                    os.path.join(output_folder, str(cnt) + "_" + str(idx) + ".png"),
                    mask,
                )
                if not saved:
                    print(f"Failed to save mask {cnt}_{idx}.png")
            cnt += 1
        cap.release()
        return np.stack(video_masks, axis=0)

    def process_image_dirs(self, image_dir):
        """Process all images in a directory and save masks"""
        image_list = sorted(glob.glob(os.path.join(image_dir, "*.png")))
        cnt = 0
        for image_path in tqdm(image_list, desc="Processing images"):
            frame = cv2.imread(image_path)
            if frame is None:
                continue

            try:
                masks = self.add_frame(frame)
                if len(masks) > 0:  # Only save if we have valid masks
                    for idx, mask in enumerate(masks):
                        output_path = os.path.join(
                            image_dir, str(cnt) + "_" + str(idx) + ".png"
                        )
                        print(f"Saving mask to: {output_path}")
                        cv2.imwrite(output_path, mask)
            except Exception as e:
                print(f"Error processing frame {cnt}: {str(e)}")
                import traceback

                traceback.print_exc()

            cnt += 1


def filter_out_overlapping_part_in_larger_ones(masks):
    """
    Process a set of object masks:
    1. Sort the masks by area size (descending order).
    2. Remove overlapping regions from larger masks with smaller masks.

    Args:
        masks (np.ndarray): A NumPy array of size (N, H, W, 1) where 0 means nothing and 1 means accept.

    Returns:
        np.ndarray: Processed masks of the same shape as input.
    """
    # Flatten the last dimension for easier processing (N, H, W)
    masks = masks.squeeze(-1)

    # Compute the area of each mask
    areas = np.sum(masks, axis=(1, 2))

    # Sort masks by area (ascending order)
    sorted_indices = np.argsort(areas)
    sorted_masks = masks[sorted_indices]

    # Create an empty array to store the processed masks
    processed_masks = np.zeros_like(sorted_masks)

    # Iterate through sorted masks and remove overlaps
    for i, mask in enumerate(sorted_masks):
        # Subtract all smaller masks that have already been added
        for j in range(i):
            mask = np.where(sorted_masks[j] > 0, 0, mask)
        # Add the updated mask to the processed masks
        processed_masks[i] = mask

    # Restore the original order
    restored_masks = np.zeros_like(processed_masks)
    for i, idx in enumerate(sorted_indices):
        restored_masks[idx] = processed_masks[i]

    # Add the last dimension back to match the input shape
    return restored_masks[..., np.newaxis]


# def filter_mask_by_area_ratio(masks, area_threshold=0.001):
#     """Filter masks based on the area ratio of the mask to the image, omitting masks with an area ratio below the threshold."""
#     T,N,C,H,W = masks.shape
#     area_threshold = H*W*area_threshold  # Minimum area for a mask to be valid

#     # Calculate the area of each mask (sum over H and W)
#     mask_area = masks.sum(dim=(-2, -1))  # Shape: [B, T, N, C]

#     # Create a boolean mask for masks with area >= threshold
#     valid_mask = mask_area >= area_threshold  # Shape: [B, T, N, C]

#     # Expand valid_mask to match the original tensor's shape
#     valid_mask_expanded = valid_mask.unsqueeze(-1).unsqueeze(-1)  # Add H and W dimensions
#     valid_mask_expanded = valid_mask_expanded.expand_as(masks)

#     # Set invalid masks to 0
#     masked_tensor = masks * valid_mask_expanded
#     return masked_tensor
from scipy.ndimage import label


def filter_mask_by_area_ratio(masks, area_threshold=0.001):
    """
    Filter masks based on the area ratio of connected components of the mask to the image,
    omitting connected components with an area ratio below the threshold.

    Args:
        masks (torch.Tensor): A tensor of shape [T, N, C, H, W] containing binary masks.
        area_threshold (float): The minimum area ratio for a connected component to be valid.

    Returns:
        torch.Tensor: A tensor of the same shape as input, with filtered masks.
    """
    T, N, C, H, W = masks.shape
    area_threshold = H * W * area_threshold  # Convert area ratio to absolute area

    # Convert masks to numpy for processing connected components
    masks_np = masks.cpu().numpy()

    # Initialize the output tensor
    filtered_masks = np.zeros_like(masks_np)

    # Iterate over each mask in the batch
    for t in range(T):
        for n in range(N):
            for c in range(C):
                # Get the binary mask for the current frame
                mask = masks_np[t, n, c]

                # Label connected components in the mask
                labeled_mask, num_features = label(mask)

                # Iterate over each connected component
                for component_id in range(1, num_features + 1):
                    # Extract the current connected component
                    component = labeled_mask == component_id

                    # Calculate the area of the component
                    component_area = component.sum()

                    # Retain the component only if its area is above the threshold
                    if component_area >= area_threshold:
                        filtered_masks[t, n, c] += component

    # Convert the filtered masks back to a torch tensor
    filtered_masks = torch.from_numpy(filtered_masks).to(masks.device)

    return filtered_masks


def remove_all_zero_masks(masks):
    """
    移除所有全零的图像掩码
    :param masks: 形状为 (N, H, W, 1) 的掩码数组
    :return: 移除全零掩码后的掩码数组
    """
    # 确保掩码是 numpy 数组
    masks = np.array(masks)

    # 找到所有非全零掩码的索引
    non_zero_indices = [i for i in range(masks.shape[0]) if np.any(masks[i])]

    # 选择非全零掩码
    non_zero_masks = masks[non_zero_indices]

    return non_zero_masks


def separate_connected_components(input_masks):
    """
    Separates unconnected parts of an image mask into individual object masks.

    Args:
        input_masks (numpy.ndarray): Input masks of shape (N, H, W, 1).

    Returns:
        numpy.ndarray: Output masks of shape (N', H, W, 1), where N' >= N.
    """
    N, H, W, C = input_masks.shape
    assert C == 1, "Input masks must have a single channel (C=1)."

    output_masks = []

    for i in range(N):
        # Extract the single mask
        mask = input_masks[i, :, :, 0]

        # Perform connected-component labeling
        labeled_mask, num_features = label(mask)

        # Create a separate mask for each connected component
        for j in range(1, num_features + 1):
            component_mask = (labeled_mask == j).astype(np.uint8)
            output_masks.append(
                component_mask[..., np.newaxis]
            )  # Add channel dimension back

    # Stack all the output masks into a single array
    output_masks = np.stack(output_masks, axis=0)

    return output_masks


if __name__ == "__main__":
    # Create processor without initial bboxes
    processor = OnlineProcessor(
        model_cfg="sam2_hiera_l.yaml",
        sam2_checkpoint="related_projects/segment/checkpoints/sam2_hiera_large.pt",
    )
    import sys

    sys.path.append("utils")
    from debug_utils import visualize_and_save_masks

    # video_id = 'video_EiYKGXdvcmtlcl8xNTJfZXBfMTBfMDZfMDZfMjIQ-AIYmQMgCDDyCyogZjUyNTg1Mjg3YTE1NzZiODVkZmJjMjk5YjQ5ODExZmM='
    # video_id = 'video_EiQKGXdvcmtlcl8wMDFfZXBfMTRfMDZfMDZfMjIQIxguIAMw6AkqIGY1MjU4NTI4N2ExNTc2Yjg1ZGZiYzI5OWI0OTgxMWZj'
    # Initialize with text prompt to automatically detect objects
    image_path = "original_images"
    images = os.listdir(image_path)
    for i, image in enumerate(images[-1:]):
        image = cv2.imread(os.path.join(image_path, image))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (640*2, 360*2))
        processor.reset(
            frame=image,
            text_prompt="object.robot.",  # Adjust this prompt based on what objects you want to detect
            confidence_threshold=0.05,
            id=i,
        )
        masks = processor.add_frame(image)
        masks //= masks.max()
        filtered_masks = masks
        filtered_masks = filter_out_overlapping_part_in_larger_ones(masks)
        filtered_masks = torch.from_numpy(filtered_masks).permute(0,3,1,2).unsqueeze(0)
        filtered_masks = filter_mask_by_area_ratio(filtered_masks, area_threshold=1e-4)
        filtered_masks = filtered_masks.squeeze(0).permute(0,2,3,1).numpy()
        filtered_masks = separate_connected_components(filtered_masks)
        filtered_masks = remove_all_zero_masks(filtered_masks)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        visualize_and_save_masks(filtered_masks, image, f"output_images/{i}_th_frame")

    # Process all images in directory

    # processor.process_image_dirs('')
    # processor.process_video(
    #     "/home/ziheng/taichang/projects/language-table/segment/language_table.mp4",
    #     "output_frames",
    # )
    # for i in range(len(os.listdir('/home/ziheng/taichang/projects/language-table/tmp'))):
    #     dir = os.listdir('/home/ziheng/taichang/projects/language-table/tmp')[i]
    #     frame = cv2.imread(f'/home/ziheng/taichang/projects/language-table/tmp/'+dir)
    #     processor.add_frame(frame)

    """.git/
    
    1. init 
    2. reset 
    3. for loop
        add_frame 
    
    """
