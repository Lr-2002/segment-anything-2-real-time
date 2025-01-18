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

from .sam2.build_sam import build_sam2_camera_predictor
from .grounding_processor import GroundingDINOProcessor
import time
from collections import defaultdict


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


def filter_out_overlapping_part_in_masks(masks, keep_larger=True):
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
    sorted_indices = np.argsort(areas) if keep_larger else np.argsort(areas)[::-1]
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
    N, H, W, C = masks.shape
    area_threshold = H * W * area_threshold  # Convert area ratio to absolute area

    # Initialize the output tensor
    filtered_masks = np.zeros_like(masks)

    # Iterate over each mask in the batch
    for n in range(N):
        for c in range(C):
            # Get the binary mask for the current frame
            mask = masks[n, :, :, c]

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
                    filtered_masks[n, :, :, c] += component

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


def count_gray(image_hsv, mask=None):
    """
    计算图像中灰色像素的数量
    :param image_hsv: 输入 HSV 图像，形状为 (H, W, 3)
    :return: 灰色像素的数量
    """
    # 提取 H 和 S 通道
    h_channel = image_hsv[:, :, 0]
    s_channel = image_hsv[:, :, 1]
    v_channel = image_hsv[:, :, 2]

    # 计算灰色像素的数量
    gray_mask = (0 < v_channel) & (s_channel < 128) & (h_channel < 30)
    if mask is not None:
        gray_mask = gray_mask & mask[:, :, 0]
    gray_pixels = gray_mask.sum()

    return gray_pixels


def get_mask_part_with_same_x_coords(mask1, mask2):
    """
    获取与 mask1 具有相同 x 坐标的 mask2 部分
    :param mask1: 输入掩码1，形状为 (H, W)
    :param mask2: 输入掩码2，形状为 (H, W)
    :return: 与 mask1 具有相同 x 坐标的 mask2 部分，形状为 (H, W)
    """
    # 确保掩码是 numpy 数组
    mask1 = np.array(mask1)
    mask2 = np.array(mask2)

    # 获取 mask1 中非零像素的 x 坐标
    x_coords = np.where(mask1)[1]

    # 创建一个与 mask2 形状相同的空掩码
    mask_part = np.zeros_like(mask2)

    # 将 mask2 中与 mask1 具有相同 x 坐标的部分设置为 1
    mask_part[:, x_coords] = mask2[:, x_coords]

    return mask_part


def expand_mask_from_HW_to_NHWC(mask):
    mask = np.expand_dims(mask, axis=0)  # 添加第一个维度
    mask = np.expand_dims(mask, axis=-1)  # 添加最后一个维度
    return mask


def find_connected_components(mask, tolerance=2):
    """
    Finds connected components in a binary mask with a specified tolerance.

    Args:
        mask (np.ndarray): A binary mask of shape (H, W).
        tolerance (int): The tolerance in pixels for connecting components.

    Returns:
        tuple: A labeled mask and the number of connected components.
               - labeled_mask: A 2D array where each connected component is labeled with a unique integer.
               - num_components: The total number of connected components.
    """
    # Create a structuring element for dilation (square of size (2 * tolerance + 1))
    structuring_element = cv2.getStructuringElement(
        cv2.MORPH_RECT, (2 * tolerance + 1, 2 * tolerance + 1)
    )

    # Dilate the mask to connect components within the tolerance
    dilated_mask = cv2.dilate(mask.astype(np.uint8), structuring_element)

    # Find connected components in the dilated mask
    labeled_mask, num_components = label(dilated_mask)

    return labeled_mask, num_components


def split_mask_objects_in_same_color_by_brightness(image, mask):
    """
    Splits a binary mask into individual object masks, ensuring each object has a bright and dark part.

    Args:
        image (np.ndarray): A 3D array representing the RGB image.
        mask (np.ndarray): A binary mask indicating the presence of objects in the image, shape (H, W, 1).

    Returns:
        list: A list of binary masks, each containing a single object.
    """
    # Remove the last dimension of the mask if it is (H, W, 1)
    if mask.shape[-1] == 1:
        mask = mask.squeeze(-1)

    masked_image = image * mask[..., np.newaxis]
    num_nonzero_pixels = np.count_nonzero(mask)

    split_masks = []

    for object_id in range(mask.shape[0]):
        # Convert the image to HSV
        image_hsv = cv2.cvtColor(masked_image[object_id], cv2.COLOR_RGB2HSV)
        object_mask = mask[object_id]
        gray_num = count_gray(image_hsv)

        # If the mask is mostly gray( means this is a robot arm), return the original mask
        if gray_num > 0.5 * num_nonzero_pixels:
            split_masks.append(mask[object_id][..., np.newaxis])
            continue

        bright_threshold = int(255 * 0.75)  # less is dark part, more is bright part
        if object_id == 2:
            print(f"bright_threshold: {bright_threshold}")
        # Split the mask into bright and dark parts
        bright_parts = (image_hsv[:, :, 2] > bright_threshold) & object_mask
        dark_parts = ~bright_parts & object_mask

        bright_parts = filter_mask_by_area_ratio(
            expand_mask_from_HW_to_NHWC(bright_parts), area_threshold=1e-4
        )[0, :, :, 0]
        dark_parts = filter_mask_by_area_ratio(
            expand_mask_from_HW_to_NHWC(dark_parts), area_threshold=1e-4
        )[0, :, :, 0]

        # Label connected components in bright and dark parts
        labeled_bright, num_bright = find_connected_components(
            bright_parts, tolerance=1
        )
        labeled_dark, num_dark = find_connected_components(dark_parts, tolerance=1)

        # Determine which part has more components
        if num_bright > num_dark:
            larger_part, larger_num = labeled_bright, num_bright
            smaller_part = dark_parts
        else:
            larger_part, larger_num = labeled_dark, num_dark
            smaller_part = bright_parts

        # Split the smaller part to match the number of components in the larger part
        for i in range(1, larger_num + 1):
            larger_component_mask = (larger_part == i).astype(np.uint8)
            smaller_component_mask = get_mask_part_with_same_x_coords(
                larger_component_mask, smaller_part
            )

            combined_mask = larger_component_mask | smaller_component_mask
            split_masks.append(combined_mask[..., np.newaxis])

    return np.stack(split_masks)


def remove_edge_lines(mask, length=4):
    H, W = mask.shape
    result = np.zeros_like(mask)
    for r in range(H):
        for c in range(W):
            non_zero = 0
            for dr, dc in [
                [-length // 2, 0],
                [length // 2, 0],
                [0, -length // 2],
                [0, length // 2],
            ]:
                r_start = r if dr > 0 else r + dr
                r_end = r + dr if dr > 0 else r
                c_start = c if dc > 0 else c + dc
                c_end = c + dc if dc > 0 else c
                if (
                    0 <= r + dr < H
                    and 0 <= c + dc < W
                    and (mask[r_start : r_end + 1, c_start : c_end + 1] == 1).all()
                ):
                    non_zero += 1
            if non_zero >= 3:
                result[r, c] = 1

    return result & mask


def get_robot_arm_by_color(image):
    """
    获取图像中的机械臂部分
    :param image: 输入图像，形状为 (H, W, 3)
    :return: 机械臂部分的掩码，形状为 (H, W)
    """
    robot_arm_colors = set()
    for i in range(75, 256):
        if i == 239:
            continue
        robot_arm_colors.add((i, i, i))

    robot_arm_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    for r in range(image.shape[0]):
        for c in range(image.shape[1]):
            if tuple(image[r, c].tolist()) in robot_arm_colors:
                robot_arm_mask[r, c] = 1
    robot_arm_mask = remove_edge_lines(robot_arm_mask, 10)
    robot_arm_mask = expand_mask_from_HW_to_NHWC(robot_arm_mask)
    return robot_arm_mask


def check_mask_all_in_center_panel(mask, polygon_points):
    """
    检查掩码中的所有点是否都在给定的多边形内
    :param mask: 输入掩码，形状为 (H, W)
    :param polygon_points: 多边形的顶点列表
    :return: 布尔值，表示掩码是否完全在多边形内
    """
    points = np.array(polygon_points, dtype=np.int32)

    # Create poly mask
    poly_mask = np.zeros(mask.shape, dtype=np.uint8)

    # Draw filled polygon
    cv2.fillPoly(poly_mask, [points], 1)

    outsize_panel = 1 - poly_mask

    return ((mask & outsize_panel) == 0).all()


def filter_object_masks_by_position(masks):
    # 创建一个空掩码数组
    result_masks = np.zeros_like(masks)

    # 逐个检查掩码
    for i in range(masks.shape[0]):
        mask = masks[i]
        not_empty = mask.sum() > 0

        if not_empty and check_mask_all_in_center_panel(
            mask, [(139, 37), (13, 334), (628, 334), (522, 37)]
        ):
            result_masks[i] = mask

    return result_masks


def filter_out_single_side_masks(image, masks):
    """
    Filter out masks that are entirely composed of a single color.

    Args:
        image (np.ndarray): The input RGB image of shape (H, W, 3).
        masks (np.ndarray): The input masks of shape (N, H, W, 1).

    Returns:
        np.ndarray: The filtered masks of shape (N, H, W, 1).
    """
    filtered_masks = np.zeros_like(masks)
    # Convert the image to HSV color space
    for i in range(masks.shape[0]):
        mask = masks[i]
        masked_image = image * mask
        image_hsv = cv2.cvtColor(masked_image, cv2.COLOR_RGB2HSV)

        v_channel = image_hsv[:, :, 2]
        counter = defaultdict(int)

        for r in range(v_channel.shape[0]):
            for c in range(v_channel.shape[1]):
                if masked_image[r, c].sum() == 0:
                    continue
                if v_channel[r, c] > 0.75 * 255:
                    counter["bright"] += 1
                else:
                    counter["dark"] += 1
        max_color_count = max(counter.values())
        if max_color_count < 0.8 * mask.sum():
            filtered_masks[i] = mask

    filtered_masks = remove_all_zero_masks(filtered_masks)
    return filtered_masks


def resize_masks(masks, new_height, new_width):
    resized = []
    for mask in masks:
        resized_mask = cv2.resize(mask, (new_width, new_height))
        resized.append(resized_mask[..., np.newaxis])
    return np.stack(resized)


if __name__ == "__main__":
    # Create processor without initial bboxes
    processor = OnlineProcessor(
        model_cfg="sam2_hiera_l.yaml",
        sam2_checkpoint="related_projects/segment/checkpoints/sam2_hiera_large.pt",
    )
    import sys

    sys.path.append("utils")
    from debug_utils import visualize_and_save_masks
    from lt_sim_pixel_value_and_rule_based_mask_cleaning import (
        clean_masks_by_pixel_values_and_rules,
    )

    # video_id = 'video_EiYKGXdvcmtlcl8xNTJfZXBfMTBfMDZfMDZfMjIQ-AIYmQMgCDDyCyogZjUyNTg1Mjg3YTE1NzZiODVkZmJjMjk5YjQ5ODExZmM='
    # video_id = 'video_EiQKGXdvcmtlcl8wMDFfZXBfMTRfMDZfMDZfMjIQIxguIAMw6AkqIGY1MjU4NTI4N2ExNTc2Yjg1ZGZiYzI5OWI0OTgxMWZj'
    # Initialize with text prompt to automatically detect objects
    image_path = "original_images"
    images = os.listdir(image_path)
    test = ["original_image_10.png"]
    for i, image in enumerate(test):  # images[-1:]):
        image = cv2.imread(
            "/home/ziheng/oawm_dev/output/original_image.png"
        )  # os.path.join(image_path, image))
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (640, 360))  # (480, 480))#(640, 360)
        processor.reset(
            frame=image,
            text_prompt="robot.object.",  # Adjust this prompt based on what objects you want to detect
            confidence_threshold=0.05,
            id=i,
        )
        masks = processor.add_frame(image)

        filtered_masks = clean_masks_by_pixel_values_and_rules(image, masks)
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
