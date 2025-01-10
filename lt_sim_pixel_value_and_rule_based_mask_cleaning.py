import numpy as np
from scipy.ndimage import label
import cv2
from collections import defaultdict
import torch

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
            mask = masks[n,:,:, c]

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
                    filtered_masks[n,:,:, c] += component


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
        gray_mask = gray_mask & mask[:,:,0]
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
    structuring_element = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * tolerance + 1, 2 * tolerance + 1))
    
    # Dilate the mask to connect components within the tolerance
    dilated_mask = cv2.dilate(mask.astype(np.uint8), structuring_element)
    
    # Find connected components in the dilated mask
    labeled_mask, num_components = label(dilated_mask)
    
    return labeled_mask, num_components

def split_mask_objects_in_same_color_by_brightness(image, mask, bright_threshold=0.75):
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
    bright_threshold = int(255 * bright_threshold)
    
    for object_id in range(mask.shape[0]):
        # Convert the image to HSV
        image_hsv = cv2.cvtColor(masked_image[object_id], cv2.COLOR_BGR2HSV)
        object_mask = mask[object_id]
        gray_num = count_gray(image_hsv)
        
        # If the mask is mostly gray( means this is a robot arm), return the original mask
        if gray_num > 0.5 * num_nonzero_pixels:
            split_masks.append(mask[object_id][..., np.newaxis])
            continue
        
         # less is dark part, more is bright part
        if object_id == 2:
            print(f"bright_threshold: {bright_threshold}")
        # Split the mask into bright and dark parts
        bright_parts = (image_hsv[:, :, 2] > bright_threshold) & object_mask
        dark_parts = ~bright_parts & object_mask
        
        bright_parts = filter_mask_by_area_ratio(expand_mask_from_HW_to_NHWC(bright_parts), area_threshold=1e-4)[0,:,:,0]
        dark_parts = filter_mask_by_area_ratio(expand_mask_from_HW_to_NHWC(dark_parts), area_threshold=1e-4)[0,:,:,0]
        
    
        # Label connected components in bright and dark parts
        labeled_bright, num_bright = find_connected_components(bright_parts, tolerance=0)
        labeled_dark, num_dark = find_connected_components(dark_parts, tolerance=0)
        
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
            smaller_component_mask = get_mask_part_with_same_x_coords(larger_component_mask, smaller_part)
            
            combined_mask = larger_component_mask | smaller_component_mask
            split_masks.append(combined_mask[..., np.newaxis])
    
    return np.stack(split_masks)

def remove_edge_lines(mask, length=4):
    H,W = mask.shape
    result = np.zeros_like(mask)
    for r in range(H):
        for c in range(W):
            non_zero = 0
            for dr, dc in [[-length//2,0],[length//2,0],[0,-length//2],[0,length//2]]:
                r_start = r if dr > 0 else r+dr
                r_end = r+dr if dr > 0 else r
                c_start = c if dc > 0 else c+dc
                c_end = c+dc if dc > 0 else c
                if 0<=r+dr<H and 0<=c+dc<W and (mask[r_start:r_end+1, c_start:c_end+1] == 1).all():
                    non_zero += 1
            if non_zero >=3:
                result[r,c] = 1

    return result & mask

def get_robot_arm_by_color(image):
    """
    获取图像中的机械臂部分
    :param image: 输入图像，形状为 (H, W, 3)
    :return: 机械臂部分的掩码，形状为 (H, W)
    """
    robot_arm_colors = set()
    for i in range(75,256):
        if i == 239:
            continue
        robot_arm_colors.add((i, i, i))
    
    robot_arm_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    for r in range(image.shape[0]):
        for c in range(image.shape[1]):
            if tuple(image[r, c].tolist()) in robot_arm_colors:
                robot_arm_mask[r, c] = 1
    robot_arm_mask = remove_edge_lines(robot_arm_mask,10)    
    robot_arm_mask = expand_mask_from_HW_to_NHWC(robot_arm_mask)
    return robot_arm_mask

def check_mask_all_in_center_panel(mask, polygon_points):
    """
    检查掩码中的所有点是否都在给定的多边形内
    :param mask: 输入掩码，形状为 (H, W)
    :param polygon_points: 多边形的顶点列表
    :return: 布尔值，表示掩码是否完全在多边形内
    """
    points = (np.array(polygon_points) * np.array([mask.shape[1], mask.shape[0]])).astype(np.int32)
    
    # Create poly mask
    poly_mask = np.zeros(mask.shape, dtype=np.uint8)
    
    # Draw filled polygon
    cv2.fillPoly(poly_mask, [points], 1)
    
    outsize_panel = 1 - poly_mask
    
    
    
    return ((mask & outsize_panel)==0).all()

def filter_object_masks_by_position(masks):

    # 创建一个空掩码数组
    result_masks = np.zeros_like(masks)

    # 逐个检查掩码
    for i in range(masks.shape[0]):
        mask = masks[i]
        not_empty = mask.sum() > 0
        
        
        if not_empty and check_mask_all_in_center_panel(mask, [(139/640, 37/360), (13/640, 334/360), (628/640, 334/360), (522/640, 37/360)]):
            result_masks[i] = mask
            

    return result_masks


def filter_out_single_side_masks(image, masks, bright_threshold=0.75, single_size_threshold=0.9):
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
        masked_image = image*mask
        image_hsv = cv2.cvtColor(masked_image, cv2.COLOR_BGR2HSV)

        v_channel = image_hsv[:, :, 2]
        if v_channel.max() <= 1:
            v_channel = v_channel * 255
        counter = defaultdict(int)
        
        for r in range(v_channel.shape[0]):
            for c in range(v_channel.shape[1]):
                if masked_image[r,c].sum() == 0:
                    continue
                if v_channel[r,c] > bright_threshold * 255:
                    counter['bright'] += 1
                else:
                    counter['dark'] += 1
        max_color_count = max(counter.values())
        if max_color_count < single_size_threshold * mask.sum():
            filtered_masks[i] = mask

    filtered_masks = remove_all_zero_masks(filtered_masks)
    return filtered_masks

def resize_masks(masks, new_height, new_width):
    resized = []
    for mask in masks:
        resized_mask = cv2.resize(mask, (new_width, new_height))
        resized.append(resized_mask[..., np.newaxis])
    return np.stack(resized)

def filter_out_panels(image, masks):
    """
    Filter out masks that are entirely composed of a single color.

    Args:
        image (np.ndarray): The input RGB image of shape (H, W, 3).
        masks (np.ndarray): The input masks of shape (N, H, W, 1).

    Returns:
        np.ndarray: The filtered masks of shape (N, H, W, 1).
    """
    
    for i in range(masks.shape[0]):
        masked_image = image*masks[i]
        condition = (masked_image == 47).all(axis=-1)
        masks[i][condition] *= 0
        
    filtered_masks = remove_all_zero_masks(masks)
    return filtered_masks

def filter_out_robot_arm_and_gray_background(image, masks):
    """
    Filter out masks that are entirely composed of gray only, implying they are background or robot arm.
    This is done by checking if the mask is mostly gray.

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
        masked_image_flatten = torch.from_numpy(image*mask).reshape(-1, 3)
        count= torch.where((masked_image_flatten[:,0] == masked_image_flatten[:,1]) & 
                           (masked_image_flatten[:,1] == masked_image_flatten[:,2]) &
                           masked_image_flatten[:,0] != 0, 1, 0).sum()
        if count < 0.5 * mask.sum():
            filtered_masks[i] = mask
        
        

    filtered_masks = remove_all_zero_masks(filtered_masks)
    return filtered_masks

def clean_masks_by_pixel_values_and_rules(image, object_masks, bright_threshold=0.75, single_size_threshold=0.95, resize_mask_to_accelerate=False):
    """
    Post-process a set of object masks to remove unwanted regions.

    Args:
        image (np.ndarray): The input RGB image of shape (H, W, 3).
        masks (np.ndarray): The input masks of shape (N, H, W, 1).

    Returns:
        np.ndarray: The processed masks of shape (N, H, W, 1).
    """
    from time import time
    start_time = time()
    object_masks //= object_masks.max()
    input_image_shape = image.shape
    if resize_mask_to_accelerate:
        resized_image = cv2.resize(image, (128, 128))
        filtered_masks = resize_masks(object_masks, 128, 128)
    else:
        resized_image = image
        filtered_masks = object_masks
    filtered_masks = remove_all_zero_masks(filtered_masks)
    
    filtered_masks = filter_out_single_side_masks(resized_image, filtered_masks, bright_threshold=bright_threshold, single_size_threshold=single_size_threshold)
    filtered_masks = filter_out_overlapping_part_in_masks(filtered_masks)
    
    filtered_masks = filter_out_robot_arm_and_gray_background(resized_image, filtered_masks)
    
    filtered_masks = filter_mask_by_area_ratio(filtered_masks, area_threshold=1e-4)

    filtered_masks = separate_connected_components(filtered_masks)

    filtered_masks = filter_mask_by_area_ratio(filtered_masks, area_threshold=5e-4)
    
    filtered_masks = split_mask_objects_in_same_color_by_brightness(resized_image, filtered_masks, bright_threshold=bright_threshold)
    filtered_masks = filter_out_overlapping_part_in_masks(filtered_masks, False)
    filtered_masks = filter_object_masks_by_position(filtered_masks)
    filtered_masks = filter_mask_by_area_ratio(filtered_masks, area_threshold=5e-4)
    
    if resize_mask_to_accelerate:
        filtered_masks = resize_masks(filtered_masks, input_image_shape[0], input_image_shape[1])

    filtered_masks = filter_out_panels(image, filtered_masks)
    filtered_masks = filter_mask_by_area_ratio(filtered_masks, area_threshold=5e-4)
    filtered_masks = remove_all_zero_masks(filtered_masks)
    
    robot_masks = get_robot_arm_by_color(image)
    filtered_masks = np.stack([robot_masks[0], *[mask for mask in filtered_masks]])
    end_time = time()
    print(f"Post-process time: {end_time - start_time}")
    return filtered_masks