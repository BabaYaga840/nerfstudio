import torch
import numpy as np
from lpm.zones_projection import region2zone_3d, zones3d_projection

def create_3d_roi(images, bounding_boxes, cameras):
    """
    Creates a 3D Region of Interest (ROI) from an array of images and bounding boxes.

    Args:
        images (list of np.ndarray): Array of N images.
        bounding_boxes (list of list): Array of bounding boxes for each image. 
                                       Each bounding box is [x_min, y_min, x_max, y_max].
        cameras (list): List of camera objects corresponding to each image. 
                        Each camera must have attributes like `rays` and intrinsic/extrinsic parameters.

    Returns:
        zones3d (list of torch.Tensor): List of 3D bounding zones for the ROI, where each zone is a tensor [xmin, ymin, zmin, xmax, ymax, zmax].
    """
    zones3d = []

    # Loop through all pairs of images
    for i, (image_i, bbox_i, camera_i) in enumerate(zip(images, bounding_boxes, cameras)):
        for j, (image_j, bbox_j, camera_j) in enumerate(zip(images, bounding_boxes, cameras)):
            # Skip if comparing the same image
            if i >= j:
                continue

            # Project the paired bounding boxes into 3D
            region_i = [int(bbox_i[0]), int(bbox_i[1]), int(bbox_i[2]), int(bbox_i[3])]
            region_j = [int(bbox_j[0]), int(bbox_j[1]), int(bbox_j[2]), int(bbox_j[3])]
            
            # Compute the 3D region for this pair
            zone3d = region2zone_3d(camera_i, camera_j, region_i, region_j)
            if zone3d is not None:
                zones3d.append(zone3d)

    return zones3d


def nms_3d(zones3d, iou_threshold=0.5):
    """
    Applies Non-Maximum Suppression (NMS) to 3D bounding boxes.

    Args:
        zones3d (list of torch.Tensor): List of 3D bounding boxes, where each box is [xmin, ymin, zmin, xmax, ymax, zmax].
        iou_threshold (float): Threshold for Intersection over Union (IoU) to suppress overlapping boxes.

    Returns:
        final_boxes (list of torch.Tensor): List of 3D bounding boxes after NMS.
    """
    if not zones3d:
        return []

    # Convert list to tensor for easier computation
    boxes = torch.stack(zones3d)  # Shape: (N, 6)

    # Compute the volume of each box
    volumes = (boxes[:, 3] - boxes[:, 0]) * (boxes[:, 4] - boxes[:, 1]) * (boxes[:, 5] - boxes[:, 2])

    # Sort by volume (or confidence, if available)
    sorted_indices = torch.argsort(volumes, descending=True)
    boxes = boxes[sorted_indices]

    selected_indices = []

    while len(boxes) > 0:
        # Select the box with the largest volume
        selected_indices.append(sorted_indices[0])

        # Compute IoU of the selected box with the rest
        ious = compute_3d_iou(boxes[0], boxes[1:])

        # Suppress boxes with IoU greater than the threshold
        keep_indices = torch.where(ious <= iou_threshold)[0]
        boxes = boxes[1:][keep_indices]
        sorted_indices = sorted_indices[1:][keep_indices]

    # Return the final selected boxes
    final_boxes = [zones3d[idx] for idx in selected_indices]
    return final_boxes


def compute_3d_iou(box_a, boxes_b):
    """
    Computes the Intersection over Union (IoU) between a 3D bounding box and a list of 3D bounding boxes.

    Args:
        box_a (torch.Tensor): The reference 3D box [xmin, ymin, zmin, xmax, ymax, zmax].
        boxes_b (torch.Tensor): The list of 3D boxes [xmin, ymin, zmin, xmax, ymax, zmax].

    Returns:
        iou (torch.Tensor): IoU values for each box in `boxes_b` with respect to `box_a`.
    """
    # Compute intersection coordinates
    inter_min = torch.max(box_a[:3], boxes_b[:, :3])  # [xmin, ymin, zmin]
    inter_max = torch.min(box_a[3:], boxes_b[:, 3:])  # [xmax, ymax, zmax]
    inter_dims = torch.clamp(inter_max - inter_min, min=0)  # [width, height, depth]

    # Intersection volume
    intersection = inter_dims[:, 0] * inter_dims[:, 1] * inter_dims[:, 2]

    # Union volume
    volume_a = (box_a[3] - box_a[0]) * (box_a[4] - box_a[1]) * (box_a[5] - box_a[2])
    volume_b = (boxes_b[:, 3] - boxes_b[:, 0]) * (boxes_b[:, 4] - boxes_b[:, 1]) * (boxes_b[:, 5] - boxes_b[:, 2])
    union = volume_a + volume_b - intersection

    # IoU
    iou = intersection / union
    return iou

def get_roi(images, bounding_boxes, cameras):
    zones_3d = create_3d_roi(images, bounding_boxes, cameras)
    final_bboxes = nms_3d(zones_3d, iou_threshold = 0.5)
    return final_bboxes



"""from lpm.region_matching import get_paired_regions
from lpm.zones_projection import zones3d_projection

###   SEGMENT   ###

current_view_regions, referred_view_regions = get_paired_regions(
    segmentation_regions_current_view,  # Segmented regions for current view
    current_view_points.cuda(),
    referred_view_points.cuda()
)

zones3d = zones3d_projection(
    current_view_cam,
    referred_view_cam,
    current_view_regions,
    referred_view_regions
)"""
