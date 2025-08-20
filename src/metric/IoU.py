import torch

from src.loss.utils.occupancy import compute_occupancy

def compute_IoU(pred_mesh, gt_mesh, pts):
    with torch.no_grad():
        gt_occ = compute_occupancy(pts, gt_mesh.vertices, gt_mesh.faces)
        pred_occ = compute_occupancy(pts, pred_mesh.vertices, pred_mesh.faces)
        intersection = torch.logical_and(gt_occ, pred_occ).sum()
        union = torch.logical_or(gt_occ, pred_occ).sum()
        iou = intersection.float() / union.float()
    return iou