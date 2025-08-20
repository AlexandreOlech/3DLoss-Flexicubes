import torch
import kaolin
from src.loss.utils.distance import compute_sdf
from src.loss.utils.sampling import sample_random_points
from src.loss.utils.geometry import *

def sdf_loss_l1(gt_mesh, pred_mesh, num_samples):
    with torch.no_grad():
        pts = sample_random_points(num_samples, gt_mesh)
        gt_sdf = compute_sdf(pts, gt_mesh.vertices, gt_mesh.faces)
    pred_sdf = compute_sdf(pts, pred_mesh.vertices, pred_mesh.faces)
    return torch.nn.functional.l1_loss(pred_sdf, gt_sdf)

def sdf_loss_l2(gt_mesh, pred_mesh, num_samples):
    with torch.no_grad():
        pts = sample_random_points(num_samples, gt_mesh)
        gt_sdf = compute_sdf(pts, gt_mesh.vertices, gt_mesh.faces)
    pred_sdf = compute_sdf(pts, pred_mesh.vertices, pred_mesh.faces)
    return torch.nn.functional.mse_loss(pred_sdf, gt_sdf)

def sdf_loss_smooth_l1(gt_mesh, pred_mesh, num_samples, beta):
    with torch.no_grad():
        pts = sample_random_points(num_samples, gt_mesh)
        gt_sdf = compute_sdf(pts, gt_mesh.vertices, gt_mesh.faces)
    pred_sdf = compute_sdf(pts, pred_mesh.vertices, pred_mesh.faces)
    return torch.nn.functional.smooth_l1_loss(pred_sdf, gt_sdf, beta=beta)

def sdf_loss_surf(gt_pts_surf, pred_mesh):
    pred_sdf = compute_sdf(gt_pts_surf, pred_mesh.vertices, pred_mesh.faces)
    return (torch.abs(pred_sdf)).mean()

def sdf_loss_random(random_pts, gt_sdf_random, pred_mesh):
    pred_sdf = compute_sdf(random_pts, pred_mesh.vertices, pred_mesh.faces)
    with torch.no_grad():
        good_sign_mask = (pred_sdf * gt_sdf_random > 0)
    return torch.nn.functional.l1_loss(pred_sdf[good_sign_mask], gt_sdf_random[good_sign_mask])

def sdf_loss_new(gt_mesh, pred_mesh, num_samples):
    surf_weight = 0.6
    with torch.no_grad():
        pts_random = (torch.rand((num_samples//2,3),device='cuda') - 0.5) * 2
        pts_surface = kaolin.ops.mesh.sample_points(gt_mesh.vertices.unsqueeze(0), gt_mesh.faces, num_samples//2)[0].squeeze(0)
        gt_sdf_random = compute_sdf(pts_random, gt_mesh.vertices, gt_mesh.faces)
    pred_sdf_random = compute_sdf(pts_random, pred_mesh.vertices, pred_mesh.faces)
    with torch.no_grad():
        good_sign_mask = (pred_sdf_random * gt_sdf_random > 0)
    loss_random = torch.nn.functional.l1_loss(pred_sdf_random[good_sign_mask], gt_sdf_random[good_sign_mask])
    pred_sdf_surf = compute_sdf(pts_surface, pred_mesh.vertices, pred_mesh.faces)
    loss_surf = (torch.abs(pred_sdf_surf)).mean()
    return (1 - surf_weight)*loss_random + surf_weight*loss_surf

def sdf_loss_mixed(gt_mesh, pred_mesh, num_samples):
    surf_weight = 0.5
    good_sign_weight = 0.3
    bad_sign_weight = 0.1
    with torch.no_grad():
        pts = sample_random_points(num_samples, gt_mesh)
        gt_sdf = compute_sdf(pts, gt_mesh.vertices, gt_mesh.faces)
        # gt_sdf = 2*((gt_sdf > 0).float() - 0.5)
    pred_sdf = compute_sdf(pts, pred_mesh.vertices, pred_mesh.faces)
    with torch.no_grad():
        good_sign_mask = (gt_sdf*pred_sdf > 0)
    good_sign_loss = torch.nn.functional.l1_loss(pred_sdf[good_sign_mask], gt_sdf[good_sign_mask])
    bad_sign_loss = ((pred_sdf[~good_sign_mask])**2).mean()
    return good_sign_weight * good_sign_loss + bad_sign_weight * bad_sign_loss

def occupancy_loss(gt_mesh, pred_mesh, num_samples):
    with torch.no_grad():
        pts = sample_random_points(num_samples, gt_mesh)
        gt_sdf = compute_sdf(pts, gt_mesh.vertices, gt_mesh.faces)
        gt_occ = (gt_sdf < 0).float()
    pred_occ_logits = 1000*compute_sdf(pts, pred_mesh.vertices, pred_mesh.faces)
    return torch.nn.functional.binary_cross_entropy_with_logits(pred_occ_logits, gt_occ)

def chamfer_normals_loss(gt_mesh, pred_mesh, num_samples):
    
    # Get Normals
    pred_mesh.auto_normals()

    # Sample clouds from both meshes
    with torch.no_grad():
        gt_pts, gt_pts_faces = kaolin.ops.mesh.sample_points(gt_mesh.vertices.unsqueeze(0), gt_mesh.faces, num_samples//2) # (B, N//2, 3)
        pred_pts, pred_pts_faces = kaolin.ops.mesh.sample_points(pred_mesh.vertices.unsqueeze(0), pred_mesh.faces, num_samples//2) # (B, N//2, 3)
    # gt_pts = gt_mesh.vertices.unsqueeze(0)
    # pred_pts = pred_mesh.vertices.unsqueeze(0)


    # Index the vertices by faces
    gt_face_vertices = kaolin.ops.mesh.index_vertices_by_faces(gt_mesh.vertices.unsqueeze(0), gt_mesh.faces) # (B, F, 3, 3) (3 vertices of dim 3 for each face)
    pred_face_vertices = kaolin.ops.mesh.index_vertices_by_faces(pred_mesh.vertices.unsqueeze(0), pred_mesh.faces) # (B, F, 3, 3)
    
    # Get distances and nearest object (vertex, edge, face) of the gt mesh to the pred cloud
    # (B, N//2) , (B, N//2), (B, N//2)
    nearest_in_gt_dist, nearest_in_gt_index, nearest_in_gt_dist_type = kaolin.metrics.trianglemesh.point_to_mesh_distance(pred_pts, gt_face_vertices)
    # Get distances and nearest object (vertex, edge, face) of the pred mesh to the gt cloud
    # (B, N//2) , (B, N//2), (B, N//2)
    nearest_in_pred_dist, nearest_in_pred_index, nearest_in_pred_dist_type = kaolin.metrics.trianglemesh.point_to_mesh_distance(gt_pts, pred_face_vertices)
    
    # Estimate the normals in both clouds using KNN/PCA
    # pred_pts_normals = estimate_normals(pred_pts.squeeze(0)) # (N//2, 3) (doesn't support batch yet)
    # gt_pts_normals = estimate_normals(gt_pts.squeeze(0)) # (N//2, 3) (doesn't support batch yet)

    # TODO : use gt/pred_faces to get gt/pred_pts_normals
    gt_pts_normals = gt_mesh.nrm[gt_pts_faces].squeeze(0)
    pred_pts_normals = pred_mesh.nrm[pred_pts_faces].squeeze(0)

    # Compute the face normals in both meshes using the analytical method (could be improved to only compute normals at needed faces)
    # Could also be improved by using normals of edges of vertices when the closest object is not a face
    # gt_mesh.auto_normals() # ground truth mesh should already have normals computed
    #pred_mesh.auto_normals()
    gt_face_normals = (gt_mesh.nrm[nearest_in_gt_index].squeeze(0)) # (N//2, 3) (doesn't support batch yet)
    pred_face_normals = (pred_mesh.nrm[nearest_in_pred_index].squeeze(0)) # (N//2, 3) (doesn't support batch yet)

    # Compute the normals loss, which penalizes differences between normals using negative cosine similarity
    hadamard_1 = gt_face_normals * pred_pts_normals # (N//2, 3)
    hadamard_2 = pred_face_normals * gt_pts_normals # (N//2, 3)
    hadamard = torch.cat((hadamard_1, hadamard_2), dim=0) # (N, 3)

    nrm_loss = 1 - torch.abs(hadamard).sum(dim=1) # (N)
    nrm_loss = nrm_loss.mean()

    # Compute the distance loss
    dist_loss = (((nearest_in_gt_dist)**2).mean() + ((nearest_in_pred_dist)**2).mean())/2

    # Return the total loss
    return dist_loss, nrm_loss

def dvf_loss(gt_mesh, pred_mesh, num_samples):
    """ Displacement Vector Field (DVF) loss
    For each evaluation sample, it penalizes the 
    squared l2 distances between the nearest point from
    the ground truth mesh vs the nearest point from the
    predicted mesh.
    """
    with torch.no_grad():
        pts = sample_random_points(num_samples, gt_mesh)
        gt_face_vertices = kaolin.ops.mesh.index_vertices_by_faces(gt_mesh.vertices.clone().unsqueeze(0), gt_mesh.faces)
        _, gt_nearest_face_indices, gt_dist_types = kaolin.metrics.trianglemesh.point_to_mesh_distance(pts.unsqueeze(0), gt_face_vertices)
    pred_face_vertices = kaolin.ops.mesh.index_vertices_by_faces(pred_mesh.vertices.clone().unsqueeze(0), pred_mesh.faces)
    _, pred_nearest_face_indices, pred_dist_types = kaolin.metrics.trianglemesh.point_to_mesh_distance(pts.unsqueeze(0), pred_face_vertices)
    gt_nearest_pts = nearest_points_from_nearest_faces(pts, gt_face_vertices, gt_nearest_face_indices, gt_dist_types)
    pred_nearest_pts = nearest_points_from_nearest_faces(pts, pred_face_vertices, pred_nearest_face_indices, pred_dist_types)
    # Return the sum of squared l2 distances between nearest points
    return ((gt_nearest_pts - pred_nearest_pts)**2).sum(dim=-1).mean()