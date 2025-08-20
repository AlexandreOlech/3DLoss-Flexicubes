import torch
import kaolin

from src.loss.utils.sampling import sample_random_points

def project_points_to_triangles(points, triangles):
    """
    Projects each point in `points` onto the corresponding triangle in `triangles`.
    Args:
        points: (N, 3) Tensor of points
        triangles: (N, 3, 3) Tensor of triangle vertices
    Returns:
        (N, 3) Tensor of projected points
    """
    A = triangles[:, 0]
    B = triangles[:, 1]
    C = triangles[:, 2]
    AB = B - A
    AC = C - A
    AP = points - A

    d1 = (AB * AP).sum(-1)
    d2 = (AC * AP).sum(-1)
    d3 = (AB * AB).sum(-1)
    d4 = (AB * AC).sum(-1)
    d5 = (AC * AC).sum(-1)

    denom = d3 * d5 - d4 * d4
    v = (d5 * d1 - d4 * d2) / denom
    w = (d3 * d2 - d4 * d1) / denom
    u = 1.0 - v - w

    # Clamp barycentric coordinates to the triangle
    u = u.clamp(0, 1)
    v = v.clamp(0, 1)
    w = w.clamp(0, 1)
    total = u + v + w
    u /= total
    v /= total
    w /= total

    return u[:, None] * A + v[:, None] * B + w[:, None] * C

def project_points_to_edges(pts, a, b):
    """
    Projects each point in `pts` onto the corresponding edge defined by (a, b).
    
    Args:
        pts: (N, 3) tensor of 3D points
        a: (N, 3) tensor of edge start points
        b: (N, 3) tensor of edge end points

    Returns:
        (N, 3) tensor of projected points
    """
    ab = b - a
    ap = pts - a

    ab_norm_sq = (ab ** 2).sum(dim=-1, keepdim=True)
    t = (ap * ab).sum(dim=-1, keepdim=True) / ab_norm_sq
    t_clamped = t.clamp(0.0, 1.0)

    return a + t_clamped * ab


def nearest_points_from_nearest_faces(pts, face_vertices, nearest_face_indices, dist_types, device="cuda"):
    pred_nearest_pts = torch.zeros_like(pts, device=device)

    # Type 0 : the distance is from a point on the surface of the triangle.
    mask = dist_types.squeeze(0) == 0
    nearest_faces_vertices = face_vertices.squeeze(0)[nearest_face_indices.squeeze(0)[mask]]
    projected_pts = project_points_to_triangles(
        pts[mask],
        nearest_faces_vertices
    )
    pred_nearest_pts[mask] = projected_pts

    # Types 1 to 3 : the distance is from a point to a vertices.
    for i in range(1,4):
        mask = dist_types.squeeze(0) == i
        nearest_vertices = face_vertices.squeeze(0)[nearest_face_indices.squeeze(0)[mask]][:,i-1]
        pred_nearest_pts[mask] = nearest_vertices

    # Types 4 to 6 : the distance is from a point to an edge.
    # Assuming the convention that the edges are ordered as AB, BC, CA
    # AB
    mask = dist_types.squeeze(0) == 4
    a = face_vertices.squeeze(0)[nearest_face_indices.squeeze(0)[mask]][:,0]
    b = face_vertices.squeeze(0)[nearest_face_indices.squeeze(0)[mask]][:,1]
    projected_pts = project_points_to_edges(
        pts[mask],
        a,
        b
    )
    pred_nearest_pts[mask] = projected_pts
    # BC
    mask = dist_types.squeeze(0) == 5
    b = face_vertices.squeeze(0)[nearest_face_indices.squeeze(0)[mask]][:,1]
    c = face_vertices.squeeze(0)[nearest_face_indices.squeeze(0)[mask]][:,2]
    projected_pts = project_points_to_edges(
        pts[mask],
        b,
        c
    )
    pred_nearest_pts[mask] = projected_pts
    # CA
    mask = dist_types.squeeze(0) == 6
    c = face_vertices.squeeze(0)[nearest_face_indices.squeeze(0)[mask]][:,2]
    a = face_vertices.squeeze(0)[nearest_face_indices.squeeze(0)[mask]][:,0]
    projected_pts = project_points_to_edges(
        pts[mask],
        c,
        a
    )
    pred_nearest_pts[mask] = projected_pts

    return pred_nearest_pts