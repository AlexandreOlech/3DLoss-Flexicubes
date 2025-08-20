import torch
import kaolin

def compute_occupancy(points, vertices, faces):
    with torch.no_grad():
        return (kaolin.ops.mesh.check_sign(vertices.unsqueeze(0), faces, points.unsqueeze(0))).float()