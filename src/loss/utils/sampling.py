import torch
import kaolin

def sample_random_points(n, mesh):
    pts_random = (torch.rand((n//2,3),device='cuda') - 0.5) * 2
    pts_surface = kaolin.ops.mesh.sample_points(mesh.vertices.unsqueeze(0), mesh.faces, n//2)[0].squeeze(0)
    pts_surface += torch.randn_like(pts_surface) * 0.05
    pts = torch.cat([pts_random, pts_surface])
    return pts