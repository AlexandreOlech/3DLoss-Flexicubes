import torch
import trimesh

def normalize_mesh_by_bbox(mesh : trimesh.Trimesh):
    bbox = mesh.bounding_box.bounds
    bbox_center = bbox.mean(axis=0)
    bbox_size = bbox.max(axis=0) - bbox.min(axis=0)

    mesh.apply_translation(-bbox_center)
    mesh.apply_scale(1.0 / max(bbox_size))
    return mesh

def dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sum(x*y, -1, keepdim=True)

def length(x: torch.Tensor, eps: float =1e-8) -> torch.Tensor:
    return torch.sqrt(torch.clamp(dot(x,x), min=eps)) # Clamp to avoid nan gradients because grad(sqrt(0)) = NaN

def safe_normalize(x: torch.Tensor, eps: float =1e-8) -> torch.Tensor:
    return x / length(x, eps)

class Mesh:
    def __init__(self, vertices, faces):
        self.vertices = vertices
        self.faces = faces
        
    def auto_normals(self):
        v0 = self.vertices[self.faces[:, 0], :]
        v1 = self.vertices[self.faces[:, 1], :]
        v2 = self.vertices[self.faces[:, 2], :]
        nrm = safe_normalize(torch.linalg.cross(v1 - v0, v2 - v0))
        self.nrm = nrm

def load_mesh(path, device):
    gt_mesh = trimesh.load(path)
    if isinstance(gt_mesh, trimesh.Scene):
        gt_mesh = trimesh.util.concatenate([g for g in gt_mesh.geometry.values()])
    gt_mesh = normalize_mesh_by_bbox(gt_mesh)
    gt_mesh.vertices *= 1.8
    vertices = torch.tensor(gt_mesh.vertices, device=device, dtype=torch.float)
    faces = torch.tensor(gt_mesh.faces, device=device, dtype=torch.long)
    return Mesh(vertices, faces)