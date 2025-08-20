# import os
#BASE_DIR = "/home/local/user/aoh2/3DLoss"
#os.chdir(BASE_DIR)
import os
import torch
import pandas as pd

from src.flexicubes.flexicubes import FlexiCubes
from src.mesh.load_mesh import Mesh, load_mesh
from src.loss.utils.sampling import sample_random_points
from src.metric.IoU import compute_IoU
from src.loss.loss import sdf_loss_smooth_l1
import trimesh

def lr_schedule(iter):
    return max(0.0, 10**(-(iter)*0.0002)) # Exponential falloff from [1.0, 0.1] over 5k epochs.

def fit_object(ref_mesh, learning_rate, voxel_grid_res, num_samples, num_iters, log_dir, loss_fn, approach, device, save_final_mesh):

    losses = []
    IoUs = []
    iou_path = os.path.join(log_dir, f"{approach}_IoU.csv")
    loss_path = os.path.join(log_dir, f"{approach}_loss.csv")

    # ps.init()
    # ps.set_navigation_style("first_person")

    # ==============================================================================================
    #  Settings
    # ==============================================================================================


    # ==============================================================================================
    #  Load and Transform the Ground Truth Mesh
    # ==============================================================================================

    gt_mesh = load_mesh(ref_mesh, device)
    gt_mesh.auto_normals()
    
    # ==============================================================================================
    #  Create Flexicubes class, grid and input parameters
    # ==============================================================================================
    fc = FlexiCubes(device)
    
    x_nx3, cube_fx8 = fc.construct_voxel_grid(voxel_grid_res)
    x_nx3 *= 2

    weight = torch.zeros((cube_fx8.shape[0], 21), dtype=torch.float, device='cuda') 
    weight = torch.nn.Parameter(weight.clone().detach(), requires_grad=True)
    deform = torch.nn.Parameter(torch.zeros_like(x_nx3), requires_grad=True)

    s = torch.rand_like(x_nx3[:,0]) - 0.1
    s = torch.nn.Parameter(s.clone().detach(), requires_grad=True)
    params = [s,weight,deform]
    
    # ==============================================================================================
    #  Setup optimizer and scheduler
    # ==============================================================================================

    optimizer = torch.optim.Adam(params, lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda x : lr_schedule(x))

    # ========================================================================
    #  Optimization
    # ========================================================================
    for it in range(num_iters):
        grid_verts = x_nx3 + (2-1e-8) / (voxel_grid_res * 2) * torch.tanh(deform)
        vertices, faces, _ = fc(grid_verts, s, cube_fx8, voxel_grid_res, beta_fx12=weight[:,:12], alpha_fx8=weight[:,12:20], gamma_f=weight[:,20], training=True)
        pred_mesh = Mesh(vertices, faces)
        loss = loss_fn(gt_mesh, pred_mesh, num_samples)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if (it+1)%10 == 0:
            IoU_pts = sample_random_points(num_samples, gt_mesh)
            IoU = compute_IoU(pred_mesh, gt_mesh, IoU_pts)
            losses.append(loss.item())
            IoUs.append(IoU)
    object_name = os.path.splitext(os.path.basename(ref_mesh))[0]
    iterations = list(range(10, num_iters + 1, 10))
    iou_values = [x.item() if torch.is_tensor(x) else x for x in IoUs]
    loss_values = losses

    # Append IoU
    if os.path.exists(iou_path):
        df_iou = pd.read_csv(iou_path)
    else:
        df_iou = pd.DataFrame({'iteration': iterations})
    df_iou[object_name] = iou_values
    df_iou.to_csv(iou_path, index=False)

    # Append loss
    if os.path.exists(loss_path):
        df_loss = pd.read_csv(loss_path)
    else:
        df_loss = pd.DataFrame({'iteration': iterations})
    df_loss[object_name] = loss_values
    df_loss.to_csv(loss_path, index=False)

    if save_final_mesh:
        mesh_np = trimesh.Trimesh(vertices=pred_mesh.vertices.detach().cpu().numpy(), faces=pred_mesh.faces.detach().cpu().numpy())
        mesh_np.export(os.path.join(log_dir, f"{approach}_{os.path.basename(ref_mesh)}"))


def fit_object_smooth_l1(ref_mesh, learning_rate, voxel_grid_res, num_samples, num_iters, log_dir, approach, beta, device, save_final_mesh):

    losses = []
    IoUs = []
    iou_path = os.path.join(log_dir, f"{approach}_beta_{beta}_IoU.csv")
    loss_path = os.path.join(log_dir, f"{approach}_beta_{beta}_loss.csv")

    # ps.init()
    # ps.set_navigation_style("first_person")

    # ==============================================================================================
    #  Settings
    # ==============================================================================================


    # ==============================================================================================
    #  Load and Transform the Ground Truth Mesh
    # ==============================================================================================

    gt_mesh = load_mesh(ref_mesh, device)
    gt_mesh.auto_normals()
    
    # ==============================================================================================
    #  Create Flexicubes class, grid and input parameters
    # ==============================================================================================
    fc = FlexiCubes(device)
    
    x_nx3, cube_fx8 = fc.construct_voxel_grid(voxel_grid_res)
    x_nx3 *= 2

    weight = torch.zeros((cube_fx8.shape[0], 21), dtype=torch.float, device='cuda') 
    weight = torch.nn.Parameter(weight.clone().detach(), requires_grad=True)
    deform = torch.nn.Parameter(torch.zeros_like(x_nx3), requires_grad=True)

    s = torch.rand_like(x_nx3[:,0]) - 0.1
    s = torch.nn.Parameter(s.clone().detach(), requires_grad=True)
    params = [s,weight,deform]
    
    # ==============================================================================================
    #  Setup optimizer and scheduler
    # ==============================================================================================

    optimizer = torch.optim.Adam(params, lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda x : lr_schedule(x))

    # ========================================================================
    #  Optimization
    # ========================================================================
    for it in range(num_iters):
        grid_verts = x_nx3 + (2-1e-8) / (voxel_grid_res * 2) * torch.tanh(deform)
        vertices, faces, _ = fc(grid_verts, s, cube_fx8, voxel_grid_res, beta_fx12=weight[:,:12], alpha_fx8=weight[:,12:20], gamma_f=weight[:,20], training=True)
        pred_mesh = Mesh(vertices, faces)
        loss = sdf_loss_smooth_l1(gt_mesh, pred_mesh, num_samples, beta)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if (it+1)%10 == 0:
            IoU_pts = sample_random_points(num_samples, gt_mesh)
            IoU = compute_IoU(pred_mesh, gt_mesh, IoU_pts)
            losses.append(loss.item())
            IoUs.append(IoU)
    object_name = os.path.splitext(os.path.basename(ref_mesh))[0]
    iterations = list(range(10, num_iters + 1, 10))
    iou_values = [x.item() if torch.is_tensor(x) else x for x in IoUs]
    loss_values = losses

    # Append IoU
    if os.path.exists(iou_path):
        df_iou = pd.read_csv(iou_path)
    else:
        df_iou = pd.DataFrame({'iteration': iterations})
    df_iou[object_name] = iou_values
    df_iou.to_csv(iou_path, index=False)

    # Append loss
    if os.path.exists(loss_path):
        df_loss = pd.read_csv(loss_path)
    else:
        df_loss = pd.DataFrame({'iteration': iterations})
    df_loss[object_name] = loss_values
    df_loss.to_csv(loss_path, index=False)

    if save_final_mesh:
        mesh_np = trimesh.Trimesh(vertices=pred_mesh.vertices.detach().cpu().numpy(), faces=pred_mesh.faces.detach().cpu().numpy())
        mesh_np.export(os.path.join(log_dir, f"{approach}_beta_{beta}_{os.path.basename(ref_mesh)}"))