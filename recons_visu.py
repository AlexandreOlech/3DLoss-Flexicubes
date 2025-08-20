import os
from src.loss.loss import sdf_loss_l1, sdf_loss_l2
from smooth_l1_tuning import run_for_beta
from l1_vs_l2 import run_for_loss

if __name__ == "__main__":

    learning_rate = 0.01
    voxel_grid_res = 64
    num_samples = 10000
    num_iters = 150
    log_dir = "logs"
    device = 'cuda'
    chosen_objects = ['block.obj', 'fandisk.obj', 'rolling_stage100K.obj', 'rocker_arm.obj', 'armchair.obj']
    save_final_mesh = True

    obj_paths = [os.path.join("data", f) for f in chosen_objects]
    os.makedirs(log_dir, exist_ok=True)

    betas = [0.001, 0.01, 0.02, 0.03, 0.04]
    approach = "smooth_l1"
    for beta in betas:
        run_for_beta(approach, obj_paths, learning_rate, voxel_grid_res, num_samples, num_iters, log_dir, beta, device, save_final_mesh)

    approach = "l1"
    loss_fn = sdf_loss_l1
    run_for_loss(approach, loss_fn, obj_paths, learning_rate, voxel_grid_res, num_samples, num_iters, log_dir, device, save_final_mesh)

    approach = "l2"
    loss_fn = sdf_loss_l2
    run_for_loss(approach, loss_fn, obj_paths, learning_rate, voxel_grid_res, num_samples, num_iters, log_dir, device, save_final_mesh)