import os

from src.loss.loss import sdf_loss_l1, sdf_loss_l2
from src.optimize.optimize import fit_object

def run_for_loss(approach, loss_fn, obj_paths, learning_rate, voxel_grid_res, num_samples, num_iters, log_dir, device, save_final_mesh=False):
    processed_objects_file_path = f"{approach}_processed_objects.txt"
    processed_objects = set()
    if os.path.exists(processed_objects_file_path):
        with open(processed_objects_file_path, 'r') as f:
            processed_objects = set(line.strip() for line in f)
    processable_obj_paths = [obj_path for obj_path in obj_paths if obj_path not in processed_objects]
    for ref_mesh in processable_obj_paths:
        fit_object(ref_mesh, learning_rate, voxel_grid_res, num_samples, num_iters, log_dir, loss_fn, approach, device, save_final_mesh)
        with open(processed_objects_file_path, 'a') as f:
            f.write(ref_mesh + '\n')

if __name__ == "__main__":

    learning_rate = 0.01
    voxel_grid_res = 64
    num_samples = 10000
    num_iters = 500
    log_dir = "logs"
    device = 'cuda'

    obj_paths = [os.path.join("data", f) for f in os.listdir("data")]
    os.makedirs(log_dir, exist_ok=True)

    approach = "l1"
    loss_fn = sdf_loss_l1
    run_for_loss(approach, loss_fn, obj_paths, learning_rate, voxel_grid_res, num_samples, num_iters, log_dir, device)

    approach = "l2"
    loss_fn = sdf_loss_l2
    run_for_loss(approach, loss_fn, obj_paths, learning_rate, voxel_grid_res, num_samples, num_iters, log_dir, device)