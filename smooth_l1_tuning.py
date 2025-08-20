import os
from src.optimize.optimize import fit_object_smooth_l1

def run_for_beta(approach, obj_paths, learning_rate, voxel_grid_res, num_samples, num_iters, log_dir, beta, device, save_final_mesh=False):
    processed_objects_file_path = f"{approach}_beta_{beta}_processed_objects.txt"
    processed_objects = set()
    if os.path.exists(processed_objects_file_path):
        with open(processed_objects_file_path, 'r') as f:
            processed_objects = set(line.strip() for line in f)
    processable_obj_paths = [obj_path for obj_path in obj_paths if obj_path not in processed_objects]
    for ref_mesh in processable_obj_paths:
        fit_object_smooth_l1(ref_mesh, learning_rate, voxel_grid_res, num_samples, num_iters, log_dir, approach, beta, device, save_final_mesh)
        with open(processed_objects_file_path, 'a') as f:
            f.write(ref_mesh + '\n')

if __name__ == "__main__":

    learning_rate = 0.01
    voxel_grid_res = 64
    num_samples = 10000
    num_iters = 150
    log_dir = "logs"
    device = 'cuda'

    obj_paths = [os.path.join("data", f) for f in os.listdir("data")]
    os.makedirs(log_dir, exist_ok=True)

    betas = [0.001, 0.01, 0.02, 0.03, 0.04]
    approach = "smooth_l1"

    for beta in betas:
        run_for_beta(approach, obj_paths, learning_rate, voxel_grid_res, num_samples, num_iters, log_dir, beta, device)