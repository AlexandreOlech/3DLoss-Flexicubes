import os
import math
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":

    plots_dir = r"E:\Fx_losses\plots"
    iou_subplots_path = os.path.join(plots_dir, "iou_subplots.png")
    avg_iou_plot_path = os.path.join(plots_dir, "avg_IoU.png")

    os.makedirs(plots_dir, exist_ok=True)

    IoU_path_l1 = r"E:\Fx_Losses\logs\l1_IoU.csv"
    loss_path_l1 = r"E:\Fx_Losses\logs\l1_loss.csv"

    IoU_path_l2 = r"E:\Fx_Losses\logs\l2_IoU.csv"
    loss_path_l2 = r"E:\Fx_Losses\logs\l2_loss.csv"

    IoU_l1 = pd.read_csv(IoU_path_l1)
    loss_l1 = pd.read_csv(loss_path_l1)

    IoU_l2 = pd.read_csv(IoU_path_l2)
    loss_l2 = pd.read_csv(loss_path_l2)

    iterations = [10*(i+1) for i in range(50)]
    objects = list(IoU_l1.columns)[1:]

    chunk_size = 18
    num_chunks = math.ceil(len(objects) / chunk_size)

    # IoU Subplots
    for chunk_idx in range(num_chunks):
        chunk_objects = objects[chunk_idx * chunk_size : (chunk_idx + 1) * chunk_size]
        fig, axs = plt.subplots(nrows=6, ncols=3, figsize=(25,15))

        for ax, object in zip(axs.flatten(), chunk_objects):
            ax.plot(iterations, IoU_l1[object], label="L1")
            ax.plot(iterations, IoU_l2[object], label="L2")
            ax.set_title(object)
            ax.set_xlabel("Iterations")
            ax.set_ylabel("IoU")
            ax.legend()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        subplot_path = os.path.join(plots_dir, f"iou_subplots_{chunk_idx}.png")
        plt.savefig(subplot_path)
        plt.close()

        # Loss subplot
        fig_loss, axs_loss = plt.subplots(nrows=6, ncols=3, figsize=(25, 15))
        for ax, obj in zip(axs_loss.flatten(), chunk_objects):
            ax.plot(iterations, loss_l1[obj], label="L1")
            ax.plot(iterations, loss_l2[obj], label="L2")
            ax.set_title(obj)
            ax.set_xlabel("Iterations")
            ax.set_ylabel("Loss")
            ax.legend()
        for ax in axs_loss.flatten()[len(chunk_objects):]:
            ax.axis("off")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        loss_subplot_path = os.path.join(plots_dir, f"loss_subplots_{chunk_idx}.png")
        plt.savefig(loss_subplot_path)
        plt.close(fig_loss)

    # Average IoU
    IoU_L1_avg = IoU_l1[objects].mean(axis=1) # Goal : shape (num_objects)
    IoU_L2_avg = IoU_l2[objects].mean(axis=1) # Goal : shape (num_objects)

    plt.figure()
    plt.xlabel("Iterations")
    plt.ylabel("Average IoU")
    plt.plot(iterations, IoU_L1_avg, label="L1")
    plt.plot(iterations, IoU_L2_avg, label="L2")
    plt.legend()
    plt.savefig(avg_iou_plot_path)
    plt.close()