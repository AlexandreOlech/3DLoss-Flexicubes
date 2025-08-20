import os
import math
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":

    plots_dir = r"E:\Fx_losses\plots"
    iou_subplots_path = os.path.join(plots_dir, "iou_subplots_smooth_l1_tuning.png")
    avg_iou_plot_path = os.path.join(plots_dir, "avg_IoU_smooth_l1_tuning.png")

    os.makedirs(plots_dir, exist_ok=True)

    IoU_path_l1 = r"E:\Fx_Losses\logs\l1_IoU.csv"
    loss_path_l1 = r"E:\Fx_Losses\logs\l1_loss.csv"

    IoU_path_l2 = r"E:\Fx_Losses\logs\l2_IoU.csv"
    loss_path_l2 = r"E:\Fx_Losses\logs\l2_loss.csv"

    IoU_path_beta001 = r"E:\Fx_Losses\logs\smooth_l1_beta_0.001_IoU.csv"
    IoU_path_beta002 = r"E:\Fx_Losses\logs\smooth_l1_beta_0.01_IoU.csv"
    IoU_path_beta003 = r"E:\Fx_Losses\logs\smooth_l1_beta_0.02_IoU.csv"
    IoU_path_beta004 = r"E:\Fx_Losses\logs\smooth_l1_beta_0.03_IoU.csv"
    IoU_path_beta005 = r"E:\Fx_Losses\logs\smooth_l1_beta_0.04_IoU.csv"

    loss_path_beta001 = r"E:\Fx_Losses\logs\smooth_l1_beta_0.001_loss.csv"
    loss_path_beta002 = r"E:\Fx_Losses\logs\smooth_l1_beta_0.01_loss.csv"
    loss_path_beta003 = r"E:\Fx_Losses\logs\smooth_l1_beta_0.02_loss.csv"
    loss_path_beta004 = r"E:\Fx_Losses\logs\smooth_l1_beta_0.03_loss.csv"
    loss_path_beta005 = r"E:\Fx_Losses\logs\smooth_l1_beta_0.04_loss.csv"

    IoU_l1 = (pd.read_csv(IoU_path_l1)).head(15)
    loss_l1 = (pd.read_csv(loss_path_l1)).head(15)

    IoU_l2 = (pd.read_csv(IoU_path_l2)).head(15)
    loss_l2 = (pd.read_csv(loss_path_l2)).head(15)

    IoU_beta001 = pd.read_csv(IoU_path_beta001)
    loss_beta001 = pd.read_csv(loss_path_beta001)

    IoU_beta002 = pd.read_csv(IoU_path_beta002)
    loss_beta002 = pd.read_csv(loss_path_beta002)

    IoU_beta003 = pd.read_csv(IoU_path_beta003)
    loss_beta003 = pd.read_csv(loss_path_beta003)

    IoU_beta004 = pd.read_csv(IoU_path_beta004)
    loss_beta004 = pd.read_csv(loss_path_beta004)

    IoU_beta005 = pd.read_csv(IoU_path_beta005)
    loss_beta005 = pd.read_csv(loss_path_beta005)

    iterations = [10*(i+1) for i in range(15)]
    objects = list(IoU_beta001.columns)[1:]

    chunk_size = 18
    num_chunks = math.ceil(len(objects) / chunk_size)

    # IoU Subplots
    for chunk_idx in range(num_chunks):
        chunk_objects = objects[chunk_idx * chunk_size : (chunk_idx + 1) * chunk_size]
        fig, axs = plt.subplots(nrows=6, ncols=3, figsize=(25,15))

        for ax, object in zip(axs.flatten(), chunk_objects):
            ax.plot(iterations, IoU_l1[object], label="L1")
            ax.plot(iterations, IoU_l2[object], label="L2")
            ax.plot(iterations, IoU_beta001[object], label=r"$\beta=0.001$")
            ax.plot(iterations, IoU_beta002[object], label=r"$\beta=0.01$")
            ax.plot(iterations, IoU_beta003[object], label=r"$\beta=0.02$")
            ax.plot(iterations, IoU_beta004[object], label=r"$\beta=0.03$")
            ax.plot(iterations, IoU_beta005[object], label=r"$\beta=0.04$")
            ax.set_title(object)
            ax.set_xlabel("Iterations")
            ax.set_ylabel("IoU")
            ax.legend()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        subplot_path = os.path.join(plots_dir, f"iou_subplots_smooth_l1_tuning_{chunk_idx}.png")
        plt.savefig(subplot_path)
        plt.close()

        # Loss subplot
        fig_loss, axs_loss = plt.subplots(nrows=6, ncols=3, figsize=(25, 15))
        for ax, obj in zip(axs_loss.flatten(), chunk_objects):
            ax.plot(iterations, loss_l1[obj], label="L1")
            ax.plot(iterations, loss_l2[obj], label="L2")
            ax.plot(iterations, loss_beta001[object], label=r"$\beta=0.001$")
            ax.plot(iterations, loss_beta002[object], label=r"$\beta=0.01$")
            ax.plot(iterations, loss_beta003[object], label=r"$\beta=0.02$")
            ax.plot(iterations, loss_beta004[object], label=r"$\beta=0.03$")
            ax.plot(iterations, loss_beta005[object], label=r"$\beta=0.04$")
            ax.set_title(obj)
            ax.set_xlabel("Iterations")
            ax.set_ylabel("Loss")
            ax.legend()
        for ax in axs_loss.flatten()[len(chunk_objects):]:
            ax.axis("off")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        loss_subplot_path = os.path.join(plots_dir, f"loss_subplots_smooth_l1_tuning_{chunk_idx}.png")
        plt.savefig(loss_subplot_path)
        plt.close(fig_loss)

    # Average IoU
    IoU_L1_avg = IoU_l1[objects].mean(axis=1)
    IoU_L2_avg = IoU_l2[objects].mean(axis=1)
    IoU_beta001_avg = IoU_beta001[objects].mean(axis=1)
    IoU_beta002_avg = IoU_beta002[objects].mean(axis=1)
    IoU_beta003_avg = IoU_beta003[objects].mean(axis=1)
    IoU_beta004_avg = IoU_beta004[objects].mean(axis=1)
    IoU_beta005_avg = IoU_beta005[objects].mean(axis=1)

    plt.figure()
    plt.xlabel("Iterations")
    plt.ylabel("Average IoU")
    plt.plot(iterations, IoU_L1_avg, label="L1")
    plt.plot(iterations, IoU_L2_avg, label="L2")
    plt.plot(iterations, IoU_beta001_avg, label=r"$\beta=0.001$")
    plt.plot(iterations, IoU_beta002_avg, label=r"$\beta=0.01$")
    plt.plot(iterations, IoU_beta003_avg, label=r"$\beta=0.02$")
    plt.plot(iterations, IoU_beta004_avg, label=r"$\beta=0.03$")
    plt.plot(iterations, IoU_beta005_avg, label=r"$\beta=0.04$")
    plt.legend()
    plt.savefig(avg_iou_plot_path)
    plt.close()