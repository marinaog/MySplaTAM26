scene = "boxes"

hdr_folder = f"experiments/rawslam/{scene}_raw_logexp_3/eval_rerun"
ldr_folder = f"experiments/rawslam/{scene}/eval_rerun"

import numpy as np
import matplotlib.pyplot as plt
import os

def load_per_frame(folder, filename):
    path = os.path.join(folder, filename)
    data = np.loadtxt(path)
    frames = data[:, 0].astype(int)
    values = data[:, 1]
    return frames, values

hdr_psnr_frames, hdr_psnr = load_per_frame(hdr_folder, "psnr_hdr_per_frames.txt")
ldr_psnr_frames, ldr_psnr = load_per_frame(ldr_folder, "psnr_per_frames.txt")

hdr_ssim_frames, hdr_ssim = load_per_frame(hdr_folder, "ssim_per_frames.txt")
ldr_ssim_frames, ldr_ssim = load_per_frame(ldr_folder, "ssim_per_frames.txt")

hdr_lpips_frames, hdr_lpips = load_per_frame(hdr_folder, "lpips_per_frames.txt")
ldr_lpips_frames, ldr_lpips = load_per_frame(ldr_folder, "lpips_per_frames.txt")

fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=False)

hdr_color, ldr_color = "tab:blue", "tab:orange"

axs[0].plot(hdr_psnr_frames, hdr_psnr, color=hdr_color, label=f"HDR (mean={hdr_psnr.mean():.2f})")
axs[0].plot(ldr_psnr_frames, ldr_psnr, color=ldr_color, label=f"LDR (mean={ldr_psnr.mean():.2f})")
axs[0].axhline(hdr_psnr.mean(), color=hdr_color, linestyle="--", linewidth=1)
axs[0].axhline(ldr_psnr.mean(), color=ldr_color, linestyle="--", linewidth=1)
axs[0].set_ylabel("PSNR (dB)")
axs[0].set_xlabel("Frame")
axs[0].legend()
axs[0].set_title("PSNR per frame")

axs[1].plot(hdr_ssim_frames, hdr_ssim, color=hdr_color, label=f"HDR (mean={hdr_ssim.mean():.3f})")
axs[1].plot(ldr_ssim_frames, ldr_ssim, color=ldr_color, label=f"LDR (mean={ldr_ssim.mean():.3f})")
axs[1].axhline(hdr_ssim.mean(), color=hdr_color, linestyle="--", linewidth=1)
axs[1].axhline(ldr_ssim.mean(), color=ldr_color, linestyle="--", linewidth=1)
axs[1].set_ylabel("SSIM")
axs[1].set_xlabel("Frame")
axs[1].legend()
axs[1].set_title("SSIM per frame")

axs[2].plot(hdr_lpips_frames, hdr_lpips, color=hdr_color, label=f"HDR (mean={hdr_lpips.mean():.3f})")
axs[2].plot(ldr_lpips_frames, ldr_lpips, color=ldr_color, label=f"LDR (mean={ldr_lpips.mean():.3f})")
axs[2].axhline(hdr_lpips.mean(), color=hdr_color, linestyle="--", linewidth=1)
axs[2].axhline(ldr_lpips.mean(), color=ldr_color, linestyle="--", linewidth=1)
axs[2].set_ylabel("LPIPS")
axs[2].set_xlabel("Frame")
axs[2].legend()
axs[2].set_title("LPIPS per frame")

fig.suptitle(f"Metrics per frame — {scene}", fontsize=14)
fig.tight_layout()
os.makedirs("plots", exist_ok=True)
plt.savefig(f"plots/{scene}_metrics_per_frame.png", bbox_inches="tight", dpi=150)
plt.close()
