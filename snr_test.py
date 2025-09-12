import numpy as np

def compute_snr(x, xhat):
    diff = x - xhat
    snr = 10 * np.log10(np.sum(x**2) / np.sum(diff**2))
    return snr

x = np.random.randn(2, 4, 10)
xhat = x.copy()
noise_sigs = [0.1, 0.1, 0.1, 0.1]
for i, noise_sig in enumerate(noise_sigs):
    noise = np.random.randn(2, 10) * noise_sig
    xhat[:, i, :] += noise
    frame_snr = compute_snr(x[:, i, :], xhat[:, i, :])
    print(f"Frame {i} SNR: {frame_snr}")

global_snr = compute_snr(x, xhat)
print(f"Global SNR: {global_snr}")


print(10 * np.log10(0.012047/3e-5))