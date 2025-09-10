accelerate launch --config_file=configs/accelerate.yaml CardiacVAE/scripts/train_vae.py --config=configs/vae_l1_only_real.yaml
accelerate launch --config_file=configs/accelerate.yaml CardiacVAE/scripts/train_vae.py --config=configs/vae_l1_only_complex.yaml
accelerate launch --config_file=configs/accelerate.yaml CardiacVAE/scripts/train_vae.py --config=configs/vae_l1_and_lpips_real.yaml
accelerate launch --config_file=configs/accelerate.yaml CardiacVAE/scripts/train_vae.py --config=configs/vae_l1_and_lpips_complex.yaml
accelerate launch --config_file=configs/accelerate.yaml CardiacVAE/scripts/train_vae.py --config=configs/vae_l1_and_lpips_and_kl_real.yaml
accelerate launch --config_file=configs/accelerate.yaml CardiacVAE/scripts/train_vae.py --config=configs/vae_l1_and_lpips_and_kl_complex.yaml
