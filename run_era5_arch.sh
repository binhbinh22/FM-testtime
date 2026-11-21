#!/bin/bash
#SBATCH --job-name=era5_arch
#SBATCH --output=era5_arch_%j.log
#SBATCH --error=era5_arch_%j.err
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --time=12:00:00

# ============================================================
# ARCH-ABLATION ERA5 (z500/z1000/t850 + hydrostatic, TimesFM-3.0)
# Chay tren server A100 / SLURM. Submit:  sbatch run_era5_arch.sh
# ============================================================
source /home/user18/miniconda3/etc/profile.d/conda.sh
conda activate stllm
cd /home/user18/binhnkt

# stream log realtime (khong buffer) de theo doi tien trinh
export PYTHONUNBUFFERED=1

# --- deps (idempotent, chi cai neu thieu) ---
pip install -q git+https://github.com/google-research/timesfm.git
pip install -q xarray netcdf4 scikit-learn scipy

# --- DATA: kernel tim era5_test_2017_2018_hydro.nc trong cwd (~/binhnkt) ---
# .nc bi gitignore nen KHONG co qua git. Neu chua co tren server, scp 1 lan:
#   scp era5_test_2017_2018_hydro.nc user18@172.16.3.8:~/binhnkt/
if ! find . -name era5_test_2017_2018_hydro.nc | grep -q .; then
  echo "!!! KHONG thay era5_test_2017_2018_hydro.nc trong ~/binhnkt"
  echo "    scp file nay len server roi sbatch lai (xem comment ben tren)."
  exit 1
fi

python -u datasets/era5/reports/kaggle_kernel_era5_arch_hydro.py
