# ============================================================
# COLAB: chay ARCH-ABLATION ERA5 (hydrostatic, TimesFM-3.0)
# Runtime -> Change runtime type -> T4 GPU
# Dan tung cell (hoac ca khoi) vao Colab.
# ============================================================

# --- Cell 1: cai dat ---
# !pip install -q git+https://github.com/google-research/timesfm.git
# !pip install -q xarray netcdf4 scikit-learn scipy

# --- Cell 2: lay data ERA5 hydro tu Kaggle dataset (can kaggle.json) ---
# import kagglehub
# path = kagglehub.dataset_download("ngkhacthaibinh/era5-z500-z1000-t850-hydro")
# print("data o:", path)   # chua era5_test_2017_2018_hydro.nc
# # Hoac: tu upload era5_test_2017_2018_hydro.nc len /content (Files -> Upload).

# --- Cell 3: chay kernel ---
# Upload file kaggle_kernel_era5_arch_hydro.py len /content roi:
# %run kaggle_kernel_era5_arch_hydro.py
#
# Kernel tu tim era5_test_2017_2018_hydro.nc trong /content (va cac thu muc con),
# tu ghi output ra /content/era5_arch_hydro_results.json + _summary.csv.
#
# Neu data nam o path khac (vd kagglehub tra ve /root/.cache/...), copy vao /content:
#   import shutil, glob, os
#   src = glob.glob(os.path.join(path, "**", "era5_test_2017_2018_hydro.nc"), recursive=True)[0]
#   shutil.copy(src, "/content/")
print("Xem huong dan trong cac comment ben tren. Chay tren Colab T4 GPU.")
