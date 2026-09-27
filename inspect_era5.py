import numpy as np
import xarray as xr


def inspect(nc_path: str = "era5_test_2017_2018.nc"):
    ds = xr.open_dataset(nc_path)
    print(ds)

    print("\n--- coords ---")
    for c in ds.coords:
        vals = ds.coords[c].values
        print(c, vals.shape, vals[:3] if vals.size > 3 else vals)

    print("\n--- time range ---")
    print(ds.time.values[0], "->", ds.time.values[-1], " n_steps=", ds.time.size)

    print("\n--- lat/lon sizes ---")
    print("lat size:", ds.latitude.size)
    print("lon size:", ds.longitude.size)

    print("\n--- data ranges / NaN check ---")
    for v in ds.data_vars:
        da = ds[v]
        print(
            f"{v}: min={float(da.min()):.3f} max={float(da.max()):.3f} "
            f"mean={float(da.mean()):.3f} nan_count={int(np.isnan(da.values).sum())}"
        )

    print("\n--- lat spacing (check equiangular w/ poles) ---")
    lat = ds.latitude.values
    print(np.diff(lat)[:5], "...", np.diff(lat)[-5:])

    print("\n--- lon spacing ---")
    lon = ds.longitude.values
    print(np.diff(lon)[:5])

    print("\n--- dt between steps (hours) ---")
    t = ds.time.values
    dt = (t[1:] - t[:-1]).astype("timedelta64[h]")
    print(np.unique(dt))


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "era5_test_2017_2018.nc"
    inspect(path)
