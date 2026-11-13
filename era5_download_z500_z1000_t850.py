import gcsfs
import xarray as xr

"""
Tai z500, z1000 (geopotential 500/1000 hPa) va t850 (nhiet do 850 hPa) tu
WeatherBench2 ERA5 (5.625 do, 64x32) ve local - PHUC VU verifier Hydrostatic
Balance (phi(500)-phi(1000) ~ R*T850*ln(2)).

FILE MOI, KHONG sua down.py/gcp.py cu (chi tai t2m/u10/v10). Dung cung bucket
GCS an danh, cung cac moc thoi gian val/test de nhat quan:
    val  : 2015-01-01 -> 2016-12-31  (tham chieu, fit/do correlation)
    test : 2017-01-01 -> 2018-12-31  (danh gia)

Luu y: 'geopotential' trong zarr la phi (m^2/s^2, da nhan g), KHONG phai
geopotential height z (m). Verifier dung thang phi nay.
"""

ZARR_PATH = "gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_with_poles_conservative.zarr"


def download_slice(out_path: str, start_date: str, end_date: str):
    print(f"1. Ket noi GCS an danh ({start_date} -> {end_date})...")
    fs = gcsfs.GCSFileSystem(token="anon")
    ds = xr.open_zarr(fs.get_mapper(ZARR_PATH), consolidated=True)

    print("2. Chon bien + muc + lat cat thoi gian (lazy, chi tai subset)...")
    ds_slice = ds.sel(time=slice(start_date, end_date))
    z = ds_slice["geopotential"].sel(level=[500, 1000])      # (time, 2, lon, lat)
    t850 = ds_slice["temperature"].sel(level=850).drop_vars("level")

    out = xr.Dataset({
        "z500": z.sel(level=500).drop_vars("level"),
        "z1000": z.sel(level=1000).drop_vars("level"),
        "t850": t850,
    })

    print(f"3. Tai + ghi -> {out_path} (co the vai phut)...")
    out.to_netcdf(out_path, engine="netcdf4")
    print(f"   Xong. Kich thuoc bien: {dict(out.sizes)}")


if __name__ == "__main__":
    download_slice("era5_test_2017_2018_hydro.nc", "2017-01-01", "2018-12-31")
    download_slice("era5_val_2015_2016_hydro.nc", "2015-01-01", "2016-12-31")
