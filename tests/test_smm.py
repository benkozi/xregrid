import numpy as np
import pytest
import xarray as xr
from xregrid import SMM, create_global_grid, Regridder
import os

def test_smm_eager_lazy_comparison(tmp_path):
    """
    Test SMM with both Eager (NumPy) and Lazy (Dask) data as per Aero Protocol.
    """
    weights_file = str(tmp_path / "test_weights.nc")
    
    # 1. Create a weight file using Regridder
    ds_src = create_global_grid(10.0, 10.0)
    ds_tgt = create_global_grid(20.0, 20.0)
    
    # This generates the weights file
    Regridder(ds_src, ds_tgt, filename=weights_file, reuse_weights=True)
    
    assert os.path.exists(weights_file)
    
    # 2. Initialize SMM
    smm = SMM(weights_file)
    
    # 3. Create Eager (NumPy) source data
    # Source grid 10x10 global: lat=18, lon=36
    shape_src = (18, 36)
    data_np = np.random.rand(*shape_src).astype(np.float32)
    da_eager = xr.DataArray(
        data_np,
        dims=("lat", "lon"),
        name="test_data"
    )
    
    # 4. Create Lazy (Dask) source data
    da_lazy = da_eager.chunk({"lat": 9, "lon": 36})
    
    # 5. Apply SMM
    res_eager = smm(da_eager)
    res_lazy = smm(da_lazy)
    
    # 6. Verify identical results
    xr.testing.assert_allclose(res_eager, res_lazy.compute())
    
    # 7. Verify shape
    # Target grid 20x20 global: lat=9, lon=18
    assert res_eager.shape == (9, 18)
    assert res_eager.dims == ("lat", "lon")

def test_smm_history_update(tmp_path):
    """Verify that SMM updates the history attribute."""
    weights_file = str(tmp_path / "test_weights.nc")
    ds_src = create_global_grid(10.0, 10.0)
    ds_tgt = create_global_grid(20.0, 20.0)
    Regridder(ds_src, ds_tgt, filename=weights_file, reuse_weights=True)
    
    smm = SMM(weights_file)
    da = xr.DataArray(
        np.random.rand(18, 36),
        dims=("lat", "lon"),
        attrs={"history": "original history"}
    )
    
    res = smm(da)
    assert "Regridded using xregrid.SMM" in res.attrs["history"]
    assert "original history" in res.attrs["history"]

def test_smm_mismatched_shape(tmp_path):
    """Verify that SMM raises error on mismatched source shape."""
    weights_file = str(tmp_path / "test_weights.nc")
    ds_src = create_global_grid(10.0, 10.0)
    ds_tgt = create_global_grid(20.0, 20.0)
    Regridder(ds_src, ds_tgt, filename=weights_file, reuse_weights=True)
    
    smm = SMM(weights_file)
    # n_a should be 18*36 = 648. Provide wrong size.
    da = xr.DataArray(np.random.rand(10, 10), dims=("lat", "lon"))
    
    with pytest.raises(ValueError, match="Source spatial size"):
        smm(da)

def test_smm_time_level_varying(tmp_path):
    """Verify SMM with time and level varying data."""
    weights_file = str(tmp_path / "test_weights.nc")
    ds_src = create_global_grid(10.0, 10.0)
    ds_tgt = create_global_grid(20.0, 20.0)
    Regridder(ds_src, ds_tgt, filename=weights_file, reuse_weights=True)
    
    smm = SMM(weights_file)
    
    # Create data with (time, level, lat, lon)
    nt, nl, nlat, nlon = 2, 3, 18, 36
    data = np.random.rand(nt, nl, nlat, nlon).astype(np.float32)
    da = xr.DataArray(
        data,
        dims=("time", "level", "lat", "lon"),
        coords={
            "time": np.arange(nt),
            "level": np.arange(nl),
            "lat": ds_src.lat,
            "lon": ds_src.lon
        },
        name="test_varying"
    )
    
    # 1. Test with both time and level specified
    res = smm(da, time_dim="time", level_dim="level")
    # Target grid 20x20: lat=9, lon=18
    assert res.shape == (nt, nl, 9, 18)
    assert res.dims == ("time", "level", "lat", "lon")
    assert "time" in res.coords
    assert "level" in res.coords
    
    # 2. Test with only time specified (level becomes spatial) -> This should fail because spatial size won't match
    # spatial size = nl * nlat * nlon = 3 * 18 * 36 = 1944 != 648
    with pytest.raises(ValueError, match="Source spatial size"):
        smm(da, time_dim="time")

    # 3. Test with Dask
    da_lazy = da.chunk({"time": 1, "level": 1})
    res_lazy = smm(da_lazy, time_dim="time", level_dim="level")
    assert res_lazy.chunks is not None
    xr.testing.assert_allclose(res, res_lazy.compute())

def test_smm_exact_reproduction_lazy(tmp_path):
    """
    Verify that SMM reproduces values exactly for identity regridding with Dask.
    """
    weights_file = str(tmp_path / "identity_weights_lazy.nc")
    ds_src = create_global_grid(10.0, 10.0)
    ds_tgt = create_global_grid(10.0, 10.0)
    Regridder(ds_src, ds_tgt, filename=weights_file, reuse_weights=True)
    smm = SMM(weights_file)
    
    nt, nl, nlat, nlon = 2, 3, 18, 36
    data = np.random.rand(nt, nl, nlat, nlon).astype(np.float32)
    da_src = xr.DataArray(
        data,
        dims=("time", "level", "lat", "lon"),
        coords={
            "time": np.arange(nt),
            "level": np.arange(nl),
            "lat": ds_src.lat,
            "lon": ds_src.lon
        },
        name="test_exact_lazy"
    ).chunk({"time": 1, "level": 1})
    
    res = smm(da_src, time_dim="time", level_dim="level")
    assert res.chunks is not None
    
    np.testing.assert_allclose(da_src.values, res.compute().values, atol=1e-6)

def test_smm_exact_reproduction(tmp_path):
    """
    Verify that SMM reproduces values exactly for identity regridding.
    
    Test setup:
    - Source: 10.0 x 10.0 global grid (18x36)
    - Target: 10.0 x 10.0 global grid (18x36)
    - Data: (time=2, level=3, lat=18, lon=36)
    - Each (time, level) slice has unique, distinguishable values.
    """
    weights_file = str(tmp_path / "identity_weights.nc")
    
    # 1. Create identical grids
    ds_src = create_global_grid(10.0, 10.0)
    ds_tgt = create_global_grid(10.0, 10.0)
    
    # 2. Generate weights (should be identity mapping)
    Regridder(ds_src, ds_tgt, filename=weights_file, reuse_weights=True)
    
    # 3. Initialize SMM
    smm = SMM(weights_file)
    
    # 4. Create source data (2 time steps, 3 levels)
    nt, nl, nlat, nlon = 2, 3, 18, 36
    data = np.zeros((nt, nl, nlat, nlon), dtype=np.float32)
    
    # Fill each (time, level) slice with unique values
    # e.g., value = time * 10 + level + lat*0.01 + lon*0.0001
    for t in range(nt):
        for l in range(nl):
            # Base value for this slice
            base = t * 10.0 + l
            # Unique spatial pattern
            spatial = np.random.rand(nlat, nlon).astype(np.float32)
            data[t, l, :, :] = base + spatial
            
    da_src = xr.DataArray(
        data,
        dims=("time", "level", "lat", "lon"),
        coords={
            "time": np.arange(nt),
            "level": np.arange(nl),
            "lat": ds_src.lat,
            "lon": ds_src.lon
        },
        name="test_exact"
    )
    
    # 5. Apply SMM
    res = smm(da_src, time_dim="time", level_dim="level")
    
    # 6. Verify identical results and dimensions
    # ESMF bilinear on identical grids should be very close to identity
    # We compare only the data values because SMM might not preserve spatial coordinates 
    # if they are not explicitly handled in the reconstruction logic for now.
    np.testing.assert_allclose(da_src.values, res.values, atol=1e-6)
    
    # Check non-spatial coordinates are preserved
    assert "time" in res.coords
    assert "level" in res.coords
    
    assert res.dims == ("time", "level", "lat", "lon")
    assert res.shape == (nt, nl, nlat, nlon)
    
    # 7. Check specific slice uniqueness
    for t in range(nt):
        for l in range(nl):
            expected_mean = (t * 10.0 + l) + 0.5
            actual_mean = float(res.isel(time=t, level=l).mean())
            # Mean should be around base + 0.5 since np.random.rand mean is ~0.5
            assert np.isclose(actual_mean, expected_mean, atol=0.1)
