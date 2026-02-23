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
    ).stack(spatial=("lat", "lon"))
    
    # 4. Create Lazy (Dask) source data
    da_lazy = da_eager.chunk({"spatial": 100})
    
    # 5. Apply SMM
    res_eager = smm(da_eager)
    res_lazy = smm(da_lazy)
    
    # 6. Verify identical results
    xr.testing.assert_allclose(res_eager, res_lazy.compute())
    
    # 7. Verify shape
    # Target grid 20x20 global: lat=9, lon=18 -> 162
    assert res_eager.shape == (162,)
    assert "target_spatial" in res_eager.dims

def test_smm_history_update(tmp_path):
    """Verify that SMM updates the history attribute."""
    weights_file = str(tmp_path / "test_weights.nc")
    ds_src = create_global_grid(10.0, 10.0)
    ds_tgt = create_global_grid(20.0, 20.0)
    Regridder(ds_src, ds_tgt, filename=weights_file, reuse_weights=True)
    
    smm = SMM(weights_file)
    da = xr.DataArray(
        np.random.rand(18*36),
        dims=("spatial",),
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
    da = xr.DataArray(np.random.rand(100), dims=("spatial",))
    
    with pytest.raises(ValueError, match="Source spatial size"):
        smm(da)
