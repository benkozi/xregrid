from pathlib import Path

import xarray as xr


class SMM:

    def __init__(self, weights: Path) -> None:
        """
        Initialize the SMM object with source data and weights.

        Parameters:
        -----------
        weights : Path
            Path to the netCDF ESMF weight file.
        """
        ...

    def __call__(self, src: xr.DataArray) -> xr.DataArray:
        """
        Apply the SMM to the source data array creating a regridding destination array.

        Parameters:
        -----------
        src : xr.DataArray
            The source data array to be regridded.

        Returns:
        --------
        xr.DataArray
            The regridded destination data array.
        """
        ...