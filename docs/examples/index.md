# Examples Gallery

This gallery contains practical examples of using XRegrid for various earth science applications. Each example is a complete, runnable script that demonstrates specific features and use cases.

## Gallery Overview

The examples are organized by complexity and use case:

### Basic Examples
- **Rectilinear Grid Regridding**: Standard atmospheric model regridding
- **Conservative Regridding**: Flux-conserving interpolation for precipitation
- **Global Grid with Periodicity**: Handling global grids correctly

### Advanced Examples
- **Curvilinear Ocean Grids**: Working with ORCA family ocean model grids
- **Unstructured Grids**: MPAS and ICON model regridding
- **High-Resolution Climate Data**: Optimizing for large datasets

### Performance Examples
- **Weight Reuse**: Optimizing repeated regridding operations
- **Dask Integration**: Parallel processing of large datasets
- **Memory Management**: Handling ultra-high-resolution grids

### Real-World Applications
- **CMIP6 Data Processing**: Working with climate model output
- **Reanalysis Regridding**: ERA5, MERRA2, and other reanalysis products
- **Satellite Data Integration**: Combining different satellite products

## Running the Examples

All examples are designed to be self-contained and runnable. To run an example:

1. **Navigate to the script location**:
   ```bash
   cd docs/examples/scripts/
   ```

2. **Run the example**:
   ```bash
   python plot_basic_regridding.py
   ```

3. **View the output**: Most examples generate plots and save them as PNG files.

## Example Data

Many examples use synthetic data for demonstration purposes. For real-world applications, you can:

- Download sample datasets from the links provided in each example
- Use your own NetCDF files by modifying the data loading sections
- Access public datasets through libraries like `intake` or `xarray.tutorial`

## Contributing Examples

We welcome new examples! To contribute:

1. Create a new Python script in `docs/examples/scripts/`
2. Follow the naming convention: `plot_<descriptive_name>.py`
3. Include comprehensive docstrings and comments
4. Test that the example runs successfully
5. Submit a pull request

### Example Template

```python
"""
Example Title
=============

Brief description of what this example demonstrates.

This example shows:
- Key feature 1
- Key feature 2
- Key feature 3

Data requirements:
- Input data format
- Required variables
- Approximate data size
"""

# Your example code here
```

## Performance Notes

When running the examples:

- **Memory requirements** are noted in each example
- **Execution time** estimates are provided for reference
- **Hardware recommendations** are given for resource-intensive examples
- Use **weight reuse** when running examples multiple times

## Troubleshooting

If you encounter issues running examples:

1. **Check dependencies**: Ensure all required packages are installed
2. **Verify data availability**: Some examples download data automatically
3. **Monitor memory usage**: Large examples may require substantial RAM
4. **Check file permissions**: Ensure write access for output files

For additional help, see the [User Guide](../user-guide/quickstart.md) or submit an issue on GitHub.