#!/bin/bash
set -e

echo "XRegrid Documentation Builder"
echo "=============================="

# Activate conda environment
echo "Activating conda environment..."
conda activate xregrid

# Install documentation dependencies if needed
echo "Checking documentation dependencies..."
pip install mkdocs mkdocs-material mkdocs-gallery mkdocs-autorefs mkdocstrings mkdocstrings-python matplotlib --quiet

# Install package in development mode
echo "Installing xregrid package..."
pip install -e . --quiet

# Clean previous build
echo "Cleaning previous build..."
rm -rf site/

# Build documentation
echo "Building documentation..."
mkdocs build --clean

echo ""
echo "Documentation built successfully!"
echo "To serve locally: mkdocs serve"
echo "To deploy: mkdocs gh-deploy"
echo ""
echo "Built site is in: ./site/"