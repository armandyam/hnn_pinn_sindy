#!/bin/bash

echo "Installing AI Safety and Physics project dependencies..."

# Check if pip3 is available
if command -v pip3 &> /dev/null; then
    PIP_CMD="pip3"
elif command -v pip &> /dev/null; then
    PIP_CMD="pip"
else
    echo "Error: pip not found. Please install Python and pip first."
    exit 1
fi

# Install dependencies
echo "Installing Python dependencies..."
$PIP_CMD install -r requirements.txt

# Create necessary directories
echo "Creating project directories..."
mkdir -p data
mkdir -p models/saved
mkdir -p results
mkdir -p plots

echo "Installation completed successfully!"
echo ""
echo "To test the installation, run:"
echo "python3 test_pipeline.py"
echo ""
echo "To run the full pipeline, run:"
echo "python3 main.py test" 