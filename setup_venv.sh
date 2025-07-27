#!/bin/bash

echo "Setting up AI Safety and Physics project with virtual environment..."

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 not found. Please install Python 3 first."
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "Creating project directories..."
mkdir -p data
mkdir -p models/saved
mkdir -p results
mkdir -p plots

echo ""
echo "Virtual environment setup completed successfully!"
echo ""
echo "To activate the virtual environment in the future, run:"
echo "source venv/bin/activate"
echo ""
echo "To test the installation, run:"
echo "source venv/bin/activate && python test_pipeline.py"
echo ""
echo "To run the full pipeline, run:"
echo "source venv/bin/activate && python main.py test"
echo ""
echo "To deactivate the virtual environment, run:"
echo "deactivate" 