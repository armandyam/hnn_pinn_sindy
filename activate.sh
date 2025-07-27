#!/bin/bash

# Activate virtual environment and run the provided command
source venv/bin/activate

# If no arguments provided, just activate the environment
if [ $# -eq 0 ]; then
    echo "Virtual environment activated."
    echo "You can now run Python commands."
    echo "To deactivate, run: deactivate"
    exec bash
else
    # Run the provided command
    exec "$@"
fi 