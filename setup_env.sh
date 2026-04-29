#!/bin/bash
set -e
echo "Installing AI Video Generator dependencies..."
python3 -m pip install --upgrade pip
python3 -m pip install --no-cache-dir -r requirements.txt
echo "Installation complete!"
