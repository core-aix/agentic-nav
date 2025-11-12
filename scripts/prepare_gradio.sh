#!/bin/bash

set -e

echo "Initializing and updating git submodules..."
git submodule update --init --recursive

echo "Building Gradio frontend..."
cd gradio
bash scripts/build_frontend.sh
cd ..
