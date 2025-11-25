#!/bin/bash

set -e

# Only initialize submodules if not in Docker (gradio folder not present)
if [ ! -d "gradio/.git" ]; then
    echo "Initializing and updating git submodules..."
    git submodule update --init --recursive
    cd gradio
    echo "Pinned gradio version to GIT revision 648169d85fbeeffc184115c4c92b12957f2a162f (Nov. 12, 2025)"
    git checkout 648169d85fbeeffc184115c4c92b12957f2a162f
    cd ..
fi

echo "Building Gradio frontend..."
cd gradio
bash scripts/build_frontend.sh
cd ..