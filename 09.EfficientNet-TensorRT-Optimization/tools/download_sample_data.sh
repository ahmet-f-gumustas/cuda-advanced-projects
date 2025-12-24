#!/bin/bash

# Download sample images and class labels for testing

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/../data"

mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

echo "Downloading ImageNet class labels..."
if [ ! -f "imagenet_classes.txt" ]; then
    curl -sL https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt -o imagenet_classes.txt
    echo "Downloaded imagenet_classes.txt"
else
    echo "imagenet_classes.txt already exists"
fi

echo "Downloading sample images..."

# Sample images from various sources
IMAGES=(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg:cat.jpg"
    "https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/YellowLabradorLooking_new.jpg/1200px-YellowLabradorLooking_new.jpg:dog.jpg"
)

for item in "${IMAGES[@]}"; do
    url="${item%%:*}"
    filename="${item##*:}"

    if [ ! -f "$filename" ]; then
        echo "Downloading $filename..."
        curl -sL "$url" -o "$filename" || echo "Failed to download $filename"
    else
        echo "$filename already exists"
    fi
done

echo ""
echo "Sample data downloaded to: $DATA_DIR"
echo ""
echo "Usage:"
echo "  ./build/bin/efficientnet_trt --model models/efficientnet_b0.onnx --image data/cat.jpg --labels data/imagenet_classes.txt"
