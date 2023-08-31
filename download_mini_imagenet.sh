#!/bin/bash


echo "Creating folders..."
mkdir datasets
cd datasets
mkdir mini_imagenet
cd mini_imagenet

echo "Downloading mini imagenet..."
wget https://www.kaggle.com/datasets/arjunashok33/miniimagenet/download?datasetVersionNumber=1
echo "Extracting dataset..."
unzip archive.zip
rm archive.zip

echo "Sample backgrounds downloaded"