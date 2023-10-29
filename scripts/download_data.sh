#!/bin/bash

cd "$(dirname "$0")/.."
dir=$(pwd)

# Download the France shapefile
mkdir -p files
cd files

if [ -d "france_shape" ]; then
    echo "Files france_shape already exists!"
else
    wget https://stacks.stanford.edu/file/druid:gy606rg8152/data.zip -O france_shape.zip
    unzip france_shape.zip -d france_shape
    rm france_shape.zip
fi

# Download the MiniFrance dataset
cd $dir
mkdir -p data/MiniFrance
cd data/MiniFrance

if [ -d "labeled_train" ]; then
    echo "MiniFrance labeled_train already exists!"
else
    wget "https://ieee-dataport.s3.amazonaws.com/competition/21720/labeled_train.zip?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAJOHYI4KJCE6Q7MIQ%2F20231028%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20231028T203506Z&X-Amz-SignedHeaders=Host&X-Amz-Expires=86400&X-Amz-Signature=8dcee82f258bf963ed06dc46583a6a87dd45cc50dbb1bb8f95ddfc1b25376e65" -O labeled_train.zip
    unzip labeled_train.zip
    rm labeled_train.zip
fi

if [ -d "unlabeled_train" ]; then
    echo "MiniFrance unlabeled_train already exists!"
else
    wget "https://ieee-dataport.s3.amazonaws.com/competition/21720/unlabeled_train.zip?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAJOHYI4KJCE6Q7MIQ%2F20231028%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20231028T203506Z&X-Amz-SignedHeaders=Host&X-Amz-Expires=86400&X-Amz-Signature=53b1ca8e64f337aa1644be0454480b8120618ace711f2ef8e4dce11fe1c2a20d" -O unlabeled_train.zip
    unzip unlabeled_train.zip
    rm unlabeled_train.zip
fi

if [ -d "val" ]; then
    echo "MiniFrance val already exists!"
else
    wget "https://ieee-dataport.s3.amazonaws.com/competition/21720/val.zip?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAJOHYI4KJCE6Q7MIQ%2F20231028%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20231028T203506Z&X-Amz-SignedHeaders=Host&X-Amz-Expires=86400&X-Amz-Signature=cae7798a584e7c3b35166b09a99289365f1e3881ccd72e73d68ef813c9da5800" -O val.zip
    unzip val.zip -d val
    rm val.zip
fi

if [ -d "test" ]; then
    echo "MiniFrance test already exists!"
else
    wget "https://ieee-dataport.s3.amazonaws.com/competition/21720/test.zip?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAJOHYI4KJCE6Q7MIQ%2F20231028%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20231028T203506Z&X-Amz-SignedHeaders=Host&X-Amz-Expires=86400&X-Amz-Signature=c8ba5176e40658a8f8b911f0bebcdfe6629ec6dfe0f83d394c61854bf589c368" -O test.zip
    unzip test.zip
    rm test.zip
fi