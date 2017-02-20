#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../" && pwd )"
cd $DIR

FILE=inpaintCenterModels.tar.gz
URL=https://drive.google.com/file/d/0BxYj-YwDqh45WDAwenZEYlhXUTg/view?usp=sharing
CHECKSUM=632b1cfcc9bc59212c77ca138235d125

if [ ! -f $FILE ]; then
  echo "Downloading the Inpainting Center Fill models (504MB)..."
  wget $URL -O $FILE
  echo "Unzipping..."
  tar zxvf $FILE
  echo "Downloading Done."
else
  echo "File already exists. Checking md5..."
fi

os=`uname -s`
if [ "$os" = "Linux" ]; then
  checksum=`md5sum $FILE | awk '{ print $1 }'`
elif [ "$os" = "Darwin" ]; then
  checksum=`cat $FILE | md5`
elif [ "$os" = "SunOS" ]; then
  checksum=`digest -a md5 -v $FILE | awk '{ print $4 }'`
fi
if [ "$checksum" = "$CHECKSUM" ]; then
  echo "Checksum is correct. File was correctly downloaded."
  exit 0
else
  echo "Checksum is incorrect. DELETE and download again."
fi


