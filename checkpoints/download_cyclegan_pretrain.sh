echo "Specified cyclegan_inference_eg"
ID=1emmVAkbaShfDT3E6tubSIkMLld5FbWOT
ZIP_FILE=./checkpoints/cyclegan_inference_eg.zip
TARGET_DIR=./checkpoints/cyclegan_inference_eg/

pip install gdown
gdown -id $ID -O $ZIP_FILE
mkdir $TARGET_DIR
unzip $ZIP_FILE -d ./checkpoints/
rm $ZIP_FILE

