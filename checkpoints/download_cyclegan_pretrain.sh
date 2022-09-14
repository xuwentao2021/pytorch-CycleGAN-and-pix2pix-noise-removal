echo "Specified moe_cyclegan_inference_eg"
URL=https://drive.google.com/uc?export=download&id=1emmVAkbaShfDT3E6tubSIkMLld5FbWOT
ZIP_FILE=./checkpoints/moe_cyclegan_inference_eg.zip
TARGET_DIR=./checkpoints/moe_cyclegan_inference_eg/
wget -N $URL -O $ZIP_FILE
mkdir $TARGET_DIR
unzip $ZIP_FILE -d ./checkpoints/
rm $ZIP_FILE
