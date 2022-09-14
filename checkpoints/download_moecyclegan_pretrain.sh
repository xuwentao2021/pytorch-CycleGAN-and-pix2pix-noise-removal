echo "Specified deepmoe_cyclegan_inference_eg"

ID=1ITzTjiEWtuFNBahzeLPoNFiY7XSQCGKC
ZIP_FILE=./checkpoints/deepmoe_cyclegan_inference_eg.zip
TARGET_DIR=./checkpoints/deepmoe_cyclegan_inference_eg/

pip install gdown
gdown --id $ID -O $ZIP_FILE
mkdir $TARGET_DIR
unzip $ZIP_FILE -d $TARGET_DIR
rm $ZIP_FILE
