MODEL='<model_nme>'
BASE_PATH='<base path>'
DATASET_TYPE='<dataset names separated by commas>' #example 'knee-coronal-pd','knee-coronal-pd-fs'

BATCH_SIZE=2
NUM_EPOCHS=100
DEVICE='cuda:0'
ACC_FACTOR= '<acceleration factors considered for training>' #' example 4x','5x','8x'
MASK_TYPE='<mask type>' #example 'cartesian', 'gaussian','radial'

EXP_DIR='/<path to store the model>/'${MODEL}'/'
NUM_COILS='coil configurations ie numbers of coils separated by commas' #example - '7','10','12' 
TRAIN_PATH=${BASE_PATH}'/datasets/'
VALIDATION_PATH=${BASE_PATH}'/datasets/'

# the datasets must be stored in a hierarchy - <dataset_name>/<mask_type name>/<acceleration factors/ this folder may contain the train and validation folders 
# example <base_path>/multi_coil_knee/coronal_pd/cartesian/acc_4x/train and <base path>/ multi_coil_knee/coronal_pd/cartesian/acc_4x/train
# inside the train and validation files the h5 files must  be stored


echo python train.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --train-path ${TRAIN_PATH} --validation-path ${VALIDATION_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} --mask_type ${MASK_TYPE}
python train.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --train-path ${TRAIN_PATH} --validation-path ${VALIDATION_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} --mask_type ${MASK_TYPE} --num_coils ${NUM_COILS}

