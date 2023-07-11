MODEL='<model_nme>'
BASE_PATH='<base path>'

BATCH_SIZE=1

CHECKPOINT='<path to where the model folder is present>'${MODEL}'/best_model.pt'
DEVICE='cuda:0'

for NUM_COILS in <'multiple coil configurarions'> #'4' '5' '6' '7' '8' '9' '10' '11' '12' '13' '14' '15'
    do
    for DATASET_TYPE in 'dataset names within single quotes separated by a blank space'#'knee-mc-coronal-pd' 'knee-mc-coronal-pd-fs'
        do
        for MASK_TYPE in '<mask names within single quotes separated by a blank space>' #'cartesian'
            do 
            for ACC_FACTOR in <'acceleration factor names within single quotes separated by a blank space'> #'4x' '5x' '7x' '9x'
                do 
                echo ${DATASET_TYPE}','${MASK_TYPE}','${ACC_FACTOR} 
                OUT_DIR='<path to where model is stored>'${MODEL}'/results/'${DATASET_TYPE}'/'${MASK_TYPE}'/acc_'${ACC_FACTOR}'/'${NUM_COILS}
                DATA_PATH=${BASE_PATH}'/'${DATASET_TYPE}'/'${MASK_TYPE}'/acc_'${ACC_FACTOR}'/validation'
                REPORT_PATH='<path to where model is stored>/'${MODEL}'/'
                echo python valid_measures.py --checkpoint ${CHECKPOINT} --out-dir ${OUT_DIR} --batch-size ${BATCH_SIZE} --device ${DEVICE} --data-path ${DATA_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} --num_coils ${NUM_COILS} 
                python valid_measures.py --checkpoint ${CHECKPOINT} --out-dir ${OUT_DIR} --batch-size ${BATCH_SIZE} --device ${DEVICE} --data-path ${DATA_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} --num_coils ${NUM_COILS} --report-path ${REPORT_PATH} 
                done 
            done 
        done
    done

