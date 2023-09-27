#!/bin/bash

#SBATCH --account=def-rrabba
#SBATCH --time=1-20:00:00               # time (DD-HH:MM)
#SBATCH --cpus-per-task=2               # CPU cores/threads
#SBATCH --gres=gpu:1                    # number of GPU(s) per node
#SBATCH --mem=32G                      # memory (per node)
#SBATCH --job-name=DGC_July24
#SBATCH --output=outlog/%x-%j.log



model_name="GRUGCN"
max_epoch=200


for dataset_name in aion aragon bancor centra cindicator coindash dgd iconomi mathoverflow CollegeMsg adex aeternity
do 
    # start_time="$(data -u +%u)"

    echo " >>> MODEL: $model_name"
    echo " >>> DATA: $dataset_name"

    echo "===================================================================================="
    echo "===================================================================================="
    echo ""
    echo " ***** $model_name: $dataset_name *****"

    # command to run the model
    for seed in 43 44 45 46 47
    do
        echo "=========================================="
        echo " >>> Seed: $seed"
        echo " >>> MODEL: $model_name"
        echo " >>> DATA: $dataset_name"
        echo "=========================================="
        python train_tgc_end_to_end.py --model "$model_name" --seed "$seed"  --dataset "$dataset_name" --max_epoch "$max_epoch" 
        echo
        echo 
    done

    # end_time="$(date -u +%s)"
    # elapsed="$(($end_time-$start_time))"
    # echo "Model: $model_name, Data: $data: Elapsed Time: $elapsed seconds."
    echo ""
    echo "===================================================================================="
    echo "===================================================================================="

done