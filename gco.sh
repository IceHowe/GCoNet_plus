#!/bin/bash
#SBATCH -n 10
#SBATCH --gres=gpu:v100:1
#SBATCH --time=48:00:00


# Run python script
method="gconet_$1"
size=256
epochs=350
val_last=50

# Train
python3 train.py --trainset DUTS_class --size ${size} --ckpt_dir /gemini/output/ckpt/${method} --epochs ${epochs} --val_dir tmp4val_${method}

# # # Show validation results
# # python collect_bests.py

# Test
for ((ep=${epochs}-${val_last};ep<${epochs};ep++))
do
pred_dir=/gemini/output/preds/${method}/ep${ep}
rm -rf ${pred_dir}
python3 test.py --pred_dir ${pred_dir} --ckpt /gemini/output/ckpt/${method}/ep${ep}.pth --size ${size}
done


# Eval
cd evaluation
CUDA_VISIBLE_DEVICES=0 python3 main.py --methods ${method}
python3 sort_results.py
# python select_results.py
cd ..



nvidia-smi
hostname
