#!/bin/bash


# 1) pre-process a continuous-time dataset: generate positive and negative edgelist
python utils/preprocess_ct_data.py --dataset=canVote --neg_sample=rnd


# 2) read a continuous-time dataset: convert the continuous time edgeslit to a discrete-time snapshot series
python utils/read_continuous_data.py --dataset=canVote --neg_sample=rnd --e_type=p
python utils/read_continuous_data.py --dataset=canVote --neg_sample=rnd --e_type=n


# 3) run a model on the snapshots of a DTDG dataset; requires a negative sampling strategy if a CTDG is specified
# HTGN
python main.py --model=HTGN --seed=1024 --neg_sample=rnd --dataset=canVote
python main.py --model=HTGN --seed=710 --dataset=aion

# EvolveGCN
python ./baselines/run_evolvegcn_baselines.py \
          --model=EGCN \
          --dataset=${dataset} \
          --device_id=${device_id}
# GRUGCN
python main.py \
          --model=GRUGCN \
          --dataset=${dataset} \
          --device_id=${device_id}
# GAE VGAE
python ./baselines/run_static_baselines.py \
          --model=GAE \
          --dataset=${dataset} \
          --device_id=${device_id}

python ./baselines/run_static_baselines.py \
          --model=VGAE \
          --dataset=${dataset} \
          --device_id=${device_id}