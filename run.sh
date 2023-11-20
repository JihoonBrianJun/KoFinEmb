# CUDA_VISIBLE_DEVICES=0,1,3 torchrun --nnodes 1 --nproc_per_node 3 run.py --run_mode train
CUDA_VISIBLE_DEVICES=3 torchrun --nnodes 1 --nproc_per_node 1 run.py --run_mode eval