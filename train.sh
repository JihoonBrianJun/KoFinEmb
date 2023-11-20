# CUDA_VISIBLE_DEVICES=2,3 python3 trainer.py
# CUDA_VISIBLE_DEVICES=2,3 python3 llama_test.py
# CUDA_VISIBLE_DEVICES=3 torchrun --nnodes 1 --nproc_per_node 1 trainer.py --enable_fsdp --low_cpu_fsdp
CUDA_VISIBLE_DEVICES=3 torchrun --nnodes 1 --nproc_per_node 1 trainer.py