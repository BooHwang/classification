export CUDA_VISIBLE_DEVICES=1,2,4,5
export OMP_NUM_THREADS=2


torchrun --master_port=21271 \
         --nproc_per_node=4 \
         train.py \