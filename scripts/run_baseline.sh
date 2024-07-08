export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=48
# export ENABLE_INTRA_NODE_COMM=1
torchrun --standalone --nproc_per_node=4 --master_port=13456 tests/baseline_benchmark.py --B 2 --checkpoint_path /home/jianc2/FastHesse/checkpoints/meta-llama/Llama-2-7b-hf/model.pth --rank_group 0 1 2 3 --printoutput --compile