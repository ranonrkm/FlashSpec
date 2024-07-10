export CUDA_VISIBLE_DEVICES=6,7,8,9
export OMP_NUM_THREADS=48
# export ENABLE_INTRA_NODE_COMM=1
# torchrun --standalone --nproc_per_node=4 --master_port=13456 tests/longspec_benchmark.py --B 2 --target /home/jianc2/FastHesse/checkpoints/meta-llama/Llama-2-7b-hf/model.pth --model /home/jianc2/FastHesse/checkpoints/princeton-nlp/Sheared-LLaMA-1.3B/model.pth --rank_group 0 1 2 3 --printoutput --gamma 10 --M 4085
torchrun --standalone --nproc_per_node=4 --master_port=13456 tests/longspec_benchmark.py --B 16 --target /home/jianc2/FastHesse/checkpoints/meta-llama/Llama-2-7b-hf/model.pth --model /home/jianc2/FastHesse/checkpoints/JackFram/llama-68m/model.pth --rank_group 0 1 2 3 --printoutput --gamma 1 --M 4091 --benchmark --compile
# torchrun --standalone --nproc_per_node=4 --master_port=13456 tests/longspec_benchmark.py --B 1 --target /home/jianc2/FastHesse/checkpoints/meta-llama/Llama-2-7b-hf/model.pth --model /home/jianc2/FastHesse/checkpoints/TinyLlama/TinyLlama_v1.1/model.pth --rank_group 0 1 2 3 --printoutput --gamma 1 --M 4094 --benchmark
