export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=48
export CUDA_LAUNCH_BLOCKING=1
# export ENABLE_INTRA_NODE_COMM=1
# torchrun --standalone --nproc_per_node=4 --master_port=13456 tests/longspec_benchmark.py --B 2 --target /home/jianc2/FastHesse/checkpoints/meta-llama/Llama-2-7b-hf/model.pth --model /home/jianc2/FastHesse/checkpoints/princeton-nlp/Sheared-LLaMA-1.3B/model.pth --rank_group 0 1 2 3 --printoutput --gamma 10 --M 4085
# torchrun --standalone --nproc_per_node=1 --master_port=13456 tests/longspec_benchmark.py --B 2 --target /home/jianc2/FastHesse/checkpoints/meta-llama/Llama-2-7b-hf/model.pth --model /home/jianc2/FastHesse/checkpoints/JackFram/llama-68m/model.pth --rank_group 0 --draft_ranks 0 --gamma 1 --prefix_len 4000 --gen_len 90 --benchmark --printoutput --streamingllm_budget 256 --compile
torchrun --standalone --nproc_per_node=4 --master_port=13456 tests/longspec_benchmark.py --B 1 --target /home/jianc2/FastHesse/checkpoints/meta-llama/Llama-2-7b-hf/model.pth --model /home/jianc2/FastHesse/checkpoints/TinyLlama/TinyLlama_v1.1/model.pth --rank_group 0 1 2 3 --draft_ranks 0 1 --gamma 1 --prefix_len 4000 --gen_len 90 --benchmark
