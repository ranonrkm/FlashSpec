export CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS=48
# export ENABLE_INTRA_NODE_COMM=1
# torchrun --standalone --nproc_per_node=1 --master_port=13456 tests/test_latency.py --M 512 --declen_list 1 --P 511 --B 128 --checkpoint_path /home/jianc2/FastHesse/checkpoints/JackFram/llama-68m/model.pth --rank_group 0 --compile

# torchrun --standalone --nproc_per_node=1 --master_port=13456 tests/test_latency_draft.py --M 4096 --declen_list 1 --P 4000 --B 128 --checkpoint_path /home/jianc2/FastHesse/checkpoints/JackFram/llama-68m/model.pth --rank_group 0 --compile

# torchrun --standalone --nproc_per_node=1 --master_port=13456 tests/test_latency.py --M 512 --declen_list 1 --P 511 --B 128 --checkpoint_path /home/jianc2/FastHesse/checkpoints/TinyLlama/TinyLlama_v1.1/model.pth --rank_group 0 --compile

# torchrun --standalone --nproc_per_node=8 --master_port=13456 tests/test_latency.py --M 4096 --declen_list 1 2 --P 4000 --B 1 --checkpoint_path ../FastHesse/checkpoints/meta-llama/Llama-2-7b-hf/model.pth --rank_group 0 1 2 3 4 5 6 7 --compile

# torchrun --standalone --nproc_per_node=4 --master_port=13456 tests/test_latency.py --M 256 --declen_list 1 --P 251 --B 128 --checkpoint_path checkpoints/TinyLlama/TinyLlama_v1.1/model.pth --rank_group 0 1 2 3 --compile

# torchrun --standalone --nproc_per_node=1 --master_port=13456 tests/test_latency.py --M 16004 --declen_list 1 2 3 4 --P 16000 --B 16 --checkpoint_path checkpoints/gradientai/Llama-3-8B-Instruct-Gradient-1048k/model.pth --rank_group 0 --compile
ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=4 --master_port=13456 tests/test_latency.py --M 16004 --declen_list 1 2 3 4 --P 16000 --B 64 --checkpoint_path checkpoints/gradientai/Llama-3-8B-Instruct-Gradient-1048k/model.pth --rank_group 0 1 2 3 --compile

torchrun --standalone --nproc_per_node=1 --master_port=13456 tests/test_latency_profile.py --M 4004 --declen_list 1 2 3 4 --P 4000 --B 1 --checkpoint_path ../FastHesse/checkpoints/meta-llama/Llama-2-7b-hf/model.pth --rank_group 0 --compile

# torchrun --standalone --nproc_per_node=7 --master_port=13456 tests/test_latency.py --maxlen 272 --declen_list 1 4 8 --prefixlen 128 --batch 16 --checkpoint_path checkpoints/meta-llama/Llama-2-70b-hf/model.pth --rank_group 0 1 2 3 4 5 6 --compile

# torchrun --standalone --nproc_per_node=6 --master_port=13456 tests/test_latency.py --maxlen 272 --declen_list 1 4 8 --prefixlen 128 --batch 16 --checkpoint_path checkpoints/meta-llama/Llama-2-70b-hf/model.pth --rank_group 0 1 2 3 4 5 --compile

# torchrun --standalone --nproc_per_node=5 --master_port=13456 tests/test_latency.py --maxlen 272 --declen_list 1 4 8 --prefixlen 128 --batch 16 --checkpoint_path checkpoints/meta-llama/Llama-2-70b-hf/model.pth --rank_group 0 1 2 3 4 --compile

# torchrun --standalone --nproc_per_node=4 --master_port=13456 tests/test_latency.py --maxlen 272 --declen_list 1 4 8 --prefixlen 128 --batch 16 --checkpoint_path checkpoints/meta-llama/Llama-2-70b-hf/model.pth --rank_group 0 1 2 3 --compile