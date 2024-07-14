import os
import torch
import torch.distributed as dist
from torch.distributed import _functional_collectives as funcol

def main():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(f"cuda:{local_rank}")
    dist.init_process_group("nccl")

    draft_group = dist.new_group([0, 1, 2, 3])
    target_group = dist.new_group([4, 5, 6, 7])

    inp = torch.full((128, 128), rank, dtype=torch.bfloat16, device="cuda")
    dist.all_reduce(inp)
    expect = sum(range(world_size))
    assert inp.eq(expect).all()

    if 0 <= rank < 4:
        inp = torch.full((128, 128), rank, dtype=torch.bfloat16, device="cuda")
        result = funcol.all_reduce(inp, "sum", draft_group)
        expect = sum(range(4))
        assert result.eq(expect).all()
    else:
        inp = torch.full((128, 128), rank, dtype=torch.bfloat16, device="cuda")
        result = funcol.all_reduce(inp, "sum", draft_group)
        expect = sum(range(4, 8))
        assert result.eq(expect).all()

    torch.cuda.synchronize()
    print("success")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()