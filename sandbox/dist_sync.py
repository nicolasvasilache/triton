import os
from functools import partial

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from triton import runtime
from triton.testing import _summarize_statistics

def setup(rank, world_size):
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')
    # rccl creates 8 threads and 100 streams making it hard to see traces.
    # since we only need the synchronization part, just use gloo for rdv for now.
    # dist.init_process_group("nccl", rank=rank, world_size=world_size, device_id=device)
    dist.init_process_group("gloo", rank=rank, world_size=world_size, device_id=device)
    print(f"Rank {rank}: Setup device {device}")
    return device

def teardown(rank, device):
    print(f"Rank {rank}: Teardown device {device}")
    dist.destroy_process_group()

def with_device_sync(device, fn):
    torch.cuda.synchronize(device)
    # with_cpu_sync(fn)
    fn()
    torch.cuda.synchronize(device)

def with_cpu_sync(device, fn):
    dist.barrier()
    with_device_sync(device, fn)
    # fn()
    dist.barrier()

def with_stream(stream, fn):
    stream.synchronize()
    with torch.cuda.stream(stream):
        fn()
    stream.synchronize()


def report(rank, start_event, end_event):
    elapsed_time = start_event.elapsed_time(end_event)
    print(f"Rank {rank}: completed in {elapsed_time:.3f} ms")


def record_copy_synchronize_global_sync(input_tensor, output_tensor, device, start_event, end_event):
    start_event.record()
    output_tensor.copy_(input_tensor)
    end_event.record()
    torch.cuda.synchronize(device)

def record_copy_synchronize_stream_sync(input_tensor, output_tensor, stream, start_event, end_event):
    def fn():
        start_event.record()
        output_tensor.copy_(input_tensor)
        end_event.record()
    with_stream(stream, fn)


def test_basic_barrier_sync(rank, device, input_tensor, output_tensor, start_event, end_event):
    print(f"Rank {rank}: start test_basic_barrier_sync")   
    with_cpu_sync(device, partial(record_copy_synchronize_global_sync, input_tensor, output_tensor, device, start_event, end_event))
    with_cpu_sync(device, partial(report, rank, start_event, end_event))
    

def test_stream_based_sync(rank, device, input_tensor, output_tensor, stream, start_event, end_event):
    print(f"Rank {rank}: start test_stream_based_sync")
    with_cpu_sync(device, partial(record_copy_synchronize_stream_sync, input_tensor, output_tensor, stream, start_event, end_event))
    with_cpu_sync(device, partial(report, rank, start_event, end_event))


def precise_sync_example(rank, world_size, tensor_size):
    device = setup(rank, world_size)
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    input_tensor = torch.randn(*tensor_size, device=device)
    output_tensor = torch.zeros_like(input_tensor)
    
    # Run all synchronization methods
    test_basic_barrier_sync(rank, device, input_tensor, output_tensor, start_event, end_event)
    test_basic_barrier_sync(rank, device, input_tensor, output_tensor, start_event, end_event)
    test_basic_barrier_sync(rank, device, input_tensor, output_tensor, start_event, end_event)
    test_basic_barrier_sync(rank, device, input_tensor, output_tensor, start_event, end_event)
    test_basic_barrier_sync(rank, device, input_tensor, output_tensor, start_event, end_event)
    test_basic_barrier_sync(rank, device, input_tensor, output_tensor, start_event, end_event)

    stream = torch.cuda.Stream(device=device)
    test_stream_based_sync(rank, device, input_tensor, output_tensor, stream, start_event, end_event)
    test_stream_based_sync(rank, device, input_tensor, output_tensor, stream, start_event, end_event)
    test_stream_based_sync(rank, device, input_tensor, output_tensor, stream, start_event, end_event)
    test_stream_based_sync(rank, device, input_tensor, output_tensor, stream, start_event, end_event)
    test_stream_based_sync(rank, device, input_tensor, output_tensor, stream, start_event, end_event)
    test_stream_based_sync(rank, device, input_tensor, output_tensor, stream, start_event, end_event)
    
    teardown(rank, device)


def do_dist_synchronized_bench(fn, device, warmup=25, rep=100, grad_to_none=None, quantiles=None, return_mode="mean"):
    assert return_mode in ["min", "max", "mean", "median", "all"]

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    with_cpu_sync(device, fn)

    cache = runtime.driver.active.get_empty_cache_for_benchmark()

    ##################################################################
    # Estimate the runtime of the function
    ##################################################################
    def warm():
        start_event.record()
        for _ in range(5):
            runtime.driver.active.clear_cache(cache)
            fn()
        end_event.record()

    with_cpu_sync(device, warm)
    estimate_ms = start_event.elapsed_time(end_event) / 5

    ##################################################################
    # compute number of warmup and repeat
    ##################################################################
    # n_warmup = max(1, int(warmup / estimate_ms))
    # n_repeat = max(1, int(rep / estimate_ms))
    # # Warm-up
    # for _ in range(n_warmup):
    #     fn()
    #
    # This is unsafe in distributed mode, we need to have the same number of iterations
    n_repeat = rep

    start_event = [torch.cuda.Event(enable_timing=True) for i in range(n_repeat)]
    end_event = [torch.cuda.Event(enable_timing=True) for i in range(n_repeat)]

    # Benchmark
    def bench():
        for i in range(n_repeat):
            # we don't want `fn` to accumulate gradient values
            # if it contains a backward pass. So we clear the
            # provided gradients
            if grad_to_none is not None:
                for x in grad_to_none:
                    x.grad = None
            # we clear the L2 cache before each run
            runtime.driver.active.clear_cache(cache)
            # record time of `fn`
            def doit():
                start_event[i].record()
                fn()
                end_event[i].record()
            
            with_cpu_sync(device, doit)

    bench()

    times = [s.elapsed_time(e) for s, e in zip(start_event, end_event)]
    return _summarize_statistics(times, quantiles, return_mode)


def test():
    world_size = min(4, torch.cuda.device_count())
    
    mp.spawn(
        precise_sync_example,
        args=(world_size, (2**30, )),
        nprocs=world_size,
        join=True
    )


if __name__ == "__main__":
    test()
