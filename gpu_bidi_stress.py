import torch
import time
import argparse
import gc
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--iterations', type=int, default=200)
parser.add_argument('--size-gb', type=float, default=4.0)
parser.add_argument('--num-streams', type=int, default=4)
parser.add_argument('--pause', type=float, default=0.1)
args = parser.parse_args()

if not torch.cuda.is_available():
    print('No CUDA available; exiting')
    sys.exit(1)

device = torch.device('cuda')
print(f'Using device: {torch.cuda.get_device_name(0)}')
print(f'CUDA capability: {torch.cuda.get_device_capability(0)}')

# total elements (float32)
elements = int(args.size_gb * (1024**3) / 4)
num_streams = max(1, args.num_streams)
chunk_elems = elements // num_streams

print(f'Iterations: {args.iterations}, Size: {args.size_gb} GB, Streams: {num_streams}, Chunk elems: {chunk_elems}')

# prepare pinned host src and dst, and device buffers
h_srcs = [torch.empty(chunk_elems, dtype=torch.float32, pin_memory=True).uniform_() for _ in range(num_streams)]
h_dsts = [torch.empty(chunk_elems, dtype=torch.float32, pin_memory=True) for _ in range(num_streams)]
d_bufs = [torch.empty(chunk_elems, dtype=torch.float32, device=device) for _ in range(num_streams)]
streams = [torch.cuda.Stream() for _ in range(num_streams)]

for i in range(args.iterations):
    print(f'\n=== iter {i+1}/{args.iterations} ===')
    t0 = time.time()
    # queue H2D and D2H on each stream
    for j in range(num_streams):
        s = streams[j]
        with torch.cuda.stream(s):
            # host -> device
            d_bufs[j].copy_(h_srcs[j], non_blocking=True)
            # device -> host back to dst
            h_dsts[j].copy_(d_bufs[j], non_blocking=True)
    # wait for all
    torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0
    total_gb = args.size_gb * 2  # H2D + D2H
    bw = total_gb / dt if dt > 0 else float('inf')
    print(f'Roundtrip time: {dt:.3f}s, effective bidirectional BW: {bw:.2f} GB/s')

    # occasionally do a small compute to exercise device
    if i % 10 == 0:
        a = torch.randn((1024,1024), device=device)
        b = torch.randn((1024,1024), device=device)
        _ = a @ b
        torch.cuda.synchronize()
        del a, b

    # free and collect occasionally
    if i % 50 == 0:
        torch.cuda.empty_cache()
        gc.collect()

    time.sleep(args.pause)

print('\nCompleted bidirectional stress loop')