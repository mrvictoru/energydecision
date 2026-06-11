import torch
import time
import argparse
import gc
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--iterations', type=int, default=100)
parser.add_argument('--size-gb', type=float, default=1.0)
parser.add_argument('--matdim', type=int, default=2048)
parser.add_argument('--pause', type=float, default=1.0)
args = parser.parse_args()

if not torch.cuda.is_available():
    print('No CUDA available; exiting')
    sys.exit(1)

device = torch.device('cuda')
print(f'Using device: {torch.cuda.get_device_name(0)}')
print(f'CUDA capability: {torch.cuda.get_device_capability(0)}')

# number of float32 elements to reach approx size-gb
elements = int(args.size_gb * (1024**3) / 4)

for i in range(args.iterations):
    print('\n=== iter {}/{} ==='.format(i+1, args.iterations))
    try:
        # allocate pinned host memory and fill
        print(f'Allocating {args.size_gb} GB pinned host tensor...')
        cpu = torch.empty(elements, dtype=torch.float32, pin_memory=True).uniform_()

        # Host -> Device copy (pinned memory to stress PCIe DMA)
        t0 = time.time()
        gpu = cpu.to(device, non_blocking=True)
        torch.cuda.synchronize()
        t1 = time.time()
        dt = t1 - t0
        bw = args.size_gb / dt if dt>0 else float('inf')
        print(f'Host->Device copy: {dt:.3f}s, {bw:.2f} GB/s')

        # compute-heavy op to stress VRAM
        print(f'Running matmul {args.matdim}x{args.matdim} on device...')
        a = torch.randn((args.matdim, args.matdim), device=device)
        b = torch.randn((args.matdim, args.matdim), device=device)
        t0 = time.time()
        _ = a @ b
        torch.cuda.synchronize()
        t1 = time.time()
        print(f'Matmul time: {t1-t0:.3f}s')

        # free memory
        del cpu, gpu, a, b
        torch.cuda.empty_cache()
        gc.collect()

        time.sleep(args.pause)
    except Exception as e:
        print('Exception during iteration:', e)
        raise

print('\nCompleted stress loop')
