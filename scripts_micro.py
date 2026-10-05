import time, torch
def bench(dev, n=20000):
    x = torch.ones(8, device=dev)
    for _ in range(200): x = x + 1
    if dev == "cuda": torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n): x = x + 1
    if dev == "cuda": torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1e6
print(f"cpu-op {bench('cpu'):.1f}us  cuda-op {bench('cuda'):.1f}us  threads {torch.get_num_threads()}")
