cd /mnt/d/arc-prize-2026/third_party/CompressARC
for t in 1 4; do
  echo "threads=$t: $(DEV=cpu THREADS=$t timeout 60 ~/arc-venv/bin/python -u bench_one.py training 007bbfb7 30 2>&1 | tail -1)"
done
