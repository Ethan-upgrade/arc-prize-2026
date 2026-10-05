cd /mnt/d/arc-prize-2026
mkdir -p runs/ab
N=5
for i in $(seq 0 $((N-1))); do
  ~/arc-venv/bin/python -u -m arcsolver.ab --steps 500 --shard $i/$N --out runs/ab/shard$i.jsonl > runs/ab/shard$i.log 2>&1 &
done
wait
echo ALL_DONE
