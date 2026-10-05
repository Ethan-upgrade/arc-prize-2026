cd /mnt/d/arc-prize-2026
timeout 200 ~/arc-venv/bin/python -u -m arcsolver.ab --tasks 7b80bb43 --steps 30 --out /tmp/smoke.jsonl 2>&1 | tail -5
