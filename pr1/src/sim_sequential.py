"""
Послідовна (baseline) реалізація броунівського руху.

Не використовує потоки. Всі частинки обробляються одна за одною на кожному
кроці часу. Це еталон для перевірки коректності та порівняння продуктивності.
"""
from __future__ import annotations
import time
import numpy as np

from crystal import (
    CrystalConfig, make_initial_state, total_particles,
    choose_direction, reflected_move, SimulationResult,
)


def run_sequential(config: CrystalConfig) -> SimulationResult:
    crystal, particles = make_initial_state(config)
    initial_count = total_particles(crystal)

    snapshots = [crystal.copy()]   # знімок №0 — початковий стан
    snapshot_steps = [0]

    t0 = time.perf_counter()

    for step in range(1, config.n_steps + 1):
        for p in particles:
            if p.rng.random() > config.move_probability:
                continue
            dr, dc = choose_direction(p.rng)
            new_r, new_c = reflected_move(p.row, p.col, dr, dc, config.grid_size)
            if (new_r, new_c) != (p.row, p.col):
                # Атомарна операція в послідовному режимі — просто read-modify-write
                crystal[p.row, p.col] -= 1
                crystal[new_r, new_c] += 1
                p.row, p.col = new_r, new_c

        if step % config.snapshot_every == 0:
            snapshots.append(crystal.copy())
            snapshot_steps.append(step)

    elapsed = time.perf_counter() - t0
    final_count = total_particles(crystal)

    return SimulationResult(
        config=config,
        snapshots=snapshots,
        snapshot_steps=snapshot_steps,
        final_crystal=crystal,
        initial_count=initial_count,
        final_count=final_count,
        elapsed=elapsed,
        method_name="sequential",
        integrity_ok=(initial_count == final_count),
    )


if __name__ == "__main__":
    cfg = CrystalConfig(grid_size=50, n_particles=1000, n_steps=200, snapshot_every=20)
    r = run_sequential(cfg)
    print(f"  sequential: t={r.elapsed:.2f}s  initial={r.initial_count}  "
          f"final={r.final_count}  integrity={'OK ✓' if r.integrity_ok else 'FAIL ✗'}")
    print(f"  snapshots: {len(r.snapshots)} taken at steps {r.snapshot_steps}")
