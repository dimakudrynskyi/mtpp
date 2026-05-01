"""
SOLVED багатопоточна симуляція: 1 потік на 1 частинку, з повним захистом
від race condition та deadlock.

Рішення:
  1. Per-cell locks: для кожної клітинки — окремий threading.Lock. При русі
     частинка повинна захопити ДВА замки: для старої і нової клітинки.

  2. Уникнення DEADLOCK через RESOURCE ORDERING (Coffman, 1971):
     завжди беремо замки в порядку зростання плоского індексу cell_idx =
     row * GRID + col. Це усуває умову circular wait — два потоки, що
     рухають у протилежних напрямках, гарантовано візьмуть замки в
     одному й тому ж порядку.

  3. Atomic update: всередині критичної секції — стандартний read-modify-write
     без yield-точок (під захистом замків).

Поки потік чекає на замок, інші потоки можуть прогресувати, тому загальна
паралельність зберігається. Це КРАЩЕ за глобальний м'ютекс на весь crystal.
"""
from __future__ import annotations
import threading
import time
import numpy as np

from crystal import (
    CrystalConfig, make_initial_state, total_particles,
    choose_direction, reflected_move, SimulationResult, Particle,
)


def _particle_worker_solved(p: Particle, crystal: np.ndarray, cell_locks: list,
                              config: CrystalConfig,
                              step_barrier: threading.Barrier,
                              stop_event: threading.Event,
                              grid_size: int):
    for _ in range(config.n_steps):
        if stop_event.is_set():
            break
        if p.rng.random() <= config.move_probability:
            dr, dc = choose_direction(p.rng)
            new_r, new_c = reflected_move(p.row, p.col, dr, dc, grid_size)
            if (new_r, new_c) != (p.row, p.col):
                # Обчислюємо плоскі індекси клітинок
                src_idx = p.row * grid_size + p.col
                dst_idx = new_r * grid_size + new_c
                # ↓↓↓ Resource Ordering: завжди в порядку зростання індексу ↓↓↓
                first_idx, second_idx = sorted([src_idx, dst_idx])
                with cell_locks[first_idx]:
                    with cell_locks[second_idx]:
                        # Тепер критична секція безпечна:
                        # лише ми володіємо обома клітинками.
                        crystal[p.row, p.col] -= 1
                        crystal[new_r, new_c] += 1
                p.row, p.col = new_r, new_c

        try:
            step_barrier.wait(timeout=10.0)
        except threading.BrokenBarrierError:
            break


def run_solved_threaded(config: CrystalConfig, timeout_s: float = 60.0) -> SimulationResult:
    crystal, particles = make_initial_state(config)
    initial_count = total_particles(crystal)

    # Окремий замок на кожну клітинку — fine-grained locking
    n_cells = config.grid_size * config.grid_size
    cell_locks = [threading.Lock() for _ in range(n_cells)]
    # snapshot_lock: захищає момент копіювання кристалу для знімка
    snapshot_lock = threading.Lock()

    snapshots = [crystal.copy()]
    snapshot_steps = [0]

    step_barrier = threading.Barrier(config.n_particles + 1)
    stop_event = threading.Event()

    threads = []
    for p in particles:
        t = threading.Thread(target=_particle_worker_solved,
                              args=(p, crystal, cell_locks, config,
                                    step_barrier, stop_event, config.grid_size),
                              daemon=True)
        threads.append(t)

    t0 = time.perf_counter()
    for t in threads:
        t.start()

    try:
        for step in range(1, config.n_steps + 1):
            try:
                step_barrier.wait(timeout=timeout_s)
            except threading.BrokenBarrierError:
                stop_event.set()
                break
            if step % config.snapshot_every == 0:
                # Знімок робимо за бар'єром — всі частинки завершили крок,
                # ніхто не пише. Snapshot_lock тут — додаткова страховка.
                with snapshot_lock:
                    snapshots.append(crystal.copy())
                    snapshot_steps.append(step)
    except threading.BrokenBarrierError:
        stop_event.set()

    for t in threads:
        t.join(timeout=2.0)

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
        method_name="solved_threaded (per-cell locks + ordering)",
        integrity_ok=(initial_count == final_count),
        extra={"n_threads": config.n_particles, "n_locks": n_cells},
    )


if __name__ == "__main__":
    cfg = CrystalConfig(grid_size=20, n_particles=200, n_steps=100, snapshot_every=20)
    r = run_solved_threaded(cfg)
    print(f"  solved_threaded: t={r.elapsed:.2f}s  initial={r.initial_count}  "
          f"final={r.final_count}  integrity={'OK ✓' if r.integrity_ok else 'CORRUPTED ✗'}")
