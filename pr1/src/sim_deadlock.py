"""
DEADLOCK демонстрація: два замки без resource ordering.

Кожна частинка бере spectacle обидва замки, але в порядку (src, dst), без
сортування індексів. Дві частинки в сусідніх клітинках, які рухаються
назустріч одна одній, утворюють циклічну залежність:
    Потік A: lock(cell_X) → намагається lock(cell_Y)
    Потік B: lock(cell_Y) → намагається lock(cell_X)
   → DEADLOCK.

Час від часу симуляція виходить у deadlock — це детектується через
timeout на step_barrier. Це версія для звіту, не для production.
"""
from __future__ import annotations
import threading
import time
import numpy as np

from crystal import (
    CrystalConfig, make_initial_state, total_particles,
    choose_direction, reflected_move, SimulationResult, Particle,
)


def _particle_worker_deadlock(p: Particle, crystal: np.ndarray, cell_locks: list,
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
                src_idx = p.row * grid_size + p.col
                dst_idx = new_r * grid_size + new_c
                # ↓↓↓ DEADLOCK PATTERN: НЕ сортуємо за індексом! ↓↓↓
                src_lock = cell_locks[src_idx]
                dst_lock = cell_locks[dst_idx]
                src_lock.acquire()
                # Невелика затримка ↑ збільшує ймовірність deadlock'у
                time.sleep(0.0001)
                dst_lock.acquire()
                try:
                    crystal[p.row, p.col] -= 1
                    crystal[new_r, new_c] += 1
                finally:
                    dst_lock.release()
                    src_lock.release()
                p.row, p.col = new_r, new_c

        try:
            step_barrier.wait(timeout=15.0)
        except threading.BrokenBarrierError:
            break


def run_deadlock_threaded(config: CrystalConfig, timeout_s: float = 8.0) -> SimulationResult:
    """
    Запускає симуляцію з deadlock-патерном (без resource ordering).
    Використовуємо короткий timeout: при deadlock швидко виходимо й
    позначаємо результат deadlocked=True.
    """
    crystal, particles = make_initial_state(config)
    initial_count = total_particles(crystal)

    n_cells = config.grid_size * config.grid_size
    cell_locks = [threading.Lock() for _ in range(n_cells)]

    snapshots = [crystal.copy()]
    snapshot_steps = [0]

    step_barrier = threading.Barrier(config.n_particles + 1)
    stop_event = threading.Event()

    threads = []
    for p in particles:
        t = threading.Thread(target=_particle_worker_deadlock,
                              args=(p, crystal, cell_locks, config,
                                    step_barrier, stop_event, config.grid_size),
                              daemon=True)
        threads.append(t)

    t0 = time.perf_counter()
    for t in threads:
        t.start()

    deadlocked = False
    completed = 0
    try:
        for step in range(1, config.n_steps + 1):
            try:
                step_barrier.wait(timeout=timeout_s)
            except threading.BrokenBarrierError:
                deadlocked = True
                stop_event.set()
                break
            completed = step
            if step % config.snapshot_every == 0:
                snapshots.append(crystal.copy())
                snapshot_steps.append(step)
    except threading.BrokenBarrierError:
        deadlocked = True
        stop_event.set()

    # Розблоковуємо застрягле барʼєрне очікування примусово
    step_barrier.abort()
    # Daemon-threads загинуть при виході; .join з малим таймаутом
    for t in threads:
        t.join(timeout=0.1)

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
        method_name="deadlock_threaded (no resource ordering)",
        integrity_ok=(initial_count == final_count),
        extra={"deadlocked": deadlocked, "n_threads": config.n_particles,
               "completed_steps": completed},
    )


if __name__ == "__main__":
    # Беремо невелику сітку, щоб збільшити ймовірність deadlock'у
    cfg = CrystalConfig(grid_size=10, n_particles=300, n_steps=100, snapshot_every=20)
    r = run_deadlock_threaded(cfg)
    extra = r.extra
    print(f"  deadlock_threaded: t={r.elapsed:.2f}s  "
          f"completed_steps={extra['completed_steps']}/{cfg.n_steps // cfg.snapshot_every}  "
          f"deadlocked={extra['deadlocked']}")
