"""
Масштабована симуляція: WORKER POOL з фіксованою кількістю потоків,
незалежно від кількості частинок.

Це ускладнена задача (15 балів): «Забезпечення процесу масштабування
для атомів — вирішення проблеми 1 атом = 1 потік».

Проблема naive-підходу:
  При N=10000 частинок створюється 10000 потоків. Кожен потік займає
  ~8 МБ стеку + структури ОС. На 32-біт системі це переповнить адресний
  простір, на 64-біт — створить шалений overhead на context switching.
  GIL Python робить додаткову суцільну серіалізацію.

Рішення — Worker Pool + Chunking:
  1. W потоків (W ≪ N), де W ~= кількість CPU-ядер.
  2. На кожному кроці часу частинки розбиваються на W приблизно рівних
     батчів (chunks). Кожен потік обробляє свій батч.
  3. Барʼєр між кроками гарантує консистентний знімок.
  4. Той самий resource ordering на per-cell locks.

Додаткова оптимізація: зменшуємо contention, використовуючи atomic
операції з модуля threading. Однак Python не має нативних atomic,
тому використовуємо "global crystal lock" для лайт-секцій або
fine-grained locks. Тут демонструємо fine-grained.
"""
from __future__ import annotations
import threading
import time
import numpy as np

from crystal import (
    CrystalConfig, make_initial_state, total_particles,
    choose_direction, reflected_move, SimulationResult, Particle,
)


def _process_batch(particles: list, crystal: np.ndarray, cell_locks: list,
                    grid_size: int, move_probability: float):
    """
    Обробка батчу частинок одним потоком worker'а: цикл по частинках,
    кожна — read direction → compute new pos → grab two locks → update.
    """
    for p in particles:
        if p.rng.random() > move_probability:
            continue
        dr, dc = choose_direction(p.rng)
        new_r, new_c = reflected_move(p.row, p.col, dr, dc, grid_size)
        if (new_r, new_c) == (p.row, p.col):
            continue
        src_idx = p.row * grid_size + p.col
        dst_idx = new_r * grid_size + new_c
        # Resource ordering — те саме рішення deadlock'у, що й у sim_solved
        first_idx, second_idx = sorted([src_idx, dst_idx])
        with cell_locks[first_idx]:
            with cell_locks[second_idx]:
                crystal[p.row, p.col] -= 1
                crystal[new_r, new_c] += 1
        p.row, p.col = new_r, new_c


def run_pool_threaded(config: CrystalConfig, n_workers: int = 4) -> SimulationResult:
    """
    Worker Pool: фіксована кількість n_workers потоків, незалежно
    від config.n_particles.
    """
    crystal, particles = make_initial_state(config)
    initial_count = total_particles(crystal)

    n_cells = config.grid_size * config.grid_size
    cell_locks = [threading.Lock() for _ in range(n_cells)]

    snapshots = [crystal.copy()]
    snapshot_steps = [0]

    # Розбиваємо частинки на W батчів — round-robin для рівномірності
    batches = [particles[i::n_workers] for i in range(n_workers)]

    # Барʼєр на n_workers + 1 (головний потік)
    step_barrier = threading.Barrier(n_workers + 1)
    next_step = threading.Event()

    def worker(batch):
        for _ in range(config.n_steps):
            _process_batch(batch, crystal, cell_locks,
                           config.grid_size, config.move_probability)
            step_barrier.wait()       # сигналізуємо: батч завершено
            next_step.wait()          # чекаємо дозвіл на наступний крок

    threads = [threading.Thread(target=worker, args=(b,), daemon=True)
               for b in batches]

    t0 = time.perf_counter()
    for t in threads:
        t.start()

    for step in range(1, config.n_steps + 1):
        step_barrier.wait()           # всі worker'и завершили свій батч цього кроку
        if step % config.snapshot_every == 0:
            snapshots.append(crystal.copy())
            snapshot_steps.append(step)
        # Скидаємо next_step щоб наступного кроку всі знов чекали
        next_step.clear()
        step_barrier.reset()
        next_step.set()                # дозволяємо наступний крок

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
        method_name=f"pool_threaded (W={n_workers} workers)",
        integrity_ok=(initial_count == final_count),
        extra={"n_workers": n_workers, "n_particles": config.n_particles,
               "particles_per_worker": config.n_particles / n_workers},
    )


if __name__ == "__main__":
    cfg = CrystalConfig(grid_size=20, n_particles=200, n_steps=100, snapshot_every=20)
    for w in [1, 2, 4, 8]:
        r = run_pool_threaded(cfg, n_workers=w)
        print(f"  pool W={w}: t={r.elapsed:.2f}s  initial={r.initial_count}  "
              f"final={r.final_count}  integrity={'OK ✓' if r.integrity_ok else 'FAIL ✗'}")
