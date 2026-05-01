"""
NAIVE багатопоточна симуляція: 1 потік на 1 частинку, без синхронізації.

Демонструє дві проблеми:
  1. RACE CONDITION: дві частинки, що знаходяться в одній клітинці, виконують
     crystal[r,c] -= 1 одночасно. У CPython частина операцій захищена GIL,
     але time.sleep(0) між read і write створює yield-точку — інший потік
     встигає взяти таке саме значення → одне зменшення «втрачається»,
     частинка зникає.

  2. Вузьке місце генератора: усі потоки використовують один глобальний
     random.Random; ефективно вони серіалізуються через GIL під час кожного
     виклику.

Архітектура — synchronization барʼєр на кожному кроці:
  • На кожному кроці N_PARTICLES потоків роблять свій рух.
  • Бар'єр чекає, поки всі завершать крок.
  • Головний потік знімає snapshot.

Це дає чесне порівняння з sequential: однакова кількість кроків часу.
"""
from __future__ import annotations
import threading
import time
import numpy as np

from crystal import (
    CrystalConfig, make_initial_state, total_particles,
    choose_direction, reflected_move, SimulationResult, Particle,
)


def _particle_worker_naive(p: Particle, crystal: np.ndarray, config: CrystalConfig,
                            step_barrier: threading.Barrier,
                            stop_event: threading.Event):
    """
    Потік однієї частинки. На кожному кроці:
      1) обирає напрямок і обчислює нову позицію
      2) виконує crystal[old] -= 1; crystal[new] += 1   ← RACE!
      3) чекає на бар'єр, щоб усі потоки завершили крок
    """
    for _ in range(config.n_steps):
        if stop_event.is_set():
            break
        if p.rng.random() <= config.move_probability:
            dr, dc = choose_direction(p.rng)
            new_r, new_c = reflected_move(p.row, p.col, dr, dc, config.grid_size)
            if (new_r, new_c) != (p.row, p.col):
                # ↓↓↓ КРИТИЧНА СЕКЦІЯ БЕЗ ЗАХИСТУ ↓↓↓
                old_val_src = crystal[p.row, p.col]
                # time.sleep(0) — явна yield-точка для надійної демонстрації race;
                # без неї GIL CPython часто захищає це від переривання.
                time.sleep(0)
                crystal[p.row, p.col] = old_val_src - 1   # «втрачаємо» -1, якщо інший
                                                            # потік взяв таке саме значення

                old_val_dst = crystal[new_r, new_c]
                time.sleep(0)
                crystal[new_r, new_c] = old_val_dst + 1
                # ↑↑↑ КРИТИЧНА СЕКЦІЯ БЕЗ ЗАХИСТУ ↑↑↑

                p.row, p.col = new_r, new_c

        # Барʼєр: усі потоки чекають один одного перед переходом до наступного кроку.
        # Це робить snapshot'и консистентними між кроками.
        try:
            step_barrier.wait(timeout=10.0)
        except threading.BrokenBarrierError:
            break


def run_naive_threaded(config: CrystalConfig, timeout_s: float = 60.0) -> SimulationResult:
    """
    NAIVE: 1 потік на 1 частинку, без жодного захисту від race condition.
    """
    crystal, particles = make_initial_state(config)
    initial_count = total_particles(crystal)

    snapshots = [crystal.copy()]
    snapshot_steps = [0]

    # +1 для головного потоку, який знімає знімки
    step_barrier = threading.Barrier(config.n_particles + 1)
    stop_event = threading.Event()

    threads = []
    for p in particles:
        t = threading.Thread(target=_particle_worker_naive,
                              args=(p, crystal, config, step_barrier, stop_event),
                              daemon=True)
        threads.append(t)

    t0 = time.perf_counter()
    for t in threads:
        t.start()

    try:
        for step in range(1, config.n_steps + 1):
            # Чекаємо на бар'єр — всі частинки завершили цей крок
            try:
                step_barrier.wait(timeout=timeout_s)
            except threading.BrokenBarrierError:
                stop_event.set()
                break
            if step % config.snapshot_every == 0:
                # Snapshot читається ЗА БАР'ЄРОМ — у момент, коли жоден потік
                # не пише. Однак сам snapshot теж робиться без явного lock'а
                # — інша проблема, що може дати неконсистентний знімок.
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
        method_name="naive_threaded (1 thread / particle, NO sync)",
        integrity_ok=(initial_count == final_count),
        extra={"n_threads": config.n_particles},
    )


if __name__ == "__main__":
    cfg = CrystalConfig(grid_size=20, n_particles=200, n_steps=100, snapshot_every=20)
    r = run_naive_threaded(cfg)
    print(f"  naive_threaded: t={r.elapsed:.2f}s  initial={r.initial_count}  "
          f"final={r.final_count}  loss={r.initial_count - r.final_count}  "
          f"integrity={'OK ✓' if r.integrity_ok else 'CORRUPTED ✗'}")
