"""
Структура двовимірного кристалу та правила руху частинок (атомів домішки).

Кристал — це сітка n×n клітинок. Кожна клітинка містить ціле число —
кількість частинок у ній. Сумарна кількість частинок зберігається на всіх
кроках симуляції (інваріант).

Правила руху:
  • На кожному кроці часу частинка обирає напрямок випадково з 4 можливих:
    UP (-1, 0), DOWN (+1, 0), LEFT (0, -1), RIGHT (0, +1).
  • Якщо рух виводить за межу — частинка лишається на місці (відбиття).
  • У клітинці може бути будь-яка кількість частинок.

Інваріанти:
  • sum(crystal) == initial_count в усіх знімках.
  • Кожна частинка має координати (row, col) ∈ [0, GRID) × [0, GRID).
"""
from __future__ import annotations
from dataclasses import dataclass, field
import random
import numpy as np

# Напрямки руху: row_delta, col_delta
DIRECTIONS = [(-1, 0), (+1, 0), (0, -1), (0, +1)]   # UP, DOWN, LEFT, RIGHT
DIR_NAMES = ["UP", "DOWN", "LEFT", "RIGHT"]


@dataclass
class CrystalConfig:
    """Параметри кристалу та симуляції."""
    grid_size: int = 50            # розмір сітки n×n
    n_particles: int = 1000        # кількість частинок
    n_steps: int = 200             # кроків часу симуляції
    snapshot_every: int = 20       # робити знімок кожні N кроків
    move_probability: float = 1.0  # ймовірність руху (1.0 — частинка завжди пробує рух)
    seed: int = 42


@dataclass
class Particle:
    """Одна частинка-домішка з власним станом."""
    pid: int
    row: int
    col: int
    rng: random.Random = field(default_factory=random.Random)


def make_initial_state(config: CrystalConfig) -> tuple[np.ndarray, list[Particle]]:
    """
    Початковий стан: кристал як 2D-масив (int32), частинки розкидаємо
    рівномірно випадково по всіх клітинках.
    """
    rng = random.Random(config.seed)
    crystal = np.zeros((config.grid_size, config.grid_size), dtype=np.int32)
    particles = []
    for pid in range(config.n_particles):
        r = rng.randint(0, config.grid_size - 1)
        c = rng.randint(0, config.grid_size - 1)
        crystal[r, c] += 1
        # Кожна частинка отримує свій під-генератор з унікальним seed —
        # це робить симуляцію детермінованою (важливо для seed-ускладнення).
        particles.append(Particle(pid=pid, row=r, col=c,
                                   rng=random.Random(config.seed * 10000 + pid)))
    return crystal, particles


def total_particles(crystal: np.ndarray) -> int:
    """Сума всіх клітинок — має дорівнювати n_particles на всіх кроках."""
    return int(crystal.sum())


def choose_direction(rng: random.Random) -> tuple[int, int]:
    """Випадковий напрямок (UP/DOWN/LEFT/RIGHT)."""
    return DIRECTIONS[rng.randint(0, 3)]


def reflected_move(row: int, col: int, dr: int, dc: int, grid_size: int) -> tuple[int, int]:
    """
    Спробувати рух (dr, dc); якщо за межу — повернути стару позицію (відбиття).
    """
    new_r = row + dr
    new_c = col + dc
    if 0 <= new_r < grid_size and 0 <= new_c < grid_size:
        return new_r, new_c
    return row, col   # відбиття: лишаємось на місці


@dataclass
class SimulationResult:
    """Результат однієї симуляції."""
    config: CrystalConfig
    snapshots: list[np.ndarray]       # моментальні знімки кристалу
    snapshot_steps: list[int]          # на якому кроці зроблено знімок
    final_crystal: np.ndarray          # фінальний стан
    initial_count: int                 # кількість частинок на старті
    final_count: int                   # після завершення (має == initial_count)
    elapsed: float                     # час виконання, секунди
    method_name: str                   # назва методу для звіту
    integrity_ok: bool                 # чи зберігся інваріант
    extra: dict = field(default_factory=dict)
