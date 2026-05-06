"""
Задача 1: Демонстрація Race Condition та Deadlock на банківських переказах.

Реалізовано п'ять версій:
  1. naive_race        — без жодної синхронізації → втрата/створення коштів (RACE)
  2. naive_deadlock    — два замки на пару рахунків у різному порядку → DEADLOCK
  3. ordered_locks     — упорядкований захват замків (по id) → коректно й без deadlock
  4. global_lock       — один глобальний замок на всю систему → коректно, але повільно
  5. lock_free_atomic  — компенсаційні транзакції з retry на CAS-подібному примітиві

У кожній версії N_THREADS потоків паралельно виконують N_TRANSFERS_PER_THREAD
переказів між двома випадковими рахунками. Інваріант, що перевіряється:
сумарний баланс системи має залишатися незмінним.
"""
from __future__ import annotations
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

# ---------- Параметри ----------
N_ACCOUNTS = 150
INITIAL_BALANCE_RANGE = (100, 10_000)
N_THREADS = 1024            # > 1000, як вимагає завдання
N_TRANSFERS_PER_THREAD = 50
SEED = 42


@dataclass
class Account:
    id: int
    balance: float
    lock: threading.RLock = field(default_factory=threading.RLock)


def make_accounts(seed: int = SEED) -> list[Account]:
    rnd = random.Random(seed)
    return [Account(id=i,
                    balance=float(rnd.randint(*INITIAL_BALANCE_RANGE)))
            for i in range(N_ACCOUNTS)]


def total_balance(accounts: list[Account]) -> float:
    return sum(a.balance for a in accounts)


# =============================================================================
# 1. NAIVE — повна відсутність синхронізації (RACE CONDITION)
# =============================================================================

def transfer_naive(src: Account, dst: Account, amount: float) -> bool:
    """
    КЛАСИЧНА RACE CONDITION:
      Між читанням src.balance і записом нового значення інший потік може
      зробити свій переказ. Тоді обидва побачать однаковий баланс і обидва
      його зменшать — гроші зникнуть. Аналогічно при додаванні до dst —
      гроші можуть продублюватися.

    У CPython арифметика над float в одному виразі частково захищена GIL,
    тому для надійної демонстрації RACE розбиваємо операцію на окремі
    кроки із мінімальною yield-точкою (time.sleep(0)) між ними. У реальних
    банківських системах операції завжди такі ж нерозривно-неатомарні:
    повноцінна валідація, обчислення комісій, журналювання тощо.
    """
    if src.balance < amount:
        return False
    # ↓↓↓ КРИТИЧНА СЕКЦІЯ БЕЗ ЗАХИСТУ ↓↓↓
    new_src = src.balance - amount   # READ + COMPUTE
    time.sleep(0)                    # явна yield-точка → інший потік може втрутитися
    new_dst = dst.balance + amount   # READ + COMPUTE
    time.sleep(0)
    src.balance = new_src             # WRITE
    dst.balance = new_dst             # WRITE
    # ↑↑↑ КРИТИЧНА СЕКЦІЯ БЕЗ ЗАХИСТУ ↑↑↑
    return True


# =============================================================================
# 2. NAIVE з ДВОМА ЗАМКАМИ — показує DEADLOCK
# =============================================================================

def transfer_naive_locks(src: Account, dst: Account, amount: float) -> bool:
    """
    КЛАСИЧНА DEADLOCK-ПАСТКА:
      Потік A робить переказ X→Y: бере lock(X), потім намагається lock(Y).
      Потік B одночасно робить переказ Y→X: бере lock(Y), потім намагається lock(X).
      Обидва тримають по одному замку і чекають на другий → ВЗАЄМНЕ БЛОКУВАННЯ.
    """
    with src.lock:
        # Затримка збільшує ймовірність deadlock'у
        time.sleep(0.0001)
        with dst.lock:
            if src.balance < amount:
                return False
            src.balance -= amount
            dst.balance += amount
            return True


# =============================================================================
# 3. ORDERED LOCKS — рішення deadlock через глобальне впорядкування ресурсів
# =============================================================================

def transfer_ordered_locks(src: Account, dst: Account, amount: float) -> bool:
    """
    РІШЕННЯ DEADLOCK ЧЕРЕЗ ВПОРЯДКУВАННЯ РЕСУРСІВ (Resource Ordering):
      Завжди беремо замки в порядку зростання id. Тоді цикл очікування
      неможливий — потоки утворюють directed acyclic graph залежностей.
      Це класичний метод Coffman et al., 1971: усуває умову «circular wait».
    """
    first, second = (src, dst) if src.id < dst.id else (dst, src)
    with first.lock:
        with second.lock:
            if src.balance < amount:
                return False
            src.balance -= amount
            dst.balance += amount
            return True


# =============================================================================
# 4. GLOBAL LOCK — найпростіше, але серіалізує всю систему
# =============================================================================

_GLOBAL_LOCK = threading.RLock()


def transfer_global_lock(src: Account, dst: Account, amount: float) -> bool:
    """
    Один м'ютекс на всю систему: deadlock неможливий (тільки один lock),
    race condition неможливий (взаємне виключення). Але throughput падає
    до рівня послідовного виконання.
    """
    with _GLOBAL_LOCK:
        if src.balance < amount:
            return False
        src.balance -= amount
        dst.balance += amount
        return True


# =============================================================================
# 5. LOCK-FREE через try_lock + retry — оптимістичний підхід
# =============================================================================

def transfer_try_lock(src: Account, dst: Account, amount: float, max_retries: int = 100) -> bool:
    """
    ОПТИМІСТИЧНИЙ ПІДХІД:
      Намагаємося захопити обидва замки через acquire(blocking=False).
      Якщо не вдалося — відпускаємо те, що тримали, і повторюємо. Deadlock
      неможливий, бо потік ніколи не блокується «всередині» транзакції.
      Проте при високій конкуренції можливі livelock та starvation.
    """
    for attempt in range(max_retries):
        if src.lock.acquire(blocking=False):
            try:
                if dst.lock.acquire(blocking=False):
                    try:
                        if src.balance < amount:
                            return False
                        src.balance -= amount
                        dst.balance += amount
                        return True
                    finally:
                        dst.lock.release()
            finally:
                src.lock.release()
        # Невелика експоненційна затримка перед retry (зменшує livelock)
        time.sleep(0.00001 * (attempt + 1))
    return False


# =============================================================================
# Обгортка-runner
# =============================================================================

def run_threads(accounts: list[Account],
                transfer_fn: Callable[[Account, Account, float], bool],
                n_threads: int = N_THREADS,
                n_transfers: int = N_TRANSFERS_PER_THREAD,
                deadlock_timeout: float = 30.0) -> tuple[float, int, int]:
    """
    Запускає n_threads потоків, кожен робить n_transfers переказів.
    Повертає (час, успішних, спроб). Якщо потоки не завершаться за
    deadlock_timeout — виявлено deadlock.
    """
    successes = [0] * n_threads
    rnd_master = random.Random(SEED + 1)
    # Заздалегідь генеруємо списки операцій, щоб усі потоки мали детерміністичні
    # послідовності, незалежні один від одного
    seeds = [rnd_master.randint(0, 2**31) for _ in range(n_threads)]

    def worker(tid: int):
        rnd = random.Random(seeds[tid])
        ok = 0
        for _ in range(n_transfers):
            i = rnd.randint(0, N_ACCOUNTS - 1)
            j = rnd.randint(0, N_ACCOUNTS - 1)
            if i == j:
                continue
            amount = float(rnd.randint(1, 50))
            if transfer_fn(accounts[i], accounts[j], amount):
                ok += 1
        successes[tid] = ok

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()

    # Чекаємо на завершення з таймаутом — якщо deadlock, переб'ємо
    deadlocked = False
    deadline = t0 + deadlock_timeout
    for t in threads:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            deadlocked = True
            break
        t.join(timeout=remaining)
        if t.is_alive():
            deadlocked = True
            break

    elapsed = time.perf_counter() - t0
    total_attempts = n_threads * n_transfers
    total_ok = sum(successes)
    if deadlocked:
        return -1.0, total_ok, total_attempts  # маркер deadlock'у
    return elapsed, total_ok, total_attempts


# =============================================================================
# Демо — запуск однієї версії
# =============================================================================

def demo_one(name: str, transfer_fn: Callable, **kwargs) -> dict:
    accounts = make_accounts()
    initial_total = total_balance(accounts)
    print(f"\n=== {name} ===")
    print(f"  Initial total: {initial_total:.2f} ({len(accounts)} accounts)")
    elapsed, ok, attempts = run_threads(accounts, transfer_fn, **kwargs)
    final_total = total_balance(accounts)
    is_deadlock = (elapsed < 0)
    discrepancy = final_total - initial_total

    print(f"  Final total:   {final_total:.2f}")
    print(f"  Discrepancy:   {discrepancy:+.2f} "
          f"({'OK ✓' if abs(discrepancy) < 0.01 else 'CORRUPTED ✗'})")
    print(f"  Successful:    {ok} / {attempts}")
    if is_deadlock:
        print(f"  DEADLOCK DETECTED (timeout)")
    else:
        print(f"  Time:          {elapsed:.2f}s")
        if elapsed > 0:
            print(f"  Throughput:    {ok / elapsed:.0f} transfers/s")

    return {
        "name": name,
        "initial_total": initial_total,
        "final_total": final_total,
        "discrepancy": discrepancy,
        "successful": ok,
        "attempts": attempts,
        "time": elapsed,
        "deadlock": is_deadlock,
    }


if __name__ == "__main__":
    import json
    from pathlib import Path

    Path("results").mkdir(exist_ok=True)
    results = []

    # 1. RACE без жодного захисту
    results.append(demo_one("naive_race (без синхронізації)", transfer_naive))

    # 2. DEADLOCK через два замки
    print("\n[!] Очікується deadlock через 30 секунд...")
    results.append(demo_one("naive_deadlock (два замки)",
                             transfer_naive_locks,
                             n_transfers=20,        # менше — швидше демонструється
                             deadlock_timeout=15.0))

    # 3. ORDERED — рішення
    results.append(demo_one("ordered_locks (рішення deadlock)", transfer_ordered_locks))

    # 4. GLOBAL LOCK
    results.append(demo_one("global_lock (повна серіалізація)", transfer_global_lock))

    # 5. TRY-LOCK
    results.append(demo_one("try_lock (оптимістичний)", transfer_try_lock))

    Path("results/task1_demo.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False))
    print("\n✓ Saved: results/task1_demo.json")
