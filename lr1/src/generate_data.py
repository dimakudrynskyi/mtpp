"""Генерація випадкових текстових файлів і вкладеної структури директорій."""
from __future__ import annotations
import random
import string
from pathlib import Path


def random_word(rnd: random.Random, min_len: int = 3, max_len: int = 10) -> str:
    n = rnd.randint(min_len, max_len)
    return "".join(rnd.choices(string.ascii_lowercase, k=n))


def random_text(rnd: random.Random, n_words: int) -> str:
    words = [random_word(rnd) for _ in range(n_words)]
    out = []
    i = 0
    while i < len(words):
        k = rnd.randint(8, 15)
        sentence = " ".join(words[i:i + k]).capitalize() + "."
        out.append(sentence)
        i += k
    return " ".join(out)


def generate(root: Path, n_files: int = 1000, depth: int = 3, seed: int = 42) -> int:
    rnd = random.Random(seed)
    root.mkdir(parents=True, exist_ok=True)

    dirs = [root]
    for _ in range(depth):
        new = []
        for d in dirs:
            for j in range(rnd.randint(2, 4)):
                sub = d / f"dir_{j}"
                sub.mkdir(exist_ok=True)
                new.append(sub)
        dirs.extend(new)

    total_words = 0
    for i in range(n_files):
        d = rnd.choice(dirs)
        n_words = rnd.randint(50, 500)
        text = random_text(rnd, n_words)
        total_words += len(text.split())
        with open(d / f"file_{i:04d}.txt", "w", encoding="utf-8") as f:
            f.write(text)
    return total_words


if __name__ == "__main__":
    import sys
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("test_data")
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    total = generate(root, n_files=n)
    print(f"Згенеровано {n} файлів у {root}, всього слів: {total}")
