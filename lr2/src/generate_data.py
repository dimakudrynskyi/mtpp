"""Генерація тестових даних для Lab 2."""
from __future__ import annotations
import random
import string
from pathlib import Path
import numpy as np
from PIL import Image


# ---------- HTML-документи ----------

HTML_TAGS = ["div", "span", "p", "a", "img", "h1", "h2", "h3", "ul", "li",
             "table", "tr", "td", "th", "button", "input", "form", "section",
             "article", "header", "footer", "nav", "main", "aside", "code"]


def random_text(rnd: random.Random, n_words: int = 10) -> str:
    return " ".join(
        "".join(rnd.choices(string.ascii_lowercase, k=rnd.randint(3, 8)))
        for _ in range(n_words)
    )


def random_html(rnd: random.Random, n_tags: int) -> str:
    parts = ["<!DOCTYPE html>", "<html><head><title>Test</title></head><body>"]
    for _ in range(n_tags):
        tag = rnd.choice(HTML_TAGS)
        if tag in ("img", "input"):
            parts.append(f"<{tag} />")
        else:
            parts.append(f"<{tag}>{random_text(rnd, rnd.randint(2, 8))}</{tag}>")
    parts.append("</body></html>")
    return "\n".join(parts)


def generate_html_files(root: Path, n_files: int = 400, seed: int = 42):
    """Створює n_files великих HTML-файлів. BeautifulSoup-парсинг достатньо
    важкий, щоб 400 файлів давали ~30с sequential."""
    rnd = random.Random(seed)
    root.mkdir(parents=True, exist_ok=True)
    for i in range(n_files):
        if rnd.random() < 0.2:
            n_tags = rnd.randint(5000, 8000)
        else:
            n_tags = rnd.randint(1500, 3000)
        html = random_html(rnd, n_tags)
        (root / f"page_{i:04d}.html").write_text(html, encoding="utf-8")
    print(f"  HTML: {n_files} файлів у {root}")


# ---------- Великий масив чисел ----------

def generate_array(path: Path, n: int = 5_000_000, seed: int = 42):
    """Зберігає масив у .npy. Розподіл — суміш нормального та uniform."""
    rng = np.random.default_rng(seed)
    half = n // 2
    a = np.concatenate([
        rng.normal(loc=100, scale=30, size=half),
        rng.uniform(low=0, high=200, size=n - half),
    ]).astype(np.float64)
    rng.shuffle(a)
    np.save(path, a)
    print(f"  Array: {n:,} чисел у {path} (mean={a.mean():.2f})")


# ---------- Матриці для множення ----------

def generate_matrices(path_a: Path, path_b: Path, n: int = 1024, seed: int = 42):
    """Зберігає дві NxN-матриці у .npy."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n)).astype(np.float64)
    B = rng.standard_normal((n, n)).astype(np.float64)
    np.save(path_a, A)
    np.save(path_b, B)
    print(f"  Matrices: 2 × {n}×{n} ({A.nbytes / 1024**2:.1f} MB кожна)")


# ---------- Зображення для Pipeline ----------

def generate_images(root: Path, n: int = 60, size: int = 800, seed: int = 42):
    """Створює n кольорових JPEG-зображень розміру size×size."""
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        # Випадкові гладкі градієнти + шум — щоб JPEG не стискав до нічого
        base = rng.integers(0, 256, size=(size // 8, size // 8, 3), dtype=np.uint8)
        img = Image.fromarray(base, mode="RGB").resize((size, size), Image.BILINEAR)
        # Додамо трохи шуму
        arr = np.asarray(img, dtype=np.int16)
        noise = rng.integers(-15, 16, size=arr.shape, dtype=np.int16)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        Image.fromarray(arr).save(root / f"img_{i:03d}.jpg", "JPEG", quality=85)
    print(f"  Images: {n} зображень {size}×{size} у {root}")


if __name__ == "__main__":
    base = Path("test_data")
    generate_html_files(base / "html", n_files=1500)
    generate_array(base / "array.npy", n=5_000_000)
    generate_matrices(base / "matA.npy", base / "matB.npy", n=1024)
    generate_images(base / "images", n=60, size=800)
    print("\n✓ Усі тестові дані згенеровано")
