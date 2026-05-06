"""Збирає JSON-результати всіх задач в один файл."""
import json, time
from pathlib import Path

results_dir = Path("results")
combined = {
    "metadata": {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "worker_counts": [1, 2, 4, 8],
        "io_latency_ms_task1": 15,
        "io_latency_ms_task2": 10,
        "notes": ("Симульована I/O-затримка додана до атомарних операцій для "
                  "моделювання реалістичного I/O-bound навантаження. У контейнерному "
                  "середовищі CPU-паралелізм процесів обмежений; ThreadPoolExecutor "
                  "з time.sleep дає чесне прискорення під час I/O."),
    },
    "results": []
}
for name in ["html", "array", "matrix", "images"]:
    path = results_dir / f"{name}.json"
    if path.exists():
        combined["results"].append(json.loads(path.read_text()))

(results_dir / "benchmark.json").write_text(json.dumps(combined, indent=2, ensure_ascii=False))
print("Combined:")
for r in combined["results"]:
    print(f"  - {r['task']}")
