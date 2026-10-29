"""Experiment tracking from the first run (spec §13.2). JSONL by default; wandb/mlflow optional."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class Tracker:
    def __init__(self, backend: str, run_dir: str | Path, project: str = "sbim-cv-layer",
                 config: dict | None = None, name: str | None = None):
        self.backend = backend
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.run_dir / "metrics.jsonl", "a", encoding="utf-8")
        self._ext = None
        if backend == "wandb":
            import wandb
            self._ext = wandb.init(project=project, config=config, name=name, dir=str(self.run_dir))
        elif backend == "mlflow":
            import mlflow
            mlflow.set_experiment(project)
            self._ext = mlflow.start_run(run_name=name)
            mlflow.log_dict(config or {}, "config.json")
        elif backend != "jsonl":
            raise ValueError(f"unknown tracking backend {backend!r}")
        if config is not None:
            (self.run_dir / "config.json").write_text(json.dumps(config, indent=2, default=str))

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        rec = {"t": time.time(), "step": step, **metrics}
        self._fh.write(json.dumps(rec, default=str) + "\n")
        self._fh.flush()
        if self.backend == "wandb":
            import wandb
            wandb.log(metrics, step=step)
        elif self.backend == "mlflow":
            import mlflow
            mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))}, step=step or 0)

    def close(self) -> None:
        self._fh.close()
        if self.backend == "wandb":
            import wandb
            wandb.finish()
        elif self.backend == "mlflow":
            import mlflow
            mlflow.end_run()
