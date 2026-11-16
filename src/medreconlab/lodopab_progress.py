"""Transactional per-slice results for interruptible LoDoPaB runs."""
import json
import sqlite3
from collections.abc import Callable, Iterable
from contextlib import closing
from pathlib import Path


def collect_slices(
    indices: Iterable[int],
    evaluate_slice: Callable[[int], list[dict]],
    checkpoint: Path | None,
    configuration: dict,
) -> list[dict]:
    if checkpoint is None:
        return [row for index in indices for row in evaluate_slice(index)]

    with closing(sqlite3.connect(checkpoint)) as db:
        signature = json.dumps(configuration, sort_keys=True)
        with db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("CREATE TABLE IF NOT EXISTS run (id INTEGER PRIMARY KEY, settings TEXT)")
            db.execute("CREATE TABLE IF NOT EXISTS slices (id INTEGER PRIMARY KEY, metrics TEXT)")
            saved = db.execute("SELECT settings FROM run WHERE id = 1").fetchone()
            if saved is None:
                db.execute("INSERT INTO run VALUES (1, ?)", (signature,))
            elif saved[0] != signature:
                raise ValueError("Checkpoint settings differ. Use the original settings or a new output directory.")

        rows = []
        for index in indices:
            saved = db.execute("SELECT metrics FROM slices WHERE id = ?", (index,)).fetchone()
            if saved is None:
                result = evaluate_slice(index)
                with db:
                    db.execute("INSERT OR IGNORE INTO slices VALUES (?, ?)",
                               (index, json.dumps(result)))
                saved = db.execute("SELECT metrics FROM slices WHERE id = ?", (index,)).fetchone()
            rows.extend(json.loads(saved[0]))
        return rows
