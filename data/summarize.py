"""Print domain and overall percentages from the aggregate evaluation counts."""
import csv
import sys
from pathlib import Path
from statistics import mean


DOMAINS = ("PatternNet", "RSICD", "NWPU45", "MLRSNet")


def harmonic(base, new):
    return 2 * base * new / (base + new) if base + new else 0.0


def summaries(path):
    scores = {}
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            task, source, target, subset = (row[k] for k in ("task", "source", "target", "subset"))
            correct, total = int(row["correct"]), int(row["total"])
            key = (task, target, subset)
            if row["method"] != "BRTP":
                raise ValueError("Expected a BRTP aggregate count")
            if key in scores or not 0 <= correct <= total or total <= 0:
                raise ValueError(f"Invalid counts or duplicate row: {row}")
            if source != (target if task == "B2N" else "PatternNet"):
                raise ValueError(f"Unexpected source: {row}")
            scores[key] = 100 * correct / total
    expected = {("B2N", d, subset) for d in DOMAINS for subset in ("base", "new")}
    expected.update((task, d, "all") for task in ("CD", "DG") for d in DOMAINS)
    if set(scores) != expected:
        raise ValueError("Unexpected BRTP task/domain combinations")
    base = [scores["B2N", d, "base"] for d in DOMAINS]
    new = [scores["B2N", d, "new"] for d in DOMAINS]
    for domain, b, n in zip((*DOMAINS, "Overall"), (*base, mean(base)), (*new, mean(new))):
        for metric, value in (("Base", b), ("New", n), ("H", harmonic(b, n))):
            yield "BRTP", "B2N", domain, metric, value
    for task in ("CD", "DG"):
        values = [scores[task, d, "all"] for d in DOMAINS]
        for domain, value in zip((*DOMAINS, "Target mean"), (*values, mean(values[1:]))):
            yield "BRTP", task, domain, "Accuracy", value


if __name__ == "__main__":
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(("method", "task", "domain", "metric", "percent"))
    for *fields, value in summaries(Path(__file__).with_name("aggregate_counts.csv")):
        writer.writerow((*fields, f"{value:.2f}"))
