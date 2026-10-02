"""
split_dataset.py

Splits the synthetic dataset into train/val sets by DAY, not by row.
This matters: if you split randomly by row, the LSTM can effectively see
"the future" of the same day's bunching pattern during training and get
artificially inflated validation accuracy. Splitting by day forces the
model to generalize across different traffic/passenger realizations.

Usage:
    python split_dataset.py --in ../data/synthetic_route.csv \
        --train-out ../data/train.csv --val-out ../data/val.csv --val-frac 0.2
"""

import argparse
import csv
from collections import defaultdict


def split_by_day(in_path, train_out, val_out, val_frac=0.2, seed=42):
    rows_by_day = defaultdict(list)
    with open(in_path) as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            rows_by_day[int(row["day"])].append(row)

    days = sorted(rows_by_day.keys())
    n_val_days = max(1, int(len(days) * val_frac))

    # Use the LAST n_val_days as validation (chronological holdout) --
    # more realistic than a random day split, since in deployment you're
    # always predicting forward in time from what you've trained on.
    train_days = days[:-n_val_days]
    val_days = days[-n_val_days:]

    def write(path, day_list):
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for d in day_list:
                writer.writerows(rows_by_day[d])

    write(train_out, train_days)
    write(val_out, val_days)

    n_train_rows = sum(len(rows_by_day[d]) for d in train_days)
    n_val_rows = sum(len(rows_by_day[d]) for d in val_days)

    print(f"Train: days {train_days[0]}-{train_days[-1]} ({len(train_days)} days, {n_train_rows} rows) -> {train_out}")
    print(f"Val:   days {val_days[0]}-{val_days[-1]} ({len(val_days)} days, {n_val_rows} rows) -> {val_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split synthetic dataset by day into train/val")
    parser.add_argument("--in", dest="in_path", type=str, default="../data/synthetic_route.csv")
    parser.add_argument("--train-out", type=str, default="../data/train.csv")
    parser.add_argument("--val-out", type=str, default="../data/val.csv")
    parser.add_argument("--val-frac", type=float, default=0.2)
    args = parser.parse_args()

    split_by_day(args.in_path, args.train_out, args.val_out, args.val_frac)
