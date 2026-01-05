#!/usr/bin/env python3
from collections import Counter

INPUT_FILE = "perf-symbols.txt"
TOP_N = 20

def main():
    counter = Counter()
    total = 0
    with open(INPUT_FILE, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) >= 6:
                sym = parts[4]
                if sym == "[unknown]" and len(parts) > 5:
                    sym = parts[5]
                counter[sym] += 1
                total += 1

    if total == 0:
        print(f"no samples found in {INPUT_FILE}")
        return

    print(f"Top {TOP_N} symbols ({total} samples):")
    for sym, cnt in counter.most_common(TOP_N):
        pct = cnt / total * 100.0
        print(f"{cnt:7d} samples ({pct:5.2f}%) {sym}")


if __name__ == "__main__":
    main()
