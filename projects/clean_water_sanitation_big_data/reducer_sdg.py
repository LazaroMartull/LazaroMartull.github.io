#!/usr/bin/env python3
import sys

current_region = None
total = 0.0
count = 0
header_printed = False

for line in sys.stdin:
    region, value = line.strip().split("\t")

    try:
        value = float(value)
    except:
        continue

    if not header_printed:
        print("region_sdg,average_wat_basal_t")
        header_printed = True

    if current_region is None:
        current_region = region

    if region != current_region:
        avg = total / count
        print(f"{current_region},{avg}")
        current_region = region
        total = 0.0
        count = 0

    total += value
    count += 1

# Flush the final region
if current_region is not None and count > 0:
    avg = total / count
    print(f"{current_region},{avg}")
