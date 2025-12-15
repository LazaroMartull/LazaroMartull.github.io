#!/usr/bin/env python3
import sys

current_key = None
total = 0.0
count = 0
header_printed = False

for line in sys.stdin:
    key, value = line.strip().split("\t")
    
    try:
        value = float(value)
    except:
        continue

    if not header_printed:
        print("region_sdg,year,avg_wat_basal_t")
        header_printed = True

    if current_key is None:
        current_key = key

    if key != current_key:
        region, year = current_key.split(",")
        avg = total / count
        print(f"{region},{year},{avg}")
        current_key = key
        total = 0.0
        count = 0

    total += value
    count += 1

# Final flush
if current_key is not None and count > 0:
    region, year = current_key.split(",")
    avg = total / count
    print(f"{region},{year},{avg}")
