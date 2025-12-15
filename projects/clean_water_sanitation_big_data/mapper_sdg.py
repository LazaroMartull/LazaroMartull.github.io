#!/usr/bin/env python3
import sys
import csv

header = True
for line in sys.stdin:
    try:
        row = next(csv.reader([line]))
        if header:
            if "region_sdg" in row and "wat_basal_t" in row:
                region_idx = row.index("region_sdg")
                value_idx = row.index("wat_basal_t")
                header = False
                continue
        # Extract columns
        region = row[region_idx].strip()
        value = row[value_idx].strip()

        # Skip blanks or missing values
        if value == "" or value.lower() in ["na", "null"]:
            continue

        # Validate number
        try:
            float(value)
        except:
            continue

        print(f"{region}\t{value}")
    except:
        continue
