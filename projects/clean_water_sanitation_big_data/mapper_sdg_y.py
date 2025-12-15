#!/usr/bin/env python3
import sys
import csv

header = True
for line in sys.stdin:
    try:
        row = next(csv.reader([line]))
        
        if header:
            if "region_sdg" in row and "year" in row and "wat_basal_t" in row:
                region_idx = row.index("region_sdg")
                year_idx = row.index("year")
                value_idx = row.index("wat_basal_t")
                header = False
                continue

        region = row[region_idx].strip()
        year = row[year_idx].strip()
        value = row[value_idx].strip()

        # Skip blanks or NA-type values
        if value == "" or value.lower() in ["na", "null"]:
            continue

        try:
            float(value)
        except:
            continue

        print(f"{region},{year}\t{value}")

    except:
        continue
