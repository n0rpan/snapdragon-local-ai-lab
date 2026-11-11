#!/usr/bin/env bash
# Builds out/summary.csv from data/exports/*.csv (see README.md).
set -eu
mkdir -p out
{
  echo "region,month,orders,total"
  awk -F, 'FNR > 1 && $3 != "" {
    k = $2 "," substr($4, 1, 7); n[k]++; s[k] += $3
  } END {
    for (k in n) printf "%s,%d,%.2f\n", k, n[k], s[k]
  }' data/exports/*.csv | sort
} > out/summary.csv
