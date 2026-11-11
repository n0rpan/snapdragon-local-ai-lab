#!/usr/bin/env bash
# Builds out/summary.csv from every data/exports/*.csv file (rules in README.md).
set -eu
cd "$(dirname "$0")/.."
mkdir -p out
python3 -I - <<'PY'
import csv
import glob
import io
import os
from decimal import Decimal


def load(path):
    with open(path, encoding='utf-8-sig', newline='') as fh:
        text = fh.read()
    header_line = text.split('\n', 1)[0].rstrip('\r')
    delim = ';' if ';' in header_line else ','
    comma_decimal = delim == ';'
    rows = [r for r in csv.reader(io.StringIO(text, newline=''), delimiter=delim)
            if any(c.strip() for c in r)]
    col = {}
    for i, name in enumerate(rows[0]):
        key = name.strip().lower()
        col['order_date' if key == 'date' else key] = i
    out = []
    for raw in rows[1:]:
        def get(name):
            return raw[col[name]].strip()
        amount = get('amount')
        if amount == '':
            continue
        if comma_decimal:
            amount = amount.replace(',', '.')
        date = get('order_date')
        if comma_decimal:
            day, month, year = date.split('/')
            date = f'{year}-{month}-{day}'
        out.append(dict(order_id=get('order_id'), region=get('region').lower(),
                        month=date[:7], amount=Decimal(amount),
                        updated_at=get('updated_at'), file=os.path.basename(path)))
    return out


best = {}
for path in sorted(glob.glob('data/exports/*.csv')):
    for rec in load(path):
        key = (rec['updated_at'], rec['file'])
        cur = best.get(rec['order_id'])
        if cur is None or key > cur[0]:
            best[rec['order_id']] = (key, rec)

groups = {}
for _, rec in best.values():
    g = groups.setdefault((rec['region'], rec['month']), [0, Decimal('0')])
    g[0] += 1
    g[1] += rec['amount']

with open('out/summary.csv', 'w', encoding='utf-8', newline='') as fh:
    w = csv.writer(fh, lineterminator='\n')
    w.writerow(['region', 'month', 'orders', 'total'])
    for region, month in sorted(groups):
        n, total = groups[(region, month)]
        w.writerow([region, month, n, str(total.quantize(Decimal('0.01')))])
PY
