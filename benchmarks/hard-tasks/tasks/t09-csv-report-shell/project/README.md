# Order export report

`data/exports/` holds order exports from several systems. Every `*.csv` file in that
directory is one export. `scripts/build_report.sh` reads all of them and writes
`out/summary.csv` (create `out/` if needed). Run it from the project root:

    bash scripts/build_report.sh

It must exit 0 and overwrite `out/summary.csv`.

## Input files

Exports come in three layouts. Detect the layout from each file's header line, never from its file name.

- Delimiter: `;` if the header line contains `;`, otherwise `,`.
- Header names: match case-insensitively after trimming whitespace. Column order varies between files.
  Required columns: `order_id`, `region`, `amount`, `order_date`, `updated_at`. Some exports call
  `order_date` just `Date`.
- Amounts: decimal point in `,`-delimited files; decimal comma in `;`-delimited files (`12,50` means 12.50).
  At most two decimal places. May be negative (refunds). May be empty.
- Dates: `order_date` is `YYYY-MM-DD` in `,`-delimited files and `DD/MM/YYYY` in `;`-delimited files.
  `updated_at` is ISO `YYYY-MM-DDTHH:MM:SS` in every file.
- Some files start with a UTF-8 byte-order mark and/or use CRLF line endings.
- Fields may be double-quoted, using normal CSV rules (`"North, East"`).

## Rules

Apply these steps in order:

1. Read every `*.csv` in `data/exports/`. Trim surrounding whitespace from every field.
2. Drop rows whose amount is empty. Dropped rows take no part in step 3.
3. De-duplicate by `order_id` across all files. Keep only the row with the latest `updated_at`
   (ISO strings compare lexicographically). If `updated_at` is equal, keep the row from the file whose
   name sorts last (plain byte order). Afterwards each order_id appears once.
4. Region: trim whitespace and lowercase. Change nothing else (do not collapse inner spaces).
5. Group the kept rows by (region, month), where month is the `YYYY-MM` of the kept row's order date.
   - `orders` = number of kept rows in the group (one per distinct order).
   - `total` = sum of the kept amounts in exact decimal arithmetic (no floats), printed with exactly
     2 decimals and no thousands separator: `-3.50`, `0.00`, `1234.00`.

## Output: out/summary.csv

- Header: `region,month,orders,total`
- One row per (region, month), sorted by region, then by month (plain string order).
- Standard CSV quoting: a region containing a comma is quoted, e.g. `"north, east"`.
- LF line endings and a trailing newline.

## Example

Illustration only; this file is not in `data/exports/`. Input `example.csv`:

```
order_id,region,amount,order_date,updated_at
1,North,10.00,2025-12-31,2026-01-02T09:00:00
1,north ,12.50,2025-12-31,2026-01-03T08:00:00
2,South,,2026-01-04,2026-01-04T10:00:00
3,South,-3.50,2026-01-05,2026-01-05T10:00:00
4,"North, East",5.00,2026-01-06,2026-01-06T11:00:00
```

Output `out/summary.csv`:

```
region,month,orders,total
north,2025-12,1,12.50
"north, east",2026-01,1,5.00
south,2026-01,1,-3.50
```
