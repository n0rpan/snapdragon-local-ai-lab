#!/usr/bin/env python3
"""REAL-1 tuning recommendations from an organic_report.py JSON (numbers only, offline).

Inputs: the organic report (--report, JSON from `organic_report.py --json`; refused unless status is ok)
and the AG-2 fit constants (LONGCTX-ZERO-H sec. 2). Outputs compaction points for fast and deep,
the --mtp-max-ctx decision, and every lever worth >= 3% of session wall. Each figure is tagged [M]
(from the organic report or a measured fit) or [E] (a model estimate or an assumed constant), and the
inputs are printed first so the reader can see what was used.

The session model is a paper model, not a measurement: decode ms per token at context c is
(verify_ms + us_per_ctx * c / 1000) / (1 + acceptance); prefill ms per token is base + 0.224 * depth_K.
Nothing here runs a model.
"""
import argparse
import json
import sys

REPORT_SCHEMA = 'organic-report/1'  # must match organic_report.SCHEMA
CAPS_K = (24, 32, 40, 48, 64, 96, 128, 256)
FAST_CAPS_K = (24, 32, 40, 48, 64)    # fast = short compaction point
DEEP_CAPS_K = (64, 96, 128, 256)      # deep = long compaction point
CURRENT_K = {'fast': 48, 'deep': 100}  # [E] current plan values (V013-PLAN sec. 3)
PREFIX_TOK = 5500      # [E] static prefix (LONGCTX-ZERO-H sec. 1)
SUMMARY_TOK = 2000     # [E] compaction summary decoded at full context (sec. 2)
TAIL_TOK = 4000        # [E] tail re-prefilled after the cut (sec. 2)
APPEND_TOK = 2000      # [E] per-turn append, REAL-0 mean; replaced by the organic p50 when present [M]
OUTPUT_TOK = 600       # [E] per-turn output, REAL-0 mean; replaced by the organic p50 when present [M]
PREFILL_PER_DEPTH_K = 0.224  # [E] ms per token per K of depth (sec. 2)
LEVER_MIN = 0.03
BAND_UPPER_K = {'<8K': 8, '8K-32K': 32, '32K-64K': 64, '64K-128K': 128, '128K-256K': 256}
# AG-2 per-verify fits (sec. 2): ms + us per context token; acceptance from AG-2 (Ornith measured, Qwen estimated)
AG2 = {
    'ornith': {'verify_ms': 29.3, 'us_per_ctx': 0.600, 'accept': 0.61, 'prefill_base_ms': 3.0,
               'fit_tag': 'M to 32K, D beyond', 'accept_tag': 'M'},
    'qwen': {'verify_ms': 41.1, 'us_per_ctx': 0.563, 'accept': 0.70, 'prefill_base_ms': 3.6,
             'fit_tag': 'M to 32K, D beyond', 'accept_tag': 'E'},
}


def decode_ms(model, ctx, plain=None):
    """ms per emitted token at context `ctx`. With plain=(a_ms, b_us) the non-MTP decode fit is used."""
    if plain is not None:
        return plain[0] + plain[1] * ctx / 1000.0
    m = AG2[model]
    return (m['verify_ms'] + m['us_per_ctx'] * ctx / 1000.0) / (1.0 + m['accept'])


def prefill_ms(model, depth_tok):
    return AG2[model]['prefill_base_ms'] + PREFILL_PER_DEPTH_K * depth_tok / 1024.0


def simulate(model, cap_k, turns, append=APPEND_TOK, output=OUTPUT_TOK):
    """Minutes of engine time for a session of `turns` turns with compaction at cap_k K. Returns (minutes, compactions)."""
    cap = cap_k * 1024
    ctx, total, comps = PREFIX_TOK, 0.0, 0
    for _ in range(turns):
        total += append * prefill_ms(model, ctx)
        ctx += append
        total += output * decode_ms(model, ctx)
        ctx += output
        if ctx > cap:
            total += SUMMARY_TOK * decode_ms(model, ctx)
            rep = SUMMARY_TOK + TAIL_TOK
            total += rep * prefill_ms(model, rep)
            ctx = PREFIX_TOK + rep
            comps += 1
    return total / 60000.0, comps


def parse_plain(values):
    """--plain-fit MODEL:A_MS:B_US_PER_CTX_TOKEN, repeatable (a measured non-MTP decode fit)."""
    fits = {}
    for raw in values or []:
        try:
            name, a, b = raw.split(':')
            fits[name] = (float(a), float(b))
        except ValueError:
            raise SystemExit(f'bad --plain-fit {raw!r}: want MODEL:A_MS:B_US')
        if name not in AG2:
            raise SystemExit(f'bad --plain-fit model {name!r}')
    return fits


def mtp_decision(report, plain_fits):
    """Recommendation for --mtp-max-ctx. Needs a measured non-MTP fit; without one it is not decided."""
    bands = report.get('mtp_acceptance_by_band') or {}
    lines = []
    if not plain_fits:
        lines.append('mtp-max-ctx: NOT DECIDED (no measured non-MTP decode fit given; AG-2 found MTP 1 wins to 32K) [E]')
        for lbl, v in bands.items():
            lines.append(f'  acceptance {lbl}: {v["acceptance"]} over {v["drafts"]} drafts [M]')
        return lines
    for name, plain in plain_fits.items():
        # MTP wins in a band when acceptance > verify(ctx) / plain(ctx) - 1. --mtp-max-ctx is the top of the
        # last band before the first band where MTP loses (bands in context order).
        last_win, first_loss = None, None
        for lbl, v in bands.items():
            upper_k = BAND_UPPER_K.get(lbl)
            if upper_k is None:
                continue
            ctx = upper_k * 1024
            verify = AG2[name]['verify_ms'] + AG2[name]['us_per_ctx'] * ctx / 1000.0
            breakeven = verify / decode_ms(name, ctx, plain) - 1.0
            lines.append(f'  {name} {lbl}: acceptance {v["acceptance"]} [M] vs break-even {breakeven:.3f} [E]')
            if v['acceptance'] > breakeven:
                if first_loss is None:
                    last_win = upper_k
            elif first_loss is None:
                first_loss = lbl
        if first_loss is None:
            verdict = 'no cutoff (MTP wins in every measured band) [E]'
        elif last_win is None:
            verdict = f'MTP loses from the first band ({first_loss}); keep MTP off [E]'
        else:
            verdict = f'{last_win}K [E]'
        lines.append(f'mtp-max-ctx {name}: {verdict}')
    return lines


def levers(report):
    shares = report.get('wall_shares') or {}
    out = []
    for name, value in shares.items():
        if value is None:
            out.append(f'{name}: not measured')
            continue
        out.append(f'{name}: {value} of wall [M] -> worth >= 3%: {"yes" if value >= LEVER_MIN else "no"}')
    out.append('compaction re-prefill: not measurable in the ledger yet (no turn_kind rows in REAL-1 traffic) [E]')
    return out


def recommend(report, plain_fits):
    append = report['new_tokens_per_turn']['p50'] or APPEND_TOK
    output = report['output_tokens_per_turn']['p50'] or OUTPUT_TOK
    tag = '[M]' if report['new_tokens_per_turn']['p50'] else '[E]'
    lines = [f'inputs: sessions {report["sessions"]} [M], append/turn {append} {tag}, output/turn {output} '
             f'{"[M]" if report["output_tokens_per_turn"]["p50"] else "[E]"}, prefix {PREFIX_TOK} [E]']
    for model in AG2:
        lines.append(f'{model}: verify {AG2[model]["verify_ms"]} ms + {AG2[model]["us_per_ctx"]} us/ctx '
                     f'({AG2[model]["fit_tag"]}), acceptance {AG2[model]["accept"]} [{AG2[model]["accept_tag"]}]')
    for turns in (50, 200):
        lines.append(f'session {turns} turns, minutes by cap [E]:')
        for model in AG2:
            cells = []
            for k in CAPS_K:
                minutes, comps = simulate(model, k, turns, append, output)
                cells.append(f'{k}K {minutes:.1f}m/{comps}c')
            lines.append(f'  {model}: ' + '  '.join(cells))
    for model in AG2:
        fast = min(FAST_CAPS_K, key=lambda k: simulate(model, k, 50, append, output)[0])
        deep = min(DEEP_CAPS_K, key=lambda k: simulate(model, k, 200, append, output)[0])
        lines.append(f'{model}: fast compaction {fast}K (current {CURRENT_K["fast"]}K) [E]; '
                     f'deep compaction {deep}K (current {CURRENT_K["deep"]}K) [E]')
    lines.extend(mtp_decision(report, plain_fits))
    lines.append('levers:')
    lines.extend('  ' + x for x in levers(report))
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--report', required=True, help='organic_report.py --json output')
    ap.add_argument('--plain-fit', action='append', metavar='MODEL:A_MS:B_US',
                    help='measured non-MTP decode fit (repeatable); without it --mtp-max-ctx is not decided')
    args = ap.parse_args(argv)
    with open(args.report, encoding='utf-8') as stream:
        report = json.load(stream)
    if report.get('schema') != REPORT_SCHEMA or report.get('status') != 'ok':
        print('refusing: report is not an ok organic report', file=sys.stderr)
        return 2
    print('\n'.join(recommend(report, parse_plain(args.plain_fit))))
    return 0


if __name__ == '__main__':
    sys.exit(main())
