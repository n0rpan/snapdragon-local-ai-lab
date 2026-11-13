"""O-27: the user's Flash cache preference (auto / 20 / 18 / 16 / 14 GiB) for the oxe-server tier decision.

The native engine decides the tier before loading. Explicit 20/18 preferences first use a MemTotal
working-set budget with a desktop reserve (default 3 GiB, configurable per profile). If that does not fit,
they fall back to the conservative MemAvailable rule, as auto/16/14 always do. No fitting tier means exit 74.
Without an explicit environment/profile preference, the controller keeps the installed fixed-cache path.
This module is pure and model-free:

- read_pref(env, flash_profile): the preference; env LOCAL_AI_FLASH_CACHE_PREF overrides the profile field
  oxe_flash.cache_pref; default 'auto'. The controller only applies this when enabled() is true.
  Enabled profiles expose GET/POST /_local/flash/cache and local-ai flash-cache --pref 20; changing the
  preference unloads idle Flash before the next load, without changing its desktop reserve or memory guards.
- apply_pref(args, pref): the oxe argv with `--flash-cache-gib auto` and, unless auto, `--flash-cache-pref N`.
- smallest_tier_need_bytes(): what the controller's own pre-launch admission must require when the engine picks the
  tier (the 14 GiB tier's need; the engine refuses below it anyway).
- status_from_props(props): the `/props` engine_status.flash_cache object -> a small dict for the status endpoint / UI
  ('Flash cache: 16 GiB (preferred 20 does not fit)').

Needs mirror crates/qwen4exp/src/pf_plan.rs (cache + dense/base 4.825 + model growth 3.50 + harness growth 0.80
(F118) + floor 1.0 GiB at ctx <= 16384); the tests pin both copies to the same numbers.
"""

import math

ENV = 'LOCAL_AI_FLASH_CACHE_PREF'
PREFS = ('auto', '20', '18', '16', '14')
TIERS_GIB = (20, 18, 16, 14)
GIB = 1024 ** 3
DENSE_BASE_GIB = 4.825
MODEL_GROWTH_GIB = 3.50
HARNESS_GROWTH_GIB = 0.80
FLOOR_GIB = 1.0
# O-27 rev: an explicit 20 / 18 preference uses the swap-tolerant rule (engine --flash-desktop-floor-gib, default 3.0)
SWAP_TOLERANT_TIERS_GIB = (20, 18)
SWAP_TOLERANT_TIERS_GIB_STR = ('20', '18')
DESKTOP_FLOOR_GIB = 3.0
# S2 (v0.1.1): the MemTotal rule assumes idle desktop pages can go to swap. Without swap they cannot, and the
# kernel OOM killer picks desktop victims instead. The lab side therefore admits 20/18 on MemTotal only when
# SwapFree covers what must leave RAM (the working set minus MemAvailable, plus the 1 GiB host floor) plus
# this margin, and does not offer 20/18 at all on a host without swap. The engine rule (pf_plan.rs) gets the
# same check in the engine hotfix; until then the controller never passes --flash-cache-pref 20/18 to an
# engine whose mirror decision is not swap-tolerant.
SWAP_MARGIN_GIB = 1.0
REASON_TEXT = {
    'auto': 'largest tier that fits',
    'preferred': 'your preference',
    'preferred-does-not-fit': 'preferred {pref} GiB does not fit now',
    'preferred-swap-tolerant': 'your preference; idle desktop pages may go to swap',
    'refuse': 'not even 14 GiB fits now',
}


def _norm(value):
    if value is None:
        return None
    v = str(value).strip().lower()
    for suffix in ('gib', 'gb', 'g'):
        if v.endswith(suffix) and v != suffix:
            v = v[:-len(suffix)].strip()
            break
    if v not in PREFS:
        raise ValueError('flash_cache_pref_invalid')
    return v


def read_pref(env=None, flash_profile=None):
    """'auto' | '20' | '18' | '16' | '14'. The environment wins over the profile; an invalid value raises
    ValueError('flash_cache_pref_invalid') instead of silently loading another size."""
    env = env if env is not None else {}
    if env.get(ENV, '').strip():
        return _norm(env[ENV])
    if isinstance(flash_profile, dict) and flash_profile.get('cache_pref') is not None:
        return _norm(flash_profile['cache_pref'])
    return 'auto'


def enabled(env=None, flash_profile=None):
    """Absent preference fields preserve the installed fixed-cache path."""
    return bool((env or {}).get(ENV, '').strip()) or (
        isinstance(flash_profile, dict) and flash_profile.get('cache_pref') is not None)


def desktop_floor(flash_profile):
    value = flash_profile.get('desktop_floor_gib', DESKTOP_FLOOR_GIB)
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 16:
        raise ValueError('flash_desktop_floor_invalid')
    return float(value)


def apply_pref(args, pref):
    """A copy of the oxe argv: every `--flash-cache-gib X` becomes `--flash-cache-gib auto` (added when missing), any
    old `--flash-cache-pref` is dropped, and `--flash-cache-pref N` is appended unless pref is 'auto'."""
    pref = _norm(pref)
    out, i, seen = [], 0, False
    while i < len(args):
        a = args[i]
        if a in ('--flash-cache-gib', '--flash-cache') and i + 1 < len(args):
            if not seen:
                out += ['--flash-cache-gib', 'auto']
                seen = True
            i += 2
            continue
        if a == '--flash-cache-pref' and i + 1 < len(args):
            i += 2
            continue
        out.append(a)
        i += 1
    if not seen:
        out += ['--flash-cache-gib', 'auto']
    if pref != 'auto':
        out += ['--flash-cache-pref', pref]
    return out


def tier_need_gib(cache_gib, n_ctx=16384):
    kv_extra = max(0, n_ctx - 16384) * 12 * (512 + 512 + 256) * 2 / GIB
    return cache_gib + DENSE_BASE_GIB + kv_extra + MODEL_GROWTH_GIB + HARNESS_GROWTH_GIB + FLOOR_GIB


def smallest_tier_need_bytes(n_ctx=16384):
    """Pre-launch admission when the engine chooses the tier: the 14 GiB tier's need, in bytes (rounded up)."""
    return int(-(-tier_need_gib(TIERS_GIB[-1], n_ctx) * GIB // 1))


def swap_tolerant_budget_gib(mem_total_bytes, cache_gib, desktop_floor_gib=DESKTOP_FLOOR_GIB, n_ctx=16384):
    kv_extra = max(0, n_ctx - 16384) * 12 * (512 + 512 + 256) * 2 / GIB
    return mem_total_bytes / GIB - (cache_gib + DENSE_BASE_GIB + kv_extra + MODEL_GROWTH_GIB) - desktop_floor_gib


def swap_needed_bytes(available_bytes, cache_gib, n_ctx=16384):
    """Bytes that must leave RAM (to swap) for this tier: working set + host floor - MemAvailable, >= 0."""
    working_set = tier_need_gib(cache_gib, n_ctx) - HARNESS_GROWTH_GIB  # includes FLOOR_GIB
    return max(0, int(working_set * GIB) - int(available_bytes))


def swap_covers(available_bytes, swap_free_bytes, cache_gib, n_ctx=16384):
    """S2: SwapFree must hold what has to leave RAM, plus SWAP_MARGIN_GIB. Unknown swap = no swap."""
    if type(swap_free_bytes) is not int or swap_free_bytes <= 0 or type(available_bytes) is not int:
        return False
    return swap_free_bytes >= swap_needed_bytes(available_bytes, cache_gib, n_ctx) + int(SWAP_MARGIN_GIB * GIB)


def offered_prefs(swap_total_bytes):
    """The preferences the controller offers on this host: no 20/18 without swap (S2)."""
    if type(swap_total_bytes) is int and swap_total_bytes > 0:
        return list(PREFS)
    return [p for p in PREFS if p not in ('20', '18')]


def expected_tier(available_bytes, pref='auto', n_ctx=16384, mem_total_bytes=None, desktop_floor_gib=DESKTOP_FLOOR_GIB,
                  swap_free_bytes=None):
    """The engine's rule, for the UI's 'would load as' hint and the controller's launch mirror. Stricter than a
    v0.1.0 engine: the MemTotal (swap-tolerant) branch also needs swap_covers() (S2); without swap_free_bytes
    it is never taken."""
    pref = _norm(pref)
    if pref in ('20', '18') and mem_total_bytes is not None:
        for t in (x for x in SWAP_TOLERANT_TIERS_GIB if x <= int(pref)):
            if (swap_tolerant_budget_gib(mem_total_bytes, t, desktop_floor_gib, n_ctx) >= 0
                    and swap_covers(available_bytes, swap_free_bytes, t, n_ctx)):
                return t, 'preferred-swap-tolerant'
    avail = available_bytes / GIB
    fits = [t for t in TIERS_GIB if tier_need_gib(t, n_ctx) <= avail]
    if not fits:
        return None, 'refuse'
    if pref != 'auto':
        p = int(pref)
        return (p, 'preferred') if p in fits else (fits[0], 'preferred-does-not-fit')
    return fits[0], 'auto'


def status_from_props(props):
    """The controller status field from oxe-server's /props, or None (older engine / not Flash)."""
    if not isinstance(props, dict):
        return None
    es = props.get('engine_status')
    fc = es.get('flash_cache') if isinstance(es, dict) else None
    if not isinstance(fc, dict) or fc.get('reason') not in REASON_TEXT:
        return None
    tier, pref = fc.get('cache_gib'), fc.get('pref_gib')
    if tier is not None and (type(tier) not in (int, float) or tier not in TIERS_GIB):
        return None
    tier = int(tier) if tier is not None else None
    if pref is not None and (type(pref) not in (int, float) or pref not in TIERS_GIB):
        return None
    pref_s = 'auto' if pref is None else str(int(pref))
    reason = fc['reason']
    text = REASON_TEXT[reason].format(pref=pref_s)
    label = f'{tier} GiB ({text})' if tier is not None else f'not loaded ({text})'
    margin = fc.get('margin_gib')
    return {
        'tier_gib': tier,
        'pref': pref_s,
        'reason': reason,
        'margin_gib': round(margin, 2) if type(margin) in (int, float) and math.isfinite(margin) else None,
        'label': label,
        'choices': list(PREFS),
        # G-F127 (v0.1.3 engine): the engine's own guard decision, when it reports one
        'swap_guard': fc.get('swap_guard') if fc.get('swap_guard') in ('swap-tolerant', 'memavailable', 'refuse') else None,
    }


def catalog_status(config, actual=None, env=None):
    """Configured preference and an explicitly estimated working set, without any I/O."""
    flash = config.get('oxe_flash')
    if not isinstance(flash, dict):
        return None
    result = dict(actual or {})
    if enabled(env, flash):
        pref = read_pref(env, flash)
        result.setdefault('pref', pref)
        result['preference_control'] = flash.get('cache_pref') is not None and not bool((env or {}).get(ENV, '').strip())
        result['desktop_floor_gib'] = desktop_floor(flash)
        tier = result.get('tier_gib') or (int(pref) if pref != 'auto' else None)
        if tier is not None:
            result['estimated_working_set_gib'] = round(
                tier_need_gib(tier, config['context_size']) - HARNESS_GROWTH_GIB - FLOOR_GIB, 3)
    return result or None
