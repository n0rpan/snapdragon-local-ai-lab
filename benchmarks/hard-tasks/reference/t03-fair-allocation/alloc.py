"""Weighted max-min fair allocation of an integer budget.

allocate(budget: int, requests: list[tuple[str, int, int]]) -> dict[str, int]

requests is a list of (name, demand, weight) tuples. The result maps every
requester name to the integer number of units it receives.

1. Validation. Raise ValueError if any of these fails (check before computing):
   - budget is an int >= 0
   - every demand is an int >= 0
   - every weight is an int >= 1
   - names are unique
   bool is NOT an int here: True/False are rejected for budget, demand and weight.
   An empty requests list is valid and returns {}.

2. Real-valued target (exact, use fractions.Fraction, never floats).
   Let D = sum of all demands and S = min(budget, D).
   Find a level L >= 0 with sum(x_i) == S, where x_i = min(demand_i, weight_i * L).
   Requesters with demand 0 get x_i = 0. If S == D, every requester gets x_i = demand_i.

3. Integer rounding.
   Start with f_i = floor(x_i). Let R = S - sum(f_i). Hand out R extra units, one
   each, in this order of preference among requesters: largest fractional part
   (x_i - f_i) first, then larger weight first, then name ascending (string order).
   Take the first R in that order.
   A requester never gets more than its demand and receives at most one extra unit.
   R is always less than the number of requesters with a nonzero fractional part
   (the fractional parts sum to the integer R and each is < 1), so this is well defined.

4. Result. Every requester name is a key. sum(result) == S, and
   0 <= result[name] <= demand for every requester.
"""
from fractions import Fraction


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def allocate(budget, requests):
    if not _is_int(budget) or budget < 0:
        raise ValueError("budget must be an int >= 0")
    seen = set()
    clean = []
    for name, demand, weight in requests:
        if not _is_int(demand) or demand < 0:
            raise ValueError("demand must be an int >= 0")
        if not _is_int(weight) or weight < 1:
            raise ValueError("weight must be an int >= 1")
        if name in seen:
            raise ValueError("duplicate name: %r" % (name,))
        seen.add(name)
        clean.append((name, demand, weight))
    if not clean:
        return {}

    total = sum(d for _, d, _ in clean)
    target = min(budget, total)

    # Water filling: f(L) = sum(min(d, w*L)) is piecewise linear and nondecreasing.
    # Walk the saturation breakpoints t = d/w in ascending order.
    order = sorted(clean, key=lambda r: Fraction(r[1], r[2]))
    saturated = 0
    active_weight = sum(w for _, _, w in clean)
    level = None
    for _, d, w in order:
        t = Fraction(d, w)
        if saturated + t * active_weight >= target:
            level = Fraction(target - saturated, active_weight)
            break
        saturated += d
        active_weight -= w
    # The last breakpoint always reaches the target, so level is set.

    exact = {name: min(Fraction(d), w * level) for name, d, w in clean}
    floors = {name: v.numerator // v.denominator for name, v in exact.items()}
    weights = {name: w for name, _, w in clean}
    extra = target - sum(floors.values())
    ranked = sorted(
        (name for name in exact if exact[name] > floors[name]),
        key=lambda name: (-(exact[name] - floors[name]), -weights[name], name),
    )
    result = dict(floors)
    for name in ranked[:extra]:
        result[name] += 1
    return result
