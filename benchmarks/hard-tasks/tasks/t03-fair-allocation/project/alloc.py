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


def allocate(budget, requests):
    raise NotImplementedError
