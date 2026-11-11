"""Rate-limit analysis over request logs.

An event is (timestamp: float, client: str, path: str). Events are in arbitrary order.
"""


def flag_bursts(events, window, limit):
    """Return the sorted indices of events whose burst count is greater than `limit`.

    For event i, count(i) is the number of events j in the whole list (any position, so
    events with an equal timestamp that appear later in the list count too) with the same
    client and  t_i - window < t_j <= t_i  (window > 0 is needed for event i to count itself;
    with window 0 the interval is empty and count(i) is 0).
    Event i is flagged if count(i) > limit.
    """
    flagged = []
    for i, (t_i, client, _) in enumerate(events):
        count = 0
        for t_j, other, _ in events:
            if other == client and t_i - window < t_j <= t_i:
                count += 1
        if count > limit:
            flagged.append(i)
    return flagged


def top_paths(events, n):
    """Return the n most frequent paths as (path, count) pairs, most frequent first.

    Ties in count are broken by path, ascending. If there are fewer than n distinct
    paths, all of them are returned.
    """
    all_paths = [path for _, _, path in events]
    pairs = [(path, all_paths.count(path)) for path in set(all_paths)]
    pairs.sort(key=lambda pc: (-pc[1], pc[0]))
    return pairs[:n]
