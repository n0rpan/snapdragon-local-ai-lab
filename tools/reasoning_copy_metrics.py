"""In-memory reasoning copy measurements on exact model token ids.

No file or network access; returns aggregate counts only. Callers must tokenize the
ephemeral reasoning, exclude the observer cost from engine timing, then discard text
and ids. N-gram coverage is an upper bound, not a speculative speedup prediction.
"""


def _tokens(values):
    if len(values) > 1_048_576 or any(type(x) is not int or x < 0 for x in values):
        raise ValueError('bounded_nonnegative_token_ids_required')
    return values


def reasoning_copy_metrics(context, reasoning):
    """N=3..8 coverage against the context preceding this reasoning, plus causal PLD.

    Coverage counts each reasoning token once when it belongs to a matching n-gram.
    PLD uses the most recent >=4-token match, drafts up to4/8 tokens from already
    available context and counts only tokens equal to the observed continuation.
    Earlier reasoning becomes eligible only after it has actually been generated.
    """
    context, reasoning = _tokens(context), _tokens(reasoning)
    rows = []
    for n in range(3, 9):
        prior = {tuple(context[i:i + n]) for i in range(len(context) - n + 1)}
        covered = bytearray(len(reasoning))
        windows = max(0, len(reasoning) - n + 1)
        matches = 0
        for i in range(windows):
            if tuple(reasoning[i:i + n]) in prior:
                matches += 1
                covered[i:i + n] = b'\1' * n
        count = sum(covered)
        rows.append({'n': n, 'windows': windows, 'matching_windows': matches,
                     'covered_tokens': count, 'copy_share': count / len(reasoning) if reasoning else 0.0})
    lookup = [_causal_lookup(context, reasoning, draft) for draft in (4, 8)]
    return {'reasoning_tokens': len(reasoning), 'preceding_context_tokens': len(context),
            'historical_ngram_coverage': rows, 'causal_prompt_lookup': lookup}


def _causal_lookup(context, reasoning, draft):
    n = 4
    known = list(context)
    # Exclude the trailing n-gram: it has no available continuation and would hide
    # an older useful match with the same key.
    index = {tuple(known[i:i + n]): i for i in range(len(known) - n)}
    position = copied = matches = lookups = 0
    accepted_lengths = [0] * (draft + 1)
    while position < len(reasoning):
        accepted = 0
        if len(known) >= n:
            source = index.get(tuple(known[-n:]))
            if source is not None:
                source += n
                if source < len(known):
                    lookups += 1
                    for j in range(min(draft, len(reasoning) - position, len(known) - source)):
                        if known[source + j] != reasoning[position + j]:
                            break
                        accepted += 1
                    accepted_lengths[accepted] += 1
                    if accepted:
                        matches += 1
                        copied += accepted
        step = max(1, accepted)
        # Update only after verification: future reasoning can never serve as a source.
        for token in reasoning[position:position + step]:
            known.append(token)
            if len(known) > n:
                start = len(known) - n - 1
                index[tuple(known[start:start + n])] = start
        position += step
    return {'min_match': n, 'max_draft': draft, 'lookups': lookups, 'successful_lookups': matches,
            'accepted_tokens': copied, 'copy_share': copied / len(reasoning) if reasoning else 0.0,
            'accepted_length_counts': accepted_lengths}
