"""Incremental build planner.

    plan(goals, rules, mtimes, phony=frozenset()) -> list[str]

Graph: `rules` maps a target name to its ordered dependency list. Any name
without an entry in `rules` is a source. `mtimes` maps names to modification
times. A target missing from `mtimes` has never been built. `phony` names are
targets that are always rebuilt; a phony name that is not a target is ignored.

Traversal: depth-first. Visit the goals in the given order. For each node, visit
its dependencies in listed order (a dependency already finished is skipped).
A target is finished when all its dependencies are finished (post-order).
Every dependency of every visited target is examined.

Errors (raised at the moment the traversal meets the problem; the first one met wins):
- A source reached as a dependency that is not in `mtimes` raises
  MissingSourceError(name). Sources not reachable from the goals are never checked.
- A goal that is neither a target nor in `mtimes` raises MissingSourceError.
- If the traversal meets a dependency that is on the current DFS stack, that is a
  cycle: raise CycleError(cycle), where `cycle` is the stack path from that node
  to itself, e.g. ['a', 'b', 'c', 'a'] (a self-loop gives ['a', 'a']).
  Only targets can be on the stack.

Staleness: a target is STALE if any of these holds:
- it is in `phony`;
- it is missing from `mtimes`;
- any of its dependencies is a stale target (a stale dependency counts as newer
  regardless of its mtime);
- any of its dependencies, which is a source or a non-stale target, has an mtime
  strictly greater than the target's mtime.
Equal mtimes are not stale.

Output: the stale targets reachable from the goals, each exactly once, in
finishing (post-order) order: dependencies before dependents. Sources never appear,
and neither do non-stale targets. A goal that is a source produces nothing (it
must still exist in `mtimes`). Duplicate goals are fine.

Scale: the planner must handle dependency chains of several thousand targets
(e.g. 3000 nested targets). Do not rely on Python's recursion limit: the
traversal must be iterative.
"""

from .errors import CycleError, MissingSourceError


def plan(goals, rules, mtimes, phony=frozenset()):
    raise NotImplementedError('plan() is not implemented yet')
