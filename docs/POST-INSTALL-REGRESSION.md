# Post-install regression smoke

`tools/post_install_regression.py` checks a changed model on the isolated
candidate controller, then the installed controller. It has a hard 300-second
budget, synthetic inputs only, pinned engine identity and actual controller
MemoryGuard checks after each case. It never installs a release.

Capture an immutable reference only after the candidate's independent task /
numeric gate passed. That receipt must contain `passed:true` and the exact
`engine_sha256`. Use fresh private output paths:

```sh
python3 tools/post_install_regression.py capture --url http://127.0.0.1:8096 \
  --alias ornith-35b-a3b-local --engine-sha256 ACTUAL_SHA \
  --qualified-receipt /private/g0/protected-proof.json --session-resume \
  --reference /private/g0/regression-reference.json --out /private/g0/pre-install.json
python3 tools/post_install_regression.py verify --url http://127.0.0.1:8092 \
  --alias ornith-35b-a3b-local --engine-sha256 ACTUAL_SHA --session-resume \
  --reference /private/g0/regression-reference.json --out /private/g0/post-install.json
```

Cases: chat; an assembled tool call and its follow-up; 4K-context raw decode;
disconnect during an 8K prefill followed by a reply in under five seconds; and,
with `--session-resume`, park → unload → reload → identical continuation with
at least 512 cached tokens. Disk-session releases must use that flag. Existing
active/queued work is refused. The controller owns the inference lease; the
caller owns the quiet window. Sessions and references stay on encrypted storage.

Raw completion cases compare generated token-ID hashes. Chat/tool cases compare
assembled-message hashes with canonical JSON arguments and transport IDs removed.
They do not enable logprobs, which would change the fast path. Native timing
fields are retained; prefill comparisons require equal new/cached token counts.
Matched timing increases beyond 25% and 50 ms fail this coarse regression smoke.
Task benchmarks remain the evidence for smaller performance changes.

PASS / FAIL / BLOCKED and per-case evidence are written mode 0600. A timeout is
BLOCKED, never a pass. Changed output or engine identity cannot update an existing
reference. Failure after installation triggers the release's guarded one-command
rollback. Full maximum-context coverage remains in tickets 027–029 / F28; this
4K context smoke is not a substitute. Integration is tracked in F-C3.
