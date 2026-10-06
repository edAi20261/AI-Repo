# QA: <title>

Verdict: PENDING
<!-- PENDING | PASS | FAIL. PASS only when checks 1 to 4 hold and no Important finding is open. -->
Round: 1
Proof commit:

## Proof: watched fail
<!-- Each proof test and the failure it produced before the code existed. -->

## Checks

| # | Check | Result | Evidence |
|---|-------|--------|----------|
| 1 | Proof tests unchanged since the proof commit; no skip, xfail, deselect or new lint ignore | | `git diff <proof commit> HEAD -- ...` |
| 2 | `scripts/verify.py` passes | | last lines of output |
| 3 | The diff matches plan.md, or the difference is under Deviations | | |
| 4 | CLAUDE.md rules the change is exposed to | | |
| 5 | Evals named in the plan | not run | |

## Findings

| # | Severity | What breaks | Failing test |
|---|----------|-------------|--------------|
<!-- Severity is Important (breaks behaviour, leaks data, breaches a rule) or Nit. -->
