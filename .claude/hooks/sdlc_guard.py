"""The squad's guardrails, as a Claude Code hook.

A subagent's instructions are advice: nothing stops a developer agent that has
been told "fix the code, not the test" from editing the test until it passes.
This hook is the part that cannot be talked out of it. It runs from
.claude/settings.json for every session, so an agent cannot switch it off by
editing its own definition, and it tells roles apart by ``agent_type``, which
Claude Code sets to the subagent's ``name``.

    PreToolUse    developer   may not change the checks that judge its work
                  qa-tester   may write tests and reports, never the code under test
                  everyone    production writes ask a human; Science Fusion never
                              reaches ingestion (CLAUDE.md, corpus licensing)
    SubagentStop  developer   cannot report done while scripts/verify.py fails

Bash is checked by reading the command, so it is a tripwire rather than a
sandbox: a determined ``python -c`` can still write a file. The backstop is the
QA agent, which checks in a fresh context that the proof tests are unchanged.
"""

from __future__ import annotations

import json
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

DEVELOPER = "developer"
QA = "qa-tester"

FILE_TOOLS = {"Edit", "MultiEdit", "Write", "NotebookEdit"}

TESTS = ("tests/",)
GUARDRAILS = (".claude/", ".github/", "pyproject.toml", "scripts/verify.py", "tests/conftest.py")
QA_WRITABLE = ("tests/", "sdlc/")
QA_PROTECTED = (
    "agent/", "shared/", "system/", "ingestion/", "scripts/", "infra/",
    "config.yaml", "Dockerfile", "Dockerfile.ingest", "requirements.txt",
    ".claude/", ".github/", "pyproject.toml", "CLAUDE.md",
)

WRITING_COMMANDS = {"rm", "mv", "cp", "truncate", "tee", "touch", "ln", "install", "patch", "dd"}
IN_PLACE_EDITORS = {"sed", "perl"}
WRITING_GIT = {"checkout", "restore", "rm", "mv", "reset", "apply", "am", "revert"}
PREFIXES = {"sudo", "env", "xargs", "command", "nohup", "time"}
REDIRECT = re.compile(r"(?<![0-9&>])>>?\s*([^\s;&|<>]+)")
SEGMENT = re.compile(r"&&|\|\||[;|\n]")

SCIENCE_FUSION = re.compile(r"science[\s_\-]*fusion", re.IGNORECASE)
INGEST = re.compile(r"ingest", re.IGNORECASE)
MANIFEST_SUFFIXES = {".tsv", ".csv", ".txt", ".json", ".yaml", ".yml"}

PROD_WRITES = [
    (re.compile(r"\b(ingest_batch|ingest_on_aws|eval_on_aws)\.sh\b"),
     "starts tasks on AWS against the production stores"),
    (re.compile(r"-m\s+ingestion\b|\bingestion/__main__\.py\b"),
     "ingestion writes ai.chunks and the Qdrant index"),
    (re.compile(r"\bbootstrap\.py\b"), "applies the schema to DATABASE_URL"),
    (re.compile(r"\breembed_chunks\.py\b.*--apply\b"), "rewrites every stored vector"),
    (re.compile(r"\baws\s+ecs\s+(update-service|run-task|stop-task|delete-\w+)\b"),
     "changes what is running in production"),
    (re.compile(r"\baws\s+s3\s+(cp|mv|rm|sync)\b|\baws\s+s3api\s+(put|delete)-"),
     "writes to S3"),
    (re.compile(r"\bpsql\b.*\b(insert|update|delete|drop|truncate|alter|create|grant)\b",
                re.IGNORECASE),
     "writes to Postgres"),
    (re.compile(r"\bgit\s+push\b.*(?:[\s:+](?:refs/heads/)?main(?=\s|$)|\s--force|\s-f(?=\s|$))"),
     "pushes to main or rewrites shared history"),
]


@dataclass(frozen=True)
class Decision:
    verdict: str
    reason: str = ""


ALLOW = Decision("allow")
SEVERITY = {"allow": 0, "ask": 1, "deny": 2}


def strictest(decisions: list[Decision]) -> Decision:
    return max(decisions, key=lambda d: SEVERITY[d.verdict], default=ALLOW)


def repo_relative(path: str) -> str | None:
    """The path from the root of the checkout that holds it, worktrees included.

    Worktrees live under .claude/worktrees/ in the main checkout, so measuring
    from the project directory would read every file in one as a guardrail.
    """
    p = Path(path.replace("\\", "/")).expanduser()
    if not p.is_absolute():
        return p.as_posix().removeprefix("./")
    p = p.resolve()
    for parent in p.parents:
        if (parent / ".git").exists():
            return p.relative_to(parent).as_posix()
    return None


def under(rel: str, prefixes: tuple[str, ...]) -> bool:
    return any(rel == p.rstrip("/") or (p.endswith("/") and rel.startswith(p)) for p in prefixes)


def check_file(role: str, tool: str, path: str) -> Decision:
    rel = repo_relative(path)
    if rel is None:
        return ALLOW
    if role == DEVELOPER:
        if under(rel, GUARDRAILS):
            return Decision("deny", f"{rel} is one of the checks that judge your work. "
                            "Changing it is a human's PR, not part of this task.")
        if under(rel, TESTS):
            creating = tool == "Write" and not Path(path).exists()
            if creating and Path(rel).name != "conftest.py":
                return ALLOW
            return Decision("deny", f"{rel} is QA's proof that the change works: fix the "
                            "code, not the test. If you believe the test is wrong, stop and "
                            "start your final message with 'BLOCKED:' saying why.")
    if role == QA and not under(rel, QA_WRITABLE):
        return Decision("deny", f"QA reports defects and proves them with tests; it does not "
                        f"fix them. {rel} is outside tests/ and sdlc/ — write the failing "
                        "test and the finding in qa.md instead.")
    return ALLOW


def segments(command: str) -> list[str]:
    return [s.strip() for s in SEGMENT.split(command) if s.strip()]


def tokens(segment: str) -> list[str]:
    try:
        return shlex.split(segment, posix=True)
    except ValueError:
        return segment.split()


def writes_its_arguments(words: list[str]) -> bool:
    while words and (words[0] in PREFIXES or "=" in words[0] or words[0].startswith("-")):
        words = words[1:]
    if not words:
        return False
    cmd, rest = Path(words[0]).name, words[1:]
    if cmd in WRITING_COMMANDS:
        return True
    if cmd in IN_PLACE_EDITORS:
        return any(re.fullmatch(r"-[a-zA-Z]*i\S*|--in-place\S*", w) for w in rest)
    if cmd == "git":
        return bool(rest) and rest[0] in WRITING_GIT
    if cmd == "find":
        return any(w in {"-delete", "-exec", "-execdir"} for w in rest)
    return False


def mutated_paths(segment: str) -> list[str]:
    found = REDIRECT.findall(segment)
    words = tokens(segment)
    if writes_its_arguments(words):
        found += [w for w in words if not w.startswith("-")]
    return [rel for t in found if (rel := repo_relative(t)) is not None]


def check_bash_role(role: str, command: str) -> Decision:
    protected = {DEVELOPER: TESTS + GUARDRAILS, QA: QA_PROTECTED}.get(role)
    if not protected:
        return ALLOW
    for segment in segments(command):
        hit = next((rel for rel in mutated_paths(segment) if under(rel, protected)), None)
        if hit:
            return Decision("deny", f"`{segment}` changes {hit}, which the {role} role may "
                            "not change. Use the file tools so the change is checked, or "
                            "stop and say why it is needed.")
    return ALLOW


def mentions_science_fusion(command: str) -> bool:
    if SCIENCE_FUSION.search(command):
        return True
    for token in tokens(command):
        p = Path(token)
        try:
            if (p.suffix in MANIFEST_SUFFIXES and p.is_file() and p.stat().st_size < 1_000_000
                    and SCIENCE_FUSION.search(p.read_text(errors="ignore"))):
                return True
        except OSError:
            continue
    return False


def check_bash_shared(command: str) -> Decision:
    if INGEST.search(command) and mentions_science_fusion(command):
        return Decision("deny", "Science Fusion (HMH, © 2017) forbids reproduction by any "
                        "information storage and retrieval system. It stays a local layout "
                        "fixture and is never ingested — see CLAUDE.md, corpus licensing.")
    reasons = [why for pattern, why in PROD_WRITES if pattern.search(command)]
    if reasons:
        return Decision("ask", "This " + "; ".join(reasons) + ". DATABASE_URL and Qdrant "
                        "point at production, so a human confirms every write.")
    return ALLOW


def pre_tool_use(payload: dict) -> Decision:
    role = payload.get("agent_type") or ""
    tool = payload.get("tool_name") or ""
    args = payload.get("tool_input") or {}
    if tool in FILE_TOOLS:
        path = args.get("file_path") or args.get("notebook_path") or ""
        return check_file(role, tool, path) if path else ALLOW
    if tool == "Bash":
        command = args.get("command") or ""
        return strictest([check_bash_role(role, command), check_bash_shared(command)])
    return ALLOW


def run_verify(cwd: str) -> tuple[int, str]:
    root = Path(cwd).resolve()
    root = next((p for p in (root, *root.parents) if (p / ".git").exists()), root)
    venv = root / ".venv" / "bin" / "python"
    python = str(venv) if venv.exists() else sys.executable
    done = subprocess.run([python, "scripts/verify.py"], cwd=root, capture_output=True,
                          text=True, check=False)
    return done.returncode, done.stdout + done.stderr


def developer_stop(payload: dict, verify=run_verify) -> dict | None:
    if payload.get("agent_type") != DEVELOPER:
        return None
    if (payload.get("last_assistant_message") or "").lstrip().startswith("BLOCKED:"):
        return None
    code, output = verify(payload.get("cwd") or ".")
    if code == 0:
        return None
    tail = "\n".join(output.splitlines()[-60:])
    return {"decision": "block",
            "reason": "scripts/verify.py fails, so the work is not done. Fix the code, not "
                      "the tests, and finish again — or, if you cannot, start your final "
                      f"message with 'BLOCKED:' and say why.\n\n{tail}"}


def main() -> int:
    try:
        payload = json.load(sys.stdin)
        event = payload.get("hook_event_name")
        if event == "PreToolUse":
            decision = pre_tool_use(payload)
            if decision.verdict != "allow":
                print(json.dumps({"hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": decision.verdict,
                    "permissionDecisionReason": decision.reason,
                }}))
        elif event == "SubagentStop":
            result = developer_stop(payload)
            if result:
                print(json.dumps(result))
        return 0
    except Exception as exc:
        # Failing open would let every guarded action through unannounced.
        print(f"sdlc_guard failed, so the action is blocked: {exc!r}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
