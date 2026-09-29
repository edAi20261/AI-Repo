"""The developer/QA squad's guardrails and artifacts, enforced.

An agent's instructions are advice. What keeps a developer agent from editing
the test it is failing, QA from quietly fixing the code it is judging, or either
from writing to production is .claude/hooks/sdlc_guard.py — and a guard nobody
tests is one more described contract (CLAUDE.md rule 2).
"""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
GUARD_PATH = ROOT / ".claude" / "hooks" / "sdlc_guard.py"
SDLC = ROOT / "sdlc"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


guard = _load("sdlc_guard", GUARD_PATH)

EXISTING_TEST = str(ROOT / "tests" / "unit" / "test_grade.py")


def _file(role: str, tool: str, path: str) -> str:
    return guard.pre_tool_use({"agent_type": role, "tool_name": tool,
                               "tool_input": {"file_path": path}}).verdict


def _bash(role: str, command: str) -> str:
    return guard.pre_tool_use({"agent_type": role, "tool_name": "Bash",
                               "tool_input": {"command": command}}).verdict


# -- developer ---------------------------------------------------------------

@pytest.mark.parametrize("tool", ["Edit", "MultiEdit", "Write"])
def test_the_developer_cannot_change_an_existing_test_to_make_it_pass(tool):
    """"Fix the code, not the test" is only an instruction until something refuses.

    The playbook's own warning: an agent fixing code must not be able to weaken
    the check on that code, or a green suite proves nothing.
    """
    assert _file("developer", tool, EXISTING_TEST) == "deny"


def test_the_developer_may_add_a_new_test_file():
    """Tests for internals are part of building; only QA's proof is locked."""
    new = str(ROOT / "tests" / "unit" / "test_a_file_that_does_not_exist_yet.py")
    assert _file("developer", "Write", new) == "allow"


def test_the_developer_cannot_add_a_conftest_that_deselects_the_proof():
    """A new conftest.py can skip any test without touching it."""
    new = str(ROOT / "tests" / "unit" / "some_new_dir" / "conftest.py")
    assert _file("developer", "Write", new) == "deny"


@pytest.mark.parametrize("path", [
    "pyproject.toml",
    ".github/workflows/deploy.yml",
    ".claude/hooks/sdlc_guard.py",
    ".claude/settings.json",
    ".claude/agents/developer.md",
    "scripts/verify.py",
    "tests/conftest.py",
])
def test_the_developer_cannot_loosen_the_checks_that_judge_it(path):
    """A new ignore in pyproject.toml silences lint as surely as a skipped test."""
    assert _file("developer", "Edit", str(ROOT / path)) == "deny"


def test_the_developer_may_change_the_code_under_test():
    assert _file("developer", "Edit", str(ROOT / "agent" / "graph.py")) == "allow"


@pytest.mark.parametrize("command", [
    "sed -i 's/==/!=/' tests/unit/test_grade.py",
    "rm tests/unit/test_grade.py",
    "rm -rf tests",
    "git checkout HEAD~1 -- tests/unit/test_grade.py",
    "git restore tests/unit/test_grade.py",
    "echo 'pass' > tests/unit/test_grade.py",
    "cat patch.txt | tee tests/unit/test_grade.py",
    "find tests -name '*.py' -delete",
    "cd /tmp && sed -i.bak 's/a/b/' tests/unit/test_grade.py",
    f"sed -i 's/a/b/' {EXISTING_TEST}",
    "echo '[tool.ruff]' >> pyproject.toml",
])
def test_the_developer_cannot_change_a_test_through_the_shell(command):
    """Blocking Edit and leaving `sed -i` open would block nothing."""
    assert _bash("developer", command) == "deny"


@pytest.mark.parametrize("command", [
    "cat tests/unit/test_grade.py",
    "ls -ln tests/",
    "sed -n 1,20p tests/unit/test_grade.py",
    "grep -rin 'skip' tests/ 2>&1",
    ".venv/bin/python scripts/verify.py tests/unit/test_grade.py",
    "git diff abc123 HEAD -- tests/",
    "git checkout -b cursor/new-branch",
    "pip install -r requirements-dev.txt",
])
def test_reading_the_tests_is_not_mistaken_for_changing_them(command):
    """A guard that refuses to let the developer read the tests gets bypassed."""
    assert _bash("developer", command) == "allow"


def test_a_worktree_is_not_read_as_a_guardrail(tmp_path):
    """Claude Code puts worktrees under .claude/worktrees/ in the main checkout.

    Measured from the project directory, every file in one starts with
    ".claude/" — and a developer in a worktree could edit nothing at all.
    """
    (tmp_path / ".git").mkdir()
    tree = tmp_path / ".claude" / "worktrees" / "feature"
    (tree / "tests" / "unit").mkdir(parents=True)
    (tree / ".git").write_text("gitdir: ../../../.git/worktrees/feature\n")
    existing = tree / "tests" / "unit" / "test_x.py"
    existing.write_text("")

    assert _file("developer", "Edit", str(tree / "agent" / "graph.py")) == "allow"
    assert _file("developer", "Edit", str(existing)) == "deny"


# -- QA ------------------------------------------------------------------------

@pytest.mark.parametrize("path", ["agent/graph.py", "shared/config.py", "config.yaml",
                                  ".claude/agents/qa-tester.md"])
def test_qa_cannot_fix_the_code_it_is_judging(path):
    """A QA verdict on code QA edited is the developer's verdict with extra steps."""
    assert _file("qa-tester", "Edit", str(ROOT / path)) == "deny"


@pytest.mark.parametrize("path", ["tests/unit/test_grade.py", "tests/conftest.py",
                                  "sdlc/changes/some-change/qa.md"])
def test_qa_may_write_tests_and_its_report(path):
    assert _file("qa-tester", "Edit", str(ROOT / path)) == "allow"


def test_qa_cannot_fix_the_code_through_the_shell():
    assert _bash("qa-tester", "sed -i 's/0.5/0.0/' config.yaml") == "deny"
    assert _bash("qa-tester", "pip install -r requirements.txt") == "allow"


# -- everyone ------------------------------------------------------------------

@pytest.mark.parametrize("role", ["developer", "qa-tester", ""])
def test_science_fusion_never_reaches_ingestion(role, tmp_path):
    """CLAUDE.md, corpus licensing: its copyright page forbids exactly this pipeline.

    It holds for the main session too, and for a manifest that only names the
    book inside the file.
    """
    direct = "python -m ingestion s3://bucket/Science_Fusion_G5.pdf --lecture-id 1 --subject-id 2"
    assert _bash(role, direct) == "deny"

    manifest = tmp_path / "lessons.tsv"
    manifest.write_text("s3://bucket/unit1.pdf\t1\t2\tScience Fusion Unit 1\n")
    assert _bash(role, f"scripts/ingest_batch.sh {manifest}") == "deny"


@pytest.mark.parametrize("command", [
    "python -m ingestion s3://bucket/unit1.pdf --lecture-id 1 --subject-id 2",
    "scripts/ingest_batch.sh lessons.tsv",
    "python scripts/bootstrap.py --postgres",
    ".venv/bin/python scripts/reembed_chunks.py --apply",
    "aws ecs update-service --cluster edu-ai-ai --force-new-deployment",
    "aws s3 rm s3://bucket/figures --recursive",
    "psql $DATABASE_URL -c 'DELETE FROM ai.chunks'",
    "git push origin main",
    "git push origin HEAD:main",
    "git push --force origin cursor/x",
])
@pytest.mark.parametrize("role", ["developer", "qa-tester", ""])
def test_a_production_write_waits_for_a_human(role, command):
    """CLAUDE.md: DATABASE_URL and Qdrant are production; writes get confirmed first."""
    assert _bash(role, command) == "ask"


@pytest.mark.parametrize("command", [
    "git push -u origin cursor/main-menu-fix-3502",
    ".venv/bin/python scripts/reembed_chunks.py",
    "psql $DATABASE_URL -c 'SELECT created_at, updated_at FROM ai.chunks LIMIT 1'",
])
def test_reads_and_feature_branches_do_not_wait_for_a_human(command):
    """A gate that asks about everything trains people to approve without reading."""
    assert _bash("developer", command) == "allow"


# -- the developer's stop gate -------------------------------------------------

def _stop(message: str = "Done.", role: str = "developer") -> dict:
    return {"hook_event_name": "SubagentStop", "agent_type": role, "cwd": str(ROOT),
            "last_assistant_message": message, "stop_hook_active": False}


def test_the_developer_cannot_report_done_while_verify_fails():
    """"Run the tests before reporting done" was a line in a prompt. Now it is a gate."""
    result = guard.developer_stop(_stop(), verify=lambda cwd: (1, "FAILED test_x - assert 1 == 2"))
    assert result["decision"] == "block"
    assert "FAILED test_x" in result["reason"]


def test_the_developer_finishes_red_only_by_saying_it_is_blocked():
    """Without a way out, a developer facing a wrong test loops until the cap."""
    def must_not_run(cwd):
        raise AssertionError("verify ran for a developer that declared itself blocked")

    assert guard.developer_stop(_stop("BLOCKED: the proof test asserts the old API"),
                                verify=must_not_run) is None


def test_a_green_verify_lets_the_developer_finish():
    assert guard.developer_stop(_stop(), verify=lambda cwd: (0, "== verify passed")) is None


def test_the_stop_gate_is_the_developers_alone():
    def must_not_run(cwd):
        raise AssertionError("verify ran for an agent that is not the developer")

    assert guard.developer_stop(_stop(role="qa-tester"), verify=must_not_run) is None


# -- the hook as Claude Code runs it -------------------------------------------

def _run_hook(stdin: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(GUARD_PATH)], input=stdin,
                          capture_output=True, text=True, check=False)


def test_the_hook_answers_in_claude_codes_protocol():
    """An allow must print nothing: an explicit "allow" would skip the permission prompt."""
    denied = _run_hook(json.dumps({"hook_event_name": "PreToolUse", "agent_type": "developer",
                                   "tool_name": "Edit",
                                   "tool_input": {"file_path": EXISTING_TEST}}))
    decision = json.loads(denied.stdout)["hookSpecificOutput"]
    assert denied.returncode == 0
    assert decision["hookEventName"] == "PreToolUse"
    assert decision["permissionDecision"] == "deny"
    assert decision["permissionDecisionReason"]

    allowed = _run_hook(json.dumps({"hook_event_name": "PreToolUse", "agent_type": "developer",
                                    "tool_name": "Bash", "tool_input": {"command": "ls"}}))
    assert allowed.returncode == 0
    assert allowed.stdout == ""


def test_a_guard_that_cannot_read_its_input_blocks_rather_than_allows():
    """CLAUDE.md rule 7. A guard that fails open is not a guard."""
    broken = _run_hook("not json")
    assert broken.returncode == 2
    assert "blocked" in broken.stderr


# -- the artifacts -----------------------------------------------------------------

PROOF_ID = re.compile(r"(tests/[\w/]+\.py)::(\w+)")


def field(text: str, name: str) -> str:
    match = re.search(rf"^{name}:[ \t]*(.*)$", text, re.MULTILINE)
    return match.group(1).strip() if match else ""


def proof_gaps(change: Path, root: Path = ROOT) -> list[str]:
    """What a PASS verdict claims that the repository does not show."""
    plan = (change / "plan.md").read_text()
    gaps = []
    if field(plan, "Status") != "approved" or not field(plan, "Approved by"):
        gaps.append("plan.md is not approved by a named human")
    proof = plan.split("## Proof", 1)[-1].split("\n## ", 1)[0]
    ids = PROOF_ID.findall(re.sub(r"<!--.*?-->", "", proof, flags=re.DOTALL))
    if not ids:
        gaps.append("plan.md lists no proof tests")
    for file, name in ids:
        test_file = root / file
        if not test_file.exists() or not re.search(rf"def {name}\(", test_file.read_text()):
            gaps.append(f"{file}::{name} does not exist")
    return gaps


def test_every_template_carries_the_fields_the_squad_reads():
    """The commands, the agents and the tests below all key on these lines."""
    templates = SDLC / "templates"
    assert field((templates / "intent.md").read_text(), "Status") == "draft"
    plan = (templates / "plan.md").read_text()
    assert field(plan, "Status") == "draft"
    assert "Approved by:" in plan
    assert "## Proof" in plan
    assert "## Deviations" in plan
    qa = (templates / "qa.md").read_text()
    assert field(qa, "Verdict") == "PENDING"
    assert "Proof commit:" in qa


def test_a_pass_verdict_without_its_proof_on_disk_is_caught(tmp_path):
    """The check below must be able to fail, or it passes for the wrong reason."""
    change = tmp_path / "sdlc" / "changes" / "demo"
    change.mkdir(parents=True)
    (change / "plan.md").write_text(
        "Status: approved\nApproved by: Mahmoud\n\n## Proof\n"
        "- tests/unit/test_demo.py::test_the_thing_is_proved\n\n## Risks\n")
    assert proof_gaps(change, root=tmp_path) == [
        "tests/unit/test_demo.py::test_the_thing_is_proved does not exist"]

    (tmp_path / "tests" / "unit").mkdir(parents=True)
    (tmp_path / "tests" / "unit" / "test_demo.py").write_text(
        "def test_the_thing_is_proved():\n    pass\n")
    assert proof_gaps(change, root=tmp_path) == []


def test_a_change_passed_by_qa_has_its_proof_in_the_repository():
    """A PASS names tests. If they are not here, the verdict is not evidence."""
    for change in sorted(p for p in (SDLC / "changes").glob("*") if p.is_dir()):
        for name, allowed in [("intent.md", {"draft", "accepted", "closed"}),
                              ("plan.md", {"draft", "approved"})]:
            if (change / name).exists():
                assert field((change / name).read_text(), "Status") in allowed, change / name
        qa = change / "qa.md"
        if not qa.exists():
            continue
        verdict = field(qa.read_text(), "Verdict")
        assert verdict in {"PENDING", "PASS", "FAIL"}, qa
        if verdict == "PASS":
            assert proof_gaps(change) == [], change.name
