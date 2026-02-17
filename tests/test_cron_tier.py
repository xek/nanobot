"""Tests for cron job tier support."""

import json
from pathlib import Path

from nanobot.cron.types import CronJob, CronPayload, CronSchedule, CronJobState, CronStore
from nanobot.cron.service import CronService
from nanobot.agent.tools.cron import CronTool


# === CronPayload tier field ===

def test_cron_payload_tier_default_none():
    p = CronPayload()
    assert p.tier is None


def test_cron_payload_tier_set():
    p = CronPayload(tier="quick")
    assert p.tier == "quick"


# === CronTool parameter schema ===

def test_cron_tool_schema_includes_tier():
    svc = CronService(Path("/tmp/test_cron_store.json"))
    tool = CronTool(svc)
    props = tool.parameters["properties"]
    assert "tier" in props
    assert props["tier"]["enum"] == ["quick", "normal", "deep"]


# === CronTool._add_job passes tier ===

def test_cron_tool_add_job_with_tier(tmp_path):
    store_path = tmp_path / "jobs.json"
    svc = CronService(store_path)
    tool = CronTool(svc)
    tool.set_context("cli", "direct")

    result = tool._add_job(
        message="check weather",
        every_seconds=3600,
        cron_expr=None,
        at=None,
        tier="quick",
    )
    assert "tier: quick" in result

    jobs = svc.list_jobs()
    assert len(jobs) == 1
    assert jobs[0].payload.tier == "quick"


def test_cron_tool_add_job_without_tier(tmp_path):
    store_path = tmp_path / "jobs.json"
    svc = CronService(store_path)
    tool = CronTool(svc)
    tool.set_context("cli", "direct")

    result = tool._add_job(
        message="remind me",
        every_seconds=60,
        cron_expr=None,
        at=None,
    )
    assert "tier:" not in result

    jobs = svc.list_jobs()
    assert len(jobs) == 1
    assert jobs[0].payload.tier is None


def test_cron_tool_add_job_invalid_tier(tmp_path):
    store_path = tmp_path / "jobs.json"
    svc = CronService(store_path)
    tool = CronTool(svc)
    tool.set_context("cli", "direct")

    result = tool._add_job(
        message="remind me",
        every_seconds=60,
        cron_expr=None,
        at=None,
        tier="invalid",
    )
    # Invalid tier is treated as None
    assert "tier:" not in result
    jobs = svc.list_jobs()
    assert jobs[0].payload.tier is None


# === CronTool._list_jobs shows tier ===

def test_cron_tool_list_jobs_shows_tier(tmp_path):
    store_path = tmp_path / "jobs.json"
    svc = CronService(store_path)
    tool = CronTool(svc)
    tool.set_context("cli", "direct")

    tool._add_job("task A", every_seconds=60, cron_expr=None, at=None, tier="deep")
    tool._add_job("task B", every_seconds=120, cron_expr=None, at=None)

    listing = tool._list_jobs()
    assert "tier: deep" in listing
    # task B has no tier, so no "tier:" label for it
    lines = listing.strip().split("\n")
    task_b_line = [l for l in lines if "task B" in l][0]
    assert "tier:" not in task_b_line


# === CronService serialization round-trip ===

def test_cron_service_serializes_tier(tmp_path):
    store_path = tmp_path / "jobs.json"
    svc = CronService(store_path)

    svc.add_job(
        name="test-job",
        schedule=CronSchedule(kind="every", every_ms=60000),
        message="hello",
        tier="normal",
    )

    # Force a fresh load
    raw = json.loads(store_path.read_text())
    assert raw["jobs"][0]["payload"]["tier"] == "normal"

    # Load from disk
    svc2 = CronService(store_path)
    jobs = svc2.list_jobs()
    assert len(jobs) == 1
    assert jobs[0].payload.tier == "normal"


def test_cron_service_loads_legacy_jobs_without_tier(tmp_path):
    """Old job files without a tier field should load without error."""
    store_path = tmp_path / "jobs.json"
    legacy_data = {
        "version": 1,
        "jobs": [{
            "id": "abc123",
            "name": "legacy",
            "enabled": True,
            "schedule": {"kind": "every", "everyMs": 60000},
            "payload": {
                "kind": "agent_turn",
                "message": "hello",
                "deliver": False,
            },
            "state": {},
            "createdAtMs": 0,
            "updatedAtMs": 0,
            "deleteAfterRun": False,
        }],
    }
    store_path.write_text(json.dumps(legacy_data))

    svc = CronService(store_path)
    jobs = svc.list_jobs()
    assert len(jobs) == 1
    assert jobs[0].payload.tier is None


# === CronTool.execute passes tier ===

async def test_cron_tool_execute_add_with_tier(tmp_path):
    store_path = tmp_path / "jobs.json"
    svc = CronService(store_path)
    tool = CronTool(svc)
    tool.set_context("cli", "direct")

    result = await tool.execute(
        action="add",
        message="daily report",
        cron_expr="0 9 * * *",
        tier="deep",
    )
    assert "tier: deep" in result
    jobs = svc.list_jobs()
    assert jobs[0].payload.tier == "deep"
