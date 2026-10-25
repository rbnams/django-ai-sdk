"""AutomationService: who may see, toggle and fire automations, and what they read back."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from django.test import override_settings
from django_ai_sdk.automations import Automation, AutomationService
from django_ai_sdk.automations.models import AutomationState
from django_ai_sdk.automations.registry import register, reset_registry
from django_ai_sdk.permissions import PermissionDenied
from django_ai_sdk.workflows.models import WorkflowRun
from django_ai_sdk.workflows.registry import register as register_wf
from django_ai_sdk.workflows.registry import reset_registry as reset_workflows
from django_ai_sdk.workflows.schemas import FieldDefinition, StepDefinition, WorkflowDefinition
from tests.factories.db import UserFactory

AGENT_ID = "663dd70f-65d2-58c4-9edc-037c9904d562"
WORKFLOW = "some-workflow"


@pytest.fixture(autouse=True)
def _clean_registries():
    reset_registry()
    reset_workflows()
    register_wf(
        WorkflowDefinition(
            name=WORKFLOW,
            input_fields={"prompt": FieldDefinition(type="str")},
            steps=[StepDefinition(name="result", agent_id=AGENT_ID, history=["prompt"])],
        )
    )
    register(
        type(
            "Example", (Automation,), {"name": "example", "cron": "0 9 * * *", "workflow": WORKFLOW}
        )
    )
    yield
    reset_registry()
    reset_workflows()


@pytest.fixture(autouse=True)
def _no_queue():
    with patch("django_ai_sdk.automations.runner._enqueue"):
        yield


@pytest.mark.django_db(transaction=True)
class TestList:
    async def test_anonymous_sees_nothing(self):
        assert await AutomationService.list_for_user(None) == []

    async def test_member_sees_schedule_and_enabled_source(self):
        user = await UserFactory.acreate()
        [row] = await AutomationService.list_for_user(user)
        assert row.name == "example"
        assert row.schedule.startswith("0 9 * * *")
        assert (row.enabled, row.enabled_source) == (True, "code")
        assert row.last_run is None

    @override_settings(AI_SDK_AUTOMATIONS_ENABLED=False)
    async def test_kill_switch_wins(self):
        user = await UserFactory.acreate()
        [row] = await AutomationService.list_for_user(user)
        assert (row.enabled, row.enabled_source) == (False, "kill-switch")


@pytest.mark.django_db(transaction=True)
class TestSetEnabled:
    async def test_staff_writes_the_database_layer(self):
        staff = await UserFactory.acreate(staff=True)
        row = await AutomationService.set_enabled("example", enabled=False, user=staff)
        assert (row.enabled, row.enabled_source) == (False, "db")
        assert (await AutomationState.objects.aget(name="example")).enabled is False

    async def test_member_may_not(self):
        user = await UserFactory.acreate()
        with pytest.raises(PermissionDenied):
            await AutomationService.set_enabled("example", enabled=False, user=user)

    async def test_unknown_name_is_none(self):
        staff = await UserFactory.acreate(staff=True)
        assert await AutomationService.set_enabled("nope", enabled=True, user=staff) is None

    async def test_a_tick_skips_what_the_database_turned_off(self):
        from django_ai_sdk.automations.runner import tick

        staff = await UserFactory.acreate(staff=True)
        await AutomationService.set_enabled("example", enabled=False, user=staff)
        [result] = await tick()
        assert result.reason == "disabled"


@pytest.mark.django_db(transaction=True)
class TestRuns:
    async def test_run_now_dispatches_and_lists(self):
        staff = await UserFactory.acreate(staff=True)
        [dispatched] = await AutomationService.run_now("example", user=staff)

        [listed] = await AutomationService.list_runs("example", user=staff)
        assert listed.id == dispatched.id
        assert (await AutomationService.get_run(dispatched.id, user=staff)).id == dispatched.id
        [row] = await AutomationService.list_for_user(staff)
        assert row.last_run.id == dispatched.id
        assert await AutomationService.last_tick_at() is not None

    async def test_member_may_not_run(self):
        user = await UserFactory.acreate()
        with pytest.raises(PermissionDenied):
            await AutomationService.run_now("example", user=user)

    @override_settings(AI_SDK_AUTOMATIONS_ENABLED=False)
    async def test_kill_switch_dispatches_nothing(self):
        staff = await UserFactory.acreate(staff=True)
        assert await AutomationService.run_now("example", user=staff) == []

    async def test_member_reads_only_their_own_runs(self):
        staff = await UserFactory.acreate(staff=True)
        user = await UserFactory.acreate()
        [run] = await AutomationService.run_now("example", user=staff)

        assert await AutomationService.list_runs("example", user=user) == []
        assert await AutomationService.get_run(run.id, user=user) is None

        await WorkflowRun.objects.filter(id=run.id).aupdate(user=user)
        assert [r.id for r in await AutomationService.list_runs("example", user=user)] == [run.id]

    async def test_bad_run_id_is_none(self):
        staff = await UserFactory.acreate(staff=True)
        assert await AutomationService.get_run("not-a-uuid", user=staff) is None
