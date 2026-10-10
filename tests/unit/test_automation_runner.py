"""The claim, the run-as account, and the WorkflowRuns a tick writes.

The enqueue is patched out except where it is the subject: what is under test is
which rows get written, not whether django-tasks accepts them.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest

from django_ai_sdk.automations import Automation
from django_ai_sdk.automations.models import AutomationState
from django_ai_sdk.automations.registry import register, reset_registry
from django_ai_sdk.automations.runner import claim, ensure_state, last_run_at, run_now, tick
from django_ai_sdk.workflows.schemas import FieldDefinition, StepDefinition, WorkflowDefinition
from django_ai_sdk.workflows.models import WorkflowRun
from django_ai_sdk.workflows.registry import register as register_wf
from django_ai_sdk.workflows.registry import reset_registry as reset_workflows

NOW = datetime(2026, 8, 16, 12, 0, tzinfo=UTC)

# A registry key, the shape a real reference has. Nothing executes the step here.
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
    yield
    reset_registry()
    reset_workflows()


@pytest.fixture(autouse=True)
def _no_queue():
    """Nothing here needs a worker; assert on rows, not on the backend."""
    with patch("django_ai_sdk.automations.runner._enqueue") as enqueue:
        yield enqueue


def declare(name="example", **attrs) -> Automation:
    defaults = {"name": name, "cron": "0 9 * * *", "workflow": WORKFLOW}
    cls = type("Example", (Automation,), {**defaults, **attrs})
    register(cls)
    return cls()


async def make_due(name="example", at=NOW) -> None:
    await AutomationState.objects.filter(name=name).aupdate(next_run_at=at)


@pytest.mark.django_db(transaction=True)
class TestClaim:
    async def test_two_ticks_racing_produce_exactly_one_winner(self):
        automation = declare(cron="0 * * * *")
        state, _ = await ensure_state(automation, now=NOW)
        await make_due()
        state = await AutomationState.objects.aget(id=state.id)

        first = await claim(automation, state, now=NOW)
        second = await claim(automation, state, now=NOW)

        assert (first, second) == (True, False)
        # And the cursor advanced once, not twice — a double advance would silently
        # skip a whole window.
        state = await AutomationState.objects.aget(id=state.id)
        assert state.next_run_at == NOW + timedelta(hours=1)
        assert state.last_dispatched_at == NOW


@pytest.mark.django_db(transaction=True)
class TestBootstrap:
    async def test_a_new_automation_is_scheduled_forward_not_backward(self):
        # Deploying an automation must not make it fire immediately just because its
        # schedule matched some moment before the deploy.
        state, created = await ensure_state(declare(), now=NOW)

        assert created is True
        assert state.next_run_at > NOW

    async def test_the_first_tick_dispatches_nothing(self):
        declare()
        [result] = await tick(now=NOW)
        assert result.run is None
        assert result.reason == "not due"


@pytest.mark.django_db(transaction=True)
class TestDueness:
    async def test_a_due_automation_dispatches_one_tagged_workflow_run(self):
        declare()
        await tick(now=NOW)
        await make_due()

        [result] = await tick(now=NOW)

        run = result.run
        assert run.status == WorkflowRun.Status.PENDING
        assert run.automation == "example"
        assert run.user_id is None
        assert run.inputs == {"prompt": "Run the example automation."}

    async def test_disabled_on_the_class_is_not_dispatched(self):
        declare(enabled=False)
        await tick(now=NOW)
        await make_due()

        [result] = await tick(now=NOW)

        assert result.run is None
        assert result.reason == "disabled"

    async def test_the_global_kill_switch_stops_everything(self, settings):
        settings.AI_SDK_AUTOMATIONS_ENABLED = False
        declare()
        assert await tick(now=NOW) == []

    async def test_one_broken_automation_does_not_stop_the_rest(self):
        class Broken(Automation):
            name = "broken"
            cron = "0 9 * * *"
            workflow = WORKFLOW

            def get_schedule(self):
                raise RuntimeError("boom")

        declare("healthy")
        # Registered past validation, the way a schedule that breaks only at runtime
        # would be — the tick must isolate it.
        from django_ai_sdk.automations import registry as registry_module

        registry_module._registry["broken"] = Broken()

        results = {r.name: r for r in await tick(now=NOW)}

        assert "see logs" in results["broken"].reason
        assert results["healthy"].reason == "not due"


@pytest.mark.django_db(transaction=True)
class TestBehindSchedule:
    # Fires at :30 past every hour, so a stale cursor lands on a window that is
    # distinguishable from `now`.
    HOURLY = "30 * * * *"

    async def test_it_runs_once_for_the_window_it_missed(self):
        declare(cron=self.HOURLY)
        await tick(now=NOW)
        await make_due(at=NOW - timedelta(hours=4, minutes=30))

        [result] = await tick(now=NOW)

        assert result.run is not None
        # One catch-up run, and the cursor moves to the next window after now.
        state = await AutomationState.objects.aget(name="example")
        assert state.next_run_at == NOW + timedelta(minutes=30)


@pytest.mark.django_db(transaction=True)
class TestRunAs:
    async def test_with_no_account_it_runs_as_the_app(self):
        declare()
        run = await run_now("example")
        assert run.user_id is None

    async def test_it_runs_as_the_configured_account(self, django_user_model):
        bot = await django_user_model.objects.acreate(email="bot@example.com")
        declare(run_as="bot@example.com", input="For {user}.")

        run = await run_now("example")

        assert run.user_id == bot.pk
        assert run.inputs == {"prompt": "For bot@example.com."}

    async def test_an_unknown_account_is_skipped_not_run_without_one(self, caplog):
        # Running with no user would quietly drop whatever the account may reach.
        declare(run_as="nobody@example.com")
        assert await run_now("example") is None
        assert "nobody@example.com" in caplog.text


@pytest.mark.django_db(transaction=True)
class TestInputs:
    async def test_an_input_the_workflow_does_not_declare_is_skipped(self, caplog):
        # A workflow drops undeclared inputs, which would be a run that succeeded
        # having asked nothing.
        def get_inputs(self, *, user=None, last_run_at=None):
            return {"messages": "hello"}

        declare(get_inputs=get_inputs)

        with caplog.at_level("WARNING"):
            assert await run_now("example") is None
        assert "'messages'" in caplog.text

    async def test_a_workflow_that_is_not_registered_is_skipped(self):
        declare(workflow="no-such-workflow")
        assert await run_now("example") is None

    async def test_last_run_at_is_the_last_completed_run(self):
        automation = declare()
        started = NOW - timedelta(days=1)
        await WorkflowRun.objects.acreate(
            automation="example", status=WorkflowRun.Status.COMPLETED, started_at=started
        )
        await WorkflowRun.objects.acreate(automation="example", status=WorkflowRun.Status.FAILED)

        assert await last_run_at(automation) == started


@pytest.mark.django_db(transaction=True)
class TestRunNow:
    async def test_it_leaves_the_schedule_alone(self):
        automation = declare()
        state, _ = await ensure_state(automation, now=NOW)

        await run_now("example")

        after = await AutomationState.objects.aget(id=state.id)
        assert after.next_run_at == state.next_run_at
        assert after.last_dispatched_at is not None

    async def test_it_respects_the_kill_switch(self, settings):
        settings.AI_SDK_AUTOMATIONS_ENABLED = False
        declare()
        assert await run_now("example") is None

    async def test_an_unknown_name_raises(self):
        with pytest.raises(ValueError):
            await run_now("nope")


@pytest.mark.django_db(transaction=True)
class TestEnqueueWaitsForTheCommit:
    """A worker must never be handed a run id that is still inside an open transaction."""

    @pytest.fixture(autouse=True)
    def _no_queue(self):
        # This class exercises the real _enqueue.
        yield

    def test_nothing_is_enqueued_until_the_transaction_commits(self):
        from asgiref.sync import async_to_sync
        from django.db import transaction

        declare()

        with patch("django_ai_sdk.workflows.tasks.execute_workflow") as task:
            task.enqueue.return_value.id = "task-1"
            with transaction.atomic():
                run = async_to_sync(run_now)("example")
                assert task.enqueue.call_count == 0
            assert task.enqueue.call_count == 1

        run.refresh_from_db()
        assert run.task_id == "task-1"

    def test_a_backend_that_refuses_fails_the_run(self):
        from asgiref.sync import async_to_sync

        declare()

        with patch("django_ai_sdk.workflows.tasks.execute_workflow") as task:
            task.enqueue.side_effect = RuntimeError("no backend")
            run = async_to_sync(run_now)("example")

        run.refresh_from_db()
        assert run.status == WorkflowRun.Status.FAILED
        assert "task backend" in run.error
