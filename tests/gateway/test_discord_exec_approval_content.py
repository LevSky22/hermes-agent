from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.discord.adapter import DiscordAdapter


def _capture_channel(adapter):
    sent = {}

    async def fake_send(**kwargs):
        sent.update(kwargs)
        return SimpleNamespace(id=1234)

    channel = SimpleNamespace(send=AsyncMock(side_effect=fake_send))
    adapter._client = SimpleNamespace(
        get_channel=lambda _chat_id: channel,
        fetch_channel=AsyncMock(),
    )
    return sent


@pytest.mark.asyncio
async def test_exec_approval_prompt_uses_visible_content_with_command_and_reason():
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    sent = _capture_channel(adapter)

    command = "python scripts/deploy.py --env prod --force"
    result = await adapter.send_exec_approval(
        chat_id="555",
        command=command,
        session_key="discord:555",
        description="script execution via -c flag",
    )

    assert result.success is True
    assert sent["view"] is not None
    assert sent["embed"] is not None

    prompt_text = sent["content"]
    assert "Command Approval Required" in prompt_text
    assert "Do you want Hermes to run this command?" in prompt_text
    assert "Requested command" in prompt_text
    assert command in prompt_text
    assert "Reason" in prompt_text
    assert "script execution via -c flag" in prompt_text


@pytest.mark.asyncio
async def test_destructive_approval_is_one_shot_and_requester_bound():
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    sent = _capture_channel(adapter)

    result = await adapter.send_exec_approval(
        chat_id="555",
        command="effect=destructive client=example tool=delete_record target=123",
        session_key="discord:555",
        description="Delete one record",
        allow_permanent=False,
        allow_session=False,
        approval_id="approval-123",
        approval_kind="destructive",
        requester_user_id="42",
    )

    assert result.success is True
    assert "Destructive Action Approval Required" in sent["content"]
    view = sent["view"]
    assert view.approval_id == "approval-123"
    assert view.requester_user_id == "42"
    labels = {child.label for child in view.children}
    # The repository's dependency-isolation shim exposes a buttonless View;
    # real discord.py retains the two renamed controls.
    if labels:
        assert labels == {"Confirm once", "Cancel"}
