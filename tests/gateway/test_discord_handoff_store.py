from plugins.platforms.discord.recovery import DiscordRecoveryStore


def test_message_handoff_round_trip_is_channel_scoped_and_bounded(tmp_path):
    store = DiscordRecoveryStore(hermes_home=tmp_path)

    assert store.record_message_handoff(
        channel_id="10",
        message_id="20",
        message_text="alert " + ("x" * 7000),
        context={
            "source_platform": "webhook",
            "source_route": "operations-alert",
            "delivery_id": "delivery-1",
        },
    )

    record = store.get_message_handoff(channel_id="10", message_id="20")
    assert record is not None
    assert record["message_text"].startswith("alert ")
    assert len(record["message_text"]) == 6000
    assert record["context"]["source_route"] == "operations-alert"
    assert store.get_message_handoff(channel_id="11", message_id="20") is None


def test_message_handoff_drops_oversized_or_unserializable_context(tmp_path):
    store = DiscordRecoveryStore(hermes_home=tmp_path)

    assert store.record_message_handoff(
        channel_id="10",
        message_id="21",
        message_text="alert",
        context={"too_large": "x" * 3000},
    )
    assert store.get_message_handoff(
        channel_id="10", message_id="21"
    )["context"] == {}

    assert store.record_message_handoff(
        channel_id="10",
        message_id="22",
        message_text="alert",
        context={"bad": object()},
    )
    assert store.get_message_handoff(
        channel_id="10", message_id="22"
    )["context"] == {}
