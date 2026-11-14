import dataclasses

import pytest

from slmstrict import CostConfig, FallbackConfig, MetricsConfig, StrictConfig


def test_defaults() -> None:
    config = StrictConfig()
    assert config.max_reask == 2
    assert config.local_only is False
    assert config.fallback is None
    assert config.metrics is None
    assert config.cost is None


def test_overrides_and_placeholder_sections() -> None:
    config = StrictConfig(
        max_reask=0,
        local_only=True,
        fallback=FallbackConfig(),
        metrics=MetricsConfig(),
        cost=CostConfig(),
    )
    assert config.max_reask == 0
    assert config.local_only is True
    assert config.fallback == FallbackConfig()


def test_negative_max_reask_is_rejected() -> None:
    with pytest.raises(ValueError, match="max_reask"):
        StrictConfig(max_reask=-1)


@pytest.mark.parametrize("bad", [True, 1.5, "2"])
def test_max_reask_must_be_int(bad: object) -> None:
    with pytest.raises(TypeError, match="max_reask"):
        StrictConfig(max_reask=bad)  # type: ignore[arg-type]


def test_local_only_must_be_bool() -> None:
    with pytest.raises(TypeError, match="local_only"):
        StrictConfig(local_only=1)  # type: ignore[arg-type]


def test_config_is_immutable_and_replaceable() -> None:
    config = StrictConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.max_reask = 5  # type: ignore[misc]
    assert dataclasses.replace(config, local_only=True).local_only is True
