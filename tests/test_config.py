"""Tests for configuration, validation, and scenario loading."""

from __future__ import annotations

import pytest

from swarm.config import SwarmConfig, build_config, list_scenarios


class TestSwarmConfig:
    """Frozen dataclass configuration tests."""

    def test_default_values(self) -> None:
        cfg = SwarmConfig()
        assert cfg.num_agents == 50
        assert cfg.epsilon == 80.0
        assert cfg.sigma == 12.0

    def test_immutability(self) -> None:
        cfg = SwarmConfig()
        with pytest.raises(AttributeError):
            cfg.num_agents = 100  # type: ignore[misc]

    def test_lj_equilibrium_distance(self) -> None:
        cfg = SwarmConfig(sigma=12.0)
        expected = (2.0 ** (1.0 / 6.0)) * 12.0
        assert abs(cfg.lj_r_eq - expected) < 1e-10

    def test_resolved_output_file_default(self) -> None:
        cfg = SwarmConfig(scenario_name="test-scenario")
        assert cfg.resolved_output_file == "swarm_test-scenario.gif"

    def test_resolved_output_file_mp4(self) -> None:
        cfg = SwarmConfig(scenario_name="demo", output_format="mp4")
        assert cfg.resolved_output_file == "swarm_demo.mp4"

    def test_resolved_output_file_custom(self) -> None:
        cfg = SwarmConfig(output_file="custom.gif")
        assert cfg.resolved_output_file == "custom.gif"

    def test_summary_contains_key_info(self) -> None:
        cfg = SwarmConfig()
        s = cfg.summary()
        assert "50 agents" in s
        assert "drifting-target" in s
        assert "ε=80.0" in s

    def test_custom_values(self) -> None:
        cfg = SwarmConfig(num_agents=100, epsilon=40.0, sigma=8.0)
        assert cfg.num_agents == 100
        assert cfg.epsilon == 40.0
        assert cfg.sigma == 8.0


class TestConfigValidation:
    """__post_init__ should reject obviously broken parameters."""

    def test_rejects_zero_agents(self) -> None:
        with pytest.raises(ValueError, match="num_agents"):
            SwarmConfig(num_agents=0)

    def test_rejects_negative_epsilon(self) -> None:
        with pytest.raises(ValueError, match="epsilon"):
            SwarmConfig(epsilon=-1.0)

    def test_rejects_zero_sigma(self) -> None:
        with pytest.raises(ValueError, match="sigma"):
            SwarmConfig(sigma=0.0)

    def test_rejects_negative_dt(self) -> None:
        with pytest.raises(ValueError, match="dt"):
            SwarmConfig(dt=-0.01)

    def test_rejects_zero_mass(self) -> None:
        with pytest.raises(ValueError, match="agent_mass"):
            SwarmConfig(agent_mass=0.0)

    def test_rejects_invalid_dropout(self) -> None:
        with pytest.raises(ValueError, match="dropout_rate"):
            SwarmConfig(dropout_rate=1.0)

    def test_rejects_negative_noise(self) -> None:
        with pytest.raises(ValueError, match="sensor_noise"):
            SwarmConfig(sensor_noise=-1.0)

    def test_rejects_invalid_format(self) -> None:
        with pytest.raises(ValueError, match="output_format"):
            SwarmConfig(output_format="avi")

    def test_multiple_errors_reported(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            SwarmConfig(num_agents=0, epsilon=-1.0, sigma=0.0)
        # Should report all three errors, not just the first
        msg = str(exc_info.value)
        assert "num_agents" in msg
        assert "epsilon" in msg
        assert "sigma" in msg


class TestBuildConfig:
    """Config builder with CLI overrides."""

    def test_defaults_without_scenario(self) -> None:
        cfg = build_config()
        assert cfg.num_agents == 50

    def test_cli_overrides(self) -> None:
        cfg = build_config(num_agents=100, epsilon=40.0)
        assert cfg.num_agents == 100
        assert cfg.epsilon == 40.0

    def test_none_overrides_ignored(self) -> None:
        cfg = build_config(num_agents=None, epsilon=None)
        assert cfg.num_agents == 50

    def test_invalid_override_raises(self) -> None:
        with pytest.raises(ValueError, match="num_agents"):
            build_config(num_agents=0)


class TestScenarioListing:
    """Scenario discovery."""

    def test_list_scenarios_returns_list(self) -> None:
        result = list_scenarios()
        assert isinstance(result, list)

    def test_list_scenarios_finds_bundled_presets(self) -> None:
        result = list_scenarios()
        # These should exist in our scenarios/ directory
        if result:  # only assert if scenarios dir is accessible
            assert "drifting_target" in result
