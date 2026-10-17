from __future__ import annotations

import os
from pathlib import Path

import pytest

from pilot_fixture_contract import PilotFixtureValidationError, validate_pilot_fixtures


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("Kawaii test profiles")
    group.addoption(
        "--pilot-profile",
        action="store_true",
        help="Run only the six external real-pilot integration tests after identity preflight.",
    )
    group.addoption(
        "--pilot-fixture-root",
        type=Path,
        help="Root containing workspace/kawaii/session2/source/eeg, workspace/kawaii/session2/source/behavioral_logs, and workspace/kawaii/session2/source/schedules.",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    pilot_profile = config.getoption("--pilot-profile")
    pilot_items = [item for item in items if item.get_closest_marker("pilot_data")]
    ordinary_items = [item for item in items if not item.get_closest_marker("pilot_data")]
    config._kawaii_pilot_item_count = len(pilot_items)  # type: ignore[attr-defined]

    if pilot_profile:
        selected, deselected = pilot_items, ordinary_items
    else:
        selected, deselected = ordinary_items, pilot_items

    if deselected:
        config.hook.pytest_deselected(items=deselected)
    items[:] = selected


def pytest_terminal_summary(
    terminalreporter: pytest.TerminalReporter,
    exitstatus: int,
    config: pytest.Config,
) -> None:
    del exitstatus
    count = getattr(config, "_kawaii_pilot_item_count", 0)
    terminalreporter.section("Kawaii test profile")
    if config.getoption("--pilot-profile"):
        terminalreporter.write_line(
            f"pilot integration profile REQUESTED: {count} pilot test(s) selected; "
            "fixture identity preflight is mandatory"
        )
    else:
        terminalreporter.write_line(
            f"default fixture-independent profile: {count} pilot test(s) NOT RUN "
            "and visibly deselected; use --pilot-profile to request them"
        )


@pytest.fixture(scope="session")
def pilot_fixture_root(pytestconfig: pytest.Config) -> Path:
    configured = pytestconfig.getoption("--pilot-fixture-root")
    environment = os.environ.get("KAWAII_PILOT_FIXTURE_ROOT")
    root = configured or (Path(environment) if environment else REPOSITORY_ROOT)
    try:
        verified = validate_pilot_fixtures(root)
    except PilotFixtureValidationError as error:
        pytest.fail(str(error), pytrace=False)
    terminal = pytestconfig.pluginmanager.get_plugin("terminalreporter")
    if terminal is not None:
        terminal.write_line(
            f"pilot fixture preflight: VERIFIED {verified.corpus_id} at {verified.root} "
            f"({len(verified.files)} files)"
        )
    return verified.root
