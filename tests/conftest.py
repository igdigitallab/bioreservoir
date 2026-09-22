"""Shared pytest fixtures. `anyio_backend` restricts the `anyio` pytest plugin (a transitive
dependency of fastapi/httpx/starlette, already installed — no separate `pytest-asyncio` needed) to
the `asyncio` backend only, for every `@pytest.mark.anyio` test in `bioreservoir.live`'s suite
(`trio` is never installed here, so leaving both backends enabled would just fail to collect).
"""

import pytest


@pytest.fixture
def anyio_backend():
    return "asyncio"
