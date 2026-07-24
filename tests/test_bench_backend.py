"""`virtual_mode` on the coding-agent backend, and the bench graph that needs it off.

KODA jails its file tools to the project root by default: a model-supplied
``/etc/hosts`` is resolved under ``<root>/etc/hosts``. That is right for a coding
agent pointed at someone's project, and wrong inside a Terminal-Bench container
where tasks address ``/etc``, ``/usr/local`` and ``/app`` by absolute path — the
agent gets "file not found" for files that plainly exist, which is a silent,
plausible-looking failure rather than a loud one. See bench/graph.py.
"""

from __future__ import annotations

from pathlib import Path

from coding_agent.backend import build_backend


def _read(backend, path: str):
    """Return (content, error) from a deepagents ReadResult."""
    result = backend.read(path)
    data = getattr(result, "file_data", None) or {}
    return data.get("content"), getattr(result, "error", None)


def test_virtual_mode_jails_absolute_paths(tmp_path: Path) -> None:
    """The default: an absolute path outside root resolves *inside* root."""
    outside = tmp_path / "outside"
    outside.mkdir()
    secret = outside / "real.txt"
    secret.write_text("real content")

    root = tmp_path / "project"
    root.mkdir()
    backend = build_backend(root, virtual_mode=True)

    content, error = _read(backend, str(secret))
    assert content is None
    assert error is not None  # jailed away — the file "does not exist"


def test_virtual_mode_off_reaches_the_real_filesystem(tmp_path: Path) -> None:
    """Bench mode: the same absolute path resolves to the real file."""
    outside = tmp_path / "outside"
    outside.mkdir()
    secret = outside / "real.txt"
    secret.write_text("real content")

    root = tmp_path / "project"
    root.mkdir()
    backend = build_backend(root, virtual_mode=False)

    content, error = _read(backend, str(secret))
    assert error is None
    assert content == "real content"


def test_virtual_mode_off_still_anchors_relative_paths(tmp_path: Path) -> None:
    """`root` keeps its job for relative paths — it only stops being a jail."""
    root = tmp_path / "project"
    root.mkdir()
    (root / "config.yaml").write_text("key: value")

    backend = build_backend(root, virtual_mode=False)

    content, error = _read(backend, "config.yaml")
    assert error is None
    assert content == "key: value"


async def test_bench_graph_disables_the_human_approval_gate(monkeypatch) -> None:
    """Unattended runs must gate nothing.

    KODA gates mutating tools via deepagents' HumanInTheLoopMiddleware, which
    calls LangGraph's ``interrupt()``. In a benchmark container no one can
    answer, so the graph halts at the first ``execute`` and ``ainvoke`` returns
    a two-message state with ``__interrupt__`` set — a silent zero, not a crash.
    """
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bench"))
    import graph as bench_graph

    captured: dict = {}

    async def fake_build_agent(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(bench_graph, "build_agent", fake_build_agent)
    await bench_graph.make_graph({"configurable": {"model": "ollama:glm-5.2"}})

    assert captured["interrupt_on"] == {}, "bench runs must not gate any tool"
    assert captured["virtual_mode"] is False, "bench runs need the real filesystem"
    assert captured["model"] == "ollama:glm-5.2"


def test_bench_graph_resolves_model_from_harbor_config() -> None:
    """Harbor passes `-m` through `configurable.model`; it is KODA's spec format."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bench"))
    from graph import DEFAULT_MODEL, _resolve_model

    assert _resolve_model({"configurable": {"model": "ollama:glm-5.2"}}) == "ollama:glm-5.2"
    assert _resolve_model({}) == DEFAULT_MODEL
    assert _resolve_model(None) == DEFAULT_MODEL
