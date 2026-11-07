"""The Arena's persona file must parse: a YAML typo once crashed a full run before the first persona."""

import yaml

from evals.run_evals import PERSONAS


def test_personas_parse_and_are_complete():
    personas = yaml.safe_load(PERSONAS.read_text(encoding="utf-8"))
    ids = [p["id"] for p in personas]
    assert len(ids) == len(set(ids)) >= 30
    for p in personas:
        assert {"category", "persona", "goal", "opening", "expect"} <= set(p), p["id"]
