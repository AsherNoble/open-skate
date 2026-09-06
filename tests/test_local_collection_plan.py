"""The no-Modal dataset plan stays balanced and metadata-complete."""
from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path

from bench.collect_local import make_plan
from opensk.sim.appearance import APPEARANCE_PRESETS
from opensk.sim.model.parks import PARKS


def test_default_local_plan_balances_every_domain(tmp_path):
    plan = make_plan(out=tmp_path, parks=list(PARKS),
                     appearances=list(APPEARANCE_PRESETS),
                     shards_per_domain=4, episodes_per_shard=64, seed=1000)
    counts = Counter((item["park"], item["appearance"]) for item in plan)
    assert len(plan) == 36
    assert set(counts.values()) == {4}
    assert set(counts) == {(park, appearance) for park in PARKS
                           for appearance in APPEARANCE_PRESETS}
    assert len({item["seed"] for item in plan}) == len(plan)
    assert len({item["path"] for item in plan}) == len(plan)
    assert sum(item["episodes"] for item in plan) == 2304


def test_modal_collector_exposes_park_and_appearance_without_launching_it():
    source = Path("bench/collect_modal.py").read_text()
    tree = ast.parse(source)
    functions = {node.name: node for node in tree.body
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    collect_args = [arg.arg for arg in functions["collect"].args.args]
    main_args = [arg.arg for arg in functions["main"].args.args]
    assert collect_args == ["batch", "park", "appearance"]
    assert main_args[:3] == ["batch", "park", "appearance"]
    assert "park=env.park_name" in source
    assert "appearance=env.appearance" in source
