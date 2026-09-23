"""The interface's defaults, citations and counts say what the design says.

* Defaults are the frozen annotator set and the Layer-2 arbitrator of record,
  read from `scripts/run_phase2.py`; the stub is opt-in (BUG-2026-09-23-G).
* A single annotation is cited by the fields the control console stores, plus
  the version it was shown under (BUG-2026-09-23-H).
* Counts of triples are labelled triples (BUG-2026-09-23-I).

There is no JS test runner here, so the front-end halves are guarded on the
source, as in `test_ui_graph_url.py`. Hermetic: no Ollama, no SBERT.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

STATIC = ROOT / "ui" / "static"


# ---------------------------------------------------------------- defaults

def test_the_layer2_default_of_the_cli_is_the_arbitrator_of_record():
    from scripts import run_phase2
    assert run_phase2.LAYER2_OF_RECORD == "cloud:qwen3.6:35b"   # decided 2026-09-22
    src = (ROOT / "scripts" / "run_phase2.py").read_text(encoding="utf-8")
    assert 'p.add_argument("--layer2-model", default=LAYER2_OF_RECORD' in src


def test_the_cloud_fallback_list_is_the_annotator_set_of_record():
    from scripts.run_phase2 import ANNOTATORS_OF_RECORD
    from src.providers import CLOUD_FALLBACK, CLOUD_PREFIX
    assert tuple(CLOUD_PREFIX + t for t in CLOUD_FALLBACK) == ANNOTATORS_OF_RECORD
    assert "gemma4:latest" not in CLOUD_FALLBACK     # the 8B, not the 31B


@pytest.fixture()
def client():
    ui_app = pytest.importorskip("ui.app")
    ui_app.app.config["TESTING"] = True
    with ui_app.app.test_client() as c:
        yield c


def test_the_page_defaults_to_the_models_of_record_and_leaves_the_stub_off(client):
    from scripts.run_phase2 import ANNOTATORS_OF_RECORD, LAYER2_OF_RECORD
    html = client.get("/").get_data(as_text=True)
    assert "qwen3.5:4b" not in html and "gemma3:4b" not in html and "phi4" not in html
    rec = json.loads(re.search(r"window\.__vaRecord = (\{.*?\});", html).group(1))
    assert rec == {"annotators": list(ANNOTATORS_OF_RECORD), "layer2": LAYER2_OF_RECORD}
    for field in ("p1-models", "exp-models"):
        m = re.search(rf'id="{field}" value="([^"]*)"', html)
        assert m and m.group(1).split(",") == list(ANNOTATORS_OF_RECORD), field
    for box in ("p2-skip-layer2", "exp-skip-layer2"):
        tag = re.search(rf'<input[^>]*id="{box}"[^>]*>', html).group(0)
        assert "checked" not in tag, f"{box} is ticked by default"
    matrix = html[html.index('rm-model-checks'):html.index('rm-add-combo')]
    assert re.findall(r'value="([^"]+)" checked', matrix) == list(ANNOTATORS_OF_RECORD)


def test_a_phase2_job_that_names_no_arbitrator_gets_the_one_of_record(tmp_path):
    from scripts.run_phase2 import LAYER2_OF_RECORD
    from src import reextract_worker as rw
    seen = {}

    def fake_run(doc_id, **kw):
        seen.update(kw)
        out = tmp_path / f"{doc_id}.json"
        out.write_text("{}", encoding="utf-8")
        return out

    job = rw.Job(job_id="t", kind="phase2", doc_ids=["d"], phase2_params={})
    rw._run_phase2(job, run_fn=fake_run)
    assert seen["layer2_model"] == LAYER2_OF_RECORD and seen["skip_layer2"] is False


def test_the_js_carries_no_retired_model_as_a_default():
    for name in ("app.js", "workflow.js"):
        src = (STATIC / name).read_text(encoding="utf-8")
        # FALLBACK_LOCAL lists what a local Ollama typically has; it is no
        # longer anyone's default selection
        body = "\n".join(ln for ln in src.splitlines() if "FALLBACK_LOCAL =" not in ln)
        assert "qwen3.5:4b" not in body, name
        assert "skip_layer2: true" not in body, name


# --------------------------------------------------------------- citations

def test_an_interface_citation_keeps_its_group_id_and_version(tmp_path, monkeypatch):
    ui_app = pytest.importorskip("ui.app")
    monkeypatch.setattr(ui_app, "PROMPTS", tmp_path)
    (tmp_path / "extract_v1.md").write_text("R1\n", encoding="utf-8")
    ui_app.app.config["TESTING"] = True
    cite = {"kind": "fact", "doc_id": "train-000035", "section": "preamble.recitals[2]",
            "annotator": "cloud:qwen3.6:35b", "group_id": "a7273979ef0dface",
            "version": "v2", "label": "granularity", "excerpt": "The Commission ..."}
    with ui_app.app.test_client() as c:
        r = c.post("/api/guideline_log/v3",
                   json={"add": {"op": "modify", "rule": "R7", "reason": "why",
                                 "cites": [cite]}})
    assert r.status_code == 200
    stored = r.get_json()["entries"][0]["cites"][0]
    assert stored == cite


def test_the_single_annotation_button_sends_the_identifying_fields():
    src = (STATIC / "app.js").read_text(encoding="utf-8")
    block = src[src.index('_citeButton("❝"'):]
    block = block[:block.index('"Cite this single annotation')]
    assert "group_id: f.group_id || f.fact_id" in block     # as make_baseline_reader
    assert "version:" in block
    # every citation the interface makes says which version it was read off
    assert src.count("version: window.__variant") >= 4
    glog = (STATIC / "guideline_log.js").read_text(encoding="utf-8")
    key = glog[glog.index("function citeKey("):]
    key = key[:key.index("}")]
    assert "group_id" in key and "version" in key


# ------------------------------------------------------------------ counts

def test_counts_of_triples_are_labelled_triples():
    stats = (STATIC / "analysis.js").read_text(encoding="utf-8")
    assert "Triples per annotator" in stats and "Triples per section" in stats
    assert "Fact count per annotator" not in stats and "Facts per section" not in stats
    app = (STATIC / "app.js").read_text(encoding="utf-8")
    assert '.join(" / ") + " triples"' in app            # the segment header
    assert '.join(" / ") + " facts"' not in app
