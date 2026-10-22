"""Builders for the five suites whose data is fully local after download:
JevBench public-231, Kev (6 suite-splits), Nimble holdout, VitaminC-dev-599, MASSIVE-en-350.

Sources are read as data only (no downloaded code is imported or executed).
"""

from __future__ import annotations

import collections
import hashlib
import json
import tarfile

from eval.parity.common import RAW, read_jsonl

# ----------------------------------------------------------------------------- JevBench public 231

JEVBENCH_DIR = RAW / "jevbench-repo" / "datasets" / "public"


def build_jevbench() -> tuple[list[dict], dict]:
    items = []
    for tier in ("original", "easy", "hard"):
        for r in read_jsonl(JEVBENCH_DIR / f"{tier}.jsonl"):
            q = {"type": r["question"]["type"], "instructions": r["question"]["instructions"]}
            if r["question"].get("criteria") is not None:
                q["criteria"] = r["question"]["criteria"]
            t = q["type"]
            if t == "noul":
                gold = {"yes": True, "no": False}[r["expected"]]
            elif t == "score":
                gold = int(r["expected"])
            else:
                gold = r["expected"]
            items.append({"uid": r["id"], "suite": "jevbench", "task": r["family"], "group": r.get("group"),
                          "rule": "per_question",
                          "requests": [{"state": r["state"], "questions": {"decision": q}}],
                          "golds": {"0:decision": gold},
                          "meta": {"tier": tier, "labels": r["labels"]}})
    info = {"source": "github.com/fstandhartinger/jevbench datasets/public/{original,easy,hard}.jsonl",
            "commit": (RAW / "jevbench-repo" / "COMMIT").read_text().strip(), "license": "MIT",
            "n_items": len(items), "families": len({i["task"] for i in items})}
    return items, info


# ----------------------------------------------------------------------------- Kev

KEV_SPLITS = [("decision-v7", "evals/v7/decision-v7"), ("transfer-v4", "evals/v4/transfer-v4"),
              ("transfer-v9", "evals/v9/transfer-v9")]


def build_kev() -> tuple[list[dict], dict]:
    items = []
    base = RAW / "kev-repo"
    counts = {}
    for name, path in KEV_SPLITS:
        for split in ("development", "test"):
            rows = read_jsonl(base / path / f"{split}.jsonl")
            counts[f"{name}/{split}"] = len(rows)
            for i, r in enumerate(rows):
                meta = r.get("_meta", {})
                qs, golds = {}, {}
                for qid, q in r["questions"].items():
                    qq = {k: v for k, v in q.items() if k not in ("label", "src")}
                    qs[qid] = qq
                    golds[f"0:{qid}"] = q["label"]
                items.append({"uid": f"{name}/{split}/{i:05d}", "suite": "kev", "task": f"{name}/{split}",
                              "group": meta.get("group_id"), "rule": "per_question",
                              "requests": [{"state": r["state"], "questions": qs}], "golds": golds,
                              "meta": {"variant": meta.get("variant"), "source": meta.get("source"),
                                       "src": {qid: q.get("src") for qid, q in r["questions"].items()},
                                       "kev_id": meta.get("id"),
                                       # Kev clean accuracy: clean variants of knowable records only
                                       "scored": meta.get("variant") == "clean" and meta.get("source") != "unknowable"}})
    info = {"source": "github.com/jaredpalmer/kev evals/{v7/decision-v7,v4/transfer-v4,v9/transfer-v9}/{development,test}.jsonl",
            "commit": (base / "COMMIT").read_text().strip(), "license": "Apache-2.0", "records": counts,
            "n_items": len(items)}
    return items, info


# ----------------------------------------------------------------------------- Nimble holdout

NIMBLE_EVAL = RAW / "nimble-repo" / "data" / "eval.jsonl"
NIMBLE_SHA = "8e9e48b8de5206593912ae01ddc95bd77e40ad2ecf4c9292c1711290eca0d896"
NEOHORSE_TOKENIZER = RAW / "neohorse" / "tokenizer" / "tokenizer.json"
# NeoHorse's 282-of-324 selection: 384-token state limit (NeoHorse-Jev-4B tokenizer), <= 26 candidates,
# whole families. Exact ids were not published. Rendering the state as json.dumps(ensure_ascii=False)
# reproduces 280 records with the exact choice (116) and score (54) counts; it drops two Noul families,
# of which NeoHorse kept one (112 Noul). We keep the Noul family with the smallest overshoot (392 tokens),
# which reproduces the published 282 = 116 Choice + 112 Noul + 54 Score composition exactly.
NIMBLE_EXTRA_KEEP = {"scale-diverse-273-001"}


def build_nimble() -> tuple[list[dict], dict]:
    import hashlib as _h
    raw_bytes = NIMBLE_EVAL.read_bytes()
    sha = _h.sha256(raw_bytes).hexdigest()
    if sha != NIMBLE_SHA:
        raise RuntimeError(f"Nimble holdout sha mismatch {sha}")
    rows = read_jsonl(NIMBLE_EVAL)
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(NEOHORSE_TOKENIZER))

    def ntok(state):
        s = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
        return len(tok.encode(s, add_special_tokens=False).ids)

    fam_ok = collections.defaultdict(lambda: True)
    ntoks = {}
    for r in rows:
        q = next(iter(r["input"]["questions"].values()))
        n = ntok(r["input"]["state"])
        ntoks[r["id"]] = n
        ncand = 2 if q["type"] == "noul" else len(q["criteria"])
        if n > 384 or ncand > 26:
            fam_ok[r["family"]] = False
        else:
            fam_ok[r["family"]]
    items = []
    for r in rows:
        (qid, q), = r["input"]["questions"].items()
        tgt = r["reference"]["target"]
        if q["type"] == "noul" and not isinstance(tgt, bool):
            tgt = {"true": True, "false": False}[str(tgt).lower()]
        in282 = fam_ok[r["family"]] or r["family"] in NIMBLE_EXTRA_KEEP
        items.append({"uid": r["id"], "suite": "nimble", "task": r["domain"], "group": r["family"],
                      "rule": "per_question", "requests": [r["input"]], "golds": {f"0:{qid}": tgt},
                      "meta": {"in_282": in282, "variant": r.get("variant"), "source_family": r.get("source_family"),
                               "state_tokens_neohorse": ntoks[r["id"]]}})
    comp = collections.Counter(next(iter(i["requests"][0]["questions"].values()))["type"] for i in items if i["meta"]["in_282"])
    info = {"source": "github.com/bespokelabsai/nimble data/eval.jsonl (324-record frozen holdout)",
            "commit": (RAW / "nimble-repo" / "COMMIT").read_text().strip(), "sha256": sha,
            "license": "no LICENSE file in repo (all rights reserved by default); model card is Apache-2.0. Internal evaluation only.",
            "n_items": len(items), "n_282": sum(i["meta"]["in_282"] for i in items), "composition_282": dict(comp),
            "selection_note": "282 reconstructed (see NIMBLE_EXTRA_KEEP); composition matches NeoHorse exactly, ids unverifiable"}
    return items, info


# ----------------------------------------------------------------------------- VitaminC / MASSIVE (Nimble public suite)

VITAMINC_CRITERIA = {
    "SUPPORTS": "The evidence states the claim, or the claim follows directly from the evidence.",
    "REFUTES": "The evidence states the opposite of the claim, or the claim is directly contradicted by it.",
    "NOT ENOUGH INFO": "The evidence neither establishes nor contradicts the claim. Settling it would need a"
                       " fact the evidence does not supply.",
}
VITAMINC_INSTRUCTIONS = ("Decide how the evidence bears on the claim. Judge only from the evidence text, and not"
                         " from outside knowledge about the subject.")
MASSIVE_INSTRUCTIONS = ("Which assistant domain should handle this utterance? Judge from the request itself, not"
                        " from how it is phrased.")
MASSIVE_CRITERIA = {
    "alarm": "Setting, changing, or querying alarms.",
    "audio": "Volume, mute, or audio output settings.",
    "calendar": "Events, meetings, reminders on a calendar.",
    "cooking": "Recipes and cooking instructions.",
    "datetime": "The current time, date, or time conversions.",
    "email": "Reading, sending, or checking email and contacts.",
    "general": "Small talk, jokes, greetings, or general assistant control such as repeat or confirm.",
    "iot": "Controlling lights, plugs, heating, cleaners, coffee machines, or wemo devices.",
    "lists": "Creating, querying, or removing list items.",
    "music": "Music preferences, likes, or music settings, not playing a specific item.",
    "news": "News headlines or updates.",
    "play": "Playing a specific song, podcast, radio, audiobook, or game.",
    "qa": "Factual questions, definitions, math, currency, stock prices.",
    "recommendation": "Suggestions for events, movies, or locations.",
    "social": "Social media posts or queries.",
    "takeaway": "Food orders and delivery status.",
    "transport": "Taxis, tickets, traffic, or travel directions.",
    "weather": "Weather forecasts or conditions.",
}


def _nimble_record(identifier, domain, family, state, criteria, instructions, target, reference):
    """Byte-faithful re-implementation of nimble.datasets.public_records.record_for (choice only)."""
    question = {"type": "choice", "instructions": instructions, "criteria": dict(criteria)}
    return {"id": identifier, "split": "validation", "domain": domain, "family": family, "source_family": family,
            "input": {"state": state, "questions": {"decision": question}},
            "reference": {"target": target, "human_reviewed": True, **reference}}


def _nimble_line(record) -> str:
    text = json.dumps(record, ensure_ascii=False, allow_nan=False)
    for sep in ("\u2028", "\u2029", "\u0085"):
        text = text.replace(sep, "\\u%04x" % ord(sep))
    return text + "\n"


def _select_by_manifest(records: list[dict], manifest: dict, expect_sha: str) -> tuple[list[dict], bool, str]:
    by_id = {r["id"]: r for r in records}
    chosen = [by_id[i] for i in manifest["ids"]]
    payload = "".join(_nimble_line(r) for r in chosen).encode()
    sha = hashlib.sha256(payload).hexdigest()
    if sha != expect_sha:  # try upstream order
        idset = set(manifest["ids"])
        chosen2 = [r for r in records if r["id"] in idset]
        sha2 = hashlib.sha256("".join(_nimble_line(r) for r in chosen2).encode()).hexdigest()
        if sha2 == expect_sha:
            return chosen2, True, sha2
    return chosen, sha == expect_sha, sha


def _to_items(records, suite):
    items = []
    for r in records:
        items.append({"uid": r["id"], "suite": suite, "task": r["domain"], "group": r["family"], "rule": "per_question",
                      "requests": [r["input"]], "golds": {"0:decision": r["reference"]["target"]},
                      "meta": {k: v for k, v in r["reference"].items() if k not in ("target",)}})
    return items


def build_vitaminc() -> tuple[list[dict], dict]:
    src = RAW / "vitaminc" / "vitaminc" / "dev.jsonl"
    manifest = json.loads((RAW / "nimble-repo" / "docs/assets/public-benchmarks/subsets/vitaminc-dev-manifest.json").read_text())
    records = []
    for raw in read_jsonl(src):
        records.append(_nimble_record(
            "vitaminc-" + raw["unique_id"], "vitaminc-" + raw["revision_type"], raw["case_id"],
            {"evidence": raw["evidence"], "claim": raw["claim"]}, VITAMINC_CRITERIA, VITAMINC_INSTRUCTIONS,
            raw["label"], {"source": "vitaminc", "page": raw.get("page", ""), "revision_type": raw["revision_type"],
                           "wiki_revision_id": raw.get("wiki_revision_id", "")}))
    chosen, ok, sha = _select_by_manifest(records, manifest, manifest["dataset_sha256"])
    info = {"source": "VitaminC dev (talschuster.github.io/static/vitaminc.zip), ids from bespokelabsai/nimble vitaminc-dev-manifest.json",
            "license": "CC BY-SA 3.0", "n_items": len(chosen), "dataset_sha256": sha,
            "byte_identical_to_nimble_subset": ok, "expected_sha256": manifest["dataset_sha256"]}
    return _to_items(chosen, "vitaminc"), info


def build_massive() -> tuple[list[dict], dict]:
    src = RAW / "massive" / "amazon-massive-dataset-1.1.tar.gz"
    manifest = json.loads((RAW / "nimble-repo" / "docs/assets/public-benchmarks/subsets/massive-en-US-manifest.json").read_text())
    with tarfile.open(src, "r:gz") as archive:
        member = [m for m in archive.getmembers() if m.name.endswith("/data/en-US.jsonl")][0]
        lines = archive.extractfile(member).read().decode().split("\n")
    records = []
    for line in lines:
        if not line.strip():
            continue
        raw = json.loads(line)
        if raw.get("partition") != "test" or raw.get("locale") != "en-US":
            continue
        utt = raw["utt"]
        if not isinstance(utt, str) or not utt.strip():
            continue
        records.append(_nimble_record(
            f"massive-{raw['id']}", "massive-en-US", f"massive-{raw['id']}", {"utterance": utt, "locale": "en-US"},
            MASSIVE_CRITERIA, MASSIVE_INSTRUCTIONS, raw["scenario"],
            {"source": "massive", "locale": "en-US", "intent": raw.get("intent"), "massive_id": str(raw["id"])}))
    chosen, ok, sha = _select_by_manifest(records, manifest, manifest["dataset_sha256"])
    info = {"source": "MASSIVE 1.1 test en-US (amazon-massive-dataset-1.1.tar.gz), ids from bespokelabsai/nimble massive-en-US-manifest.json",
            "license": "CC BY 4.0", "n_items": len(chosen), "dataset_sha256": sha,
            "byte_identical_to_nimble_subset": ok, "expected_sha256": manifest["dataset_sha256"]}
    return _to_items(chosen, "massive"), info
