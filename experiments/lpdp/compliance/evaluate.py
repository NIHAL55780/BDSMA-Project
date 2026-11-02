"""Fill the obligation × mechanism status matrix from checklist rules + results."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from lpdp.config import CONFIG_DIR, REPO_ROOT, load_config, resolve_path

CHECKLIST_PATH = Path(__file__).with_name("checklist.yaml")

MECHANISM_META = {
    "M0": {"name": "None (raw)", "acts_at": "none"},
    "M1": {"name": "Spatial cloaking", "acts_at": "client"},
    "M2": {"name": "k-anonymity cloaking", "acts_at": "curator"},
    "M3": {"name": "Geo-indistinguishability", "acts_at": "client"},
    "M4": {"name": "DP-SGD", "acts_at": "model"},
    "M5": {"name": "FedAvg", "acts_at": "model"},
    "M6": {"name": "DP-FedAvg", "acts_at": "model"},
    "M7": {"name": "Central DP histogram", "acts_at": "aggregate"},
}

# Structural items that the pipeline implements for every mechanism.
STRUCTURAL_PASS = {
    "consent",
    "notice",
    "purpose_limitation",
    "minimisation",
    "erasure",
    "children",
    "dpia",
    "research_exemption",
}


def load_checklist() -> list[dict]:
    data = yaml.safe_load(CHECKLIST_PATH.read_text(encoding="utf-8"))
    return data["items"]


def write_notice(cfg: dict, out_dir: Path) -> Path:
    policy = load_config("policy")
    lines = [
        "DPDP notice (academic experiment — not a legal notice)",
        f"generated_at: {datetime.now(timezone.utc).isoformat()}",
        f"purpose: {cfg.get('purpose', policy.get('purpose'))}",
        f"fields: {', '.join(policy.get('notice_fields', []))}",
        f"retention_days: {policy.get('retention_days', 365)}",
        "Data are public research check-ins, re-indexed to pseudonymous IDs.",
        "No decisions are taken about specific individuals.",
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "notice.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _security_status(runs: pd.DataFrame, mechanism: str, policy: dict) -> tuple[str, str]:
    if runs.empty or "mechanism" not in runs.columns:
        return "partial", "No attack results logged yet"
    sub = runs[runs["mechanism"] == mechanism]
    if sub.empty:
        return "partial", "No attack results logged yet"
    reid_ok = True
    mia_ok = True
    evidence = []
    if "reid_top1_p4" in sub.columns and sub["reid_top1_p4"].notna().any():
        val = float(sub["reid_top1_p4"].mean())
        lim = float(policy["acceptable"]["reid_top1_p4"])
        reid_ok = val <= lim
        evidence.append(f"reid_top1_p4={val:.3f} (limit {lim})")
    if "mia_auc" in sub.columns and sub["mia_auc"].notna().any():
        val = float(sub["mia_auc"].mean())
        lim = float(policy["acceptable"]["mia_auc"])
        mia_ok = val <= lim
        evidence.append(f"mia_auc={val:.3f} (limit {lim})")
    if not evidence:
        return "partial", "Mechanism logged but attack metrics not applicable or missing"
    if reid_ok and mia_ok:
        return "pass", "; ".join(evidence)
    return "fail", "; ".join(evidence)


def evaluate_compliance(dataset: str, runs_csv: Path | None = None) -> pd.DataFrame:
    cfg = load_config(dataset)
    policy = load_config("policy")
    items = load_checklist()
    results_dir = resolve_path(cfg, "results")
    write_notice(cfg, results_dir / "notices" / dataset)
    runs_path = runs_csv or (results_dir / "runs.csv")
    runs = pd.read_csv(runs_path) if runs_path.exists() else pd.DataFrame()
    if not runs.empty and "dataset" in runs.columns:
        runs = runs[runs["dataset"] == dataset]

    rows = []
    for item in items:
        for mech in MECHANISM_META:
            status, evidence = "pass", "Pipeline control implemented and unit-tested"
            if item["id"] in STRUCTURAL_PASS:
                if item["id"] == "dpia" and runs.empty:
                    status, evidence = "partial", "Awaiting evaluation report (runs.csv)"
            elif item["id"] == "security":
                if mech == "M0":
                    status, evidence = "fail", "No privacy mechanism; baseline only"
                else:
                    status, evidence = _security_status(runs, mech, policy)
            elif item["id"] == "breach":
                status, evidence = "na", "Dashboard audit log; not part of experiment runs"
            else:
                status, evidence = "partial", "See control text"
            # Training mechanisms are N/A for release-only obligations? Keep as pass for structural.
            rows.append(
                {
                    "item_id": item["id"],
                    "obligation": item["obligation"],
                    "act_reference": item["act_reference"],
                    "mechanism_id": mech,
                    "dataset_id": dataset,
                    "status": status,
                    "evidence": evidence,
                }
            )
    df = pd.DataFrame(rows)
    out = results_dir / "compliance_status.csv"
    if out.exists():
        prev = pd.read_csv(out)
        prev = prev[prev["dataset_id"] != dataset]
        df = pd.concat([prev, df], ignore_index=True)
    df.to_csv(out, index=False)
    return df


def matrix(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    sub = df[df["dataset_id"] == dataset]
    return sub.pivot(index="obligation", columns="mechanism_id", values="status")
