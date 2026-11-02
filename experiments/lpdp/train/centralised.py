"""Non-private centralised training for the GRU next-POI model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader

from lpdp.eval.metrics import ranking_metrics
from lpdp.models.gru_next_poi import GRUNextPOI
from lpdp.train.windows import Window, WindowDataset, collate_windows


@dataclass
class TrainResult:
    model: GRUNextPOI
    history: list[dict]
    best_acc10: float
    seconds: float
    n_params: int


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_model(model: nn.Module, windows: list[Window], batch_size: int = 64) -> dict[str, float]:
    if not windows:
        return {"acc1": 0.0, "acc5": 0.0, "acc10": 0.0, "mrr": 0.0}
    device = next(model.parameters()).device
    loader = DataLoader(
        WindowDataset(windows, seq_len=1),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_windows,
    )
    model.eval()
    acc = {"acc1": 0.0, "acc5": 0.0, "acc10": 0.0, "mrr": 0.0}
    n = 0
    with torch.no_grad():
        for batch in loader:
            logits = model(
                batch["poi_ids"].to(device),
                batch["hours"].to(device),
                batch["lengths"].to(device),
            )
            m = ranking_metrics(logits, batch["targets"].to(device))
            b = batch["targets"].size(0)
            for k, v in m.items():
                acc[k] += v * b
            n += b
    return {k: v / max(n, 1) for k, v in acc.items()}


def per_window_loss(model: nn.Module, windows: list[Window], batch_size: int = 64) -> list[tuple[int, float]]:
    """Return (user_id, mean_ce_loss) contributions per window."""
    if not windows:
        return []
    device = next(model.parameters()).device
    loader = DataLoader(
        WindowDataset(windows, seq_len=1),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_windows,
    )
    loss_fn = nn.CrossEntropyLoss(reduction="none", ignore_index=0)
    model.eval()
    rows: list[tuple[int, float]] = []
    with torch.no_grad():
        for batch in loader:
            logits = model(
                batch["poi_ids"].to(device),
                batch["hours"].to(device),
                batch["lengths"].to(device),
            )
            losses = loss_fn(logits, batch["targets"].to(device))
            for uid, loss in zip(batch["users"].tolist(), losses.cpu().tolist(), strict=False):
                rows.append((int(uid), float(loss)))
    return rows


def train_centralised(
    n_pois: int,
    train_windows: list[Window],
    val_windows: list[Window],
    cfg: dict,
    use_dp_rnn: bool = False,
) -> TrainResult:
    import time

    device = _device()
    mcfg = cfg["model"]
    model = GRUNextPOI(
        n_pois=n_pois,
        poi_dim=mcfg["poi_dim"],
        hour_dim=mcfg["hour_dim"],
        hidden=mcfg["hidden"],
        use_dp_rnn=use_dp_rnn,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=mcfg["lr"])
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    loader = DataLoader(
        WindowDataset(train_windows, seq_len=mcfg["seq_len"]),
        batch_size=mcfg["batch_size"],
        shuffle=True,
        collate_fn=collate_windows,
    )
    best_acc = -1.0
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    patience = int(mcfg.get("patience", 4))
    bad = 0
    history: list[dict] = []
    t0 = time.time()
    for epoch in range(int(mcfg["max_epochs"])):
        model.train()
        total = 0.0
        n = 0
        for batch in loader:
            opt.zero_grad(set_to_none=True)
            logits = model(
                batch["poi_ids"].to(device),
                batch["hours"].to(device),
                batch["lengths"].to(device),
            )
            loss = loss_fn(logits, batch["targets"].to(device))
            loss.backward()
            opt.step()
            total += float(loss.item()) * batch["targets"].size(0)
            n += batch["targets"].size(0)
        val = evaluate_model(model, val_windows, batch_size=mcfg["batch_size"])
        acc10 = val["acc10"]
        history.append({"epoch": epoch, "train_loss": total / max(n, 1), **val})
        if acc10 > best_acc:
            best_acc = acc10
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    return TrainResult(
        model=model,
        history=history,
        best_acc10=best_acc,
        seconds=time.time() - t0,
        n_params=n_params,
    )
