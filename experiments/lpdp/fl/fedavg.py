"""M5: FedAvg simulation — each set-A user is one client."""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from lpdp.models.gru_next_poi import GRUNextPOI
from lpdp.train.centralised import TrainResult, evaluate_model
from lpdp.train.windows import Window, WindowDataset, collate_windows


@dataclass
class FLResult(TrainResult):
    comm_bytes_per_round: int
    rounds: int
    epsilon: float | None = None


def _state_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _load(model: nn.Module, state: dict[str, torch.Tensor], device) -> None:
    model.load_state_dict({k: v.to(device) for k, v in state.items()})


def _average(states: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    keys = states[0].keys()
    out = {}
    n = float(len(states))
    for k in keys:
        acc = None
        for s in states:
            acc = s[k].float() if acc is None else acc + s[k].float()
        out[k] = (acc / n).to(states[0][k].dtype)
    return out


def _local_train(
    model: nn.Module,
    windows: list[Window],
    cfg: dict,
    device,
    epochs: int,
) -> dict[str, torch.Tensor]:
    if not windows:
        return _state_cpu(model)
    opt = torch.optim.SGD(model.parameters(), lr=cfg["fl"]["lr"])
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    loader = DataLoader(
        WindowDataset(windows, seq_len=cfg["model"]["seq_len"]),
        batch_size=min(len(windows), cfg["model"]["batch_size"]),
        shuffle=True,
        collate_fn=collate_windows,
    )
    model.train()
    for _ in range(epochs):
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
    return _state_cpu(model)


def train_fedavg(
    n_pois: int,
    windows_by_user: dict[int, list[Window]],
    val_windows: list[Window],
    cfg: dict,
    seed: int = 0,
) -> FLResult:
    rng = np.random.default_rng(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mcfg = cfg["model"]
    fl = cfg["fl"]
    model = GRUNextPOI(
        n_pois=n_pois,
        poi_dim=mcfg["poi_dim"],
        hour_dim=mcfg["hour_dim"],
        hidden=mcfg["hidden"],
    ).to(device)
    clients = [u for u, w in windows_by_user.items() if w]
    n_per_round = min(int(fl["clients_per_round"]), max(len(clients), 1))
    n_params = sum(p.numel() for p in model.parameters())
    comm = n_params * 4 * n_per_round
    history: list[dict] = []
    t0 = time.time()
    rounds = int(fl["rounds"])
    for r in range(rounds):
        chosen = rng.choice(clients, size=n_per_round, replace=False) if clients else []
        global_state = _state_cpu(model)
        local_states = []
        for uid in chosen:
            local = copy.deepcopy(model)
            _load(local, global_state, device)
            state = _local_train(local, windows_by_user[int(uid)], cfg, device, int(fl["local_epochs"]))
            local_states.append(state)
        if local_states:
            _load(model, _average(local_states), device)
        val = evaluate_model(model, val_windows, batch_size=mcfg["batch_size"])
        history.append({"round": r, **val})
    return FLResult(
        model=model,
        history=history,
        best_acc10=max((h.get("acc10", 0.0) for h in history), default=0.0),
        seconds=time.time() - t0,
        n_params=n_params,
        comm_bytes_per_round=int(comm),
        rounds=rounds,
    )
