"""M6: DP-FedAvg — clip client updates, add Gaussian noise at the server."""

from __future__ import annotations

import copy
import time

import numpy as np
import torch

from lpdp.fl.accounting import report_epsilon
from lpdp.fl.fedavg import FLResult, _load, _local_train, _state_cpu
from lpdp.models.gru_next_poi import GRUNextPOI
from lpdp.train.centralised import evaluate_model
from lpdp.train.windows import Window


def _clip_update(
    new_state: dict[str, torch.Tensor],
    old_state: dict[str, torch.Tensor],
    clip_norm: float,
) -> dict[str, torch.Tensor]:
    delta = {k: new_state[k].float() - old_state[k].float() for k in old_state}
    sq = sum(float(v.square().sum()) for v in delta.values())
    norm = sq ** 0.5
    scale = min(1.0, clip_norm / (norm + 1e-12))
    return {k: v * scale for k, v in delta.items()}


def train_dp_fedavg(
    n_pois: int,
    windows_by_user: dict[int, list[Window]],
    val_windows: list[Window],
    cfg: dict,
    noise_multiplier: float,
    seed: int = 0,
) -> FLResult:
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mcfg = cfg["model"]
    fl = cfg["fl"]
    clip_c = float(fl.get("clip_norm", 1.0))
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
    sample_rate = n_per_round / max(len(clients), 1)
    delta = 1.0 / max(len(clients), 1)
    rounds = int(fl["rounds"])
    history: list[dict] = []
    t0 = time.time()
    for r in range(rounds):
        chosen = rng.choice(clients, size=n_per_round, replace=False) if clients else []
        global_state = _state_cpu(model)
        clipped = []
        for uid in chosen:
            local = copy.deepcopy(model)
            _load(local, global_state, device)
            new_state = _local_train(local, windows_by_user[int(uid)], cfg, device, int(fl["local_epochs"]))
            clipped.append(_clip_update(new_state, global_state, clip_c))
        if clipped:
            averaged = {}
            m = float(len(clipped))
            sigma = noise_multiplier * clip_c / m
            for k in global_state:
                acc = sum(d[k] for d in clipped) / m
                noise = torch.randn_like(acc) * sigma
                averaged[k] = global_state[k].float() + acc + noise
            _load(model, {k: v for k, v in averaged.items()}, device)
        val = evaluate_model(model, val_windows, batch_size=mcfg["batch_size"])
        history.append({"round": r, **val})
    eps = report_epsilon(noise_multiplier, sample_rate, rounds, delta)
    return FLResult(
        model=model,
        history=history,
        best_acc10=max((h.get("acc10", 0.0) for h in history), default=0.0),
        seconds=time.time() - t0,
        n_params=n_params,
        comm_bytes_per_round=int(comm),
        rounds=rounds,
        epsilon=eps,
    )
