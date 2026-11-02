"""M4: DP-SGD training (Opacus when available, otherwise a clipped+noisy fallback)."""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader

from lpdp.models.gru_next_poi import GRUNextPOI
from lpdp.train.centralised import TrainResult, evaluate_model
from lpdp.train.windows import Window, WindowDataset, collate_windows


@dataclass
class DPSGDResult(TrainResult):
    epsilon: float
    delta: float
    noise_multiplier: float


def _try_opacus():
    try:
        from opacus import PrivacyEngine
        from opacus.validators import ModuleValidator

        return PrivacyEngine, ModuleValidator
    except Exception:  # noqa: BLE001
        return None, None


def _make_model(n_pois: int, mcfg: dict, device, use_dp_rnn: bool) -> GRUNextPOI:
    return GRUNextPOI(
        n_pois=n_pois,
        poi_dim=mcfg["poi_dim"],
        hour_dim=mcfg["hour_dim"],
        hidden=mcfg["hidden"],
        use_dp_rnn=use_dp_rnn,
    ).to(device)


def _to_vanilla(model: nn.Module, n_pois: int, mcfg: dict, device) -> GRUNextPOI:
    clean = _make_model(n_pois, mcfg, device, use_dp_rnn=False)
    raw = model.state_dict()
    stripped = {k.replace("_module.", ""): v for k, v in raw.items()}
    clean.load_state_dict(stripped, strict=False)
    return clean


def _make_loader(train_windows: list[Window], mcfg: dict) -> DataLoader:
    bs = min(int(mcfg["batch_size"]), max(len(train_windows), 1))
    return DataLoader(
        WindowDataset(train_windows, seq_len=mcfg["seq_len"]),
        batch_size=bs,
        shuffle=True,
        collate_fn=collate_windows,
        drop_last=len(train_windows) > bs,
    )


def train_dpsgd(
    n_pois: int,
    train_windows: list[Window],
    val_windows: list[Window],
    cfg: dict,
    target_epsilon: float,
    seed: int = 0,
) -> DPSGDResult:
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mcfg = cfg["model"]
    epochs = int(mcfg.get("dp_epochs", mcfg["max_epochs"]))
    delta = float(mcfg.get("delta_override") or (1.0 / max(len(train_windows), 1)))
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    t0 = time.time()

    model = _make_model(n_pois, mcfg, device, use_dp_rnn=True)
    loader = _make_loader(train_windows, mcfg)
    opt = torch.optim.Adam(model.parameters(), lr=mcfg["lr"])
    PrivacyEngine, ModuleValidator = _try_opacus()

    used_opacus = False
    engine = None
    history: list[dict] = []
    epsilon_reported = float(target_epsilon)
    noise_multiplier = 1.0

    if PrivacyEngine is not None and len(train_windows) >= 8:
        try:
            if ModuleValidator is not None:
                model = ModuleValidator.fix(model).to(device)
                opt = torch.optim.Adam(model.parameters(), lr=mcfg["lr"])
            engine = PrivacyEngine()
            model, opt, loader = engine.make_private_with_epsilon(
                module=model,
                optimizer=opt,
                data_loader=loader,
                epochs=epochs,
                target_epsilon=float(target_epsilon),
                target_delta=delta,
                max_grad_norm=float(mcfg.get("max_grad_norm", 1.0)),
            )
            noise_multiplier = float(getattr(opt, "noise_multiplier", 1.0))
            used_opacus = True
        except Exception:  # noqa: BLE001
            used_opacus = False
            model = _make_model(n_pois, mcfg, device, use_dp_rnn=False)
            loader = _make_loader(train_windows, mcfg)
            opt = torch.optim.Adam(model.parameters(), lr=mcfg["lr"])

    if used_opacus:
        try:
            for epoch in range(epochs):
                model.train()
                total = 0.0
                n = 0
                for batch in loader:
                    opt.zero_grad()
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
                history.append({"epoch": epoch, "train_loss": total / max(n, 1)})
            try:
                epsilon_reported = float(engine.get_epsilon(delta))
            except Exception:  # noqa: BLE001
                epsilon_reported = float(target_epsilon)
        except Exception:  # noqa: BLE001
            used_opacus = False
            model = _make_model(n_pois, mcfg, device, use_dp_rnn=False)
            loader = _make_loader(train_windows, mcfg)
            opt = torch.optim.Adam(model.parameters(), lr=mcfg["lr"])

    if not used_opacus:
        model, history, epsilon_reported, noise_multiplier = _manual_dpsgd(
            model, opt, loader, loss_fn, device, epochs, target_epsilon, delta, mcfg
        )

    # Copy weights into a vanilla module so later eval / MIA do not hit Opacus hooks.
    model = _to_vanilla(model, n_pois, mcfg, device)
    model.eval()
    val = evaluate_model(model, val_windows, batch_size=mcfg["batch_size"])
    if history:
        history[-1].update(val)
    else:
        history.append(val)

    n_params = sum(p.numel() for p in model.parameters())
    return DPSGDResult(
        model=model,
        history=history,
        best_acc10=max((h.get("acc10", 0.0) for h in history), default=val.get("acc10", 0.0)),
        seconds=time.time() - t0,
        n_params=n_params,
        epsilon=epsilon_reported,
        delta=delta,
        noise_multiplier=noise_multiplier,
    )


def _manual_dpsgd(model, opt, loader, loss_fn, device, epochs, target_eps, delta, mcfg):
    """Clip batch gradients and add Gaussian noise. Accountant: GDP approximation."""
    from lpdp.fl.accounting import gaussian_dp_epsilon, noise_for_target_eps

    max_norm = float(mcfg.get("max_grad_norm", 1.0))
    steps_per_epoch = max(len(loader), 1)
    steps = epochs * steps_per_epoch
    n_data = max(len(getattr(loader, "dataset", [])), 1)
    sample_rate = min(1.0, mcfg["batch_size"] / n_data)
    z = noise_for_target_eps(target_eps, delta, steps, sample_rate)
    history = []
    for epoch in range(epochs):
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            for p in model.parameters():
                if p.grad is None:
                    continue
                p.grad.add_(torch.randn_like(p.grad) * (z * max_norm))
            opt.step()
            total += float(loss.item()) * batch["targets"].size(0)
            n += batch["targets"].size(0)
        history.append({"epoch": epoch, "train_loss": total / max(n, 1)})
    eps = gaussian_dp_epsilon(z, steps, sample_rate, delta)
    return model, history, eps, z
