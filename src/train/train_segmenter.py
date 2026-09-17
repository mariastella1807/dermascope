"""Entrenamiento del segmentador (§4.2) y ablacion del self-attention (§4.3).

La ablacion se corre con el mismo comando y el flag `--no-attention`. Semilla, split,
aumentaciones, lr y epocas quedan fijos, asi que la diferencia en Dice e IoU entre las
dos corridas es atribuible al bloque de self-attention:

    python -m src.train.train_segmenter --config configs/segmentation.yaml
    python -m src.train.train_segmenter --config configs/segmentation.yaml --no-attention

Se usan las 10.015 imagenes con mascara, con el mismo reparto por `lesion_id` que la
tarea de clasificacion.
"""

from __future__ import annotations

import argparse
import json
import time

import torch

from src.config import load_config, resolve, set_seed
from src.eval.metrics import bce_dice_loss, dice_coefficient, iou_score


def run_epoch(model, loader, optimizer, scaler, device, cfg, train: bool):
    model.train(train)
    total_loss, dice_values, iou_values, n = 0.0, [], [], 0
    threshold = cfg.postprocess.threshold

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, enabled=scaler is not None):
                logits = model(images)
                loss = bce_dice_loss(logits, masks, cfg.train.dice_weight)

            if train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            batch = masks.size(0)
            total_loss += loss.item() * batch
            n += batch
            # Las metricas se calculan en float32 para que el umbral no dependa del autocast.
            logits_fp32 = logits.float()
            dice_values.extend(dice_coefficient(logits_fp32, masks, threshold).cpu().tolist())
            iou_values.extend(iou_score(logits_fp32, masks, threshold).cpu().tolist())

    return (
        total_loss / max(n, 1),
        sum(dice_values) / max(len(dice_values), 1),
        sum(iou_values) / max(len(iou_values), 1),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/segmentation.yaml")
    parser.add_argument(
        "--no-attention",
        action="store_true",
        help="Corrida de ablacion: identica pero sin el bloque de self-attention",
    )
    parser.add_argument("--tag", default=None, help="Sufijo del checkpoint")
    args = parser.parse_args()

    from src.data.datasets import make_dataloaders
    from src.models.segmenter import build_segmenter

    cfg = load_config(args.config)
    set_seed(cfg.seed)

    use_attention = not args.no_attention
    tag = args.tag or ("attn" if use_attention else "noattn")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device} | Self-attention: {use_attention} | tag: {tag}")

    loaders, datasets = make_dataloaders(cfg, "segmentation")
    print(
        f"Imagenes con mascara -> train {len(datasets['train'])}, "
        f"val {len(datasets['val'])}, test {len(datasets['test'])}"
    )

    model = build_segmenter(cfg, override_attention=use_attention).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parametros: {n_params / 1e6:.2f} M")

    optimizer = torch.optim.AdamW(
        model.param_groups(cfg.train.lr, cfg.train.head_lr_multiplier),
        weight_decay=cfg.train.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=cfg.train.plateau_factor,
        patience=cfg.train.plateau_patience,
    )
    scaler = (
        torch.amp.GradScaler(device.type)
        if (cfg.train.amp and device.type == "cuda")
        else None
    )

    models_dir = resolve(cfg.paths.models)
    models_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = models_dir / f"segmenter_{tag}_best.pt"

    history, best_dice, stale = [], -1.0, 0
    start = time.time()

    for epoch in range(1, cfg.train.epochs + 1):
        tr_loss, tr_dice, _ = run_epoch(
            model, loaders["train"], optimizer, scaler, device, cfg, True
        )
        va_loss, va_dice, va_iou = run_epoch(
            model, loaders["val"], optimizer, scaler, device, cfg, False
        )
        scheduler.step(va_loss)

        history.append(
            {
                "epoch": epoch,
                "train_loss": tr_loss,
                "train_dice": tr_dice,
                "val_loss": va_loss,
                "val_dice": va_dice,
                "val_iou": va_iou,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )
        print(
            f"[{epoch:02d}/{cfg.train.epochs}] train_loss={tr_loss:.4f} "
            f"val_loss={va_loss:.4f} val_dice={va_dice:.4f} val_iou={va_iou:.4f}"
        )

        if va_dice > best_dice:
            best_dice, stale = va_dice, 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val_dice": best_dice,
                    "use_self_attention": use_attention,
                    "config": dict(cfg),
                },
                ckpt_path,
            )
            print(f"  -> nuevo mejor Dice, checkpoint en {ckpt_path.name}")
        else:
            stale += 1
            if stale >= cfg.train.early_stopping_patience:
                print(f"Early stopping en la epoca {epoch}")
                break

    # Evaluacion final en test con el mejor checkpoint, no con los pesos de la ultima epoca.
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model"])
    _, te_dice, te_iou = run_epoch(model, loaders["test"], optimizer, scaler, device, cfg, False)
    print(f"\n=== Test ({tag}) === Dice={te_dice:.4f} IoU={te_iou:.4f}")

    results_dir = resolve(cfg.paths.reports) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / f"segmentation_{tag}.json").write_text(
        json.dumps(
            {
                "tag": tag,
                "use_self_attention": use_attention,
                "n_params": n_params,
                "checkpoint_mb": ckpt_path.stat().st_size / 1e6,
                "train_minutes": (time.time() - start) / 60,
                "best_val_dice": best_dice,
                "test_dice": te_dice,
                "test_iou": te_iou,
                "history": history,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Metricas en {results_dir / f'segmentation_{tag}.json'}")


if __name__ == "__main__":
    main()
