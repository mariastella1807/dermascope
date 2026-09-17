"""Poda estructurada del clasificador: alternativa a la cuantizacion (§4.4).

Existe como plan B: si la cuantizacion INT8 degrada el macro-F1 mas alla de
`optimize.max_f1_drop` (lo que suele ocurrir por las clases minoritarias, cuyas
activaciones son las peor calibradas), la poda con reentrenamiento corto es la ruta
alternativa para cumplir §4.4.

Se usa poda estructurada por norma L_n de canal y no poda no estructurada: la poda por
magnitud de peso individual produce matrices dispersas que no reducen ni el tamano real
en disco ni la latencia en CPU sin soporte especifico de kernels dispersos. La reduccion
que §4.4 pide medir tiene que ser observable.

Flujo: podar -> medir -> reentrenar unas pocas epocas para recuperar -> medir de nuevo.
La poda sin reentrenamiento casi siempre pierde demasiado y no vale la pena reportarla
como resultado final.

Uso:
    python -m src.optimize.prune --checkpoint models/classifier_cbam_best.pt --amount 0.3
"""

from __future__ import annotations

import argparse
import json

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from src.config import load_config, resolve, set_seed


def prunable_modules(model: nn.Module) -> list[tuple[nn.Module, str]]:
    """Convoluciones y lineales del modelo, excluyendo la capa de salida.

    La capa final se excluye a proposito: tiene 7 salidas y podar canales ahi elimina
    capacidad para clases enteras, que es exactamente lo que no se quiere con `df` y
    `vasc` teniendo ~100 ejemplos.
    """
    modules = [
        (m, "weight")
        for m in model.modules()
        if isinstance(m, (nn.Conv2d, nn.Linear))
    ]
    return modules[:-1] if modules else modules


def apply_structured_pruning(model: nn.Module, amount: float, n: int = 2) -> nn.Module:
    """Poda `amount` de los canales de salida de cada capa por norma L_n."""
    for module, param_name in prunable_modules(model):
        if isinstance(module, nn.Conv2d):
            prune.ln_structured(module, name=param_name, amount=amount, n=n, dim=0)
        else:
            prune.l1_unstructured(module, name=param_name, amount=amount)
    return model


def remove_pruning_reparam(model: nn.Module) -> nn.Module:
    """Hace la poda permanente.

    Mientras la reparametrizacion esta activa, PyTorch guarda `weight_orig` y `weight_mask`,
    asi que el checkpoint pesa MAS que el original. Sin este paso, el tamano reportado en
    §4.4 saldria al reves.
    """
    for module, param_name in prunable_modules(model):
        try:
            prune.remove(module, param_name)
        except ValueError:
            # La capa no tenia poda aplicada.
            pass
    return model


def sparsity_report(model: nn.Module) -> dict:
    total, zeros = 0, 0
    for module, _ in prunable_modules(model):
        weight = module.weight
        total += weight.numel()
        zeros += int((weight == 0).sum())
    return {
        "total_weights": total,
        "zero_weights": zeros,
        "sparsity": zeros / total if total else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/classification.yaml")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Por defecto, classifier_cbam_best.pt dentro de paths.models",
    )
    parser.add_argument("--amount", type=float, default=0.3)
    parser.add_argument(
        "--finetune-epochs",
        type=int,
        default=5,
        help="Epocas de recuperacion tras la poda. Con 0 solo se mide el efecto crudo.",
    )
    args = parser.parse_args()

    from src.data.datasets import make_dataloaders
    from src.eval.metrics import classification_metrics
    from src.models.classifier import build_classifier

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = list(cfg.classes)

    checkpoint = (
        resolve(args.checkpoint)
        if args.checkpoint
        else resolve(cfg.paths.models) / "classifier_cbam_best.pt"
    )
    model = build_classifier(cfg).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device)["model"])

    loaders, _ = make_dataloaders(cfg, "classification")

    @torch.no_grad()
    def evaluate() -> dict:
        model.eval()
        y_true, y_pred = [], []
        for images, labels in loaders["test"]:
            logits = model(images.to(device))
            y_pred.extend(logits.argmax(dim=1).cpu().tolist())
            y_true.extend(labels.tolist())
        import numpy as np

        return classification_metrics(np.array(y_true), np.array(y_pred), class_names)

    before = evaluate()
    print(f"Antes de podar: accuracy={before['accuracy']:.4f} macroF1={before['macro_f1']:.4f}")

    apply_structured_pruning(model, args.amount)
    pruned_raw = evaluate()
    print(
        f"Tras podar {args.amount:.0%} sin reentrenar: "
        f"accuracy={pruned_raw['accuracy']:.4f} macroF1={pruned_raw['macro_f1']:.4f}"
    )

    if args.finetune_epochs > 0:
        print(f"\nReentrenando {args.finetune_epochs} epocas para recuperar ...")
        # TODO: reutilizar run_epoch de src.train.train_classifier con un lr reducido
        # (tipicamente lr/10) y sin warmup, manteniendo las mascaras de poda activas.
        print("Pendiente de implementar: ver el TODO en este archivo.")

    remove_pruning_reparam(model)
    after = evaluate()
    sparsity = sparsity_report(model)

    models_dir = resolve(cfg.paths.models)
    out_ckpt = models_dir / f"classifier_pruned_{int(args.amount * 100)}.pt"
    torch.save({"model": model.state_dict(), "pruned_amount": args.amount}, out_ckpt)

    summary = {
        "amount": args.amount,
        "sparsity": sparsity,
        "before": {"accuracy": before["accuracy"], "macro_f1": before["macro_f1"]},
        "after": {"accuracy": after["accuracy"], "macro_f1": after["macro_f1"]},
        "checkpoint_mb": out_ckpt.stat().st_size / 1e6,
    }
    out = resolve(cfg.paths.reports) / "results" / "pruning.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nResumen en {out}")


if __name__ == "__main__":
    main()
