"""Fine-tuning del clasificador de 7 clases (§4.1) y ablacion de CBAM (§4.3).

La ablacion se corre con el mismo comando y el flag `--no-cbam`. Todo lo demas
(semilla, split, aumentaciones, lr, epocas) queda fijo, asi que la diferencia en
macro-F1 entre las dos corridas es atribuible al bloque de atencion y no al ruido de
entrenamiento.

    python -m src.train.train_classifier --config configs/classification.yaml
    python -m src.train.train_classifier --config configs/classification.yaml --no-cbam

El checkpoint se selecciona por `val_macro_f1`, no por accuracy: con nv al 67% del
dataset, la accuracy de validacion sube sola sin que el modelo aprenda las clases raras.

`train.early_stopping_patience: null` desactiva la parada temprana, para que las dos
corridas de la ablacion entrenen el mismo numero de epocas.

Reanudable: al final de cada epoca se guarda `classifier_<tag>_last.pt`. Si la sesion se
interrumpe, volver a lanzar el mismo comando continua desde la ultima epoca completada.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn

from src.config import load_config, resolve, set_seed
from src.eval.metrics import class_weights_balanced, classification_metrics
from src.train.checkpointing import cargar_estado, detener_si_no_finita, guardar_estado


def run_epoch(model, loader, criterion, optimizer, scaler, device, train: bool):
    model.train(train)
    total_loss, n = 0.0, 0
    y_true, y_pred = [], []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, enabled=scaler is not None):
                logits = model(images)
                loss = criterion(logits, labels)

            if train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            total_loss += loss.item() * labels.size(0)
            n += labels.size(0)
            y_pred.extend(logits.argmax(dim=1).cpu().tolist())
            y_true.extend(labels.cpu().tolist())

    return total_loss / max(n, 1), np.array(y_true), np.array(y_pred)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/classification.yaml")
    parser.add_argument(
        "--no-cbam",
        action="store_true",
        help="Corrida de ablacion: identica pero sin el bloque de atencion",
    )
    parser.add_argument("--tag", default=None, help="Sufijo del checkpoint")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Verificacion rapida: 1 epoca con 32 imagenes por split. No sirve como resultado.",
    )
    args = parser.parse_args()

    from src.data.datasets import make_dataloaders
    from src.models.classifier import build_classifier

    cfg = load_config(args.config)
    if args.quick:
        cfg["train"]["epochs"] = 1
    set_seed(cfg.seed)

    use_cbam = not args.no_cbam
    tag = args.tag or ("cbam" if use_cbam else "nocbam")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device} | CBAM: {use_cbam} | tag: {tag}")

    loaders, datasets = make_dataloaders(cfg, "classification", limit=32 if args.quick else None)
    class_names = list(cfg.classes)

    model = build_classifier(cfg, override_cbam=use_cbam).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    # Se vuelve a fijar la semilla DESPUES de construir el modelo. Construirlo con o sin
    # el bloque de atencion consume distinta cantidad de numeros aleatorios; sin esto, el
    # barajado de los lotes y las aumentaciones cambiarian entre las dos corridas de la
    # ablacion, y la diferencia no seria atribuible solo al bloque.
    set_seed(cfg.seed)
    print(f"Parametros: {n_params / 1e6:.2f} M")

    if cfg.train.class_weights == "balanced":
        weights = class_weights_balanced(
            datasets["train"].class_counts(cfg.model.num_classes)
        ).to(device)
        print(f"Pesos por clase: {dict(zip(class_names, weights.cpu().numpy().round(3)))}")
    else:
        weights = None

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(
        model.param_groups(cfg.train.lr, cfg.train.head_lr_multiplier),
        weight_decay=cfg.train.weight_decay,
    )
    # Si val_loss no mejora en `patience` epocas, todos los lr se multiplican por `factor`.
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
    ckpt_path = models_dir / f"classifier_{tag}_best.pt"

    if args.quick and ckpt_path.exists():
        raise SystemExit(
            f"Ya existe {ckpt_path}. --quick lo sobrescribiria con un modelo sin entrenar."
        )

    last_path = models_dir / f"classifier_{tag}_last.pt"
    history: list[dict] = []
    best_metric, epochs_without_improvement = -1.0, 0
    start_epoch, minutos_previos, detenido = 1, 0.0, False
    start = time.time()

    estado = None if args.quick else cargar_estado(
        last_path, model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler, device=device
    )
    if estado is not None:
        history, best_metric = estado["history"], estado["best_metric"]
        epochs_without_improvement, detenido = estado["stale"], estado["detenido"]
        start_epoch, minutos_previos = estado["epoch"] + 1, estado["minutos"]
        print(f"Reanudando: epocas 1-{estado['epoch']} ya completadas, mejor macro-F1 {best_metric:.4f}")

    for epoch in range(start_epoch, cfg.train.epochs + 1):
        if detenido:
            break
        train_loss, ytr, ypr = run_epoch(
            model, loaders["train"], criterion, optimizer, scaler, device, True
        )
        val_loss, yte, ype = run_epoch(
            model, loaders["val"], criterion, optimizer, scaler, device, False
        )
        detener_si_no_finita(epoch, val_loss)
        scheduler.step(val_loss)

        train_m = classification_metrics(ytr, ypr, class_names)
        val_m = classification_metrics(yte, ype, class_names)
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_macro_f1": train_m["macro_f1"],
            "val_loss": val_loss,
            "val_accuracy": val_m["accuracy"],
            "val_macro_f1": val_m["macro_f1"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(record)
        print(
            f"[{epoch:02d}/{cfg.train.epochs}] "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_acc={val_m['accuracy']:.4f} val_macroF1={val_m['macro_f1']:.4f}"
        )

        if val_m["macro_f1"] > best_metric:
            best_metric = val_m["macro_f1"]
            epochs_without_improvement = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val_macro_f1": best_metric,
                    "use_cbam": use_cbam,
                    "config": dict(cfg),
                    "class_names": class_names,
                },
                ckpt_path,
            )
            print(f"  -> nuevo mejor macro-F1, checkpoint guardado en {ckpt_path.name}")
        else:
            epochs_without_improvement += 1
            paciencia = cfg.train.early_stopping_patience
            if paciencia is not None and epochs_without_improvement >= paciencia:
                print(f"Early stopping en la epoca {epoch}")
                detenido = True

        guardar_estado(
            last_path, model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
            epoch=epoch, best_metric=best_metric, stale=epochs_without_improvement,
            history=history, minutos=minutos_previos + (time.time() - start) / 60, detenido=detenido,
        )

    # Evaluacion final en test con el mejor checkpoint, no con los pesos de la ultima epoca.
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model"])
    _, yte, ype = run_epoch(
        model, loaders["test"], criterion, optimizer, scaler, device, False
    )
    test_m = classification_metrics(yte, ype, class_names)

    # Prediccion de cada imagen de test, en el mismo orden que el split (el loader de test
    # no baraja). Permite comparar las dos corridas de la ablacion imagen por imagen y
    # estimar si la diferencia supera al azar, sin tener que volver a entrenar.
    test_predictions = {
        "image_id": datasets["test"].frame["image_id"].tolist(),
        "y_true": yte.tolist(),
        "y_pred": ype.tolist(),
    }

    print("\n=== Test ===")
    print(test_m["report"])

    results_dir = resolve(cfg.paths.reports) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "tag": tag,
        "use_cbam": use_cbam,
        "n_params": n_params,
        "checkpoint_mb": ckpt_path.stat().st_size / 1e6,
        "train_minutes": minutos_previos + (time.time() - start) / 60,
        "best_val_macro_f1": best_metric,
        "epochs_planned": cfg.train.epochs,
        "early_stopping_patience": cfg.train.early_stopping_patience,
        "test": {k: v for k, v in test_m.items() if k != "report"},
        "test_predictions": test_predictions,
        "history": history,
    }
    (results_dir / f"classification_{tag}.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(f"Metricas en {results_dir / f'classification_{tag}.json'}")
    # Entrenamiento terminado y resultados escritos: el estado para reanudar ya no sirve.
    last_path.unlink(missing_ok=True)
    print(
        "\nPara la ablacion de §4.3, compara este archivo con el de la otra corrida "
        "usando: python -m src.eval.compare_ablation"
    )


if __name__ == "__main__":
    main()
