"""Analisis complementarios sobre los modelos finales, a partir de los checkpoints.

Responde preguntas que las tablas principales no cubren:

1. **Melanoma y punto de operacion.** El modelo con CBAM detecta mas melanomas, pero
   tambien predice "melanoma" mas veces. Para separar las dos cosas se comparan medidas
   que no dependen del umbral (AUC y precision promedio de melanoma frente al resto) y el
   recall de los dos modelos con el MISMO numero de alertas: se marcan las k imagenes con
   mayor probabilidad de melanoma de cada modelo.
2. **Sensibilidad al ruido.** Fraccion de imagenes de test cuya clase predicha cambia al
   sumar ruido gaussiano de distintas intensidades.
3. **Entradas sin lesion.** Que responden el clasificador y el segmentador ante imagenes
   sin ninguna lesion.
4. **Segmentacion por clase y por tamano de lesion**, y Dice a 256 px frente a la
   resolucion original.

Todo se calcula sobre el conjunto de test, sin usarlo para elegir nada. Tarda unos
10 minutos en CPU.

    python -m src.eval.analisis_complementario
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import binomtest
from sklearn.metrics import average_precision_score, roc_auc_score

from src.config import load_config, resolve
from src.data import transforms as T
from src.data.datasets import read_mask, read_rgb
from src.explain.isolate import isolate_lesion
from src.models.classifier import build_classifier
from src.models.segmenter import build_segmenter


def load_model(build, cfg, path, **kwargs):
    model = build(cfg, **kwargs)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["model"])
    return model.eval()


@torch.no_grad()
def probabilities(model, x: torch.Tensor, batch: int = 64) -> np.ndarray:
    return torch.cat([torch.softmax(model(x[i:i + batch]), 1) for i in range(0, len(x), batch)]).numpy()


def melanoma_analysis(p_cbam, p_nocbam, y, mel: int, n_boot: int, seed: int) -> dict:
    y_mel = (y == mel).astype(int)
    out = {"melanomas_en_test": int(y_mel.sum())}
    for tag, p in [("con_cbam", p_cbam), ("sin_cbam", p_nocbam)]:
        pred = p.argmax(1)
        alerts = pred == mel
        out[tag] = {
            "veces_que_predice_melanoma": int(alerts.sum()),
            "recall": float((alerts & (y_mel == 1)).sum() / y_mel.sum()),
            "precision": float((alerts & (y_mel == 1)).sum() / max(alerts.sum(), 1)),
            "auc_melanoma": float(roc_auc_score(y_mel, p[:, mel])),
            "precision_promedio_melanoma": float(average_precision_score(y_mel, p[:, mel])),
            "auc_macro_ovr": float(roc_auc_score(y, p, multi_class="ovr", average="macro")),
        }

    # Recall con el mismo numero de alertas: el de cada modelo con la regla de la clase
    # mas probable. Asi la comparacion no depende de que un modelo alerte mas que el otro.
    out["recall_con_igual_numero_de_alertas"] = {}
    for k in sorted({out["con_cbam"]["veces_que_predice_melanoma"], out["sin_cbam"]["veces_que_predice_melanoma"]}):
        out["recall_con_igual_numero_de_alertas"][str(k)] = {
            tag: float(y_mel[np.argsort(-p[:, mel])[:k]].sum() / y_mel.sum())
            for tag, p in [("con_cbam", p_cbam), ("sin_cbam", p_nocbam)]
        }

    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if y_mel[idx].min() == y_mel[idx].max():
            continue
        diffs.append(roc_auc_score(y_mel[idx], p_cbam[idx, mel]) - roc_auc_score(y_mel[idx], p_nocbam[idx, mel]))
    out["diferencia_auc_melanoma"] = {
        "valor": out["con_cbam"]["auc_melanoma"] - out["sin_cbam"]["auc_melanoma"],
        "ic95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
        "remuestreos": len(diffs),
    }

    # McNemar sobre los melanomas con la regla de la clase mas probable (punto de operacion
    # distinto en cada modelo: por eso se reporta junto a las medidas anteriores).
    ok_c, ok_n = p_cbam.argmax(1) == mel, p_nocbam.argmax(1) == mel
    solo_c = int((ok_c & ~ok_n & (y_mel == 1)).sum())
    solo_n = int((ok_n & ~ok_c & (y_mel == 1)).sum())
    out["mcnemar_melanomas"] = {"solo_con_cbam": solo_c, "solo_sin_cbam": solo_n,
                                "p_exacto": float(binomtest(solo_c, solo_c + solo_n).pvalue)}
    return out


@torch.no_grad()
def noise_sensitivity(model, x: torch.Tensor, y: np.ndarray, levels, seed: int) -> list[dict]:
    torch.manual_seed(seed)
    base = torch.from_numpy(probabilities(model, x).argmax(1))
    span = float(x.max() - x.min())
    rows = [{"ruido_frac_rango": 0.0, "cambia_la_clase": 0.0, "accuracy": float((base.numpy() == y).mean())}]
    for level in levels:
        pred = torch.from_numpy(probabilities(model, x + torch.randn_like(x) * level * span).argmax(1))
        rows.append({"ruido_frac_rango": level, "cambia_la_clase": float((pred != base).float().mean()),
                     "accuracy": float((pred.numpy() == y).mean())})
    return rows


@torch.no_grad()
def no_lesion_inputs(classifier, segmenter, cfg_cls, cfg_seg) -> list[dict]:
    rng = np.random.default_rng(0)
    skin = np.clip(np.array([224, 172, 150]) + rng.integers(0, 8, (450, 600, 3)), 0, 255).astype(np.uint8)
    cases = {"piel_sin_lesion": skin, "imagen_negra": np.zeros((450, 600, 3), np.uint8)}
    tf_c, tf_s = T.classification_eval_transform(cfg_cls), T.segmentation_eval_transform(cfg_seg)
    post, names, rows = cfg_seg.postprocess, list(cfg_cls.classes), []
    for name, img in cases.items():
        p = torch.softmax(classifier(tf_c(img).unsqueeze(0)), 1)[0].numpy()
        prob_map = torch.sigmoid(segmenter(tf_s(img).unsqueeze(0)))[0, 0].numpy()
        iso = isolate_lesion(img, prob_map, threshold=post.threshold, keep_largest_component=post.keep_largest_component,
                             morph_close_kernel=post.morph_close_kernel, min_area_ratio=post.min_area_ratio)
        rows.append({"entrada": name, "clase_predicha": names[int(p.argmax())], "confianza": float(p.max()),
                     "segmentador_detecta_lesion": bool(iso.detected)})
    return rows


def segmentation_breakdown(test: pd.DataFrame, per_image: pd.DataFrame) -> dict:
    d = test.merge(per_image, on="image_id")
    d["fraccion_lesion"] = [read_mask(p).mean() for p in d.mask_path]
    d["tamano"] = pd.cut(d.fraccion_lesion, [0, 0.05, 0.15, 0.35, 1.0], labels=["<5%", "5-15%", "15-35%", ">35%"])
    return {
        "por_clase": d.groupby("dx").dice.agg(["mean", "median", "count"]).round(4).to_dict(orient="index"),
        "por_tamano": d.groupby("tamano", observed=True).dice.agg(["mean", "median", "count"]).round(4).to_dict(orient="index"),
        "imagenes_dice_menor_0_7": int((d.dice < 0.7).sum()),
        "imagenes_dice_menor_0_5": int((d.dice < 0.5).sum()),
        "n": len(d),
    }


@torch.no_grad()
def dice_resolution(segmenter, test: pd.DataFrame, cfg_seg, n: int, seed: int) -> dict:
    tf, th = T.segmentation_eval_transform(cfg_seg), cfg_seg.postprocess.threshold
    d256, dorig = [], []
    for _, r in test.sample(n, random_state=seed).iterrows():
        img, gt = np.array(read_rgb(r.image_path)), torch.from_numpy(read_mask(r.mask_path)).float()
        prob = torch.sigmoid(segmenter(tf(img).unsqueeze(0)))[0, 0]
        g = F.interpolate(gt[None, None], size=prob.shape, mode="nearest")[0, 0]
        p = (prob >= th).float()
        d256.append(float(2 * (p * g).sum() / (p.sum() + g.sum() + 1e-8)))
        po = (F.interpolate(prob[None, None], size=gt.shape, mode="bilinear", align_corners=False)[0, 0] >= th).float()
        dorig.append(float(2 * (po * gt).sum() / (po.sum() + gt.sum() + 1e-8)))
    return {"n": n, "dice_256px": float(np.mean(d256)), "dice_resolucion_original": float(np.mean(dorig))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cls-config", default="configs/classification.yaml")
    parser.add_argument("--seg-config", default="configs/segmentation.yaml")
    parser.add_argument("--noise-images", type=int, default=300)
    parser.add_argument("--resolution-images", type=int, default=300)
    args = parser.parse_args()

    cfg_cls, cfg_seg = load_config(args.cls_config), load_config(args.seg_config)
    seed = int(cfg_cls.seed)
    models_dir, results_dir = resolve(cfg_cls.paths.models), resolve(cfg_cls.paths.reports) / "results"
    splits = pd.read_csv(resolve(cfg_cls.paths.processed) / "splits.csv")
    test = splits[splits.split == "test"].reset_index(drop=True)
    y, mel = test.dx_idx.to_numpy(), list(cfg_cls.classes).index("mel")

    clf_c = load_model(build_classifier, cfg_cls, models_dir / "classifier_cbam_best.pt", override_cbam=True)
    clf_n = load_model(build_classifier, cfg_cls, models_dir / "classifier_nocbam_best.pt", override_cbam=False)
    seg = load_model(build_segmenter, cfg_seg, models_dir / "segmenter_attn_best.pt", override_attention=True)

    print("Clasificando el test con los dos modelos ...")
    tf = T.classification_eval_transform(cfg_cls)
    x = torch.stack([tf(read_rgb(p)) for p in test.image_path])
    p_c, p_n = probabilities(clf_c, x), probabilities(clf_n, x)

    results = {"melanoma": melanoma_analysis(p_c, p_n, y, mel, n_boot=2000, seed=seed)}

    print("Sensibilidad al ruido ...")
    idx = test.sample(args.noise_images, random_state=seed).index.to_numpy()
    results["sensibilidad_al_ruido"] = {
        "modelo": "classifier_cbam_best.pt", "imagenes": args.noise_images,
        "niveles": noise_sensitivity(clf_c, x[idx], y[idx], [0.01, 0.02, 0.05, 0.10], seed),
    }

    print("Entradas sin lesion ...")
    results["entradas_sin_lesion"] = no_lesion_inputs(clf_c, seg, cfg_cls, cfg_seg)

    print("Segmentacion por clase, tamano y resolucion ...")
    per_image = pd.DataFrame(json.loads((results_dir / "segmentation_attn.json").read_text(encoding="utf-8"))["test_per_image"])
    results["segmentacion"] = segmentation_breakdown(test, per_image)
    results["segmentacion"]["resolucion_del_dice"] = dice_resolution(seg, test, cfg_seg, args.resolution_images, seed)

    out = results_dir / "analisis_complementario.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nResultados en {out}")


if __name__ == "__main__":
    main()
