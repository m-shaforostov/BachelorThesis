# attention_map_evaluation.py

import csv
import cv2
import numpy as np
from pathlib import Path

from reveal_segmentation import make_trimap, save_colored_trimap


def trimap_to_score_mask(trimap, attention_shape):
    trimap_resized = cv2.resize(
        trimap,
        (attention_shape[1], attention_shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )

    score_mask = np.zeros_like(trimap_resized, dtype=np.float32)
    score_mask[trimap_resized == 0] = -1.0
    score_mask[trimap_resized == 128] = 0.0
    score_mask[trimap_resized == 255] = 1.0

    return score_mask, trimap_resized


def compute_attention_score(attention_mask, score_mask):
    attention = np.asarray(attention_mask, dtype=np.float32)

    attention = attention - attention.min()
    if attention.max() > 0:
        attention = attention / attention.max()

    attention_sum = attention.sum()
    if attention_sum == 0:
        return 0.0

    score = np.sum(attention * score_mask) / attention_sum
    return float(score)


def evaluate_attention_maps(
    trimap,
    attention_maps,
    output_dir,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    colored_trimap_path_out = output_dir / "generated_trimap_color.png"
    save_colored_trimap(trimap, colored_trimap_path_out)

    first_attention_mask = next(iter(attention_maps.values()))
    score_mask, _ = trimap_to_score_mask(trimap, first_attention_mask.shape)

    results = []

    for map_name, attention_mask in attention_maps.items():
        if attention_mask.shape != first_attention_mask.shape:
            raise ValueError(
                f"Attention map '{map_name}' has shape {attention_mask.shape}, "
                f"but expected {first_attention_mask.shape}."
            )

        score = compute_attention_score(attention_mask, score_mask)

        results.append({
            "attention_map": map_name,
            "score": score,
        })

    csv_path = output_dir / "attention_scores.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["attention_map", "score"],
        )
        writer.writeheader()
        writer.writerows(results)

    print("\nAttention map scores:")
    for item in results:
        print(f"{item['attention_map']}: {item['score']:.4f}")

    print(f"\nSaved scores to: {csv_path}")

    return results