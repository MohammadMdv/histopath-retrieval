"""
Gallery-side minority-class augmentation (optional, config-driven).

Rationale: in a patient-disjoint kNN gallery, rare classes have too few
exemplars, so a query may have no same-class neighbor in the top-K (a recall
ceiling that voting cannot fix). We expand under-represented classes by adding
LABEL-PRESERVING geometric variants (rotations / flips / mild crops) of their
EXISTING gallery patches. H&E patches have no canonical orientation, so these
transforms preserve the class.

What this does NOT do: it adds no new patients and no new morphology — only new
orientations of real minority patches. Each augmented vector keeps the ORIGINAL
source `path`, so /thumb still shows a genuine image and no synthetic pixels are
ever served. Expect a modest minority-recall lift, possibly at some precision
cost; measure it like any other knob.

Queries are never augmented here (that would be test-time augmentation — a
separate feature). This is applied only during index build.
"""
import random
from collections import defaultdict

from PIL import Image


def _random_spec(rng: random.Random) -> dict:
    """A non-identity, JSON-serializable geometric transform spec."""
    while True:
        spec = {
            "rot": rng.choice([0, 90, 180, 270]),
            "flip": rng.choice([None, "h", "v"]),
            "crop_scale": round(rng.uniform(0.85, 1.0), 3),
            "crop_x": round(rng.random(), 3),
            "crop_y": round(rng.random(), 3),
        }
        # Reject the no-op so an "augmented" exemplar is never a pure duplicate.
        if spec["rot"] or spec["flip"] or spec["crop_scale"] < 1.0:
            return spec


def apply_augmentation(img: Image.Image, spec: dict) -> Image.Image:
    """Deterministically apply a spec from _random_spec to a PIL image."""
    out = img
    if spec.get("rot"):
        out = out.rotate(spec["rot"], expand=True)
    flip = spec.get("flip")
    if flip == "h":
        out = out.transpose(Image.FLIP_LEFT_RIGHT)
    elif flip == "v":
        out = out.transpose(Image.FLIP_TOP_BOTTOM)
    scale = spec.get("crop_scale", 1.0)
    if scale < 1.0:
        w, h = out.size
        cw, ch = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        x = int(round(spec.get("crop_x", 0.0) * (w - cw)))
        y = int(round(spec.get("crop_y", 0.0) * (h - ch)))
        out = out.crop((x, y, x + cw, y + ch))
    return out


def build_augmentation_plan(
    records: list[dict],
    target_per_class: int,
    max_factor: int,
    seed: int = 42,
) -> tuple[list[dict], dict[str, tuple[int, int]]]:
    """
    Return (expanded_records, summary).

    expanded_records = all originals (unchanged) followed by augmented copies.
    Each augmented copy is a shallow copy of its source record plus an
    "augment" spec; build_index applies that spec to the image before encoding.

    A class is expanded UP TO target_per_class, but never beyond max_factor
    variants per source patch (so a tiny class can't collapse into near-clones
    of one image). summary maps label -> (original_count, added_count).
    """
    rng = random.Random(seed)
    by_class: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_class[r.get("label", "unknown")].append(r)

    plan = list(records)  # originals first, untouched
    summary: dict[str, tuple[int, int]] = {}

    for label, recs in by_class.items():
        n = len(recs)
        capacity = n * (max_factor - 1)            # max augmented variants allowed
        need = min(max(target_per_class - n, 0), capacity)
        for i in range(need):
            src = recs[i % n]                       # round-robin spreads variants evenly
            aug = dict(src)
            aug["augment"] = _random_spec(rng)
            plan.append(aug)
        summary[label] = (n, need)

    return plan, summary
