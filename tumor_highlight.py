"""Output 2 and 3: tumor highlight and segmentation mask.

This is a heuristic region finder, NOT a trained segmentation model. It looks
for a large, bright, contrast-enhancing mass and traces its outer boundary
from the image itself (no fixed shape, circle or coordinates).

How it works
1. Seed: find the bright blob that stands out most from its surroundings
   (ranked by area x local contrast), so small bright structures such as the
   ventricles or choroid plexus lose to a large enhancing mass.
2. Boundary: grow that seed outward with GrabCut, which follows the intensity
   edges of the mass, including irregular extensions and dimmer lobes.
3. Guard rails: if the traced region looks implausible (shrinks, or balloons
   into normal brain), fall back to the plain seed.

Limits: it works for bright / enhancing tumors. Dark, cystic or low-contrast
tumors, or slices with lots of bright bone, can be missed or mislocated.
"""
import cv2
import numpy as np

_K3 = np.ones((3, 3), np.uint8)
_K5 = np.ones((5, 5), np.uint8)


def _head_mask(g):
    """Filled mask of the head (everything brighter than the black background)."""
    head = (g > 25).astype(np.uint8) * 255
    head = cv2.morphologyEx(head, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2)
    n, lab, st, _ = cv2.connectedComponentsWithStats(head)
    if n < 2:
        return None
    head = (lab == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(head, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(head)
    cv2.drawContours(filled, cnts, -1, 255, -1)
    return filled


def _fill_holes(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(mask)
    cv2.drawContours(out, cnts, -1, 255, -1)
    return out


def _largest_overlapping(mask, seed):
    """Keep the connected component of `mask` that overlaps `seed` the most."""
    n, lab, _, _ = cv2.connectedComponentsWithStats(mask)
    if n < 2:
        return None
    best = max(range(1, n), key=lambda i: int(((lab == i) & (seed > 0)).sum()))
    if ((lab == best) & (seed > 0)).sum() == 0:
        return None
    return (lab == best).astype(np.uint8) * 255


def _smooth(mask):
    """Remove pixel-level jaggies but keep real irregular extensions."""
    m = cv2.GaussianBlur(mask, (5, 5), 0)
    m = (m > 127).astype(np.uint8) * 255
    return _fill_holes(m)


def _find_seed(g, seed_zone, local_contrast, pct=90):
    """Bright blob that stands out most from its surroundings."""
    h, w = g.shape
    t = np.percentile(g[seed_zone > 0], pct)
    b = ((g > t) & (seed_zone > 0)).astype(np.uint8) * 255
    b = cv2.morphologyEx(b, cv2.MORPH_OPEN, _K5)
    b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8), iterations=2)
    n, lab, st, _ = cv2.connectedComponentsWithStats(b)
    best, best_score = None, 0.0
    for i in range(1, n):
        area = int(st[i, cv2.CC_STAT_AREA])
        if not (0.003 * h * w < area < 0.20 * h * w):
            continue
        contrast = float(local_contrast[lab == i].mean())
        score = area * max(contrast, 0.0)
        if score > best_score:
            best, best_score = i, score
    if best is None:
        return None
    return _fill_holes((lab == best).astype(np.uint8) * 255)


def _trace_boundary(g, local_contrast, seed, loose_zone):
    """Grow the seed to the mass's real outer boundary with GrabCut."""
    h, w = g.shape
    ys, xs = np.where(seed > 0)
    pad = int(0.35 * max(xs.max() - xs.min(), ys.max() - ys.min())) + 4
    x0, x1 = max(0, xs.min() - pad), min(w, xs.max() + pad + 1)
    y0, y1 = max(0, ys.min() - pad), min(h, ys.max() + pad + 1)

    grow = int(0.03 * h)
    seed_c = seed[y0:y1, x0:x1]
    zone_c = loose_zone[y0:y1, x0:x1]
    near = cv2.dilate(seed_c, _K5, iterations=max(1, grow // 2))

    gc = np.full(seed_c.shape, cv2.GC_BGD, np.uint8)
    gc[zone_c > 0] = cv2.GC_PR_BGD
    gc[(near > 0) & (zone_c > 0)] = cv2.GC_PR_FGD
    gc[cv2.erode(seed_c, _K5, iterations=3) > 0] = cv2.GC_FGD

    feat = cv2.merge([
        g[y0:y1, x0:x1],
        cv2.GaussianBlur(g, (0, 0), 3)[y0:y1, x0:x1],
        np.clip(local_contrast[y0:y1, x0:x1] + 128, 0, 255).astype(np.uint8),
    ])
    bgm = np.zeros((1, 65), np.float64)
    fgm = np.zeros((1, 65), np.float64)
    cv2.grabCut(feat, gc, None, bgm, fgm, 6, cv2.GC_INIT_WITH_MASK)

    out_c = ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)).astype(np.uint8) * 255
    out = np.zeros_like(seed)
    out[y0:y1, x0:x1] = out_c
    return out


def _find_tumor_mask(gray):
    """Return (filled_mask, contours) for the detected mass, or (None, [])."""
    g = cv2.GaussianBlur(gray, (5, 5), 0)
    h, w = g.shape

    filled = _head_mask(g)
    if filled is None:
        return None, []
    seed_zone = cv2.erode(filled, _K5, iterations=max(1, int(0.035 * h)))
    loose_zone = cv2.erode(filled, _K5, iterations=max(1, int(0.012 * h)))
    if (seed_zone > 0).sum() < 100:
        return None, []

    background = cv2.GaussianBlur(g, (0, 0), 25).astype(np.float32)
    local_contrast = g.astype(np.float32) - background

    seed = _find_seed(g, seed_zone, local_contrast)
    if seed is None:
        return None, []
    seed_area = float((seed > 0).sum())

    solid = None
    try:
        traced = _trace_boundary(g, local_contrast, seed, loose_zone)
        traced = cv2.morphologyEx(traced, cv2.MORPH_OPEN, _K3)
        traced = _largest_overlapping(traced, seed)
        if traced is not None:
            traced = _smooth(traced)
            area = float((traced > 0).sum())
            if 0.8 * seed_area <= area <= 3.0 * seed_area:
                solid = traced
    except cv2.error:
        solid = None

    if solid is None:
        solid = _smooth(seed)

    contours, _ = cv2.findContours(solid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None, []
    return solid, contours


# Kept so any older import of the previous name still works.
_find_bright_mass = _find_tumor_mask


def make_highlight(display_rgb):
    """Take the RGB display image; return the image with a red outer contour, found: bool."""
    gray = cv2.cvtColor(display_rgb, cv2.COLOR_RGB2GRAY)
    solid, contours = _find_tumor_mask(gray)
    if solid is None or len(contours) == 0:
        return display_rgb, False
    out = display_rgb.copy()
    for contour in contours:
        cv2.drawContours(out, [contour], -1, (255, 0, 0), 2)
    return out, True


def make_segmentation(display_rgb):
    """Return a black-and-white segmentation mask for the detected region."""
    gray = cv2.cvtColor(display_rgb, cv2.COLOR_RGB2GRAY)
    solid, contours = _find_tumor_mask(gray)
    if solid is None or len(contours) == 0:
        return np.zeros_like(display_rgb), False
    return cv2.cvtColor(solid, cv2.COLOR_GRAY2RGB), True
