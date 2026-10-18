#!/usr/bin/env python3
"""Generate the RL Testbed brand assets as PNGs.

Everything is drawn programmatically (Pillow) at 4x supersampling and then
downscaled with a Lanczos filter, so the output stays crisp at every size and
the assets can be regenerated deterministically instead of being committed as
untouchable binaries.

Outputs, all into ``web/public/``:

    logo-mark.png          512x512   app icon / avatar
    logo.png              1024x256   horizontal lockup for light surfaces
    logo-dark.png         1024x256   horizontal lockup for the dark app shell
    favicon.png              64x64
    favicon-32.png           32x32
    apple-touch-icon.png    180x180
    og-image.png          1200x630   social / link preview card

Usage:
    python web/scripts/generate-assets.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# --------------------------------------------------------------------------- #
# Brand
# --------------------------------------------------------------------------- #
INDIGO = (99, 102, 241)      # #6366F1
VIOLET = (167, 139, 250)     # #A78BFA
LILAC = (199, 210, 254)      # #C7D2FE
ORANGE = (232, 111, 81)      # #E86F51
WHITE = (255, 255, 255)
BG = (7, 10, 23)             # #070A17  app background
BG_SOFT = (11, 16, 32)       # #0B1020

SS = 4  # supersample factor

REGULAR_FONT_CANDIDATES = [
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/SFNS.ttf",
    "/Library/Fonts/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]

BOLD_FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Black.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/Library/Fonts/Arial.ttf",
]


def _load(path: str, size: int, bold: bool) -> ImageFont.FreeTypeFont | None:
    if not Path(path).exists():
        return None
    # Helvetica.ttc is a TrueType collection. Index 0 = Regular, 1 = Bold.
    candidates = ([(1, size)] if bold else [(0, size)]) if path.endswith(".ttc") else [(0, size)]
    candidates += [(0, size)]  # fallback
    for index, pt in candidates:
        try:
            return ImageFont.truetype(path, pt, index=index)
        except Exception:
            continue
    return None


def font(size: float, bold: bool = True) -> ImageFont.FreeTypeFont:
    """Load the first available sans-serif font at `size` (supersampled)."""
    target = int(round(size * SS))
    sources = BOLD_FONT_CANDIDATES if bold else REGULAR_FONT_CANDIDATES
    for path in sources:
        f = _load(path, target, bold)
        if f is not None:
            return f
    return ImageFont.load_default()


def _blank(size: tuple[int, int]) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGBA", (size[0] * SS, size[1] * SS), (0, 0, 0, 0))
    return img, ImageDraw.Draw(img)


def _finish(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    return img.resize((size[0] * SS, size[1] * SS), Image.LANCZOS).resize(
        size, Image.LANCZOS
    ) if False else img.resize(size, Image.LANCZOS)


def vertical_gradient(size: int, top, bottom) -> Image.Image:
    """Smooth diagonal-ish gradient tile (top-left -> bottom-right)."""
    n = size * SS
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
    # Blend along the anti-diagonal so the gradient reads as top-left -> bottom-right.
    yy, xx = np.mgrid[0:n, 0:n]
    t = ((xx + yy) / (2.0 * n)).astype(np.float32)
    top_a = np.array(top, dtype=np.float32)
    bot_a = np.array(bottom, dtype=np.float32)
    arr = top_a[None, None, :] * (1.0 - t[..., None]) + bot_a[None, None, :] * t[..., None]
    arr = np.concatenate([arr, np.full((n, n, 1), 255.0, dtype=np.float32)], axis=-1)
    return Image.fromarray(arr.astype(np.uint8), "RGBA")


def rounded_mask(size: int, radius_ratio: float = 0.24) -> Image.Image:
    """Alpha mask with a rounded-rectangle silhouette (antialiased)."""
    n = size * SS
    mask = Image.new("L", (n, n), 0)
    d = ImageDraw.Draw(mask)
    d.rounded_rectangle(
        [0, 0, n - 1, n - 1], radius=int(n * radius_ratio), fill=255
    )
    return mask.filter(ImageFilter.GaussianBlur(radius=0.6 * SS))


# --------------------------------------------------------------------------- #
# The mark: a rounded gradient tile holding a rising learning curve that ends
# in a goal node. Reads as "reward improving over time" even at 16px.
# --------------------------------------------------------------------------- #

# Curve control points in a 100x100 design space.
CURVE = [(20, 74), (40, 66), (57, 50), (72, 30)]
GOAL = (80, 22)


def draw_mark(size: int) -> Image.Image:
    n = size * SS

    tile = vertical_gradient(size, INDIGO, VIOLET)
    mask = rounded_mask(size)
    tile.putalpha(mask)

    glyph = Image.new("RGBA", (n, n), (0, 0, 0, 0))
    d = ImageDraw.Draw(glyph)

    def P(pt):
        return (pt[0] / 100.0 * n, pt[1] / 100.0 * n)

    pts = [P(p) for p in CURVE]

    # Soft base line: the "episode axis" the curve climbs away from.
    d.line([P((14, 84)), P((86, 84))], fill=(255, 255, 255, 70),
           width=int(n * 0.035), joint="curve")

    # Rising reward curve.
    d.line(pts, fill=WHITE + (255,), width=int(n * 0.085), joint="curve")

    # Goal node at the peak: a filled dot with a halo punched out of the tile.
    gx, gy = P(GOAL)
    r_out = n * 0.115
    r_in = n * 0.062
    d.ellipse([gx - r_out, gy - r_out, gx + r_out, gy + r_out],
              fill=LILAC + (255,))
    d.ellipse([gx - r_in, gy - r_in, gx + r_in, gy + r_in],
              fill=VIOLET + (255,))

    tile = Image.alpha_composite(tile, glyph)
    return tile.resize((size, size), Image.LANCZOS)


# --------------------------------------------------------------------------- #
# Lockups
# --------------------------------------------------------------------------- #


def _cap_height(f) -> int:
    """Approximate cap height by measuring 'H' (Pillow has no real metric)."""
    img = Image.new("L", (1, 1))
    d = ImageDraw.Draw(img)
    bbox = d.textbbox((0, 0), "H", font=f)
    return bbox[3] - bbox[1]


def _text_width(f, text: str) -> int:
    img = Image.new("L", (1, 1))
    d = ImageDraw.Draw(img)
    return d.textbbox((0, 0), text, font=f)[2]


def draw_lockup(width: int, height: int, dark: bool = True) -> Image.Image:
    """Mark + wordmark on a transparent (dark=False) or app-bg background.

    All measurements are made in *output pixels*, then drawn at 4x SS. The
    composition is a two-line text block with the mark on the left, sized to
    match the cap height of "RL".
    """
    w, h = width * SS, height * SS
    fg = WHITE if dark else (17, 24, 39)
    base = Image.new("RGBA", (w, h), BG + (255,) if dark else (0, 0, 0, 0))

    # All sizes are in *output* pixels - the canvas is later upscaled by SS.
    pad_x = max(int(width * 0.04), 8)
    mark_size = int(height * 0.60)
    rl_font_pt = int(height * 0.32)         # bold "RL"
    tb_font_pt = int(height * 0.155)        # "TESTBED" subordinate

    f_rl = font(rl_font_pt, bold=True)
    f_tb = font(tb_font_pt, bold=False)

    # Vertical layout: two text lines centred, mark vertically centred.
    rl_bbox = d_text(f_rl, "RL")
    tb_bbox = d_text(f_tb, "T")
    cap_rl = rl_bbox[3] - rl_bbox[1]        # in SS units
    cap_tb = tb_bbox[3] - tb_bbox[1]
    gap_y = int(height * 0.06) * SS
    block_h = cap_rl + gap_y + cap_tb
    top = (h - block_h) // 2
    mark_y = top + (cap_rl - mark_size * SS) // 2

    # Place the mark.
    mark = draw_mark(mark_size)
    base.alpha_composite(mark, (pad_x * SS, mark_y))

    draw = ImageDraw.Draw(base)
    x_text = (pad_x + mark_size + int(height * 0.10)) * SS

    # "RL" on the top line.
    draw.text((x_text, top - rl_bbox[1]), "RL", font=f_rl, fill=fg + (255,))

    # "TESTBED" on the second line, letter-spaced.
    word = "TESTBED"
    tracking = int(height * 0.025) * SS
    cx = x_text
    for ch in word:
        ch_w = d_text(f_tb, ch)[2]
        draw.text((cx, top + cap_rl + gap_y - tb_bbox[1]), ch, font=f_tb,
                  fill=(fg[0], fg[1], fg[2], 230))
        cx += ch_w + tracking

    # Orange signature dot at the end of "TESTBED".
    dot = int(height * 0.06) * SS
    draw.ellipse(
        [cx + int(height * 0.02) * SS, top + cap_rl + gap_y - tb_bbox[1] + cap_tb // 2 - dot // 2,
         cx + int(height * 0.02) * SS + dot, top + cap_rl + gap_y - tb_bbox[1] + cap_tb // 2 + dot // 2],
        fill=ORANGE + (255,),
    )

    return base.resize((width, height), Image.LANCZOS)


def d_text(f, text: str):
    """textbbox shortcut (Pillow has no per-glyph metric helper)."""
    img = Image.new("L", (1, 1))
    d = ImageDraw.Draw(img)
    return d.textbbox((0, 0), text, font=f)


# --------------------------------------------------------------------------- #
# Social card
# --------------------------------------------------------------------------- #


def draw_og(width: int = 1200, height: int = 630) -> Image.Image:
    w, h = width * SS, height * SS
    img = Image.new("RGBA", (w, h), BG + (255,))
    d = ImageDraw.Draw(img)

    # Ambient glows echoing the app's background treatment.
    for (cx, cy, rad, col) in (
        (0.78, -0.10, 0.85, (129, 140, 248)),
        (-0.05, 0.30, 0.70, (167, 139, 250)),
        (1.02, 0.95, 0.60, (52, 211, 153)),
    ):
        glow = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow)
        gx, gy, rr = cx * w, cy * h, rad * w
        gd.ellipse([gx - rr, gy - rr * 0.72, gx + rr, gy + rr * 0.72],
                   fill=col + (int(0.13 * 255),))
        glow = glow.filter(ImageFilter.GaussianBlur(radius=0.18 * w))
        img = Image.alpha_composite(img, glow)
        d = ImageDraw.Draw(img)

    # Faint grid for a "laboratory" feel.
    step = int(w * 0.0625)
    for gx in range(step, w, step):
        d.line([(gx, 0), (gx, h)], fill=(255, 255, 255, 8), width=SS)
    for gy in range(step, h, step):
        d.line([(0, gy), (w, gy)], fill=(255, 255, 255, 8), width=SS)

    # Lockup at 1280x320 reference size, scaled to the og.
    lock_w_out = int(width * 0.50)
    lock_h_out = int(lock_w_out * (320 / 1280))
    lock = draw_lockup(lock_w_out, lock_h_out, dark=True)
    lock_x = int(width * 0.06) * SS
    lock_y = int(height * 0.18) * SS
    img.alpha_composite(lock, (lock_x, lock_y))

    d = ImageDraw.Draw(img)

    # Tagline + subhead + chips, all left-aligned below the lockup.
    f_tag = font(int(height * 0.062), bold=True)
    f_sub = font(int(height * 0.028), bold=False)
    f_chip = font(int(height * 0.030), bold=True)

    x_left = int(width * 0.06) * SS
    y_tag = lock_y + lock_h_out * SS + int(height * 0.10) * SS
    d.text((x_left, y_tag), "Laboratory for Reinforcement Learning",
           font=f_tag, fill=(226, 232, 240, 235))

    y_sub = y_tag + int(height * 0.085) * SS
    d.text((x_left, y_sub),
           "Train agents  ·  watch learning live  ·  compare experiments",
           font=f_sub, fill=(148, 163, 184, 230))

    # Capability chips: dark glass pill with light text.
    y_chips = y_sub + int(height * 0.075) * SS
    cx_chip = x_left
    for label in ("36 environments", "13 algorithms", "MuJoCo ready", "Live charts"):
        tw = d_text(f_chip, label)[2]
        pad_x_chip = int(height * 0.028) * SS
        pad_y_chip = int(height * 0.018) * SS
        pw = tw + pad_x_chip * 2
        ph = int(height * 0.072) * SS
        d.rounded_rectangle(
            [cx_chip, y_chips, cx_chip + pw, y_chips + ph],
            radius=ph // 2,
            fill=(11, 16, 32, 210),
            outline=(129, 140, 248, 110),
            width=SS,
        )
        d.text((cx_chip + pad_x_chip, y_chips + pad_y_chip),
               label, font=f_chip, fill=(241, 245, 249, 255))
        cx_chip += pw + int(height * 0.014) * SS

    return img.resize((width, height), Image.LANCZOS)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> None:
    out_dir = Path(__file__).resolve().parent.parent / "public"
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs = [
        ("logo-mark.png", lambda: draw_mark(512)),
        ("logo.png", lambda: draw_lockup(1280, 320, dark=False)),
        ("logo-dark.png", lambda: draw_lockup(1280, 320, dark=True)),
        ("favicon.png", lambda: draw_mark(64)),
        ("favicon-32.png", lambda: draw_mark(32)),
        ("apple-touch-icon.png", lambda: draw_mark(180)),
        ("og-image.png", lambda: draw_og()),
    ]

    for name, factory in jobs:
        img = factory()
        path = out_dir / name
        img.save(path, "PNG", optimize=True)
        print(f"  {name:24s} {img.size[0]}x{img.size[1]}  {path.stat().st_size / 1024:6.1f} KB")


if __name__ == "__main__":
    main()
