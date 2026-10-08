"""Training augmentations aimed at generalising to unseen clips of people in vegetation.

Two failure sources dominate on footage from a new camera/site, so the augmentations target both:
- the camera: thermal/monochrome sensors, heavy compression from messaging apps, motion and focus blur,
  low light noise, low resolution;
- the scene: people partly hidden behind grass blades in the foreground (GrassOcclusion).

Boxes are left unchanged by every transform here, which matches how annotators label a person who
is partly behind grass (the box covers the person, not just the visible pixels).
"""

from __future__ import annotations

import albumentations as A
import cv2
import numpy as np


class GrassOcclusion(A.ImageOnlyTransform):
    """Draws a band of thin, randomly leaning grass blades over part of the image."""

    def __init__(self, blades_range: tuple[int, int] = (40, 250), p: float = 0.3):
        super().__init__(p=p)
        self.blades_range = blades_range

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        rng = np.random.default_rng()
        h, w = img.shape[:2]
        out = img.copy()
        # Blade colour sampled from the image itself so the overlay matches the scene's vegetation.
        sample = img[rng.integers(0, h, 64), rng.integers(0, w, 64)].astype(np.float32)
        base = sample[np.argmax(sample[:, 1] - sample.mean(1))]  # greenest sampled pixel
        band_top = int(rng.uniform(0.2, 0.8) * h)
        for _ in range(int(rng.integers(*self.blades_range))):
            x = int(rng.integers(0, w))
            y = int(rng.integers(band_top, h))
            length = int(rng.uniform(0.05, 0.35) * h)
            lean = int(rng.normal(0, 0.15) * length)
            colour = np.clip(base * rng.uniform(0.6, 1.4), 0, 255).astype(int).tolist()
            thick = int(rng.integers(1, max(2, w // 300)))
            cv2.line(out, (x, y), (x + lean, y - length), colour, thick, cv2.LINE_AA)
        return out

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("blades_range",)


def build_augmentations(strength: float = 1.0) -> list[A.BasicTransform]:
    """strength scales every probability (0 disables the extra augmentations, 1 = defaults below)."""
    s = max(0.0, strength)
    return [
        GrassOcclusion(p=0.3 * s),
        A.ToGray(p=0.15 * s),                                    # thermal / IR / night-vision footage
        A.CLAHE(p=0.1 * s),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.3 * s),
        A.RandomGamma(gamma_limit=(60, 160), p=0.2 * s),          # dusk and overexposed midday
        A.RandomShadow(num_shadows_limit=(1, 3), p=0.15 * s),     # dappled shade under trees
        A.OneOf([A.MotionBlur(blur_limit=(3, 9)), A.Defocus(radius=(1, 3)), A.GaussianBlur(blur_limit=(3, 5))],
                p=0.2 * s),                                        # hand-held, zoomed or out-of-focus cameras
        A.GaussNoise(std_range=(0.02, 0.08), p=0.15 * s),          # low-light sensor noise
        A.Downscale(scale_range=(0.4, 0.8), p=0.15 * s),           # low-resolution sources
        A.ImageCompression(quality_range=(25, 90), p=0.3 * s),    # WhatsApp/CCTV-grade compression
    ]
