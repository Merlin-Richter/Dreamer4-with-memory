"""Recognize fixed-position card tiles by RGB template MSE; no learned classifier.

Labels 0..7 follow envs.memory_cards_env.IDENTITIES; 8 is the back; -1 means
unrecognisable. Score the inner 75% of each tile, excluding the shared border.
The threshold is an absolute MSE in [0,1] RGB, applied to the nearest template.
Border corruption is outside this check. Near-valid mixtures can be accepted.

Visual examples and checks: python -m paper_figures.card_checker
"""
import numpy as np

from envs.memory_cards_env import IDENTITIES, card_tile

BACK = 8
UNRECOGNISABLE = -1
NAMES = tuple(name for name, _ in IDENTITIES) + ('back',)


def label_name(label):
    return 'unrecognisable' if label == UNRECOGNISABLE else NAMES[label]


class CardChecker:
    def __init__(self, tile_size=32, threshold=0.001):
        if tile_size < 16 or not np.isfinite(threshold) or threshold < 0:
            raise ValueError('Need tile_size >= 16 and a finite nonnegative threshold')
        self.tile_size = tile_size
        self.threshold = threshold
        self.border = tile_size // 8
        self.templates = np.stack([np.asarray(card_tile(i, tile_size)) for i in range(8)]
                                  + [np.asarray(card_tile(None, tile_size))]).astype(np.float32) / 255

    def classify_tiles(self, tiles):
        """(...,size,size,3) uint8 or float RGB [0,1] -> labels, nearest MSE.

        Output arrays have the leading input dimensions. A single tile returns
        scalar arrays. Rejected tiles retain their nearest MSE for inspection.
        """
        tiles = np.asarray(tiles)
        if tiles.shape[-3:] != (self.tile_size, self.tile_size, 3):
            raise ValueError('Expected RGB tiles matching the template size')
        if tiles.dtype == np.uint8:
            tiles = tiles.astype(np.float32) / 255
        elif not np.issubdtype(tiles.dtype, np.floating):
            raise ValueError('Use uint8 RGB or floating point RGB in [0,1]')
        if not np.isfinite(tiles).all() or tiles.min() < 0 or tiles.max() > 1:
            raise ValueError('RGB must be finite and in [0,1]; pixels are never silently clipped')
        b = self.border
        crops = tiles.reshape(-1, self.tile_size, self.tile_size, 3)[:, b:-b, b:-b]
        templates = self.templates[:, b:-b, b:-b]
        errors = ((crops[:, None] - templates[None]) ** 2).mean(axis=(2, 3, 4))
        nearest = errors.argmin(axis=1)
        best = errors[np.arange(len(nearest)), nearest]
        labels = np.where(best <= self.threshold, nearest, UNRECOGNISABLE)
        return labels.reshape(tiles.shape[:-3]), best.reshape(tiles.shape[:-3])

    def classify_boards(self, frames):
        """(...,4*size,4*size,3) RGB -> (...,4,4) row-major tile results."""
        frames = np.asarray(frames)
        s = self.tile_size
        if frames.shape[-3:] != (4*s, 4*s, 3):
            raise ValueError('Expected a 4x4 board at the template resolution')
        tiles = frames.reshape(*frames.shape[:-3], 4, s, 4, s, 3).swapaxes(-4, -3)
        return self.classify_tiles(tiles)
