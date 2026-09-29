"""Deterministic RGB hidden-card world. Dependencies: numpy, Pillow.

Actions 0..15 reveal a row-major slot (hiding the previous card).
16 hides all cards; 17 waits. No reward, matching, or automatic termination.
The diagnostic all-face-up renderer never modifies episode visibility.
"""
from pathlib import Path
import math
from functools import lru_cache
import numpy as np
from PIL import Image, ImageDraw

IDENTITIES = (
    ("circle", "#E64045"), ("triangle", "#1877D2"),
    ("square", "#15965B"), ("star", "#EEA600"),
    ("plus", "#9145CC"), ("diamond", "#EE7133"),
    ("ring", "#009BAD"), ("heart", "#D43888"),
)
__version__ = "0.1.0"


@lru_cache(maxsize=8)
def _cached_tiles(resolution):
    tiles = tuple(np.asarray(card_tile(i, resolution//4)) for i in range(8))
    return tiles, np.asarray(card_tile(None, resolution//4))


def card_tile(identity, size):
    """Draw at 4x then downsample; identity=None has one universal back."""
    scale = 4
    im = Image.new("RGB", (size * scale, size * scale), "#172436")
    d = ImageDraw.Draw(im)
    def box(coords):
        return tuple(round(x * size * scale) for x in coords)
    def poly(points, fill):
        d.polygon([(round(x*size*scale), round(y*size*scale)) for x,y in points], fill=fill)
    d.rounded_rectangle(box((.045,.045,.955,.955)), radius=round(.1*size*scale),
                        fill="#FAF8F2" if identity is not None else "#334C70")
    if identity is None:
        d.rounded_rectangle(box((.15,.15,.85,.85)), radius=round(.06*size*scale),
                            outline="#7894B7", width=max(1,round(.035*size*scale)))
        poly([(.5,.32),(.68,.5),(.5,.68),(.32,.5)], "#7894B7")
    else:
        name, color = IDENTITIES[identity]
        if name == "circle":
            d.ellipse(box((.23,.23,.77,.77)), fill=color)
        elif name == "triangle":
            poly([(.5,.18),(.82,.77),(.18,.77)],color)
        elif name == "square":
            d.rectangle(box((.24,.24,.76,.76)),fill=color)
        elif name == "star":
            points=[]
            for i in range(10):
                a=-math.pi/2+i*math.pi/5
                r=.34 if i%2==0 else .15
                points.append((.5+r*math.cos(a),.5+r*math.sin(a)))
            poly(points,color)
        elif name == "plus":
            d.rectangle(box((.38,.19,.62,.81)),fill=color)
            d.rectangle(box((.19,.38,.81,.62)),fill=color)
        elif name == "diamond":
            poly([(.5,.16),(.79,.5),(.5,.84),(.21,.5)],color)
        elif name == "ring":
            d.ellipse(box((.20,.20,.80,.80)),fill=color)
            d.ellipse(box((.35,.35,.65,.65)),fill="#FAF8F2")
        elif name == "heart":
            # Parametric silhouette, independent of fonts or platform assets.
            points=[]
            for t in np.linspace(0,2*math.pi,160):
                x=16*math.sin(t)**3
                y=13*math.cos(t)-5*math.cos(2*t)-2*math.cos(3*t)-math.cos(4*t)
                points.append((.5+x*.020,.46-y*.020))
            poly(points,color)
    return im.resize((size,size),Image.Resampling.LANCZOS)


class MemoryCards:
    rows = cols = 4
    num_cards = 16
    hide_action = 16
    wait_action = 17
    # Optional torus board shifts (enabled with shifts=True): 18 right, 19 left, 20 down, 21 up.
    # The whole board moves one tile with wrap-around; a visible card moves with it.
    shift_actions = {18: (0, 1), 19: (0, -1), 20: (1, 0), 21: (-1, 0)}
    num_actions = 18

    def __init__(self, resolution=128, seed=None, shifts=False):
        if not isinstance(resolution, int) or resolution < 64 or resolution % 4:
            raise ValueError("resolution must be an integer >=64 divisible by 4")
        self.resolution = resolution
        self.shifts = bool(shifts)
        self.num_actions = 22 if self.shifts else 18
        self.rng = np.random.default_rng(seed)
        self._tiles, self._back = _cached_tiles(resolution)
        self.reset(seed=seed)

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._board = self.rng.integers(0, 8, size=16)  # Every card has random identity
        self.visible = None
        self.steps = 0
        return self.render()

    def step(self, action):
        if isinstance(action, (bool,np.bool_)) or not isinstance(action,(int,np.integer)) or not 0 <= action < self.num_actions:
            raise ValueError(f"action must be an integer in [0,{self.num_actions-1}]")
        if action < self.num_cards:
            self.visible = int(action)
        elif action == self.hide_action:
            self.visible = None
        elif int(action) in self.shift_actions:
            dr, dc = self.shift_actions[int(action)]
            grid = self._board.reshape(4, 4)
            self._board = np.roll(np.roll(grid, dr, axis=0), dc, axis=1).reshape(16).copy()
            if self.visible is not None:
                r, c = divmod(self.visible, 4)
                self.visible = int(((r + dr) % 4) * 4 + (c + dc) % 4)
        self.steps += 1
        return self.render()

    def render(self, *, all_face_up=False):
        frame=np.empty((self.resolution,self.resolution,3),dtype=np.uint8)
        s=self.resolution//4
        for pos,identity in enumerate(self._board):
            r,c=divmod(pos,4)
            tile=self._tiles[int(identity)] if all_face_up or pos==self.visible else self._back
            frame[r*s:(r+1)*s,c*s:(c+1)*s]=tile
        return frame

    def evaluation_state(self):
        """Privileged metadata: never feed this to a model. Returns copies."""
        return {"board": self._board.copy(),
                "visible_position": -1 if self.visible is None else self.visible,
                "steps": self.steps}



if __name__ == "__main__":
    import matplotlib.pyplot as plt

    env = MemoryCards()

    env.reset()

    for x in range(20):
        obs = env.step(np.random.randint(0, 16))

        plt.imshow(obs)
        plt.pause(1)


