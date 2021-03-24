# Written by xxx

import copy

__all__ = ["MutualTransform"]


class MutualTransform:
    """Apply the transformer more times on a same raw image."""

    def __init__(self, transformer, transformer2, times=2):
        self.transformer = transformer
        self.transformer2 = transformer2
        self.times = times

    def __call__(self, img):
        imgs = []
        for i in range(self.times):
            if i == 0:
                img_copy = copy.deepcopy(img)
                img_copy = self.transformer2(img_copy)
                imgs.append(img_copy)
            else:
                img_copy = copy.deepcopy(img)
                img_copy = self.transformer(img_copy)
                imgs.append(img_copy)

        return imgs

    def __repr__(self):
        return "Mutual Transformer"
