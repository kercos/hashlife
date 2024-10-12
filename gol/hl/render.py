import numpy as np
import matplotlib.pyplot as plt

## utility to show a point collection as an image in Matplotlib
def render_img(pts, name=None, filepath=None, show=True, force_show=True):
    pts = np.array(pts)
    pts[:, 0] -= np.min(pts[:, 0])
    pts[:, 1] -= np.min(pts[:, 1])
    grays = np.zeros((int(np.max(pts[:, 1] + 1)), int(np.max(pts[:, 0] + 1))))

    for x, y, g in pts:
        grays[int(y), int(x)] = g

    if filepath:
        fig = plt.figure()
        plt.imshow(grays, cmap="bone")
        plt.axis("off")
        plt.savefig(filepath, bbox_inches='tight')
        plt.close(fig) # so it's not shown

    if show:
        plt.figure(name, figsize=(5, 5))
        plt.axis("off")
        plt.imshow(grays, cmap="bone")
        if force_show:
            plt.show()
