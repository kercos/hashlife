import numpy as np
import matplotlib.pyplot as plt

FIG_NUM = 0

## utility to show a point collection as an image in Matplotlib
def render_img(pts):
    global FIG_NUM
    FIG_NUM += 1
    pts = np.array(pts)
    pts[:, 0] -= np.min(pts[:, 0])
    pts[:, 1] -= np.min(pts[:, 1])
    grays = np.zeros((int(np.max(pts[:, 1] + 1)), int(np.max(pts[:, 0] + 1))))

    for x, y, g in pts:
        grays[int(y), int(x)] = g

    plt.figure(FIG_NUM, figsize=(5, 5))
    plt.axis("off")

    plt.imshow(grays, cmap="bone")
