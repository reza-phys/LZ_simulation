import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.image as mpimg
img = mpimg.imread('inputs/figures_png/Fig1_combined_recoils.png')
rgb = (img[..., :3] * 255).astype(int)
blk = rgb.sum(-1) < 200
y0, y1 = 1056, 1800
for x in range(192, 200):
    rows = np.nonzero(blk[y0:y1, x])[0] + y0
    if len(rows):
        grp = np.split(rows, np.nonzero(np.diff(rows) > 1)[0] + 1)
        print("col", x, [(int(r[0]), int(r[-1])) for r in grp])
# label text blocks left of spine
txt = blk[y0:y1, 90:180].any(1)
rows = np.nonzero(txt)[0] + y0
grp = np.split(rows, np.nonzero(np.diff(rows) > 3)[0] + 1)
print("label blocks:", [(int(r[0]), int(r[-1]), float(r.mean())) for r in grp])
# also top panel for reference (labels 10^-1 .. 10^9)
txt = blk[150:930, 90:180].any(1)
rows = np.nonzero(txt)[0] + 150
grp = np.split(rows, np.nonzero(np.diff(rows) > 3)[0] + 1)
print("top label blocks:", [(int(r[0]), int(r[-1]), float(r.mean())) for r in grp])
