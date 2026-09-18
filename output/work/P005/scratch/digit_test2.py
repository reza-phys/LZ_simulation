import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.image as mpimg
img = mpimg.imread('inputs/figures_png/Fig1_combined_recoils.png')
rgb = (img[..., :3] * 255).astype(int)
H, W = rgb.shape[:2]
y0, y1 = 1061, 1794   # bottom panel curve extent (axes box)
print("sample colours: left band", rgb[1700, 195], "right band", rgb[1700, 1500], "white", rgb[1700, 800], "spine?", rgb[1700, 190], rgb[1700, 189], rgb[1700, 1648])
band = rgb[1700, 1500]
g = (np.abs(rgb - band).sum(-1) < 12)
frac = g[y0:y1].mean(0)
cols = np.nonzero(frac > 0.5)[0]
br = np.nonzero(np.diff(cols) > 1)[0]
print("gray blocks (frac>0.5):", cols[0], [(cols[b], cols[b+1]) for b in br], cols[-1])
# tick marks: dark pixels in columns left of the spine
blk = rgb.sum(-1) < 200
for x in range(176, 192):
    rows = np.nonzero(blk[y0-5:y1+5, x])[0] + y0 - 5
    # group consecutive rows
    if len(rows):
        grp = np.split(rows, np.nonzero(np.diff(rows) > 1)[0] + 1)
        print(x, [(int(r[0]), int(r[-1])) for r in grp])
# same for bottom x ticks (below the bottom spine)
for y in range(1794, 1812):
    colsb = np.nonzero(blk[y, 150:1700])[0] + 150
    if len(colsb):
        grp = np.split(colsb, np.nonzero(np.diff(colsb) > 1)[0] + 1)
        print("row", y, [(int(r[0]), int(r[-1])) for r in grp][:30])
