import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.image as mpimg
img = mpimg.imread('inputs/figures_png/Fig1_combined_recoils.png')
print(img.shape, img.dtype, img.min(), img.max())
rgb = (img[..., :3] * 255).astype(int) if img.max() <= 1.0 else img[..., :3].astype(int)
H, W = rgb.shape[:2]
# gray shading: look for pixels close to (128,128,128)+-? find the dominant gray
def mask_color(c, tol):
    return (np.abs(rgb - np.array(c)).sum(-1) < tol)
for c in [(128,128,128),(140,86,75),(148,103,189),(214,39,40),(31,119,180),(44,160,44)]:
    m = mask_color(c, 40)
    ys, xs = np.nonzero(m)
    print(c, m.sum(), (xs.min(), xs.max(), ys.min(), ys.max()) if m.sum() else None)
# brown curve: for each column, the median row of brown pixels in the bottom half
m = mask_color((140,86,75), 60)
ys, xs = np.nonzero(m)
sel = ys > H//2
ys, xs = ys[sel], xs[sel]
cols = np.unique(xs)
print("brown cols", cols.min(), cols.max(), len(cols))
# gray extents in bottom half
g = mask_color((128,128,128), 30)
gy, gx = np.nonzero(g)
sel = gy > H//2
gy, gx = gy[sel], gx[sel]
print("gray bottom-half x range", gx.min(), gx.max(), "y range", gy.min(), gy.max())
# columns that are gray in bottom half: find the two blocks
gcols = np.unique(gx)
breaks = np.nonzero(np.diff(gcols) > 1)[0]
print("gray col blocks:", gcols[0], gcols[breaks], gcols[breaks+1], gcols[-1])
# find dark tick marks left of the left spine: scan for black pixels in columns left of gray block start
left = gcols[0]
blk = (rgb.sum(-1) < 150)
for x in range(left-12, left+2):
    rows = np.nonzero(blk[H//2:, x])[0] + H//2
    print(x, len(rows), rows[:40])
