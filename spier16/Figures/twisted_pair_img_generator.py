import numpy as np
import matplotlib.pyplot as plt

ARC_ANGLE   = 30
ARC_RADIUS  = 120
TWISTS      = 3
SEPARATION  = 2
THICKNESS   = 12
COLOR_1     = '#185fa5'
COLOR_2     = '#185fa5'
OUTPUT_FILE = 'twisted_pair.svg'

STEPS = 1000
angle_rad   = np.radians(ARC_ANGLE)
# Begin at the bottom of the arc: the initial tangent points horizontally to
# the right, then the pair curves upward as it travels through the arc.
start_angle = -np.pi / 2
t = np.linspace(0, 1, STEPS)

a = start_angle + t * angle_rad
bx = ARC_RADIUS * np.cos(a)
by = ARC_RADIUS * np.sin(a)

# Correct perpendicular normal to the arc
tx = -np.sin(a)
ty =  np.cos(a)
nx = -ty
ny =  tx

wave = np.sin(t * TWISTS * 2 * np.pi)

w1x = bx + nx * wave * SEPARATION
w1y = by + ny * wave * SEPARATION
w2x = bx - nx * wave * SEPARATION
w2y = by - ny * wave * SEPARATION

fig, ax = plt.subplots(figsize=(6, 6))
ax.set_aspect('equal')
ax.axis('off')

# wire1 on top when wave > 0, wire2 on top when wave < 0
# Draw bottom layers first, then top layers
for (cx, cy, color, on_top_when_positive) in [
    (w2x, w2y, COLOR_2, False),  # wire2 bottom segments
    (w1x, w1y, COLOR_1, False),  # wire1 bottom segments
    (w2x, w2y, COLOR_2, True),   # wire2 top segments
    (w1x, w1y, COLOR_1, True),   # wire1 top segments
]:
    # wire1 is on top when wave > 0; wire2 is on top when wave < 0
    if color == COLOR_1:
        mask = wave > 0 if on_top_when_positive else wave <= 0
    else:
        mask = wave < 0 if on_top_when_positive else wave >= 0

    # Split mask into contiguous True segments
    padded = np.concatenate([[False], mask, [False]])
    starts = np.where(~padded[:-1] & padded[1:])[0]
    ends   = np.where( padded[:-1] & ~padded[1:])[0]

    for s, e in zip(starts, ends):
        # extend by 1 on each side so segments join cleanly
        s2 = max(0, s - 1)
        e2 = min(STEPS - 1, e + 1)
        ax.plot(cx[s2:e2], cy[s2:e2], color=color, linewidth=THICKNESS,
                solid_capstyle='round', solid_joinstyle='round')

plt.tight_layout(pad=0)
plt.savefig(OUTPUT_FILE, format=OUTPUT_FILE.split('.')[-1],
            bbox_inches='tight', transparent=True)
print(f'Saved to {OUTPUT_FILE}')
plt.show()
