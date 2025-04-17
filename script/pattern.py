"""
Minigrid:

Y  X ->
|
v
           dx
          |---|
          X   X   X   X
      - X   X   X   X
   dy |   X   X   X   X
      - X   X   X   X
          X   X   X   X
        X   X   X   X
          X   X   X   X
        X   X   X   X
          X   X   X   X
        X   X   X   X
          X   X   X   X
        X   X   X   X
          X   X   X   X
        X   X   X   X
          X   X   X   X
        X   X   X   X

Macrogrid:

           gapx
           |--|
        - MG  MG  MG
   gapy |
        - MG  MG  MG

          MG  MG  MG

"""
from PIL import Image, ImageDraw
import math

# Measurements in dx units
dx = 1.0
dy = dx * 0.620 # current estimate for aspect ratio...

# fairly close to actual gap sizes, probably
gapx = 0.26 # dx units
gapy = 0.3 # dy units

# *assume* that the pattern is centered at the center of the middle minigrid...
center_x = 3.3/2 # dx units
center_y = 8/2 # dy units

pts = []
for mgx in (0, -1, 1):
    for mgy in (0, -1, 1):
        tl_x = -center_x + (4 + gapx) * mgx
        tl_y = -center_y + (8 + gapy) * mgy
        for yy in range(16):
            even = (yy % 2 == 0)
            for xx in range(4):
                pts.append(((tl_x + xx + 0.5 * even) * dx, (tl_y + yy * 0.5) * dy))

def project_pts(pts3):
    res = []
    for x, y, z in pts3:
        res.append((x/z, y/z))
    return res

def render_pts(pts3, scale):
    pts = project_pts(pts3)
    xmin, xmax = min(x for (x, y) in pts), max(x for (x, y) in pts)
    ymin, ymax = min(y for (x, y) in pts), max(y for (x, y) in pts)

    margin_x = (xmax - xmin) * 0.08
    margin_y = (ymax - ymin) * 0.08
    im_ox = xmin - margin_x
    im_oy = ymin - margin_y
    im_sx = xmax - xmin + margin_x * 2
    im_sy = ymax - ymin + margin_y * 2

    im = Image.new("L", (int(round(im_sx * scale)), int(round(im_sy * scale))))
    draw = ImageDraw.Draw(im)
    for i, (x, y) in enumerate(pts):
        px = (x - im_ox) * scale
        py = (y - im_oy) * scale
        fx = px - int(px)
        fy = py - int(py)
        px = int(px)
        py = int(py)

        im.putpixel((px, py), int(round(255 * (1 - fx) * (1 - fy))))
        im.putpixel((px + 1, py), int(round(255 * fx * (1 - fy))))
        im.putpixel((px, py + 1), int(round(255 * (1 - fx) * fy)))
        im.putpixel((px + 1, py + 1), int(round(255 * fx * fy)))
        draw.text((px, py+10), str(i), fill=(255,))

    return im

def render_pts_on_reference(pts3, scale):
    pts = project_pts(pts3)
    xmin, xmax = min(x for (x, y) in pts), max(x for (x, y) in pts)
    ymin, ymax = min(y for (x, y) in pts), max(y for (x, y) in pts)

    im = Image.open("ir-whole-pattern-adjusted.png").convert("RGB")

    # minigrid (0, 0), grid point (0, 0)
    anchor_x, anchor_y = (555, 834)

    im_ox = pts[0][0] - anchor_x / scale
    im_oy = pts[0][1] - anchor_y / scale

    for x, y in pts:
        px = (x - im_ox) * scale
        py = (y - im_oy) * scale
        fx = px - int(px)
        fy = py - int(py)
        px = int(px)
        py = int(py)

        im.putpixel((px, py), (int(round(255 * (1 - fx) * (1 - fy))), 0, 0))
        im.putpixel((px + 1, py), (int(round(255 * fx * (1 - fy))), 0, 0))
        im.putpixel((px, py + 1), (int(round(255 * (1 - fx) * fy)), 0, 0))
        im.putpixel((px + 1, py + 1), (int(round(255 * fx * fy)), 0, 0))

    return im

def sphere_pt1(pt):
    # Treat x, y as being points on an image plane,
    # orthographically projected onto the surface of the sphere
    x, y = pt
    # calibrated scale
    scale = 0.0635
    x *= scale
    y *= scale
    # x, y idealized point on image plane
    # z corresponding sphere point
    z = (1 - (x * x + y * y)) ** 0.5

    return (x, y, z)

def sphere_pt2(pt):
    # Treat x, y as being independent angles on the sphere.
    # (I don't know if this is mathematically accurate!)
    x, y = pt
    scale = 3.75
    x = math.sin(math.radians(x * scale))
    y = math.sin(math.radians(y * scale))
    # x, y idealized point on image plane
    # z corresponding sphere point
    z = (1 - (x * x + y * y)) ** 0.5

    return (x, y, z)

sphere_pts = [sphere_pt1(pt) for pt in pts]

if __name__ == "__main__":
    # rendering square image
    im = render_pts([(x, y, 1) for (x, y) in pts], 100)
    im.save("pattern-square.png")


    # rendering sphere image
    # There are many ways to interpret the coordinates...
    im = render_pts([sphere_pt1(pt) for pt in pts], 500)
    im.save("pattern-sphere.png")

    #im = render_pts([sphere_pt1((x, y+dy)) for (x, y) in pts], 500)
    #im.save("pattern-sphere-shifted.png")

    im = render_pts_on_reference([sphere_pt1(pt) for pt in pts], 917)
    im.save("pattern-reference-sphere.png")
    #im.show()


    im = render_pts([sphere_pt2(pt) for pt in pts], 528)
    im.save("pattern-sphere2.png")

    #im = render_pts([sphere_pt2((x, y+dy)) for (x, y) in pts], 500)
    #im.save("pattern-sphere2-shifted.png")

    im = render_pts_on_reference([sphere_pt2(pt) for pt in pts], 910)
    im.save("pattern-reference-sphere2.png")
    im.show()

    import matplotlib.pyplot as plt
    import numpy as np
    fig = plt.figure("Pattern on the sphere")
    ax = fig.add_subplot(111, projection='3d')
    ax.cla()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_aspect('equal')
    sphere_pts = np.asarray(sphere_pts)
    ax.scatter(sphere_pts[:,0], sphere_pts[:,1], sphere_pts[:,2])
    plt.show(block=True)
