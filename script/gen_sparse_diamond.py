import numpy as np
from pattern import *

def render_pts_(pts3, scale):
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
    
    arr = np.zeros((64*9,2))
    # for i, (x, y) in enumerate(pts):
    #     arr[i] = np.array([x,y])
    # arr = arr.astype(int)
    for i, (x, y) in enumerate(pts):
        px = (x - im_ox) * scale
        py = (y - im_oy) * scale
        fx = px - int(px)
        fy = py - int(py)
        px = int(px)
        py = int(py)
        # if i == 0 or i == 4 or i == 8:
        #     print(px-572,py-665)

        im.putpixel((px, py), int(round(255 * (1 - fx) * (1 - fy))))
        im.putpixel((px + 1, py), int(round(255 * fx * (1 - fy))))
        im.putpixel((px, py + 1), int(round(255 * (1 - fx) * fy)))
        im.putpixel((px + 1, py + 1), int(round(255 * fx * fy)))
        draw.text((px, py+10), str(i), fill=(255,))

        arr[i] = np.array([px,py])
    arr = arr.astype(int)
    return arr

def render_pts_on_reference_(pts3, scale):
    pts = project_pts(pts3)
    xmin, xmax = min(x for (x, y) in pts), max(x for (x, y) in pts)
    ymin, ymax = min(y for (x, y) in pts), max(y for (x, y) in pts)

    im = Image.open("ir-whole-pattern-adjusted.png").convert("RGB")
    draw = ImageDraw.Draw(im)

    # minigrid (0, 0), grid point (0, 0)
    anchor_x, anchor_y = (555, 834)

    im_ox = pts[0][0] - anchor_x / scale
    im_oy = pts[0][1] - anchor_y / scale

    arr = np.zeros((64*9,2))
    for i, (x, y) in enumerate(pts):
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
        draw.text((px, py+10), str(i), fill=(255,))

        arr[i] = np.array([px,py])
    arr = arr.astype(int)
    return arr

idx = np.arange(64).astype(int)
pattern1_ = idx[(idx/4).astype(int) % 4==0]
pattern2_ = idx[(idx/4).astype(int) % 4==1]
pattern3_ = idx[(idx/4).astype(int) % 4==2]
pattern4_ = idx[(idx/4).astype(int) % 4==3]

pattern1_ = pattern1_.reshape(-1,4)
pattern2_ = pattern2_.reshape(-1,4)
pattern3_ = pattern3_.reshape(-1,4)
pattern4_ = pattern4_.reshape(-1,4)

diamond1_, diamond2_ = [], []
diamond3_, diamond4_ = [], []
for row in np.arange(0,len(pattern1_)-1):
    for col in range(3):
        dia1 = [pattern1_[row][col],pattern1_[row][col+1],pattern1_[row+1][col+1],pattern1_[row+1][col]]
        dia2 = [pattern2_[row][col],pattern2_[row][col+1],pattern2_[row+1][col+1],pattern2_[row+1][col]]
        dia3 = [pattern3_[row][col],pattern3_[row][col+1],pattern3_[row+1][col+1],pattern3_[row+1][col]]
        dia4 = [pattern4_[row][col],pattern4_[row][col+1],pattern4_[row+1][col+1],pattern4_[row+1][col]]
        diamond1_.append(dia1)
        diamond2_.append(dia2)
        diamond3_.append(dia3)
        diamond4_.append(dia4)
diamond1_ = np.array(diamond1_)
diamond2_ = np.array(diamond2_)
diamond3_ = np.array(diamond3_)
diamond4_ = np.array(diamond4_)


diamond1, diamond2, diamond3, diamond4 = [], [], [], []
for i in range(9):
    # add 64 to each elements in list pattern1_

    diamond1 += (diamond1_ + 64*i).tolist()
    diamond2 += (diamond2_ + 64*i).tolist()
    diamond3 += (diamond3_ + 64*i).tolist()
    diamond4 += (diamond4_ + 64*i).tolist()

def to_image_coords(xyzs):
    # divide out the z coordinate to obtain image plane coordinates
    return xyzs[..., :2] / xyzs[..., 2:3]

def get_pattern_model():
    pattern_model = np.zeros((64*9,2)).astype(int)
    pat_pts = to_image_coords(np.array(sphere_pts))
    for i, (x, y) in enumerate(pat_pts):
        px = int(round(x * 600)) + 500
        py = int(round(y * 600)) + 500
        pattern_model[i,0] = px
        pattern_model[i,1] = py
    return pattern_model

pattern_model = get_pattern_model()

with open("sparse_diamonds.py", "w") as outf:

    print("## Automatically generated by gen_diamond.py", file=outf)

    print("sphere_pts = [", file=outf)
    for (x, y, z) in sphere_pts:
        print(f"    [{x}, {y}, {z}],", file=outf)
    print("]", file=outf)
    
    print("pattern_model = [", file=outf)
    for x, y in pattern_model:
        print(f"    [{x}, {y}],", file=outf)
    print("]", file=outf)

    print("diamond1 = [", file=outf)
    for a, b, c, d in diamond1:
        print(f"    [{a}, {b}, {c}, {d}],", file=outf)
    print("]", file=outf)

    print("diamond2 = [", file=outf)
    for a, b, c, d in diamond2:
        print(f"    [{a}, {b}, {c}, {d}],", file=outf)
    print("]", file=outf)

    print("diamond3 = [", file=outf)
    for a, b, c, d in diamond3:
        print(f"    [{a}, {b}, {c}, {d}],", file=outf)
    print("]", file=outf)

    print("diamond4 = [", file=outf)
    for a, b, c, d in diamond4:
        print(f"    [{a}, {b}, {c}, {d}],", file=outf)
    print("]", file=outf)