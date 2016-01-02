#!/usr/local/bin/python

import numpy as np
from PIL import Image, ImageDraw
from random import randint

def triangle_area(p1, p2, p3):
    ab = np.array(p2) - np.array(p1)
    ac = np.array(p3) - np.array(p1)
    return np.cross(ab,ac)/2.0

def draw_rand_triangle(path):
    width = 1
    while True:
        p1 = (randint(0,64),randint(0,64))
        p2 = (randint(0,64),randint(0,64))
        p3 = (randint(0,64),randint(0,64))

        if triangle_area(p1,p2,p3) > 100:
            break

    img = Image.new("RGB", (64, 64), "white")
    draw = ImageDraw.Draw(img)
    draw.line([p1, p2, p3, p1], fill=(0,0,0), width=width)
    img.save(path)

def draw_rand_circle(path):
    p1 = (randint(0,60),randint(0,60))
    p2 = (randint(p1[0]+4,64),randint(p1[1]+4,64))

    img = Image.new("RGB", (64, 64), "white")
    draw = ImageDraw.Draw(img)
    draw.arc([p1, p2], 0, 360, fill=(0,0,0))
    img.save(path)

for n in range(1,1001):
    draw_rand_triangle("triangle/%.04d.png" % n)
    draw_rand_circle("circle/%.04d.png" % n)
