"""
Pointillism (/ˈpɔɪntɪlɪzəm/) is a technique of painting in which small, distinct dots of color are applied in patterns to form an image.

Georges Seurat and Paul Signac developed the technique in 1886, branching from Impressionism
"""


"""
To do this in Python,first step is to compute most used colors for which KMeans is used.I used color palette of 20 which means dot will be made by 20 most used colors will be present in the image. A suitable radius size is calculated for the dots according to the image size. Then we loop over the image and find the color closest to the dot, using which the circle is drawn.
"""

#Importing Libraries
import scipy.spatial
import numpy as np
import random
import cv2
import math
from sklearn.cluster import KMeans

def compute_color_probabilities(pixels,palette):
    distances = scipy.spatial.distance.cdist(pixels,palette)
    maxima = np.amax(distances,axis=1)
    distances = maxima[:,None] - distances
    summ = np.sum(distances,1)
    distances /= summ[:,None]
    return distances


def get_color_from_prob(probabilities,palette):
    probs = np.argsort(probabilities)
    i = probs[-1]
    return palette[i]


def randomized_grid(h,w,scale):
    assert (scale>0)
    r = scale//2

    grid = []
    for i in range(0,h,scale):
        for j in range(0,w,scale):
            y = random.randint(-r,r) + i
            x = random.randint(-r,r) + j
    grid.append((y%h,x%w))

    random.shuffle(grid)
    return grid

def get_color_palette(img,n=20):
    clf = KMeans(n_clusters=20)
    clf.fit(img.reshape(-1,3))
    return clf.cluster_centers_

def complement(colors):
    return 255-colors

def create_pointilism_art(image_path,primay_colors):
    img = cv2.imread(image_path)
    radius_width = int(math.ceil(max(img.shape)/1000))
    palette = get_color_palette(img,primay_colors)
    complements = complement(palette)
    palette = np.vstack((palette,complements))
    canvas = img.copy()

    grid = randomized_grid(img.shape[0],img.shape[1],scale=3)

    pixed_colors = np.array([img[x[0],x[1]] for x in grid])

    color_probabilities = compute_color_probabilities(pixed_colors,palette)

    for i,(y,x) in enumerate(grid):
        color = get_color_from_prob(color_probabilities[i],palette)

        cv2.ellipse(canvas,(x,y),(radius_width,radius_width),0,0,360,color,-1,cv2.LINE_AA)

    return canvas


res = create_pointilism_art('Chadwick Boseman.jpg',20)
cv2.imshow('res',res)

cv2.waitKey(0)
cv2.destroyAllWindows()
