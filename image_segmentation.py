from __future__ import division
import cv2
import numpy as np
import os
import sys
import argparse, time
from math import exp, pow
from augmentingPath import augmentingPath
from PRBeta import MaxFlow as   pushRelabelBeta
from Dinics import MaxFlowDinic
from tqdm import tqdm

graphCutAlgo = {"ap": augmentingPath, 
                "pr" : pushRelabelBeta,
                "di" : MaxFlowDinic}
SIGMA = 50
# LAMBDA = 1
ObjectColor, BackgroundColor = (255, 0, 0), (0, 0, 255) #In RGB Scale
ObjectCode, BackgroundCode = 1, 2
OBJ, BKG = "OBJ", "BKG"

CUTCOLOR = (0, 255, 0)

SOURCE, SINK = -2, -1
SF = 10
LOADSEEDS = False
# drawing = False

def show_image(image):
    windowname = "Segmentation"
    cv2.namedWindow(windowname, cv2.WINDOW_NORMAL)
    cv2.startWindowThread()
    cv2.imshow(windowname, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def plantSeed(image):

    def drawSeeds(x, y, pixelType):
        if pixelType == OBJ:
            color, code = ObjectColor, ObjectCode
        else:
            color, code = BackgroundColor, BackgroundCode
        cv2.circle(image, (x, y), radius, color, thickness)
        cv2.circle(seeds, (x // SF, y // SF), radius // SF, code, thickness)

    def onMouse(event, x, y, flags, pixelType):
        global drawing
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            drawSeeds(x, y, pixelType)
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            drawSeeds(x, y, pixelType)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False

    def paintSeeds(pixelType):
        print ("Planting", pixelType, "seeds")
        global drawing
        drawing = False
        windowname = "Plant " + pixelType + " seeds"
        cv2.namedWindow(windowname, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(windowname, onMouse, pixelType)
        while (1):
            cv2.imshow(windowname, image)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        cv2.destroyAllWindows()
    
    
    seeds = np.zeros(image.shape, dtype="uint8")
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    image = cv2.resize(image, (0, 0), fx=SF, fy=SF)

    radius = 10
    thickness = -1 # fill the whole circle
    global drawing
    drawing = False
    

    paintSeeds(OBJ)
    paintSeeds(BKG)
    return seeds, image



def IntDiff(ip, iq):
    penalty = 100 * exp(- pow(int(ip) - int(iq), 2) / (2 * pow(SIGMA, 2)))
    return penalty

def buildGraph(image):
    V = image.size + 2
    graph = np.zeros((V, V), dtype='int32')
    seeds, seededImage = plantSeed(image)
    K = makeLinks(graph, image, seeds)
    #makeTLinks(graph, seeds, K)
    return graph, seededImage

def makeLinks(graph, image, seeds):
    K = -10000
    row, col = image.shape
    for i in range(row):
        for j in range(col):
            x = i * col + j
            if i + 1 < row: # pixel below
                y = (i + 1) * col + j
                penalty = IntDiff(image[i][j], image[i + 1][j])
                graph[x][y] = graph[y][x] = penalty
                K = max(K, penalty)
            if j + 1 < col: # pixel to the right
                y = i * col + j + 1
                penalty = IntDiff(image[i][j], image[i][j + 1])
                graph[x][y] = graph[y][x] = penalty
                K = max(K, penalty)

    row, col = seeds.shape
    for i in range(row):
        for j in range(col):
            x = i * col + j
            if seeds[i][j] == ObjectCode:
                graph[SOURCE][x] = K
            elif seeds[i][j] == BackgroundCode:
                graph[x][SINK] = K


def displayCut(image, cuts):
    def colorPixel(i, j):
    	#input (image[i][j][0])
    	image[i][j][1] = 255
    	image[i][j][0] = image[i][j][2] = 0
    #input (image.shape)
    r, c, _ = image.shape
    #image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    for c in cuts:
        if c[0] != SOURCE and c[0] != SINK and c[1] != SOURCE and c[1] != SINK:
            colorPixel(c[0] // r, c[0] % r)
            colorPixel(c[1] // r, c[1] % r)
    return image
    


def imageSegmentation(imagefile, size=(30, 30), algo="ff"):
    pathname = os.path.splitext(imagefile)[0]
    image = cv2.imread(imagefile, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, size)
    print ('Building Graph')
    graph, seededImage = buildGraph(image)
    cv2.imwrite(pathname + "seeded.jpg", seededImage)

    global SOURCE, SINK
    SOURCE += len(graph) 
    SINK   += len(graph)
    print ('Performing graph cut')
    now = time.time()
    cuts = graphCutAlgo[algo](graph, SOURCE, SINK)
    print ('Time taken : ', time.time() - now)
    print ("cuts:")
    print (cuts)
    image = cv2.imread(imagefile)
    image = cv2.resize(image, size)
    image = displayCut(image, cuts)
    image = cv2.resize(image, (0, 0), fx=SF, fy=SF)
    #image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
    show_image(image)
    savename = pathname + "cut.jpg"
    cv2.imwrite(savename, image)
    print ("Saved image as", savename)
    

def parseArgs():
    def algorithm(string):
        if string in graphCutAlgo:
            return string
        raise argparse.ArgumentTypeError(
            "Algorithm should be one of the following:", graphCutAlgo.keys())

    parser = argparse.ArgumentParser()
    parser.add_argument("imagefile")
    parser.add_argument("--size", "-s", 
                        default=30, type=int)
    parser.add_argument("--algo", "-a", default="ap", type=algorithm)
    return parser.parse_args()

import re
def color_image(image_path, size):
	f = open('cute_peacock.txt', 'r')
	x = f.readlines()[0]
	#x = x.split()
	temp = re.findall(r'\d+', x) 
	res = list(map(int, temp)) 
	image = cv2.imread(image_path)
	image = cv2.resize(image, (size*10,size*10))
	print (image.shape)
	cords = []
	for i in range(len(res)):
		cords.append(res[i])
		if len(cords)%2==0:
			image[cords[0]][cords[1]][:] = [255,0,0]

	show_image(image)

if __name__ == "__main__":

    args = parseArgs()
    imageSegmentation(args.imagefile, (args.size, args.size), args.algo)
