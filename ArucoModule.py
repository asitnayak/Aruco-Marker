import cv2 as cv
import numpy as np
import os


def loadImgAug(path):
    myList = os.listdir(path)
    noOfMarkers = len(myList)
    print(f"Total number of markers detected : {noOfMarkers}")
    augDics = {}
    for imgPath in myList:
        key = int(os.path.splitext(imgPath)[0])
        imgAug = cv.imread(os.path.join(path, imgPath))
        augDics[key] = imgAug
    return augDics


def findArucoMarkers(img, markerSize=6, totalMarkers=250, draw=True):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    key = getattr(cv.aruco, f'DICT_{markerSize}X{markerSize}_{totalMarkers}')
    arucoDict = cv.aruco.Dictionary_get(key)
    arucoParam = cv.aruco.DetactorParameters_create()
    bboxs, ids, rejected = cv.aruco.detectMarkers(gray, arucoDict, parameters=arucoParam)

    if draw:
        cv.aruco.drawDetectedMarkers(img, bboxs)

    return [bboxs, ids]


def augmentAruco(bbox, id, img, imgAug, drawId=True):
    
    tl = bbox[0][0][0], bbox[0][0][1]
    tr = bbox[0][1][0], bbox[0][1][1]
    br = bbox[0][2][0], bbox[0][2][1]
    bl = bbox[0][3][0], bbox[0][3][1]

    h, w, c = imgAug.shape

    pts1 = np.array([tl,tr, br, bl])
    pts2 = np.array([[0,0], [w,0], [w,h], [0,h]])
    matrix, _ = cv.findHomography(pts2, pts1)
    imgOut = cv.warpPerspective(imgAug, matrix, (img.shape[1], img.shape[0]))
    cv.fillConvexPoly(img, pts1.astype(int), (0,0,0))

    imgOut = img + imgOut

    if drawId:
        cv.putText(imgOut, str(id), tl, cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

    return imgOut



def main():
    cap = cv.VideoCapture(1)
    augDics = loadImgAug("Markers")
    while True:
        success, img = cap.read()
        arucoFound = findArucoMarkers(img)

        # Loop through all the markers and augment them
        if len(arucoFound[0]) > 0:
            for bbox, id in zip(arucoFound[0], arucoFound[1]):
                if int(id) in augDics.keys():
                    img = augmentAruco(bbox, id, img, augDics[int(id)])

        cv.imread("Image", img)
        cv.waitKey(1)


if __name__ == "__main__":
    main()