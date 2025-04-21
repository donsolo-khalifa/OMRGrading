import cv2
import numpy as np


## TO STACK ALL THE IMAGES IN ONE WINDOW
def stackImages(imgArray, scale, lables=[]):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]

    if rowsAvailable:
        for x in range(rows):
            for y in range(cols):
                imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                if len(imgArray[x][y].shape) == 2:
                    imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
            hor_con[x] = np.concatenate(imgArray[x])
        ver = np.vstack(hor)
        ver_con = np.concatenate(hor)
    else:
        for x in range(rows):
            imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            if len(imgArray[x].shape) == 2:
                imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        hor_con = np.concatenate(imgArray)
        ver = hor

    if len(lables) != 0:
        eachImgWidth = int(ver.shape[1] / cols)
        eachImgHeight = int(ver.shape[0] / rows)
        for d in range(rows):
            for c in range(cols):
                cv2.rectangle(ver,
                              (c * eachImgWidth, eachImgHeight * d),
                              (c * eachImgWidth + len(lables[d][c]) * 13 + 27, 30 + eachImgHeight * d),
                              (255, 255, 255), cv2.FILLED)
                cv2.putText(ver, lables[d][c],
                            (eachImgWidth * c + 10, eachImgHeight * d + 20),
                            cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 255), 2)
    return ver


def reorder(myPoints):
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), np.int32)
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]  # Top-left point
    myPointsNew[3] = myPoints[np.argmax(add)]  # Bottom-right point
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]  # Top-right point
    myPointsNew[2] = myPoints[np.argmax(diff)]  # Bottom-left point

    return myPointsNew


def rectContour(contours):
    rectCon = []
    for i in contours:
        area = cv2.contourArea(i)
        if area > 15:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if len(approx) == 4:
                rectCon.append(i)
    rectCon = sorted(rectCon, key=cv2.contourArea, reverse=True)
    return rectCon


def getCornerPoints(cont):
    peri = cv2.arcLength(cont, True)
    approx = cv2.approxPolyDP(cont, 0.02 * peri, True)
    return approx


def splitBoxes(img, questions=5, choices=5):
    """
    Splits the given image into boxes given the number of questions (rows)
    and choices (columns).
    """
    rows = np.vsplit(img, questions)
    boxes = []
    for r in rows:
        cols = np.hsplit(r, choices)
        for box in cols:
            boxes.append(box)
    return boxes


def drawGrid(img, questions=5, choices=5):
    """
    Draws a grid on the provided image based on the number of questions (rows)
    and choices (columns).
    """
    secW = int(img.shape[1] / choices)  # Width of each cell
    secH = int(img.shape[0] / questions)  # Height of each cell

    # Draw horizontal lines
    for i in range(questions + 1):
        pt1 = (0, secH * i)
        pt2 = (img.shape[1], secH * i)
        cv2.line(img, pt1, pt2, (255, 255, 0), 2)

    # Draw vertical lines
    for j in range(choices + 1):
        pt1 = (secW * j, 0)
        pt2 = (secW * j, img.shape[0])
        cv2.line(img, pt1, pt2, (255, 255, 0), 2)

    return img


def showAnswers(img, myIndex, grading, ans, questions=5, choices=5):
    """
    Draws circles corresponding to the user's selected answers, using green for
    correct answers and red for wrong ones, while indicating the correct answer.
    """
    secW = int(img.shape[1] / choices)
    secH = int(img.shape[0] / questions)

    for x in range(questions):
        myAns = myIndex[x]
        cX = (myAns * secW) + secW // 2
        cY = (x * secH) + secH // 2
        if grading[x] == 1:
            myColor = (0, 255, 0)  # Green for correct
            cv2.circle(img, (cX, cY), 15, myColor, cv2.FILLED)
        else:
            myColor = (0, 0, 255)  # Red for wrong
            cv2.circle(img, (cX, cY), 15, myColor, cv2.FILLED)
            # Draw the correct answer indicator
            correctAns = ans[x]
            cv2.circle(img, ((correctAns * secW) + secW // 2, (x * secH) + secH // 2),
                       15, (0, 255, 0), cv2.FILLED)
