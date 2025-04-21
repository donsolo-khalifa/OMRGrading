import cv2
import numpy as np
import utils

########################################################################
# Settings
webCamFeed = False
pathImage = "phone (3).jpg"  # Path to your new sheet image
cap = cv2.VideoCapture(1)
# cap.set(10, 160)
heightImg = 700
widthImg = 700

# Configure the number of questions (rows) and choices (columns)
questions = 20  # Number of questions
choices = 5  # Number of options per question

# Answer key (adjust the index list according to the number of questions)
ans = [1, 2, 1, 1, 1, 1, 1, 1, 1, 1,1,1,1,1,1,1,1,1,1,1]
########################################################################

count = 0

while True:
    # Read the image either from webcam or from file
    if webCamFeed:
        success, img = cap.read()
    else:
        img = cv2.imread(pathImage)
    img = cv2.resize(img, (widthImg, heightImg))  # Resize image
    imgFinal = img.copy()
    imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8)  # Blank image for debugging
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)  # Gaussian blur
    imgCanny = cv2.Canny(imgBlur, 10, 70)  # Edge detection

    try:
        # Find contours in the image
        imgContours = img.copy()
        imgBigContour = img.copy()
        contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10)
        rectCon = utils.rectContour(contours)

        # Assume the largest contour is the answers area and the second is the grade box.
        biggestPoints = utils.getCornerPoints(rectCon[0])
        gradePoints = utils.getCornerPoints(rectCon[1])

        if biggestPoints.size != 0 and gradePoints.size != 0:
            # --- Warp the answers area ---
            biggestPoints = utils.reorder(biggestPoints)
            cv2.drawContours(imgBigContour, biggestPoints, -1, (0, 255, 0), 20)
            pts1 = np.float32(biggestPoints)
            pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

            # --- Warp the grade area if needed, but for score display we will use the detected coordinates ---
            cv2.drawContours(imgBigContour, gradePoints, -1, (255, 0, 0), 20)
            gradePoints = utils.reorder(gradePoints)
            # Optionally, you can still compute a warped grade image if needed:
            ptsG1 = np.float32(gradePoints)
            # ptsG2 can be set to a default size (only used if needed)
            ptsG2 = np.float32([[0, 0], [325, 0], [0, 150], [325, 150]])
            matrixG = cv2.getPerspectiveTransform(ptsG1, ptsG2)
            imgGradeDisplay = cv2.warpPerspective(img, matrixG, (325, 150))

            # --- Process the warped answers area for bubble detection ---
            imgWarpGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
            # Use OTSU for automatic threshold detection
            imgThresh = cv2.threshold(imgWarpGray, 0, 255,
                                      cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

            # imgThresh = cv2.threshold(imgWarpGray, 170, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_TRUNC)[1]

            # Split the warped threshold image into individual answer boxes.
            boxes = utils.splitBoxes(imgThresh, questions, choices)
            # Show one of the split boxes for debugging
            # cv2.imshow("Split Test", boxes[3])

            # Analyze each box: count non-zero pixels to decide marked answer.
            countR = 0
            countC = 0
            myPixelVal = np.zeros((questions, choices))
            for box in boxes:
                totalPixels = cv2.countNonZero(box)
                myPixelVal[countR][countC] = totalPixels
                countC += 1
                if countC == choices:
                    countC = 0
                    countR += 1

            # Determine which bubble is marked for each question.
            myIndex = []
            for x in range(questions):
                arr = myPixelVal[x]
                myIndexVal = np.where(arr == np.amax(arr))
                myIndex.append(myIndexVal[0][0])

            # Grade each question.
            grading = []
            for x in range(questions):
                if ans[x] == myIndex[x]:
                    grading.append(1)
                else:
                    grading.append(0)
            score = (sum(grading) / questions) * 100

            # --- Draw answers grid and circles on warped answers image ---
            utils.showAnswers(imgWarpColored, myIndex, grading, ans, questions, choices)
            utils.drawGrid(imgWarpColored, questions, choices)
            imgRawDrawings = np.zeros_like(imgWarpColored)
            utils.showAnswers(imgRawDrawings, myIndex, grading, ans, questions, choices)
            invMatrix = cv2.getPerspectiveTransform(pts2, pts1)
            imgInvWarp = cv2.warpPerspective(imgRawDrawings, invMatrix, (widthImg, heightImg))

            # --- Write the score directly on the original image using the grade box coordinates ---
            # Get the bounding rectangle for the grade box.
            x, y, w, h = cv2.boundingRect(gradePoints)
            # Calculate center or a preferred position for the score text
            centerX = x + w // 2
            centerY = y + h // 2
            # Optionally, adjust the position with an offset if needed.
            offsetX, offsetY = 15, 10  # Adjust this if the text isn't centered

            # Draw the score (adjust font scale and thickness based on your grade box size)
            if int(score)<50:
                cv2.putText(imgFinal, f"{int(score)}%", (centerX - 50 + offsetX, centerY + offsetY),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)
            else: cv2.putText(imgFinal, f"{int(score)}%", (centerX - 50 + offsetX, centerY + offsetY),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv2.LINE_AA)

            # Combine the warped bubbles overlay back onto the final image.
            imgFinal = cv2.addWeighted(imgFinal, 1, imgInvWarp, 1, 0)

            # Optionally display intermediate images for debugging:
            # cv2.imshow("Warped Answers", imgWarpColored)
            # cv2.imshow("Grade Display", imgGradeDisplay)
            # cv2.imshow("Final Result", imgFinal)

            # Image array for debugging display (stacked)
            imageArray = ([img, imgGray, imgCanny, imgContours],
                          [imgBigContour, imgThresh, imgWarpColored, imgFinal])
            cv2.imshow("Final Result", imgFinal)
        else:
            print("Grade and/or answer area not detected properly.")
    except Exception as e:
        print("Error:", e)
        imageArray = ([img, imgGray, imgCanny, imgContours],
                      [imgBlank, imgBlank, imgBlank, imgBlank])

    lables = [["Original", "Gray", "Edges", "Contours"],
              ["Biggest Contour", "Threshold", "Warped", "Final"]]
    stackedImage = utils.stackImages(imageArray, 0.5, lables)
    cv2.imshow("Result", stackedImage)

    key = cv2.waitKey(1) & 0xFF

    # Save the image when 's' key is pressed
    if key == ord('s'):
        cv2.imwrite("Scanned/myImage" + str(count) + ".jpg", imgFinal)
        cv2.rectangle(stackedImage, ((int(stackedImage.shape[1] / 2) - 230),
                                     int(stackedImage.shape[0] / 2) + 50),
                      (1100, 350), (0, 255, 0), cv2.FILLED)
        cv2.putText(stackedImage, "Scan Saved", (int(stackedImage.shape[1] / 2) - 200,
                                                 int(stackedImage.shape[0] / 2)), cv2.FONT_HERSHEY_DUPLEX, 3,
                    (0, 0, 255), 5, cv2.LINE_AA)
        cv2.imshow('Result', stackedImage)
        cv2.waitKey(300)
        count += 1

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

