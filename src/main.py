import cv2 as cv
import imutils
import pytesseract


path = "D:\\projects\\VISN-Eindopdracht\\data\\"
print(path)

img = cv.imread(path + 'suzuki_1.jpg', cv.IMREAD_COLOR)
img = cv.resize(img, (600, 400))
cv.imshow('resize', img)
cv.waitKey(0)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray = cv.bilateralFilter(gray, 11, 17, 17)

cv.imshow('gray', gray)
cv.waitKey(0)

edged = cv.Canny(gray, 30, 200)

cv.imshow('edged', edged)
cv.waitKey(0)

cnts,new = cv.findContours(edged.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
img1 = img.copy()
cv.drawContours(img1,cnts,-1,(0,255,0),3)

cv.imshow("img1",img1)
cv.waitKey(0)


cnts = sorted(cnts, key = cv.contourArea, reverse = True)[:30]
screenCnt = None #will store the number plate contour
img2 = img.copy()
cv.drawContours(img2,cnts,-1,(0,255,0),3) 
cv.imshow("img2",img2) #top 30 contours
cv.waitKey(0)

# loop over contours
for c in cnts:
  # approximate the contour
        peri = cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, 0.019 * peri, True)
        if len(approx) == 4: #chooses contours with 4 corners
                screenCnt = approx
                x,y,w,h = cv.boundingRect(c) #finds co-ordinates of the plate
                new_img=img[y:y+h,x:x+w]
                cv.imwrite('./test.png',new_img) #stores the new image
                break

#draws the selected contour on original image        
cv.drawContours(img, [screenCnt], -1, (0, 255, 0), 3)
cv.imshow("Final image with plate detected",img)
cv.waitKey(0)