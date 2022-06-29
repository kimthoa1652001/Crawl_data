import cv2
import numpy as np
import matplotlib.pylab as plt

def Canny(image_gray):
    median_intensity = np.median(image_gray)
    lower_threshold = int(max(0,(1.0-0.33)*median_intensity))
    upper_threshold = int(min(255,(1.0+0.33)*median_intensity))
    image_canny = cv2.Canny(image_gray,lower_threshold,upper_threshold)
    return image_canny
def Hough(image,image_canny):
    lines = cv2.HoughLines(image_canny, 1, np.pi / 180, 100)
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return image
def drow_the_line(img,lines):
    img = np.copy(img)
    blank_image = np.zeros((img.shape[0],img.shape[1],3), dtype=np.uint8)

    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(blank_image,(x1,y1),(x2,y2),(0,255,0),2)
    img = cv2.addWeighted(img, 0.8 , blank_image, 1,0.0)
    return img
def region_of_interest(img,vertices):
    mask = np.zeros_like(img)
    #channel_count = img.shape[2]
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img,mask)
    return masked_image

image = cv2.imread("images.jpeg")
image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#cv2.imshow("Image",image)
#image_canny = Canny(image_gray)
#image_hough = Hough(image, image_canny)

#cv2.imshow("Image_Edge", image_canny)
#cv2.imshow("Image_Hough", image_hough)
#cv2.waitKey(0)
video = cv2.VideoCapture("vd1.mov")
while True:
    ret,frame = video.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    height = hsv.shape[0]
    width = hsv.shape[1]
    region = [(0,height),
            (width/2,height/2),
            (width,height) ]
    if not ret:
        video = cv2.VideoCapture("vd1.mov")
        continue
    low_yellow = np.array([25,50,70])
    up_yellow = np.array([30,255,255])
    #low_yellow = np.array([18, 94, 140])
    #up_yellow = np.array([48,255,255])
    mask = cv2.inRange(hsv,low_yellow,up_yellow)

    edges = cv2.Canny(mask,100,200)
    crop_ = region_of_interest(edges,np.array([region], np.int32))
    #lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, maxLineGap=50)
    lines = cv2.HoughLinesP(crop_, 1, np.pi/180, 50, maxLineGap=50)
    #if lines is not None:
    #    for line in lines:
    #        x1,y1,x2,y2 = line[0]
    #        cv2.line(frame,(x1,y1), (x2,y2), (0,255,0),2)
    try:
        frame = drow_the_line(frame,lines)
    except:
        pass
    cv2.imshow("frame", frame)

    key = cv2.waitKey(25)
    if key == 27:
        break
video.release()
cv2.destroyAllWindows()

