from PIL import Image
import pytesseract
import argparse
import cv2
import os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image to be OCR'd")
args = vars(ap.parse_args())
image = cv2.imread(args["image"])

def imresize(im,sz):
	#resize an image array
	pil_im = Image.fromarray(uint8(im))
	return array(pil_im.resize(sz))


def process_without_dilation(rgb):
    hasText = 0
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY);
    morphKernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, morphKernel)
    # binarize
    _, bw = cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # connect horizontally oriented regions
    morphKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
    connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, morphKernel)
    # find contours
    mask = np.zeros(bw.shape[:2], dtype="uint8");
    _,contours, hierarchy = cv2.findContours(connected, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    # filter contours
    idx = 0
    while idx >= 0:
       x,y,w,h = cv2.boundingRect(contours[idx]);
       # fill the contour
       cv2.drawContours(mask, contours, idx, (255, 255, 255), cv2.FILLED);
       # ratio of non-zero pixels in the filled region
       r = cv2.contourArea(contours[idx])/(w*h)
       if(r > 0.45 and h > 5 and w > 5 and w > h):
         cv2.rectangle(rgb, (x,y), (x+w,y+h), (0, 255, 0), 2)
         hasText = 1
       idx = hierarchy[0][idx][0]
    cv2.imwrite('done.jpg',rgb)
    text = pytesseract.image_to_string(Image.open("done.jpg"))
    return text,rgb, hasText

def process_with_dilation(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret,thresh1 = cv2.threshold(gray,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    dilated = cv2.dilate(thresh1,kernel,iterations = 3)
    _, contours, hierarchy = cv2.findContours(dilated,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    # for each contour found, draw a rectangle around it on original image
    for contour in contours:
       # get rectangle bounding contour
       [x,y,w,h] = cv2.boundingRect(contour)
       # discard areas that are too large
       if h>300 and w>300:
         continue
       # discard areas that are too small
       if h<40 or w<40:
         continue
       # draw rectangle around contour on original image
       cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,255),2)
    #write original image with added contours to disk
    cv2.imwrite("contoured.jpg", image)
    return image
txt,img, hasText = process_without_dilation(image)
print(txt)
print(hasText)
cv2.waitKey(0)
