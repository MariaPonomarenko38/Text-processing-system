import cv2
import imutils
from skimage.filters import threshold_local
import numpy as np

from PIL import Image
from pytesseract import pytesseract

def order_points(pts):
    rect = np.zeros((4, 2), dtype = "float32")

    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

def four_point_transform(image, pts):
 
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")

    print(rect)
    print(dst)
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped

def main_func(path):
    img_path = path
    big_img = cv2.imread(img_path)

    ratio = big_img.shape[0] / 500.0
    org = big_img.copy()
    img = imutils.resize(big_img, height = 500)

    gray_img = cv2.cvtColor(img.copy(),cv2.COLOR_BGR2GRAY)

    blur_img = cv2.GaussianBlur(gray_img,(5,5),0)

    edged_img = cv2.Canny(blur_img,75,200)

    cnts, _ = cv2.findContours(edged_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
    doc = ''
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            doc = approx
            break

    p = []

    for d in doc:
        tuple_point = tuple(d[0])
        cv2.circle(img, tuple_point, 3, (0, 0, 255), 4)
        p.append(tuple_point)

    warped_image = four_point_transform(org, doc.reshape(4, 2) * ratio)
    warped_image = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)

    T = threshold_local(warped_image, 11, offset = 10, method = "gaussian")
    warped = (warped_image > T).astype("uint8") * 255

    pytesseract.tesseract_cmd = 'tesseract'
    text = pytesseract.image_to_string(imutils.resize(warped, height = 650))
    return text