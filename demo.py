import frontalize
import facial_feature_detector as feature_detection
import camera_calibration as calib
import scipy.io as io
import cv2
import numpy as np
import os
# import check_resources as check
import matplotlib.pyplot as plt


this_path = os.path.dirname(os.path.abspath(__file__))
JAWLINE_POINTS = list(range(0, 17))
RIGHT_EYEBROW_POINTS = list(range(17, 22))
LEFT_EYEBROW_POINTS = list(range(22, 27))
NOSE_POINTS = list(range(27, 36))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
MOUTH_OUTLINE_POINTS = list(range(48, 61))
MOUTH_INNER_POINTS = list(range(61, 68))

def demo():
    # check for dlib saved weights for face landmark detection
    # if it fails, dowload and extract it manually from
    # http://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2
    # check.check_dlib_landmark_weights()
    # load detections performed by dlib library on 3D model and Reference Image
    model3D = frontalize.ThreeD_Model(this_path + "/frontalization_models/model3Ddlib.mat", 'model_dlib')
    # load query image
    img = cv2.imread("datasets/train/0/00431.jpg", 1)
    height, width,c = np.shape(img)
    minV = min(height, width)
    startX = int((width - minV) / 2)
    startY = int((height - minV) / 2)
    img = img[startY:startY + minV, startX:startX + minV]

    plt.title('Query Image')
    plt.imshow(img[:, :, ::-1])
    # extract landmarks from the query image
    # list containing a 2D array with points (x, y) for each face detected in the query image
    lmarks = feature_detection.get_landmarks(img)
    # plt.figure()
    # plt.title('Landmarks Detected')
    # plt.imshow(img[:, :, ::-1])
    llmarks=[1]
    hullIndex = cv2.convexHull(lmarks[0], returnPoints=False)
    hullIndex=hullIndex.flatten()
    llmarks[0]=lmarks[0][hullIndex]
    # height,width,c=np.shape(img)
    # mask = np.zeros((height, width, c), np.uint8)
    # contour = llmarks[0]
    # contour = np.array(contour).reshape((-1, 1, 2)).astype(np.int32)
    # cv2.fillPoly(mask, [contour], (255, 255, 255))
    # mask = cv2.bitwise_not(mask)
    # img = cv2.subtract(img, mask)
    #
    # x, y, w, h = cv2.boundingRect(contour)
    # img_crop = img[y:y + h, x:x + w, :]

    plt.scatter(llmarks[0][:, 0], llmarks[0][:, 1])
    # perform camera calibration according to the first face detected
    proj_matrix, camera_matrix, rmat, tvec = calib.estimate_camera(model3D, lmarks[0])
    # load mask to exclude eyes from symmetry
    eyemask = np.asarray(io.loadmat('frontalization_models/eyemask.mat')['eyemask'])
    # perform frontalization
    frontal_raw, frontal_sym = frontalize.frontalize(img, proj_matrix, model3D.ref_U, eyemask)
    plt.figure()
    plt.title('Frontalized no symmetry')
    plt.imshow(frontal_raw[:, :, ::-1])
    plt.figure()
    plt.title('Frontalized with soft symmetry')
    cv2.imwrite('FrameFrontal.jpg', frontal_sym)
    plt.imshow(frontal_sym[:, :, ::-1])
    plt.figure()
    plt.title('contour')
    height, width, c = np.shape(frontal_sym)
    mask = np.zeros((height, width,c), np.uint8)
    contour=llmarks[0]
    contour=np.array(contour).reshape((-1, 1, 2)).astype(np.int32)
    cv2.fillPoly(mask,[contour],(255,255,255))
    mask=cv2.bitwise_not(mask)
    x, y, w, h = cv2.boundingRect(contour)
    img=cv2.subtract(frontal_sym,mask)
    img_crop = img[y:y+h, x:x+w,:]
    plt.imshow(img[:, :, ::-1])
    plt.figure()
    plt.title('contour2')
    # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    plt.imshow(img[:, :, ::-1])
    plt.show()
    return lmarks


if __name__ == "__main__":
    lmarks=demo()