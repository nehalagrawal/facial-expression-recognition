
import cv2
import numpy as np
import dlib as dl
import scipy.io as io
import os
import camera_calibration as calib
import facial_feature_detector as feature_detection
import frontalize
this_path = os.path.dirname(os.path.abspath(__file__))


# cv2.namedWindow("preview")
def run(type):
    camera = cv2.VideoCapture(0)
    status, frame = 0, 0
    if camera.isOpened():  # try to get the first frame
        status, frame = camera.read()
        cv2.imwrite('ooo.jpg', frame)
        camera.release()
    else:
        rval = False
        exit(0)
    if status:
        height, width, c = np.shape(frame)
        minV = min(height, width)
        startX = int((width - minV) / 2)
        startY = int((height - minV) / 2)
        frame = frame[startY:startY + minV, startX:startX + minV]
        # use Effective Face Frontalization in Unconstrained Images to frontalize the face
        model3D = frontalize.ThreeD_Model(this_path + "/frontalization_models/model3Ddlib.mat", 'model_dlib')
        # Whether to extract the face first or frontalize the face first is important and need to be tried
        if (type==0):
            # I proceed with Frontalization first then I extract the face
            landmarks = feature_detection.get_landmarks(frame)
            # perform camera calibration according to the first face detected
            if len(landmarks) <= 0:
                print("could not detect face landmarks")
                return
            proj_matrix, camera_matrix, rmat, tvec = calib.estimate_camera(model3D, landmarks[0])
            # load mask to exclude eyes from symmetry
            eyemask = np.asarray(io.loadmat('frontalization_models/eyemask.mat')['eyemask'])
            # perform frontalization
            frontal_raw, frontal_sym = frontalize.frontalize(frame, proj_matrix, model3D.ref_U, eyemask)
            # then I extract the face and crop it
            cv2.imwrite('FrameFrontal.jpg', frontal_sym)
            landmarks = feature_detection.get_landmarks(frontal_sym)
            if len(landmarks) <= 0:
                print("could not detect face landmarks")
                return
            hullIndex = cv2.convexHull(landmarks[0], returnPoints=False)
            hullIndex = hullIndex.flatten()
            contour = landmarks[0][hullIndex]
            # And we can choose whether to use the one with soft-symmetry or raw. I use the one with soft symmetry
            height, width, channels = np.shape(frontal_sym)
            mask = np.zeros((height, width, channels), np.uint8)
            contour = np.array(contour).reshape((-1, 1, 2)).astype(np.int32)
            cv2.fillPoly(mask, [contour], (255, 255, 255))
            cv2.imwrite('wlfae.jpg', frontal_sym)
            mask = cv2.bitwise_not(mask)
            # make the background of the face black
            spotted_frame = cv2.subtract(frontal_sym, mask)
            # make a box of the contour of the face to crop the image
            x, y, w, h = cv2.boundingRect(contour)
            if w > h:
                diff = w - h
                h = w
                y = int(y - diff / 2)
            elif h > w:
                diff = h - w
                w = h
                x = int(x - diff / 2)
            frame_crop = spotted_frame[y:y + h, x:x + w, :]
            # then I change the cropped face to a greyscale
            frame_grey = cv2.cvtColor(frame_crop, cv2.COLOR_BGR2GRAY)
            # then I use adaptive histogram equalization to normalize the brightness and contrast of the face image
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(12, 12))
            framev2 = clahe.apply(frame_grey)
            resizedFrame = cv2.resize(framev2, (48, 48), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite('FrameCompress.jpg', resizedFrame)
            cv2.imwrite('framev2.jpg', framev2)
        elif type==1:
            # I extract the face first then I frontalize it
            landmarks = feature_detection.get_landmarks(frame)
            # perform camera calibration according to the first face detected
            if len(landmarks) <= 0:
                print("could not detect face landmarks")
                return

            hullIndex = cv2.convexHull(landmarks[0], returnPoints=False)
            hullIndex = hullIndex.flatten()
            contour = landmarks[0][hullIndex]
            # And we can choose whether to use the one with soft-symmetry or raw. I use the one with soft symmetry
            height, width, channels = np.shape(frame)
            mask = np.zeros((height, width, channels), np.uint8)
            contour = np.array(contour).reshape((-1, 1, 2)).astype(np.int32)
            cv2.fillPoly(mask, [contour], (255, 255, 255))
            cv2.imwrite('wlfae.jpg', frame)
            mask = cv2.bitwise_not(mask)
            # make the background of the face black
            spotted_frame = cv2.subtract(frame, mask)
            cv2.imwrite('spotted.jpg', spotted_frame)

            proj_matrix, camera_matrix, rmat, tvec = calib.estimate_camera(model3D, landmarks[0])
            # load mask to exclude eyes from symmetry
            eyemask = np.asarray(io.loadmat('frontalization_models/eyemask.mat')['eyemask'])
            # perform frontalization
            frontal_raw, frontal_sym = frontalize.frontalize(spotted_frame, proj_matrix, model3D.ref_U, eyemask)
            # then I extract the face and crop it
            cv2.imwrite('FrameFrontal.jpg', frontal_sym)
            landmarks = feature_detection.get_landmarks(frontal_sym)
            if len(landmarks) <= 0:
                print("could not detect face landmarks")
                return
            hullIndex = cv2.convexHull(landmarks[0], returnPoints=False)
            hullIndex = hullIndex.flatten()
            contour = landmarks[0][hullIndex]
            contour = np.array(contour).reshape((-1, 1, 2)).astype(np.int32)
            # make a box of the contour of the face to crop the image
            x, y, w, h = cv2.boundingRect(contour)
            # ratio=320/height
            # x=int(x*ratio)
            # y = int(y * ratio)
            # w = int(w * ratio)
            # h = int(h * ratio)
            if w > h:
                diff = w - h
                h = w
                y = int(y - diff / 2)
            elif h > w:
                diff = h - w
                w = h
                x = int(x - diff / 2)
            frame_crop = frontal_sym[y:y + h, x:x + w, :]
            # then I change the cropped face to a greyscale
            frame_grey = cv2.cvtColor(frame_crop, cv2.COLOR_BGR2GRAY)
            # then I use adaptive histogram equalization to normalize the brightness and contrast of the face image
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(12, 12))
            framev2 = clahe.apply(frame_grey)
            resizedFrame = cv2.resize(framev2, (48, 48), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite('FrameCompress.jpg', resizedFrame)
            cv2.imwrite('framev2.jpg', framev2)

    camera.release()
    cv2.waitKey(1)
    cv2.waitKey(1)
    cv2.waitKey(1)
    cv2.waitKey(1)
    return


