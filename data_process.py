
import cv2
import glob
import numpy as np
import dlib as dl
import scipy.io as io
import os
from PIL import Image
import camera_calibration as calib
import facial_feature_detector as feature_detection
import frontalize
import csv
this_path = os.path.dirname(os.path.abspath(__file__))

database_path = r''
datasets_path = r'./datasets'
csv_file = os.path.join(database_path, 'fer2013.csv')
train_csv = os.path.join(datasets_path, 'train.csv')
val_csv = os.path.join(datasets_path, 'val.csv')
test_csv = os.path.join(datasets_path, 'test.csv')
train_set = os.path.join(datasets_path, 'train')
val_set = os.path.join(datasets_path, 'val')
test_set = os.path.join(datasets_path, 'test')
images_paths=[]
def getDataset():
    with open(csv_file) as f:
        csvr = csv.reader(f)
        header = next(csvr)
        rows = [row for row in csvr]

        trn = [row[:-1] for row in rows if row[-1] == 'Training']
        csv.writer(open(train_csv, 'w+')).writerows([header[:-1]] + trn)
        print(len(trn))

        val = [row[:-1] for row in rows if row[-1] == 'PublicTest']
        csv.writer(open(val_csv, 'w+')).writerows([header[:-1]] + val)
        print(len(val))

        tst = [row[:-1] for row in rows if row[-1] == 'PrivateTest']
        csv.writer(open(test_csv, 'w+')).writerows([header[:-1]] + tst)
        print(len(tst))
images=[]
def generateImages():
    used_save_path=''
    category=-1
    for save_path, csv_file in [(train_set, train_csv), (val_set, val_csv), (test_set, test_csv)]:
        # if not os.path.exists(save_path):
        #     os.makedirs(save_path)
        if (save_path!=used_save_path):
            images.append([])
            images_paths.append([])
            category+=1
        num = 1

        with open(csv_file) as f:
            csvr = csv.reader(f)
            header = next(csvr)
            images[category]=[[],[],[],[],[],[],[]]
            for i, (label, pixel) in enumerate(csvr):
                pixel = np.asarray([float(p) for p in pixel.split()]).reshape(48, 48)
                # print(save_path)
                # print(label)
                subfolder = os.path.join(save_path, label)
                # if not os.path.exists(subfolder):
                #     os.makedirs(subfolder)
                im = Image.fromarray(pixel).convert('L').convert('RGB')
                array = np.array(im)
                #                 print(np.shape(array))
                #                 print(im)
                opencvImage = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
                images[category][int(label)].append(opencvImage)
                image_name = os.path.join(subfolder, '{:05d}.jpg'.format(i))
                #                 print(image_name)
                images_paths[category][int(label)].append(image_name)
                # print(image_name)
                # im.save(image_name)




def getAllPath():
    for i in range(0, 7):
        WSI_MASK_PATH = 'datasets/train/' + str(i)  # 存放图片的文件夹路径
        print(WSI_MASK_PATH)
        images_paths_temp = glob.glob(os.path.join(WSI_MASK_PATH, '*.jpg'))
        images_paths_temp.sort()
        print(images_paths_temp)
        print(i)
        print(len(images_paths))
        images_paths.append(images_paths_temp)

def run(type):
    # camera = cv2.VideoCapture(0)
    # status, frame = 0, 0
    # if camera.isOpened():  # try to get the first frame
    #     status, frame = camera.read()
    #     cv2.imwrite('ooo.jpg', frame)
    #     camera.release()
    # else:
    #     rval = False
    #     exit(0)
    status=True
    for p in range(0, 3):
        for i in range(0, 7):
            for k in range(0, len(images[p][i])):
                print(k)
                image_path = images_paths[p][i][k]
                file_name = image_path[17:]
                print(image_path)
                # frame = cv2.imread(image_path)
                frame = images[p][i][k]
                if status | True:
                    height, width, c = np.shape(frame)
                    minV = min(height, width)
                    startX = int((width - minV) / 2)
                    startY = int((height - minV) / 2)
                    frame = frame[startY:startY + minV, startX:startX + minV]
                    # use Effective Face Frontalization in Unconstrained Images to frontalize the face
                    model3D = frontalize.ThreeD_Model(this_path + "/frontalization_models/model3Ddlib.mat",
                                                      'model_dlib')
                    # Whether to extract the face first or frontalize the face first is important and need to be tried
                    if (type == 0):
                        dest_path = 'datasets/train/' + str(i) + '/processedT1/'
                        if not os.path.exists(dest_path):
                            os.makedirs(dest_path)
                        # I proceed with Frontalization first then I extract the face
                        landmarks = feature_detection.get_landmarks(frame)
                        # perform camera calibration according to the first face detected
                        if len(landmarks) <= 0:
                            print("could not detect face landmarks")
                            frame_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(12, 12))
                            framev2 = clahe.apply(frame_grey)
                            cv2.imwrite(dest_path + "bad_" + file_name, framev2)
                            continue
                        proj_matrix, camera_matrix, rmat, tvec = calib.estimate_camera(model3D, landmarks[0])
                        # load mask to exclude eyes from symmetry
                        eyemask = np.asarray(io.loadmat('frontalization_models/eyemask.mat')['eyemask'])
                        # perform frontalization
                        frontal_raw, frontal_sym = frontalize.frontalize(frame, proj_matrix, model3D.ref_U, eyemask)
                        # then I extract the face and crop it
                        cv2.imwrite('FrameFrontal.jpg', frontal_sym)
                        landmarks = feature_detection.get_landmarks(frontal_sym)
                        if len(landmarks) <= 0:
                            print("could not detect face landmarks 2")
                            frame_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(12, 12))
                            framev2 = clahe.apply(frame_grey)
                            cv2.imwrite(dest_path + "bad_" + file_name, framev2)
                            continue
                        hullIndex = cv2.convexHull(landmarks[0], returnPoints=False)
                        hullIndex = hullIndex.flatten()
                        contour = landmarks[0][hullIndex]
                        # And we can choose whether to use the one with soft-symmetry or raw. I use the one with soft symmetry
                        height, width, channels = np.shape(frontal_sym)
                        mask = np.zeros((height, width, channels), np.uint8)
                        contour = np.array(contour).reshape((-1, 1, 2)).astype(np.int32)
                        cv2.fillPoly(mask, [contour], (255, 255, 255))
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
                        print(dest_path + file_name)
                        cv2.imwrite(dest_path + file_name, resizedFrame)
                    elif type == 1:
                        dest_path = 'datasets/train/' + str(i) + '/processedT2/'
                        if not os.path.exists(dest_path):
                            os.makedirs(dest_path)
                        # I extract the face first then I frontalize it
                        landmarks = feature_detection.get_landmarks(frame)
                        # perform camera calibration according to the first face detected
                        if len(landmarks) <= 0:
                            print("could not detect face landmarks")
                            frame_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(12, 12))
                            framev2 = clahe.apply(frame_grey)
                            cv2.imwrite(dest_path + "bad_" + file_name, framev2)
                            continue

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
                        frontal_raw, frontal_sym = frontalize.frontalize(spotted_frame, proj_matrix, model3D.ref_U,
                                                                         eyemask)
                        # then I extract the face and crop it
                        cv2.imwrite('FrameFrontal.jpg', frontal_sym)
                        landmarks = feature_detection.get_landmarks(frontal_sym)
                        if len(landmarks) <= 0:
                            print("could not detect face landmarks 2")
                            frame_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(12, 12))
                            framev2 = clahe.apply(frame_grey)
                            cv2.imwrite(dest_path + "bad_" + file_name, framev2)
                            continue
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
                        cv2.imwrite(dest_path + file_name, resizedFrame)
                # camera.release()
                # cv2.waitKey(1)
                # cv2.waitKey(1)
                # cv2.waitKey(1)
                # cv2.waitKey(1)
                # return



