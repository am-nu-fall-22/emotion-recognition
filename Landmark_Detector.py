import numpy as np
import dlib
import math


def Landmark_Detector(image):

    features_vector = []

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(
        "data/shape_predictor_68_face_landmarks.dat")
    detections = detector(image, 1)

    # repeat for all detected faces
    for k, d in enumerate(detections):

        # facial landmarks with the predictor class of the dlib library
        shape = predictor(image, d)
        xsequence = []
        ysequence = []

        # save x and y coordinates in two lists
        for i in range(1, 68):
            xsequence.append(float(shape.part(i).x))
            ysequence.append(float(shape.part(i).y))

        # mean of all landmarks' x and y coordinates
        xmean = np.mean(xsequence)
        ymean = np.mean(ysequence)

        # distance between each point and the mean point
        xrelative = [(x-xmean) for x in xsequence]
        yrelative = [(y-ymean) for y in ysequence]

        # prevent 'divide by 0' error
        if xsequence[26] == xsequence[29]:
            anglenose = 0
        else:
            # rotation angle
            anglenose = int(math.atan(
                (ysequence[26]-ysequence[29])/(xsequence[26]-xsequence[29]))*180/math.pi)
        if anglenose < 0:
            anglenose += 90
        else:
            anglenose -= 90

        # create feature vector
        feature_vector = []
        for x, y, w, z in zip(xrelative, yrelative, xsequence, ysequence):
            feature_vector.append(x)
            feature_vector.append(y)
            meannp = np.asarray((ymean, xmean))
            coornp = np.asarray((z, w))
            dist = np.linalg.norm(coornp-meannp)
            if (w - xmean == 0):
                if (z - ymean > 0):
                    anglerelative = 90 - anglenose
                else:
                    anglerelative = -90 - anglenose

            else:
                anglerelative = (math.atan((z-ymean)/(w-xmean))
                                 * 180/math.pi) - anglenose
            feature_vector.append(dist)
            feature_vector.append(anglerelative)

        # append to features_vector
        features_vector.append(feature_vector)

    # return error if no face is detected
    if len(detections) < 1:
        features_vector = "error"

    return features_vector
