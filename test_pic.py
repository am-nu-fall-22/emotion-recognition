import cv2
import numpy as np
import dlib
import joblib
from imutils import face_utils
import SVM_Data


def draw_landmarks(image, detector, predictor):

    # convert to grey scale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    rects = detector(gray, 0)

    # loop over the face detections
    for (i, rect) in enumerate(rects):

        # determine the facial landmarks for the face region
        shape = predictor(gray, rect)

        # convert the facial landmark (x, y)-coordinates to a NumPy array
        shape = face_utils.shape_to_np(shape)

        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        for (x, y) in shape:
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)


def write_emotion(image, clf, detector, emotions, features_vector):

    # detect faces in image
    detections = detector(image, 1)

    # enumerate on detected faces and write the emotion on top of each
    for k, d in enumerate(detections):
        # predict emotion
        index = clf.predict(
            np.array(features_vector[k]).reshape(-1, len(features_vector[k])))

        # write emotion on top of each detected face
        emotion = emotions[index[0]]
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, emotion, (d.left(), d.top()),
                    font, 1, (0, 255, 0), 3)


def main():
    # initialization
    dataset = "dataset1"
    classifier = "clf_saved.pkl"
    test_data_dir = "data/test_data/"
    pic_name = "test_pic2.jpg"
    pic_dir = test_data_dir + pic_name
    emotions = ["neutral", "anger", "contempt", "disgust",
                "fear", "happiness", "sadness", "surprise"]

    # read test image
    image = cv2.imread(pic_dir)

    # face detector
    detector = dlib.get_frontal_face_detector()

    # landmark predictor
    predictor = dlib.shape_predictor(
        "data/shape_predictor_68_face_landmarks.dat")

    # load trained classifier
    clf = joblib.load('results/%s/%s' % (dataset, classifier))

    # extract features from image
    features_vector = SVM_Data.extract_feature(pic_dir)

    # write emotion
    write_emotion(image, clf, detector, emotions, features_vector)

    # draw landmarks
    draw_landmarks(image, detector, predictor)

    # show and save resulting image
    cv2.namedWindow(pic_name)
    cv2.moveWindow(pic_name, 0, 0)
    cv2.imshow(pic_name, image)
    cv2.imwrite('results/test_data/' + pic_name, image)
    cv2.waitKey()
    cv2.destroyAllWindows()


main()
