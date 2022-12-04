import numpy as np
import SVM_Data
from sklearn.svm import SVC
import joblib

# initial variables
dataset = "dataset1"
classifier = "clf_saved.pkl"
confusion_matrix_file_name = "confusion_matrix.txt"

# linear SVM classifier
clf = SVC(kernel='linear', probability=True, tol=1e-3)

# prepare data for training and prediction sets
training_data, training_labels, prediction_data, prediction_labels = SVM_Data.SVM_Data(
    dataset)

# change to numpy array for the classifier
npar_train = np.array(training_data)
npar_trainlabs = np.array(training_labels)
npar_pred = np.array(prediction_data)

# train the classifier
clf.fit(npar_train, training_labels)

# save the classifier
clf_saved = joblib.dump(clf, "results/%s/%s" % (dataset, classifier))

# calculate the accuracy
prediction_accuracy = clf.score(npar_pred, prediction_labels)
print("accuracy: ", prediction_accuracy)

# calculate confusion matrix
Confusion_Matrix = np.zeros((8, 8))
for data, label in zip(prediction_data, prediction_labels):
    SVM_index = clf.predict(np.array(data).reshape(-1, len(data)))[0]
    Real_index = label
    Confusion_Matrix[Real_index][SVM_index] = Confusion_Matrix[Real_index][SVM_index] + 1
np.savetxt('results/%s/%s' %
           (dataset, confusion_matrix_file_name), Confusion_Matrix)
print(Confusion_Matrix)
