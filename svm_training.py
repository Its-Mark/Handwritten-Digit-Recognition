import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import svm

values = load_digits()
samples = len(values.images)
data = values.images.reshape((samples, -1))

xTra, xTes, yTra, yTes = train_test_split(data, values.target, test_size=0.5, shuffle=False)


svm_classifier = svm.SVC(gamma=0.001)
svm_classifier.fit(xTra, yTra)
pred = svm_classifier.predict(xTes)
_, axis = plt.subplots(2, 4)
image_tra = list(zip(values.images, values.target))
for ax, (image, label) in zip(axis[0, :], image_tra[:4]):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title('Training: %i' % label)

image_tes = list(zip(values.images[samples // 2:], pred))
for ax, (image, prediction) in zip(axis[1, :], image_tes[:4]):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title('Prediction: %i' % prediction)

print("\nClassification report for classifier %s:\n%s\n" % (svm_classifier, metrics.classification_report(yTes, pred)))
print("\nAccuracy of the Algorithm: ", svm_classifier.score(xTes, yTes))
plt.show()


plt.gray()
plt.matshow(values.images[35])
plt.show()

