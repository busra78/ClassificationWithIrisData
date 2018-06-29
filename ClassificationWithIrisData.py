from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import itertools
import pandas as pd
import numpy as np

#Sınıflandırıcıların import edilmesi
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

#Confusion Matrix Gösterimi İçin Gerekli Olan Fonksiyon
def plot_confusion_matrix(cm,classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



#İris Veri Setinin Yüklenmesi
iris =load_iris()
X=iris.data
Y=iris.target
class_names=iris.target_names


#İris Veri Setinden İlk 5 Verinin Bir Tabloda Gösterilmesi

df = pd.DataFrame(iris.data, columns=iris.feature_names)

df['species']= pd.Categorical.from_codes(iris.target, iris.target_names)

print("5 of Iris Dataset:")
print(df.head(5))




"""HESAPLAMA İŞLEMLERİ"""

#KNN İşlemleri

#KNN Sınıflandırıcısı İle İris Veri Setinin 4 Komşuya Göre Sınıflandırılması

knn_classifier = KNeighborsClassifier(n_neighbors = 4)

#İris Verisinin Test İşleminden Geçirilmesi ve KNN için Oluşturulacak Confusion Matrix İçin Gerekli Hesaplamaların Yapılması

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,random_state=0)

Y_pred_knn=knn_classifier.fit(X_train,Y_train).predict(X_test)

knn_cm=confusion_matrix(Y_test,Y_pred_knn)

np.set_printoptions(precision=2)


#SVM İşlemleri


#SVM Sınıflandırıcısı İle İris Veri Setinin Lineer Olarak Sınıflandırılması

svm_classifier = svm.SVC(kernel='linear', C=0.01)

#İris Verisinin Test İşleminden Geçirilmesi ve SVM için Oluşturulacak Confusion Matrix İçin Gerekli Hesaplamaların Yapılması

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0)
Y_pred_svm = svm_classifier.fit(X_train, Y_train).predict(X_test)
svm_cm = confusion_matrix(Y_test,Y_pred_svm)

np.set_printoptions(precision=2)

   
    
#Decision Tree İşlemleri

#Decision Tree Sınıflandırıcısı İle İris Veri Setinin  Sınıflandırılması

decision_tree_classifier=DecisionTreeClassifier()


#İris Verisinin Test İşleminden Geçirilmesi ve Decision Tree için Oluşturulacak Confusion Matrix İçin Gerekli Hesaplamaların Yapılması

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,random_state=0)

Y_pred_dt=decision_tree_classifier.fit(X_train,Y_train).predict(X_test)

decision_tree_cm=confusion_matrix(Y_test,Y_pred_dt)

np.set_printoptions(precision=2)

"""CONFUSION MATRIX'İN GÖSTERİLME İŞLEMLERİ"""
#KNN için Normalize Edilen ve Normalize Edilmeyen Confusion Matrix'in Gösterilmesi
print("")
print("KNN CONFUSION MATRIX")
print("")
plt.figure()
plot_confusion_matrix(knn_cm, classes=class_names,
                      title='Confusion matrix by KNN, without normalization')

plt.figure()
plot_confusion_matrix(knn_cm, classes=class_names, normalize=True,
                      title='Confusion matrix by KNN, with normalization')

plt.show()

#SVM için Normalize Edilen ve Normalize Edilmeyen Confusion Matrix'in Gösterilmesi
print("")
print("SVM CONFUSION MATRIX")
print("")
plt.figure()
plot_confusion_matrix(svm_cm, classes=class_names,
                      title='Confusion matrix by SVM, without normalization')

plt.figure()
plot_confusion_matrix(svm_cm, classes=class_names, normalize=True,
                      title='Confusion matrix by SVM, with normalization')

plt.show()


#Decision Tree için Normalize Edilen ve Normalize Edilmeyen Confusion Matrix'in Gösterilmesi
print("")
print("DECISION TREE CONFUSION MATRIX")
print("")
plt.figure()
plot_confusion_matrix(decision_tree_cm, classes=class_names,            title='Confusion matrix by Decision Tree, without normalization')

plt.figure()
plot_confusion_matrix(decision_tree_cm, classes=class_names, normalize=True,
                      title='Confusion matrix by Decision Tree, with normalization')

plt.show()




"""10 FOLD CROSS VALIDATION HESAPLANMASI VE GÖSTERİLMESİ"""
# KNN için 10 Fold Cross Validation

print("")
knn_tree_cross_val= cross_val_score(knn_classifier,iris.data,iris.target,cv=10)
print("Accuracy matrix by KNN:", str(knn_tree_cross_val))

print("Accuracy with 10-fold cross validation by KNN: %0.2f (+/- %0.2f)" % (knn_tree_cross_val.mean(), knn_tree_cross_val.std() * 2))
print("")


#SVM için 10 Fold Cross Validation
print("")
svm_cross_val= cross_val_score(svm_classifier,iris.data,iris.target,cv=10)
print("Accuracy matrix by SVM:", str(svm_cross_val))
print("Accuracy with 10-fold cross validation by SVM: %0.2f (+/- %0.2f)" % (svm_cross_val.mean(), svm_cross_val.std() * 2))

print("")

#Decision Tree için 10 Fold Cross Validation
print("")
decision_tree_cross_val=cross_val_score(decision_tree_classifier,iris.data,iris.target,cv=10)
print("Accuracy matrix by Decision Tree:", str(decision_tree_cross_val))

print("Accuracy with 10-fold cross validation by Decision Tree: %0.2f (+/- %0.2f)" % (decision_tree_cross_val.mean(), decision_tree_cross_val.std() * 2))

print("")


"""YENİ VERİ İÇİN SINIFLANDIRMA SONUÇLARININ GÖSTERİLMESİ"""

#KNN İçin Yeni Veri
print("")
new_test_data_knn = ( [[10.1, 9.4, 1.3, 6.7]])
new_test_result_knn = knn_classifier.predict(new_test_data_knn)
print("")
print("KNN CLASSIFICATION FOR  ",new_test_data_knn)
print("")

if new_test_result_knn== [0]:
    print("",new_test_data_knn,"'ya ait sınıflandırma sonucu: Setosa =>",new_test_result_knn)
elif new_test_result_knn== [1]:
    print("",new_test_data_knn,"'ya ait sınıflandırma sonucu: Versicolor=>",new_test_result_knn)
else: 
    print("",new_test_data_knn,"'ya ait sınıflandırma sonucu: Virginicia=>",new_test_result_knn)














#SVM İçin Yeni Veri
new_test_data_svm = ( [[10.1, 9.4, 1.3, 6.7]])
new_test_result_svm = svm_classifier.predict(new_test_data_svm)
print("")
print("SVM CLASSIFICATION FOR  ",new_test_data_svm)
print("")

if new_test_result_svm== [0]:
    print("",new_test_data_svm,"'ya ait sınıflandırma sonucu: Setosa =>",new_test_result_svm)
elif new_test_result_svm== [1]:
    print("",new_test_data_svm,"'ya ait sınıflandırma sonucu: Versicolor=>",new_test_result_svm)
else: 
    print("",new_test_data_svm,"'ya ait sınıflandırma sonucu: Virginicia=>",new_test_result_svm)
    
    
    
#DECISION TREE İçin Yeni Veri

new_test_data_dt = ( [[10.1, 9.4, 1.3, 6.7]])
new_test_result_dt = decision_tree_classifier.predict(new_test_data_dt)
print("")
print("DECISION TREE CLASSIFICATION FOR  ",new_test_data_dt)
print("")
if new_test_result_dt== [0]:
    print("",new_test_data_dt,"'ya ait sınıflandırma sonucu: Setosa =>",new_test_result_dt)
elif new_test_result_dt== [1]:
    print("",new_test_data_dt,"'ya ait sınıflandırma sonucu: Versicolor=>",new_test_result_dt)
else: 
    print("",new_test_data_dt,"'ya ait sınıflandırma sonucu: Virginicia=>",new_test_result_dt)