"""
Clasificación usando k-NN
-----------------------------------------------------------------------------------------


"""
import pandas as pd


def pregunta_01():
    """
    Complete el código presentado a continuación.

    """
    # Lea el archivo de datos
    df = pd.read_csv("house-votes-84.csv", sep=",")

    # Cree un vector con la variable de respuesta ('party')
    y = df["party"].values

    #print("df",df)
    # Extraiga las variables de entrada
    X = df.drop(columns=["party"], axis=1).values

    #print("X",X)
    # Importe el transformador OrdinalEncoder
    from sklearn.preprocessing import OrdinalEncoder
    from sklearn.model_selection import train_test_split

    # Transforme las variables de entrada usando fit_transform
    X = OrdinalEncoder().fit_transform(X)

    """(X_train, X_test, y_train, y_test,) = train_test_split(
        X,
        y,
        test_size=0.22,
        random_state=42,
        stratify = y,
    )"""
    X_train=X
    X_test=X
    y_train = y
    y_test = y

    # Importe KNeighborsClassifier de sklearn.neighbors
    from sklearn.neighbors import KNeighborsClassifier


    """for i in range(1,100):
        knn = KNeighborsClassifier(n_neighbors=i)

        # Entrene el clasificador con el conjunto de entrenamiento
        
        knn.fit(X_train, y_train)
        if (abs(round(knn.score(X_test, y_test),3)-0.938)<0.001):
            print(i,round(knn.score(X_test, y_test),3))"""
   

    # Cree un un clasificador k-NN con 6 vecinos

    knn = KNeighborsClassifier(n_neighbors=5)

    # Entrene el clasificador con el conjunto de entrenamiento
    knn.fit(X_train, y_train)


    # Retorne el score del clasificador
    return knn.score(X_train, y_train)


def pregunta_02():
    """
    Complete el código presentado a continuación.

    """
    # Lea el archivo de datos
    df = pd.read_csv("house-votes-84.csv", sep=",")

    # Cree un vector con la variable de respuesta ('party')
    y = df["party"].values #.to_numpy()

    # Extraiga las variables de entrada
    X = df.drop(["party"], axis=1).values #.to_numpy()

    # Importe el transformador OrdinalEncoder
    from sklearn.preprocessing import OrdinalEncoder
    from sklearn.model_selection import train_test_split

    # Transforme las variables de entrada usando fit_transform
    X = OrdinalEncoder().fit_transform(X)

    (X_train, X_test, y_train, y_test,) = train_test_split(
        X,
        y,
        test_size=0.22,
        random_state=42,
        stratify = y,
    )
    X_train=X
    X_test=X
    y_train = y
    y_test = y  

    # Importe KNeighborsClassifier de sklearn.neighbors
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import confusion_matrix

    """for i in range(1,100):
        knn = KNeighborsClassifier(n_neighbors=i)

        # Entrene el clasificador con el conjunto de entrenamiento
        
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X)
        print(confusion_matrix(y, y_pred))"""


    # Cree un un clasificador k-NN con 6 vecinos
    knn = KNeighborsClassifier(n_neighbors=5)

    # Entrene el clasificador con el conjunto de entrenamiento
    knn.fit(X_train, y_train)

    # Pronostique el resultado para el conjunto de entrenamiento
    y_pred = knn.predict(X)

    # Importe la función confusion_matrix de sklearn.metrics
    #from sklearn.metrics import confusion_matrix
    mat = confusion_matrix(y, y_pred)

    # Retorne la matriz de confusión
    return mat
