KNN
========

## Analyze different of parameters
![](img/sklearn-knn.png)

![](img/sk-knn-1.png)
![](img/sk-knn-2.png)
* The bigger Neighbors we choose, the more accurate we got.
* If Neighbors is small. choosing larger Minkowski distance's p will decrease the accuracy.

## Implementaion of KNN using Tensorflow
[code](knn.ipynb)
### Usage
1. declare a KNN Model: `knn = TF_KNeighborsClassifier()`
2. Fit KNN: `knn.fit(x_train,y_train)`
3. Predict: `y_pred = knn.predict(x_test)`
4. Scoring: `knn.score(X_test=x_test,y_test=y_test)`


### Example
```python
from knn import TF_KNeighborsClassifier

```