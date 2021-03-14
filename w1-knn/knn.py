class TF_KNeighborsClassifier:
    def __init__(self, n_neighbors=2, p=2):
        self.n_neighbors = n_neighbors
        self.p = p
    def fit(self, X, y):
        self.X = tf.convert_to_tensor(X)
        self.y = tf.convert_to_tensor(y)
    def predict(self, X_test):
        y_pred = []
        for x in X_test:
            d = tf.map_fn(self.Manhattan_distance,x) 
            min_d_idx = tf.argsort(d)[:self.n_neighbors]
            # find nearest neighbors's label 
            nearest_neighbors_label = []
            for i in min_d_idx:
                nearest_neighbors_label.append(self.y[i])
            # find mode 
            y, idx, count = tf.unique_with_counts(nearest_neighbors_label)
            #y_pred.append( y[tf.argmax(count)] ) # slower 
            y_pred.append( tf.slice(y, begin=[tf.argmax(count, 0)], size=tf.constant([1], dtype=tf.int64))[0] ) #faster
        y_pred = np.array(y_pred)
        return y_pred
    def score(self,X_test, y_test):
        return accuracy_score(self.predict(X_test), y_test)
    def Manhattan_distance(self,x_test):
        return tf.math.pow( tf.reduce_sum( tf.math.pow(tf.math.abs( tf.subtract(self.X,x_test) ),self.p) ) , 1.0/self.p)