class TF_SimpleLinearRegression:
    
    def __init__(self):
        self.a = 0
        self.b = 0
    
    def fit(self, x, y):
        x = tf.convert_to_tensor(x, dtype=tf.float64)
        y = tf.convert_to_tensor(y, dtype=tf.float64)
       
        ##### y=ax+b #####
        # some var in equation
        x_sum = tf.reduce_sum(x)
        y_sum = tf.reduce_sum(y)
        xy_sum = tf.reduce_sum( tf.math.multiply(x,y) )
        x2_sum = tf.reduce_sum( tf.math.pow(x,2) )
       
        n = x.shape[0]
        # a (slope)
        ## numerator(分子)、denominator(分母)
        a_numerator = (n * xy_sum) - (x_sum * y_sum)
        a_denominator = (n * x2_sum) - (x_sum**2)
        a = a_numerator / a_denominator

        
        b = (y_sum - (a*x_sum)) / n
        self.a = a
        self.b = b
        print("model: y={}x+{}".format(round(a.numpy(),2),round(b.numpy(),2)))
        
    def predict(self, x):
        regression = lambda x : self.b + self.a * x
        return regression(x)
    