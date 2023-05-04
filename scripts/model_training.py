"""Create Model
Author: Justin Zarb 

Based heavily on Training.py in DSR_ML_FUNDAMENTALS

Pseudocode:
- fit
- score
""" 
import numpy as np 
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression


class Model:
    def __init__(self, Xtrain, Ytrain, Xtest, Ytest):
        self.X_train = Xtrain 
        self.Y_train = Ytrain 
        self.X_test = Xtest
        self.Y_test = Ytest

    def mean(self):
        model = np.mean(self.Ytrain)

    
    def linear_regression(self, Xtrain, Ytrain, Xtest, Ytest):
        lr = LinearRegression()
        lr.fit(X_train, Y_train)
        
        Y_pred_train = lr.predict(X_train)
        Y_pred_test = lr.predict(X_test)
        rmse_train, rmse_test, mae_train, mae_test = self.regression_evaluation(
                Y_train, Y_test, Y_pred_train, Y_pred_test
                )
        return rmse_train, rmse_test, mae_train, mae_test
    
    def negative_binomial(self, model_formula, alpha):
        model = smf.glm(formula=model_formula,
                            data=train,
                            family=sm.families.NegativeBinomial(alpha=alpha))
        fitted_model = model.fit()
        predictions = results.predict(test).astype(int)
        score = eval_measures.meanabs(predictions, test.total_cases)

        return fitted_model, predictions, score
    
if __name__ == "__main__":
     pass
     # Example implementation
     model_formula = "total_cases ~ 1 + " \
                        "reanalysis_specific_humidity_g_per_kg + " \
                        "reanalysis_dew_point_temp_k + " \
                        "station_min_temp_c + " \
                        "station_avg_temp_c"
    
  




