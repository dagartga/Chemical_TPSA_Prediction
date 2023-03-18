from sklearn import ensemble
from sklearn import tree
from sklearn import linear_model

models = {"decision_tree_sq_err":tree.DecisionTreeRegressor(criterion="squared_error"),
          "decision_tree_mse":tree.DecisionTreeRegressor(criterion='friedman_mse'),
          "rf":ensemble.RandomForestRegressor(),
          "lr": linear_model.LinearRegression(),
          "lasso":linear_model.Lasso()}
