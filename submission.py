# load packages
import numpy as np
import pandas as pd
from typing import Tuple, Iterable, Union

# load datasets
# X = np.load("example_X.npy")
# treatment = np.load("example_treatment.npy")
# y = np.load("example_y.npy")
# preds = np.load("example_preds.npy")

class UpliftTreeRegressor:
    def __init__(
        self, 
        Max_depth: int = 3,                     # max tree depth 
        Min_samples_leaf: int = 1000,           # min number of values in leaf 
        Min_samples_leaf_treated: int = 300,    # min number of treatment values in leaf 
        Min_samples_leaf_control: int = 300,    # min number of control values in leaf
    ):
        self.Max_depth = Max_depth
        self.Min_samples_leaf = Min_samples_leaf
        self.Min_samples_leaf_treated = Min_samples_leaf_treated
        self.Min_samples_leaf_control = Min_samples_leaf_control
        
    def fit(
        self, 
        X: np.ndarray,            # (n * k) array with features 
        treatment: np.ndarray,    # (n) array with treatment flag 
        y: np.ndarray             # (n) array with the target 
    ) -> None: 
        # fit the model
        
        def findBestSplit(X: np.ndarray,            # (n * k) 
                          treatment: np.ndarray,    # (n) 
                          y: np.ndarray             # (n) 
        ) -> Tuple[int, float]:
            # find the best split feature and threshold by looping features & thresholds
            
            delta_delta_p_max = -1     # a flag to record the current biggest deltaDeltaP 
            feature_winner = None      # current feature for biggest ddp
            threshold_winner = None    # current threshold for biggest ddp
            
            # loops start
            for i in range(X.shape[1]):    # loop on all features
                column_values = X[:,i]
                unique_values = np.unique(column_values) 
                if len(unique_values) > 10:
                    percentiles = np.percentile(column_values, [3, 5, 10, 20, 30, 50, 70, 80, 90, 95, 97]) 
                else:
                    percentiles = np.percentile(unique_values, [10, 50, 90])
                threshold_options = np.unique(percentiles)
                for threshold in threshold_options:    # loop on all thresholds
                    left = X[column_values <= threshold]
                    right = X[column_values > threshold]
                    y_left = y[column_values <= threshold]
                    y_right = y[column_values > threshold]
                    treatment_left = treatment[column_values <= threshold]
                    treatment_right = treatment[column_values > threshold]
                    
                    # if number of values is smaller than the required, ignore it
                    if (left.shape[0] < self.Min_samples_leaf 
                        or right.shape[0] < self.Min_samples_leaf 
                        or (treatment_left == 1).sum() < self.Min_samples_leaf_treated 
                        or (treatment_right == 1).sum() < self.Min_samples_leaf_control
                       ):
                        continue
                        
                    # calculate deltaDeltaP
                    delta_delta_p = abs(
                        (y_left[treatment_left == 1].mean() - y_left[treatment_left == 0].mean()) 
                        - (y_right[treatment_right == 1].mean() - y_right[treatment_right == 0].mean())
                    )
                    
                    # renew our flags if a bigger ddp observed
                    if delta_delta_p > delta_delta_p_max:
                        delta_delta_p_max = delta_delta_p
                        feature_winner = i
                        threshold_winner = threshold
            # loops end
            
            return feature_winner, threshold_winner
    
        def build(X: np.ndarray,            # (n * k) 
                  treatment: np.ndarray,    # (n) 
                  y: np.ndarray,            # (n) 
                  depth: int                # current tree depth
        ):
            # build the tree recursively, stored in a nested dict
            
            # return as a leaf if max_depth reached or no split
            if depth > self.Max_depth:
                return {"n_items": X.shape[0], 
                        "ATE": y[treatment==1].mean() - y[treatment==0].mean()
                       }
            feat, thres = findBestSplit(X, treatment, y)
            if feat == None:
                return {"n_items": X.shape[0], 
                        "ATE": y[treatment==1].mean() - y[treatment==0].mean()
                       }
            myTree = {feat: {}}                       # initialize the output tree/subtree
            node_left_index = (X[:,feat] <= thres)    # get the left/right subtree indices
            node_right_index = (X[:,feat] > thres)    # get the left/right subtree indices
            # set values for this node, and generate subtree recursively
            myTree[feat]["n_items"] = X.shape[0]
            myTree[feat]["ATE"] = y[treatment==1].mean() - y[treatment==0].mean()
            myTree[feat]["thres"] = thres
            myTree[feat]["left"]= build(X[node_left_index], treatment[node_left_index], 
                                        y[node_left_index], depth + 1)
            myTree[feat]["right"]= build(X[node_right_index], treatment[node_right_index], 
                                         y[node_right_index], depth + 1)
            return myTree
        
        self.tree = build(X, treatment, y, 1)
        
    def predict(self, X: np.ndarray) -> Iterable[float]:
        # compute predictions
        
        tree = self.tree.copy()           # copy the final tree
        root = list(tree.keys())[0]       # get the root feature
        predictions = np.zeros(len(X))    # initialize the predictions
        
        def get_pred(row: np.ndarray, node: int, tree: Union[dict, float]) -> float:
            # get the prediction for one specific row, recursively
            
            # return the ATE value if left or right subtree is a leaf, recurse otherwise
            if row[node] <= tree[node]["thres"] and len(tree[node]["left"]) == 2:
                return tree[node]["left"]["ATE"]
            elif row[node] > tree[node]["thres"] and len(tree[node]["right"]) == 2:
                return tree[node]["right"]["ATE"]
            elif row[node] <= tree[node]["thres"]:
                return get_pred(row, list(tree[node]["left"].keys())[0], tree[node]["left"])
            else:
                return get_pred(row, list(tree[node]["right"].keys())[0], tree[node]["right"])
        
        # loop to get all predictions
        for i in range(len(X)):
            predictions[i] = get_pred(X[i], root, tree)
            
        return predictions

# instantiate a UpliftTreeRegressor
# uplift = UpliftTreeRegressor(3, 6000, 2500, 2500)
# uplift.fit(X, treatment, y)
# print(uplift.tree)
# uplift.predict(X)
# print((uplift.predict(X) - preds).sum())