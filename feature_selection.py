import mlxtend.feature_selection as fs
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def do_features_selection(modele, x_train, y_train, select_method = "forward", cv = 5, feature_names=[], scoring = 'accuracy'):
    methode = 0 if select_method =="backward" else 1

    k_features = x_train.shape[1] if methode else 1

    feat_select = fs.SequentialFeatureSelector(modele, k_features=k_features, forward=methode, scoring=scoring,
                                               cv=cv)
    feat_select = feat_select.fit(x_train, y_train)
    print(feat_select.get_metric_dict())
    fig1 = plot_sfs(feat_select.get_metric_dict(), kind='std_dev')

    # plt.ylim([0.8, 1])
    methode = "backward" if methode == 0 else 'forward'
    plt.title('Sequential {} Selection (w.StdDev)'.format(methode))
    plt.grid()
    plt.show()

    results = pd.DataFrame.from_dict(feat_select.get_metric_dict()).T
    results["avg_score"] = [np.sqrt(elt) for elt in list(results["avg_score"])]
    return results[["feature_names", "avg_score"]]