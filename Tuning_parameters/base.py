"""

Module which contains helper functions
"""

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from statistics import mean

# Data extraction

def combine_annotated_to_NNB(annotated_list, # list of annotated cells df
                             NNB_list):     # list of NNB df
    """
    Combine main cell dataframe with NNB info (annotated cells only).
    Returns output, and output_log
    """

    output = []
    output_log = []
    n=len(annotated_list)
    for i in range(0,n):

        # select NNB dataframe & remove image name
        nb_dat_ = NNB_list[i]
        nb_dat = nb_dat_.drop(columns=['Image_name'])
        nb_dat = nb_dat.rename(columns={"X": "Centroid_X",
                               "Y": "Centroid_Y"})

        # select annotated cell dataframe
        dat = annotated_list[i]

        # combine annotated cells with nb info: intersect between 2 dataframes
        combined = dat.merge(nb_dat,
                             on=['Centroid_X','Centroid_Y'],
                             how='inner',
                             validate='1:1')

        output.append(combined)
        output_log.append((dat.shape[0]==combined.shape[0]) &
                          ((dat.shape[1]+nb_dat.shape[1]-2)==combined.shape[1])) # minus 2 for centriods
    return output, output_log

def combine_annotated_to_NNB2(annotated_list, # list of annotated cells df
                              NNB_list):     # list of NNB df
    """
    Combine main cell dataframe with NNB info (annotated cells only).
    Returns output, and output_log
    """

    output = []
    output_log = []
    n=len(annotated_list)
    for i in range(0,n):

        # select NNB dataframe & remove image name
        nb_dat_ = NNB_list[i]
        nb_dat = nb_dat_.drop(columns=['slice_id'])
        nb_dat = nb_dat.rename(columns={"X": "Centroid_X",
                                        "Y": "Centroid_Y"})

        # select annotated cell dataframe
        dat = annotated_list[i]

        # combine annotated cells with nb info: intersect between 2 dataframes
        combined = dat.merge(nb_dat,
                             on=['Centroid_X','Centroid_Y'],
                             how='inner',
                             validate='1:1')

        output.append(combined)
        output_log.append((dat.shape[0]==combined.shape[0]) &
                          ((dat.shape[1]+nb_dat.shape[1]-2)==combined.shape[1])) # minus 2 for centriods
    return output, output_log




def find_hema_to_remove(hema_inputs):
    """
    Find cells with hematoxylin above criteria, put in a list needed to be removed.
    """

    # Get instances needed to be remove for each slide

    hema_to_remove = []
    for h in hema_inputs:
        h2 = h.copy()
        hema = h['Hematoxylin: Nucleus: Mean']
        # normalise
        threshold = hema.quantile(0.99)
        hema_norm = hema/threshold
        # put in new variable
        h2.loc[:,'Hematoxylin: Nucleus: Mean'] = hema_norm
        h2 = h2[h2['Hematoxylin: Nucleus: Mean']>1] # to select instances need removing (keep <=1)
        hema_to_remove.append(h2)
    return hema_to_remove




def remove_cell_hema(cell_nb_inputs,
                     hema_to_remove):
    """
    Remove cells with hema that fit the criteria, return
    cleaned outputs & removal log if it was successful or not.
    """

    cleaned_inputs = []
    removed_log = []
    for n in range(0,(len(cell_nb_inputs))):

        i = cell_nb_inputs[n] # annotated cells
        h = hema_to_remove[n] # cells we need to remove, may or may not contain annotated cells

        #Find cells that exist in both 'i' & 'h' = cells we want to remove
        to_remove = i.merge(h,
                            on=['Centroid_X', 'Centroid_Y'],
                            how='inner',
                            validate='1:1')

        #Find cells that only exist in 'i' but not in 'h' = cells we want to retain
        to_retain = i.merge(h,
                            on=['Centroid_X', 'Centroid_Y'],
                            how='left',
                            indicator=True,
                            validate='1:1')

        #Extract cells we want to retain
        retained = i[to_retain['_merge']=='left_only']

        cleaned_inputs.append(retained)
        removed_log.append((to_remove.shape[0]==(i.shape[0]-retained.shape[0])))

    return cleaned_inputs, removed_log


# General functions for classification pipeline

def get_threshold(best_params):
    """
    Get all class-specific thresholds from best parameters.
    """
    class_thresholds = []
    for i in best_params:
        t = best_params[i][0]
        class_thresholds.append(t)
    return class_thresholds


def multiclass_PR_curves(n_class, test_y_numeric, predy):
    """
    Create PR curve for each class in multi-classification problem.
    """
    precision = {}
    recall = {}
    thresh = {}

    # calculate PR curve locations
    for i in range(n_class):  # for each class, calculate roc_curve
        precision[i], recall[i], thresh[i] = precision_recall_curve(
            test_y_numeric, predy[:, i], pos_label=i)

    return precision, recall, thresh


def best_param_f_score(n_class, precision, recall, thresh):
    """
    Find the best position on the precision-recall curve
    for multi-classification problem.
    """
    # calculate f-score for each threshold
    best_params = {}
    for i in range(n_class):
        p = precision[i]
        r = recall[i]
        nu = (2*p*r)
        de = (p+r)
        f_score = np.divide(nu, de, out=np.zeros_like(nu), where=de != 0)
        t = thresh[i]
        ix = np.argmax(f_score)
        best_params[i] = (t[ix], f_score[ix], p[ix], r[ix])
    return best_params


def prob_thresholding(y_pred_prob, y_pred, threshold):
    """
    A function to perform thresholding to convert
    scores into crisp class label.
    """
    thresholded_class = []
    for i in range(0, len(y_pred_prob)):
        if(max(y_pred_prob[i]) < threshold):
            c = 'Ambiguous'
        else:
            c = y_pred[i]
        thresholded_class.append(c)
    return thresholded_class


def remove_amb_class(t_class, y_test):
    """
    Removing 'ambiguous' class from the thresholded class & y_predict.
    """
    # get indices of instances with no ambiguous label
    x = pd.Series(t_class)
    y_pred_no_amb = x[x != 'Ambiguous']
    y_pred_no_amb_indices = y_pred_no_amb.index

    # extract these instances fom y_pred
    # y_predict_no_amb = y_predict.iloc[pred_no_amb_indices]

    # subset y_test
    y_test_no_amb = y_test.iloc[y_pred_no_amb_indices]

    return (y_pred_no_amb, y_test_no_amb)


def precision_recall_auc(clf, X, Y):
    """
    Calculates precision-recall area under the curve
     of cell classification ('A', 'N', 'O', 'Ot').
    """
    # variables
    pr_score = {}

    # get y prob predictions
    y_prob_pred = clf.predict_proba(X)

    # convert true y name into numerical classes
    y_true_numeric = name_to_numeric_classes_c2(Y)

    # get number of classes
    n_class = list(set(Y))

    # create PR curve using OVR approach
    for i in range(len(n_class)):  # for each class, calculate roc_curve
        p, r, thresh = precision_recall_curve(
            y_true_numeric, y_prob_pred[:, i], pos_label=i)
        pr_score[i] = auc(r, p)  # recall on x axis, precision on y axis

    # combine all pr-scores using 'macro' method
    pr_auc = mean(pr_score.values())
    return pr_auc


def threshold_list_c2(y_pred_prob, best_params):
    """
    Applies class-specific thresholding to each detection
     from classifier 2 (A, N, O, Ot, Ambiguous).
    """
    thresholded_classes = []
    for i in y_pred_prob:  # for each detection

        # get class-specific threshold values:
        thresholds = get_threshold(best_params)

        # calculate predicted probability - threshold
        # = difference for each of the classes
        differences = (i-thresholds)/thresholds

        # count number of positive or equal (0) differences
        count = np.count_nonzero(differences >= 0)

        if (count == 1):  # only assign class when 1 class passes the threshold
            pred_class = np.argmax(differences)
        else:  # otherwise, label as ambiguous
            # (when more than 1 class passes, or when no class passes)
            pred_class = 4

        # putting prediction in a list
        thresholded_classes.append(pred_class)

    thresholded_classes_ = numeric_to_name_classes_c2(thresholded_classes)
    return thresholded_classes_


def numeric_to_name_classes_c2(numeric_classes):
    """
    Converts numeric to its corresponding name classes
     for classifier 2 (A, N, O, Ot Ambiguous).
    """
    code = {0: 'Astro', 1: 'Neuron', 2: 'Oligo', 3: 'Others', 4: 'Ambiguous'}
    output = [code[i] for i in numeric_classes]
    return output



def name_to_numeric_classes_c2(name_classes):
    """
    Converts name to its corresponding numeric classes
     for classifier 2 (CB,NFT,Others, TA Ambiguous).
    """
    code = {'Astro': 0, 'Neuron': 1, 'Oligo': 2, 'Others': 3, 'Ambiguous': 4}
    output = [code[i] for i in name_classes]
    return output


