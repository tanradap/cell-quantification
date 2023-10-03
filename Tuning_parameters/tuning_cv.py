from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import classification_report
import sys
sys.path.insert(0,
                '/Users/tanrada/OneDrive - University of Cambridge/Attachments/Jan2023/Cell_pipeline/Tuning_parameters/')
from base import *

# Tau classifier
class ClassifierTuning():
    def __init__(self, hyperparameters):
        self.pipeline = Pipeline(hyperparameters)

        # self.classifier = None
        # self.f_importance = None
        self.best_parameters = None

        self.cv_accuraciesT = None
        self.cv_reportsT = None
        self.cv_confusion_matricesT = None
        self.cv_x_testT = None
        self.cv_y_testT = None
        self.cv_y_predictsT = None
        self.cv_y_prob_predicts = None

        self.cv_accuracies = None
        self.cv_reports = None
        self.cv_confusion_matrices = None
        self.cv_x_test = None
        self.cv_y_test = None
        self.cv_y_predicts = None

    def find_bestparameters(self, X, Y):
        """
        """
        best_parameters = []
        accuracies = []
        reports = []
        confusion_matrices = []

        skf = StratifiedKFold(n_splits=10)
        skf.get_n_splits(X, Y)
        for train_index, test_index in skf.split(X, Y):
            x_train_, x_test_ = X.iloc[train_index], X.iloc[test_index]
            y_train_, y_test_ = Y[train_index], Y[test_index]

            # Specify model configuration
            classifier = self.pipeline

            # Train the classifier
            classifier.fit(x_train_, y_train_)

            # Get class probability predictions for 'test' data
            y_prob_predict = classifier.predict_proba(x_test_)

            # For thresholding:
            # convert y_test_ from name to numeric classes
            y_test_numeric = name_to_numeric_classes_c2(y_test_)

            # For thresholding
            # use predicted class probabilities
            # to calculate ROC curve for each class vs rest

            precision, recall, thresh = multiclass_PR_curves(
                                                            4,
                                                            y_test_numeric,
                                                            y_prob_predict)
            # For thresholding
            # from PR curves, find the best location for each class
            best_params_ = best_param_f_score(
                                        4,
                                        precision,
                                        recall,
                                        thresh)

            best_parameters.append(best_params_)

            # For thresholding
            # apply thresholding to each class to create crisp class label
            t_class = threshold_list_c2(y_prob_predict, best_params_)

            # Remove 'ambiguous class'
            # from t_class & y_test_ - for accuracy calculation
            (y_predict_no_amb, y_test_no_amb) = remove_amb_class(
                                                                t_class,
                                                                y_test_)

            # Calculate & put performance metric (balanced accuracy)
            #  per fold into a list
            accuracies.append(balanced_accuracy_score(
                                                     y_test_no_amb,
                                                     y_predict_no_amb
                                                     ))

            # Compute classification reports
            reports.append(classification_report(
                                                y_test_no_amb,
                                                y_predict_no_amb,
                                                output_dict=True
                                                ))

            # Create confusion matrices for default
            # & thresholded results per fold then put in a list
            cm_t = confusion_matrix(y_test_no_amb, y_predict_no_amb,
                                    labels=['Astro', 'Neuron', 'Oligo', 'Others'])
            # ,normalize='true'

            confusion_matrices.append(cm_t)
        self.cv_best_parameters = best_parameters
        # Extracting thresholds

        astro = []
        neuron = []
        oligo = []
        others = []
        for i in best_parameters:  # NON-CALIBRATED best params
            astro.append(i[0])
            neuron.append(i[1])
            oligo.append(i[2])
            others.append(i[3])

        # Finding mean across the folds
        res_astro = [sum(ele) / len(astro) for ele in zip(*astro)]
        res_neuron = [sum(ele) / len(neuron) for ele in zip(*neuron)]
        res_oligo = [sum(ele) / len(oligo) for ele in zip(*oligo)]
        res_others = [sum(ele) / len(others) for ele in zip(*others)]


        best_params_classifier = {
                                0: tuple(res_astro),
                                1: tuple(res_neuron),
                                2: tuple(res_oligo),
                                3: tuple(res_others)}

        self.best_parameters = best_params_classifier

    def cv_nothresholding(self, X, Y, X_location):

        accuracies = []
        reports = []
        confusion_matrices = []
        y_predicts = []
        y_cv_test = []
        x_cv_test = []

        skf = StratifiedKFold(n_splits=10)
        skf.get_n_splits(X, Y)
        for train_index, test_index in skf.split(X, Y):
            x_train_, x_test_ = X.iloc[train_index], X.iloc[test_index]
            y_train_, y_test_ = Y[train_index], Y[test_index]
            x_test_l = X_location.iloc[test_index]

            # Specify model configuration
            classifier = self.pipeline

            # Train the classifier
            classifier.fit(x_train_, y_train_)

            # Get class probability predictions for 'test' data
            y_predict = classifier.predict(x_test_)
            y_predicts.append(y_predict)

            # Calculate & put performance metric (balanced accuracy)
            #  per fold into a list
            accuracies.append(balanced_accuracy_score(
                                                     y_test_,
                                                     y_predict
                                                     ))

            # Compute classification reports
            reports.append(classification_report(
                                                 y_test_,
                                                 y_predict,
                                                 output_dict=True
                                                ))

            # Create confusion matrices for default
            # & thresholded results per fold then put in a list
            cm = confusion_matrix(y_test_,
                                  y_predict,
                                  labels=['Astro', 'Neuron', 'Oligo', 'Others'])
            # ,normalize='true'

            confusion_matrices.append(cm)
            y_cv_test.append(y_test_)
            x_cv_test.append(x_test_l)

        self.cv_accuracies = accuracies
        self.cv_reports = reports
        self.cv_confusion_matrices = confusion_matrices
        self.cv_y_predicts = y_predicts
        self.cv_x_test = x_cv_test
        self.cv_y_test = y_cv_test

    def cv_withthresholding(self, X, Y, best_parameters, X_location):

        accuracies = []
        reports = []
        confusion_matrices = []
        y_predicts = []
        y_cv_test = []
        x_cv_test = []
        y_prob_preds = []

        skf = StratifiedKFold(n_splits=10)
        skf.get_n_splits(X, Y)
        for train_index, test_index in skf.split(X, Y):
            x_train_, x_test_ = X.iloc[train_index], X.iloc[test_index]
            y_train_, y_test_ = Y[train_index], Y[test_index]
            x_test_l = X_location.iloc[test_index]

            # Specify model configuration
            classifier = self.pipeline

            # Train the classifier
            classifier.fit(x_train_, y_train_)

            # Get class probability predictions for 'test' data
            y_prob_predict = classifier.predict_proba(x_test_)
            y_prob_preds.append(y_prob_predict)

            # For thresholding
            # apply thresholding to each class to create crisp class label
            t_class = threshold_list_c2(y_prob_predict, best_parameters)
            y_predicts.append(t_class)
            # Remove 'ambiguous class'
            # from t_class & y_test_ - for accuracy calculation
            (y_predict_no_amb, y_test_no_amb) = remove_amb_class(
                                                                t_class,
                                                                y_test_)
            # Calculate & put performance metric (balanced accuracy)
            #  per fold into a list
            accuracies.append(balanced_accuracy_score(
                                                     y_test_no_amb,
                                                     y_predict_no_amb
                                                     ))

            # Compute classification reports
            reports.append(classification_report(
                                                y_test_no_amb,
                                                y_predict_no_amb,
                                                output_dict=True
                                                ))

            # Create confusion matrices for default
            # & thresholded results per fold then put in a list
            cm_t = confusion_matrix(y_test_no_amb, y_predict_no_amb,
                                    labels=['Astro', 'Neuron', 'Oligo', 'Others'])
            # ,normalize='true'

            confusion_matrices.append(cm_t)
            y_cv_test.append(y_test_)
            x_cv_test.append(x_test_l)

        self.cv_accuraciesT = accuracies
        self.cv_reportsT = reports
        self.cv_confusion_matricesT = confusion_matrices
        self.cv_y_predictsT = y_predicts
        self.cv_y_prob_predicts = y_prob_preds
        self.cv_x_testT = x_cv_test
        self.cv_y_testT = y_cv_test
