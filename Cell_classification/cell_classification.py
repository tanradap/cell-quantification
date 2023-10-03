import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import classification_report
from base import *


class CellClassifier():
    def __init__(self, hyperparameters):
        self.pipeline = Pipeline(hyperparameters)

        self.classifier = None
        self.f_importance = None
        self.best_parameters = None

        self.cv_accuracies = None
        self.cv_reports = None
        self.cv_confusion_matrices = None

    def train(self, X, Y):
        """
        Train the classifier to classify tau detections
        into 4 classes (Astro, Neuron, Oligo, Others) and Ambiguous.

        Attributes:
        - f_importance: feature importance
        - best_parameters: The final chosen threshold (mean across 10-folds)
        - From 10-fold cross validation,
        you can extract best_parameters, accuracies, reports,
        and confusion matrices to see training performance.
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

        # train the final model
        classifier.fit(X, Y)
        self.classifier = classifier

        # selected features from training
        selected_features_indices = classifier.named_steps['selector'].get_support(indices=True)
        selected_features = [X.columns[i] for i in selected_features_indices]
        importance = classifier.named_steps['clf'].feature_importances_
        f_importance = pd.DataFrame(data={
            'features': selected_features,
            'importance': importance})
        f_importance = f_importance.sort_values(by=['importance'],
                                                ascending=False)
        self.f_importance = f_importance

    def predict(self, inputs):

        # get class score predictions
        y_prob_predict = self.classifier.predict_proba(inputs)
        self.prob_predict = y_prob_predict

        # apply thresholding to get crisp class labels
        y_predict = threshold_list_c2(y_prob_predict, self.best_parameters)
        self.prediction = y_predict