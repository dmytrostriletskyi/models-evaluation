"""
Provide implementation of the classification evaluation.
"""
import re
from multiprocessing import (
    Process,
    Manager,
)

import numpy
import imblearn
import pandas
from sklearn.naive_bayes import (
    BernoulliNB,
    MultinomialNB,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.svm import SVC
from sklearn.metrics import (
    roc_auc_score as get_roc_auc_score,
    accuracy_score as get_accuracy_score,
    precision_score as get_precision_score,
    recall_score as get_recall_score,
    f1_score as get_f1_score,
)
from sklearn.model_selection import KFold
from tabulate import tabulate

CLASSIFIERS = {
    'BernoulliNB': BernoulliNB,
    'MultinomialNB': MultinomialNB,
    'LogisticRegression': LogisticRegression,
    'KNeighborsClassifier': KNeighborsClassifier,
    'OneVsRestClassifier': OneVsRestClassifier,
    'RandomForestClassifier': RandomForestClassifier,
    'SVC': SVC,
}

METRICS = {
    'roc_auc_score': get_roc_auc_score,
    'accuracy_score': get_accuracy_score,
    'precision_score': get_precision_score,
    'recall_score': get_recall_score,
    'f1_score': get_f1_score,
}

SAMPLINGS = {
    'SMOTE': imblearn.over_sampling.SMOTE,
    'RandomOverSampler': imblearn.over_sampling.RandomOverSampler,
    'ADASYN': imblearn.over_sampling.ADASYN,
    'BorderlineSMOTE': imblearn.over_sampling.BorderlineSMOTE,
    'SVMSMOTE': imblearn.over_sampling.SVMSMOTE,
}


class ClassificationEvaluation:
    """
    Classification evaluation implementation.
    """

    # TODO: human-readable headers.
    # TODO: method of saving.
    def __init__(self,
            data_frame,
            features_columns,
            labels_column,
            thresholds,
            classifiers=None,
            metrics=None,
            samplings=None,
            excluded_labels=None,
            cross_validation_folds=None,
            sort_by=None,
        ):
        """
        Construct the object

        Arguments:
             data_frame ():
             features_columns ():
             labels_column ():
             thresholds ():
             classifiers ():
             metrics ():
             samplings ():
             excluded_labels ():
             cross_validation_folds ():
        """
        self.features_columns = features_columns
        self.thresholds = thresholds
        self.classifiers = classifiers
        self.metrics = metrics
        self.cross_validation_folds = cross_validation_folds
        self.sort_by = sort_by

        if self.classifiers is None:
            self.classifiers = CLASSIFIERS.keys()

        if self.sort_by is None:
            self.sort_by = []

        if self.metrics is None:
            self.metrics = METRICS.keys()

        if excluded_labels is None:
            excluded_labels = []

        self.unique_labels = []

        for index, row in data_frame.iterrows():
            labels = row[labels_column].split(',')

            for label in labels:
                label = label.strip()

                if label in excluded_labels:
                    continue

                label = re.sub('[^-9A-Za-z ]', '', label)
                label = '-'.join(label.split(' '))
                label = label.lower()

                if label not in self.unique_labels:
                    self.unique_labels.append(label)
                    data_frame[f'is-{label}'] = 0

                data_frame.at[index, f'is-{label}'] = 1

        self.samplings = samplings

        if self.cross_validation_folds is None:
            self.cross_validation_folds = 2

        self.data_frame = data_frame
        self.cross_validation = KFold(n_splits=self.cross_validation_folds, shuffle=True)

        self.evaluation_data_frame = pandas.DataFrame(columns=[
            'label',
            'classifier',
            'feature',
            'threshold',
            'sampling',
            'sampling strategy',
        ])

        for metric in metrics:
            self.evaluation_data_frame[metric] = None

    def execute(self):
        """
        Execute the evaluation.
        """
        unique_labels_number = len(self.unique_labels)
        print(f'The following number of unique labels will be processed: {unique_labels_number}.')

        evaluations = []

        for unique_label in self.unique_labels:
            for classifier in self.classifiers:
                for feature_column in self.features_columns:
                    for threshold in self.thresholds:
                        evaluations.append(
                            (unique_label, classifier, feature_column, threshold),
                        )

        processes_returns = Manager().list()
        processes = []

        for evaluation in evaluations:
            process = Process(target=self._evaluate, args=(*evaluation, processes_returns))
            processes.append(process)
            process.start()

        for process in processes:
            process.join()

        for process_return in processes_returns:
            self.evaluation_data_frame = self.evaluation_data_frame.append(
                pandas.DataFrame(process_return),
                ignore_index=True,
            )

        self.evaluation_data_frame = self.evaluation_data_frame.sort_values(
            by=self.sort_by,
            ascending=False,
        )

        data_frame_to_print = tabulate(
            self.evaluation_data_frame,
            headers='keys',
            tablefmt='psql',
        )

        print(data_frame_to_print)

    def _evaluate(self, unique_label, classifier, feature_column, threshold, process_return):
        """
        Evaluate a specific label, classifier, feature_column and threshold.
        """
        classifier_class = CLASSIFIERS.get(classifier)
        classifier_instance = classifier_class(probability=True) if 'SVC' in str(classifier) else classifier_class()

        feature_series = self.data_frame[feature_column]
        label_series = self.data_frame[f'is-{unique_label}']

        for sampling in self.samplings:
            sampling_class = SAMPLINGS.get(sampling)

            for sampling_strategy in ['not majority', 'minority', 'not minority', 'all']:
                sampling_instance = sampling_class(sampling_strategy=sampling_strategy)

                multilabel_binarizer = LabelBinarizer()
                transformed_labels = multilabel_binarizer.fit_transform(y=label_series)

                metrics_scores = {}
                is_sampling_failed = False

                for train_index, test_index in self.cross_validation.split(feature_series):
                    training_dataset, valuation_dataset = (
                        feature_series.iloc[train_index],
                        feature_series.iloc[test_index],
                    )

                    transformed_training_labels, transformed_valuation_labels = (
                        transformed_labels[train_index],
                        transformed_labels[test_index],
                    )

                    tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=10000)
                    transformed_training_dataset = tfidf_vectorizer.fit_transform(training_dataset)
                    transformed_valuation_dataset = tfidf_vectorizer.transform(valuation_dataset)

                    try:
                        transformed_training_dataset, transformed_training_labels = sampling_instance.fit_resample(
                            transformed_training_dataset,
                            transformed_training_labels,
                        )

                    except TypeError:
                        transformed_training_dataset, transformed_training_labels = sampling.fit_resample(
                            transformed_training_dataset.toarray(),
                            transformed_training_labels,
                        )

                    except ValueError:
                        print(
                            f'Sampling {sampling} with strategy {sampling_strategy} for '
                            f'{unique_label}, {classifier}, {feature_column}, {threshold} '
                            f'is failed. Evaluation without sampling will happen.',
                        )

                        is_sampling_failed = True
                        break

                    classifier_instance.fit(
                        transformed_training_dataset,
                        numpy.ravel(transformed_training_labels, order='C'),
                    )

                    transformed_predicted_valuation_labels = \
                        classifier_instance.predict_proba(transformed_valuation_dataset)[::, 1]

                    transformed_predicted_valuation_labels = (
                            transformed_predicted_valuation_labels >= threshold
                    ).astype(int)

                    for metric in self.metrics:
                        metric_function = METRICS.get(metric)

                        metric_score = metric_function(
                            transformed_valuation_labels[::, 0],
                            transformed_predicted_valuation_labels,
                        )

                        if metric not in metrics_scores:
                            metrics_scores[metric] = 0

                        metrics_scores[metric] += float(
                            '{:.2f}'.format(metric_score / self.cross_validation_folds),
                        )

                if not is_sampling_failed:
                    variants = {
                        'label': [unique_label],
                        'classifier': [classifier],
                        'feature': [feature_column],
                        'threshold': [threshold],
                        'sampling': [sampling],
                        'sampling strategy': [sampling_strategy],
                        **metrics_scores,
                    }

                    process_return.append(variants)

        multilabel_binarizer = LabelBinarizer()
        transformed_labels = multilabel_binarizer.fit_transform(y=label_series)

        metrics_scores = {}

        for train_index, test_index in self.cross_validation.split(feature_series):
            training_dataset, valuation_dataset = (
                feature_series.iloc[train_index],
                feature_series.iloc[test_index],
            )

            transformed_training_labels, transformed_valuation_labels = (
                transformed_labels[train_index],
                transformed_labels[test_index],
            )

            tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=10000)
            transformed_training_dataset = tfidf_vectorizer.fit_transform(training_dataset)
            transformed_valuation_dataset = tfidf_vectorizer.transform(valuation_dataset)

            classifier_instance.fit(
                transformed_training_dataset,
                numpy.ravel(transformed_training_labels, order='C'),
            )

            transformed_predicted_valuation_labels = \
                classifier_instance.predict_proba(transformed_valuation_dataset)[::, 1]

            transformed_predicted_valuation_labels = (
                    transformed_predicted_valuation_labels >= threshold
            ).astype(int)

            for metric in self.metrics:
                metric_function = METRICS.get(metric)

                metric_score = metric_function(
                    transformed_valuation_labels[::, 0],
                    transformed_predicted_valuation_labels,
                )

                if metric not in metrics_scores:
                    metrics_scores[metric] = 0

                metrics_scores[metric] += float(
                    '{:.2f}'.format(metric_score / self.cross_validation_folds),
                )

        variants = {
            'label': [unique_label],
            'classifier': [classifier],
            'feature': [feature_column],
            'threshold': [threshold],
            'sampling': ['-'],
            'sampling strategy': ['-'],
            **metrics_scores,
        }

        process_return.append(variants)
