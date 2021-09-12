"""
Provide an example of the regular usage.
"""
import pandas

from models_evaluation import ClassificationEvaluation


if __name__ == '__main__':
    movies = pandas.read_csv('fixtures/movies.csv')
    print(movies)

    evaluation = ClassificationEvaluation(
        data_frame=movies,
        features_columns=[
            'name', 'plot',
        ],
        labels_column='genres',
        thresholds=[
            0.5, 0.6,
        ],
        samplings=[
            'SMOTE', 'ADASYN',
        ],
        classifiers=[
            'RandomForestClassifier', 'SVC',
        ],
        excluded_labels=[
            'Neorealism',
        ],
        metrics=[
            'roc_auc_score', 'accuracy_score', 'precision_score', 'recall_score', 'f1_score',
        ],
        cross_validation_folds=3,
    )
    evaluation.execute()
