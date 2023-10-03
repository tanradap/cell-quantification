from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
import numpy as np

cell_extracted_features = [
                    'Image',
                    'Name',
                    'Class',
                    'Parent',
                    'ROI',
                    'Centroid_X',
                    'Centroid_Y',
                    'Detection probability',
                    'Nucleus: Area µm^2',
                    'Nucleus: Length µm',
                    'Nucleus: Circularity',
                    'Nucleus: Solidity',
                    'Nucleus: Max diameter µm',
                    'Nucleus: Min diameter µm',
                    'Cell: Area µm^2',
                    'Cell: Length µm',
                    'Cell: Circularity',
                    'Cell: Solidity',
                    'Cell: Max diameter µm',
                    'Cell: Min diameter µm',
                    'Nucleus/Cell area ratio',
                    'Hematoxylin: Nucleus: Mean',
                    'Hematoxylin: Nucleus: Median',
                    'Hematoxylin: Nucleus: Min',
                    'Hematoxylin: Nucleus: Max',
                    'Hematoxylin: Nucleus: Std.Dev.',
                    'Hematoxylin: Cytoplasm: Mean',
                    'Hematoxylin: Cytoplasm: Median',
                    'Hematoxylin: Cytoplasm: Min',
                    'Hematoxylin: Cytoplasm: Max',
                    'Hematoxylin: Cytoplasm: Std.Dev.',
                    'Hematoxylin: Membrane: Mean',
                    'Hematoxylin: Membrane: Median',
                    'Hematoxylin: Membrane: Min',
                    'Hematoxylin: Membrane: Max',
                    'Hematoxylin: Membrane: Std.Dev.',
                    'Hematoxylin: Cell: Mean',
                    'Hematoxylin: Cell: Median',
                    'Hematoxylin: Cell: Min',
                    'Hematoxylin: Cell: Max',
                    'Hematoxylin: Cell: Std.Dev.']

# original cell classification pipeline & hyperparam space config

pipeline_cell = Pipeline([
    ('normalizer', MinMaxScaler()),
    ('selector', RFE(SVC(kernel='linear'))),
    ('clf', BalancedRandomForestClassifier())
])

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

# Number of features to consider at every split
max_features = ['auto', 'sqrt']

# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)

# Minimum number of samples required to split an internal node
min_samples_split = [2, 5, 10]

# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]

# Method for selecting samples for training each tree
bootstrap = [True, False]

# sampling strategy
sampling_strategy = ['auto', 'all', 'not majority', 'majority']

# ccp_alphas
ccp_alphas = [float(x) for x in np.linspace(start=0, stop=0.03, num=7)]

# Create the random grid
random_grid = {'selector__n_features_to_select': [30,32,34,36,38,40],
                'clf__n_estimators': n_estimators,
               'clf__max_features': max_features,
               'clf__max_depth': max_depth,
               'clf__min_samples_split': min_samples_split,
               'clf__min_samples_leaf': min_samples_leaf,
              'clf__bootstrap': bootstrap,
              'clf__random_state':[42],
               'clf__sampling_strategy':sampling_strategy,
               'clf__ccp_alpha':ccp_alphas
              }