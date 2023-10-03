from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from sklearn.decomposition import PCA

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

# Training features
training_features = ['Detection probability', 'Nucleus: Area µm^2', 'Nucleus: Length µm',
                      'Nucleus: Circularity', 'Nucleus: Solidity', 'Nucleus: Max diameter µm',
                      'Nucleus: Min diameter µm', 'Cell: Area µm^2', 'Cell: Length µm',
                      'Cell: Circularity', 'Cell: Solidity', 'Cell: Max diameter µm',
                      'Cell: Min diameter µm', 'Nucleus/Cell area ratio',
                      'Hematoxylin: Nucleus: Mean', 'Hematoxylin: Nucleus: Median',
                      'Hematoxylin: Nucleus: Min', 'Hematoxylin: Nucleus: Max',
                      'Hematoxylin: Nucleus: Std.Dev.', 'Hematoxylin: Cytoplasm: Mean',
                      'Hematoxylin: Cytoplasm: Median', 'Hematoxylin: Cytoplasm: Min',
                      'Hematoxylin: Cytoplasm: Max', 'Hematoxylin: Cytoplasm: Std.Dev.',
                      'Hematoxylin: Membrane: Mean', 'Hematoxylin: Membrane: Median',
                      'Hematoxylin: Membrane: Min', 'Hematoxylin: Membrane: Max',
                      'Hematoxylin: Membrane: Std.Dev.', 'Hematoxylin: Cell: Mean',
                      'Hematoxylin: Cell: Median', 'Hematoxylin: Cell: Min',
                      'Hematoxylin: Cell: Max', 'Hematoxylin: Cell: Std.Dev.',
                      'NN_10_um','NN_20_um','NN_30_um','NN_40_um','NN_50_um',
                      'NN_60_um','NN_70_um','NN_80_um','NN_90_um','NN_100_um']

# Hyperparameter tuning

pipeline_cell = Pipeline([
    ('normalizer', MinMaxScaler()),
    ('selector', RFE(RandomForestClassifier(random_state=42))),
    ('clf', BalancedRandomForestClassifier())
])

# # Hyper parameter space for tau

# # Features
features_to_select = [28, 30, 34, 36, 38, 40, 42, 44]

# Number of trees in random forest
n_estimators = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

# Number of features to consider at every split (sqrt(55) = ~7)
max_features = [0.2, 0.4, 0.6, 0.8, 1]

# Maximum number of levels in tree
max_depth = [5, 10, 15, 20]
max_depth.append(None)

# Minimum number of samples required to split an internal node
min_samples_split = [2, 5, 10]

# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]

# sampling strategy
sampling_strategy=['auto', 'all', 'not majority', 'majority'] #auto = not minority

# max_samples
max_samples = [0.25, 0.5, 0.75, None]

# # class weights
# class_weight = ['balanced', 'balanced_subsample', None]

# Create the random grid
random_grid = {'selector__n_features_to_select': features_to_select,
                'clf__n_estimators': n_estimators,
               'clf__max_features': max_features,
               'clf__max_depth': max_depth,
               'clf__min_samples_split': min_samples_split,
               'clf__min_samples_leaf': min_samples_leaf,
              'clf__random_state': [42],
               'clf__sampling_strategy': sampling_strategy,
               'clf__max_samples': max_samples,
             'clf__class_weight': ['balanced']
              }

# Hyperparameter tuning:  SVM

# Features
features_to_select = [28, 30, 34, 36, 38, 40, 42, 44]

components_to_select = [0.95, 0.96, 0.97, 0.98, 0.99]

C = [10**1, 10**0, 10**1, 10**2]

gamma = [10**-4, 10**-3, 10**-2, 10**-1, 10**0]


# linear SVM (non probabilistic)
pipeline_cell_linearSVM_prob_noFS = Pipeline([
    ('normalizer', MinMaxScaler()),
    ('clf', SVC(random_state=42,
                kernel='linear',
                probability=True))
])

# linear SVM (non probabilistic)
pipeline_cell_linearSVM = Pipeline([
    ('normalizer', MinMaxScaler()),
    ('selector', RFE(SVC(random_state=42,
                         kernel='linear'))),
    ('clf', SVC(random_state=42,
                kernel='linear'))
])

# linear SVM (probabilistic) + RFE
pipeline_cell_linearSVM_prob = Pipeline([
    ('normalizer', MinMaxScaler()),
    ('selector', RFE(SVC(random_state=42,
                         kernel='linear'))),
    ('clf', SVC(random_state=42,
                kernel='linear',
                probability=True))
])

# linear SVM (probabilistic) + PCA
pipeline_cell_linearSVM_prob_PCA = Pipeline([
    ('normalizer', MinMaxScaler()),
    ('selector', PCA(random_state=42)),
    ('clf', SVC(random_state=42,
                kernel='linear',
                probability=True))
])

# Create the grid
grid_linearSVM = {'selector__n_features_to_select': features_to_select,
                  'clf__C': C,
                  'clf__class_weight': ['balanced']
                  }

# Create the grid, no FS

grid_linearSVM_noFS = {'clf__C': C,
                  'clf__class_weight': ['balanced']
                  }

# Create the grid
grid_linearSVM_PCA = {'selector__n_components': components_to_select,
                  'clf__C': C,
                  'clf__class_weight': ['balanced']
                  }

# RBF-SVM
pipeline_cell_RBF_SVM = Pipeline([
    ('normalizer', MinMaxScaler()),
    ('selector', RFE(SVC(random_state=42,
                         kernel='linear'))),
    ('clf', SVC(random_state=42,
                kernel='rbf'))
])


# RBF-SVM (probabilistic)
pipeline_cell_RBF_SVM_prob = Pipeline([
    ('normalizer', MinMaxScaler()),
    ('selector', RFE(SVC(random_state=42,
                         kernel='linear'))),
    ('clf', SVC(random_state=42,
                kernel='rbf',
                probability=True))
])

# Create the random grid
grid_RBF_SVM = {'selector__n_features_to_select': features_to_select,
                'clf__C': C,
                'clf__gamma': gamma,
                'clf__class_weight': ['balanced']
               }