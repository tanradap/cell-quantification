"""
Module with useful constants
"""
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE

from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier

# features for cells
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
# NNB features
nnb_extracted_features = ['Centroid_X','Centroid_Y','NN_10_um','NN_20_um','NN_30_um',
                         'NN_40_um','NN_50_um','NN_60_um','NN_70_um','NN_80_um','NN_90_um','NN_100_um']

# DAB features
dab_features = ['Centroid_X',
                'Centroid_Y',
                'DAB: Nucleus: Mean',
                'DAB: Nucleus: Median',
                'DAB: Nucleus: Min',
                'DAB: Nucleus: Max',
                'DAB: Nucleus: Std.Dev.',
                'DAB: Cytoplasm: Mean',
                'DAB: Cytoplasm: Median',
                'DAB: Cytoplasm: Min',
                'DAB: Cytoplasm: Max',
                'DAB: Cytoplasm: Std.Dev.',
                'DAB: Membrane: Mean',
                'DAB: Membrane: Median',
                'DAB: Membrane: Min',
                'DAB: Membrane: Max',
                'DAB: Membrane: Std.Dev.',
                'DAB: Cell: Mean',
                'DAB: Cell: Median',
                'DAB: Cell: Min',
                'DAB: Cell: Max',
                'DAB: Cell: Std.Dev.']

# Cortical classifier
cortical_classifier_hyperparams = [
    ('normalizer', MinMaxScaler()),
    ('selector', RFE(estimator=RandomForestClassifier(random_state=42),
        n_features_to_select=38)),
    ('clf', BalancedRandomForestClassifier(
        random_state=42,
        sampling_strategy='not majority',
        n_estimators=900,
        min_samples_split=5,
        min_samples_leaf=4,
        max_features=0.2,
        max_depth=None,
        max_samples=0.5,
        class_weight='balanced'))]

# Occipital classifier
occipital_classifier_hyperparams = [
    ('normalizer', MinMaxScaler()),
    ('selector', RFE(estimator=RandomForestClassifier(random_state=42),
        n_features_to_select=28)),
    ('clf', BalancedRandomForestClassifier(
        random_state=42,
        sampling_strategy='auto',
        n_estimators=200,
        min_samples_split=10,
        min_samples_leaf=4,
        max_features=0.4,
        max_depth=15,
        max_samples=0.25,
        class_weight='balanced'))]

# BG classifier
bg_classifier_hyperparams = [
    ('normalizer', MinMaxScaler()),
    ('selector', RFE(estimator=RandomForestClassifier(random_state=42),
        n_features_to_select=38)),
    ('clf', BalancedRandomForestClassifier(
        random_state=42,
        sampling_strategy='not majority',
        n_estimators=600,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features=0.2,
        max_depth=10,
        max_samples=0.75,
        class_weight='balanced'))]
