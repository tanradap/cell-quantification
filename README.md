# cell-quantification

This repository contains python scripts & jupyter notebooks used to create the neuronal and glial cell classification pipeline for PSP postmortem brain. Morphological and slide staining intensity were used as features for cell classification, which were extracted from cells using QuPath software. This pipeline did not end up performing well, but the backbone is useful for other purposes e.g. creating the tau quantitifaction pipeline. Cell classes include neurons, astrocytes, oligodendrocytes, and others. Folders are detailed below:

**Tuning_parameters:** for tuning hyperparameters of the classification model.

**Sample_size_check:** scripts for checking we have annotated sufficient cells to train our model.

**Cell_classification:** for classifying cell into type-specific classes (neuron, astrocyte, oligodendrocyte, others).

**Cell_quantification:** for collating classified cells from multiple slides into a single file, and some useful plotting functions.
