Kaggle competition link：https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification

## Overview

### Competition Goal

The task is to develop a model that analyzes EEG (electroencephalography) data and predicts the probability distribution over six types of brain activity that may be harmful or indicative of seizures.

The six target classes are:

- seizure
- lpd (lateralized periodic discharges)
- gpd (generalized periodic discharges)
- lrda (lateralized rhythmic delta activity)
- grda (generalized rhythmic delta activity)
- other

Your model should predict the probability of each class for each EEG segment.

------

### Data Overview

- `train.csv`: Metadata for each EEG/spectrogram segment and corresponding expert votes.
- `test.csv`: Metadata for test EEG segments.
- `sample_submission.csv`: Submission format.
- `train_eegs/`: Raw EEG signals, sampled at 200 Hz, with 21 channels.
- `test_eegs/`: Each file contains exactly 50 seconds of EEG data.
- `train_spectrograms/` and `test_spectrograms/`: Spectrograms built from EEG signals.
- `example_figures/`: Visual examples of each brain activity type.

Each training sample corresponds to a 50-second EEG clip and a matched 10-minute spectrogram segment. Labels are based on the central 10 seconds of each segment.

------

### Prediction Requirements

Your submission must be a `.csv` file with the following columns:

```
eeg_id,seizure_vote,lpd_vote,gpd_vote,lrda_vote,grda_vote,other_vote
```

- One row per `eeg_id` in the test set.
- Each column represents the predicted probability of that class.
- Probabilities **do not have to sum to 1**.
- Your model should reflect uncertainty (e.g., ambiguous cases may have multiple high values).

------

### Evaluation Metric

Submissions are evaluated using the **Kullback–Leibler (KL) divergence** between your predicted probability vector and the observed label distribution (expert vote distribution). Lower is better.

------

### Label Characteristics

Expert labels fall into three categories:

- **Idealized**: High agreement among experts (e.g., a typical seizure).
- **Proto-patterns**: Split between one IIIC pattern and “other” (e.g., poorly formed or ambiguous activity).
- **Edge cases**: Split between two IIIC patterns (e.g., features of both seizure and LPD).

These reflect the natural ambiguity in EEG interpretation, even among specialists.

## Results (temporary)

This repository only records a phased attempt for this competition. Due to the limitation of GPU resources, the model training only went to epoch15, but the verification accuracy has reached 0.9041. 

The training details are in Logs.txt

### Test data results (only one)

```
Device used: cuda
Model loaded
Test dataset size: 1
Starting to generate predictions...
Processing progress: 0/1 batches
Prediction completed!
Submission file saved as submission.csv
Submission file shape: (1, 7)
Preview of the first 5 rows:
       eeg_id  seizure_vote  lpd_vote  gpd_vote  lrda_vote  grda_vote  \
0  3911565283      0.977652  0.003053  0.000528   0.006812   0.003766   

   other_vote  
0    0.008188  

Verify submission file:
Number of EEG IDs: 1
Column names: ['eeg_id', 'seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']
Probability sum check (first 5 rows):
0 1.0
dtype: float32
Probability sum range: 1.0000 - 1.0000

Submission complete!
```