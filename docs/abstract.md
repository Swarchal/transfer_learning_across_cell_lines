## Title:
The transferability of machine learning models to predict compound
mechanism-of-action from cellular morphology across distinct cell lines.

## Idea:
Train machine learning models which predict compound mechanism of action from
cellular morphology. See how well these transfer when asked to predict MOA
on a previously unseen cell-line with distinct morphology.

Two machine learning techniques:
    1. Extracted morphological features from cellprofiler, fed into a random
       forest classifier
    2. Pixel values (images) used to train a convolutional neural network
       (CNN).

- Train on 7 cell-lines, predict on the with-held 8th.
- Does adding additional data from other cell-lines improve classification
  accuracy? I.e predict on MDA-231, train on MDA-231, then train on MDA-231 + 1
other, then MDA-231 + 2 others. See if classification accuracy on MDA-231 cells
increases?

## Experiment:

