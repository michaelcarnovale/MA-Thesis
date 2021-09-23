# Prediction of Overall and Differential Treatment Response for Depression using Personality Facets: Evaluating Multivariable Prediction Models using Multi-Study Data
A repository for my Master's Thesis (**"Prediction of Overall and Differential Treatment Response for Depression using Personality Facets: Evaluating Multivariable Prediction Models using Multi-Study Data"**) defended for completion of my Master's degree in Clinical Psychology. This repo contains information regarding the actual background/results of my Master's Thesis and files related to its statistical analyses (i.e., Markdown files with R syntax and output). 

### The final approved and official version of my Master's thesis can be found here: https://tspace.library.utoronto.ca/handle/1807/103790
# Table of Contents
* [Description and Methods](https://github.com/michaelcarnovale/MA-Thesis#description-and-methods)
* [Files/Usage](https://github.com/michaelcarnovale/MA-Thesis#files-in-this-repo-and-usage)
* [Abstract/Brief Summary of Results](https://github.com/michaelcarnovale/MA-Thesis#abstractbrief-summary)
# Description and Methods
The Master's thesis was concerned with the following research questions:
1. Using machine learning models, to what degree of accuracy can we predict overall treatment response in patients receiving either psychotherapy or medication for depression?
2. Can we further develop good models to predict whether a given patient may respond better in one treatment vs the other (i.e., using counterfactual estimates)?

This project also involved analyzing data using the following methodologies:
* Integrating data from multiple different clinical trials for depression
  * Data included self-reported personality traits and depression symptom severity (as rated through self-reports and clinician interviews)
* Machine learning techniques and models
  * Use of elastic net penalized regression to perform both variable/feature selection and prediction (*caret* R package)
  * Use of global data partitioning (training, test) and repeated k-fold cross validation to estimate predictive accuracy
  * Comparisons with ordinary least squares regression models
  * Series of models containing various conceptual subsets of features/IVs
  * Two separate groups of models for: (a) self-reported depression severity and (b) clinician-rated depression severity
  * Use of the Personalized Advantage Index (see [here](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6314923/) and [here](https://www.tandfonline.com/doi/abs/10.1080/10503307.2018.1563312) for published examples) methodology to construct and attempt to validate models for predicting differential treatment response - that is, which treatment works best for whom?

# Files in this Repo and Usage
* [R folder](R/) - contains rendered R Markdown syntax and output viewable on GitHub
  * [Data Cleaning, Descriptives, and Inferential Stats R Markdown](R/DataCleaning_Descriptives_InferentialStats.md)
  * [Overall Treatment Response R Markdown](R/OverallTreatmentResponse.md)
  * [Differential Treatment Response R Markdown](R/DifferentialTreatmentResponse.md)

*Note: Unforuntately the data cannot be made available due to privacy issues.* 

# Abstract/Brief Summary
_The potential utility of self-reported personality in the context of a ‘personalized
medicine’ approach to the treatment of Major Depressive Disorder (MDD) remains unclear.
Specifically, although cross-sectional relations between the Five Factor Model (FFM) of
personality and MDD are well-established, treatment-related studies have had several limitations
(e.g., focused on FFM domains, less focused on optimal treatment selection). Using multi-study
data with similar methodologies, the present study was concerned with exploring whether FFM
facets can be used to robustly predict overall treatment response (OTR) and differential treatment
response (DTR). Results suggested that various FFM facets were able to best predict OTR in
HRSD scores, while pre-treatment depression was able to best predict OTR in BDI-II scores.
Results did not support the use of any of the candidate variables for DTR in BDI-II scores, while
a variety of models with different predictor combinations were supported for DTR in HRSD
scores._
