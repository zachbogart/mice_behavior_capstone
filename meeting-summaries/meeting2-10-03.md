# Meeting 1 (10-03-2018)

## Summary
For our second meeting, we solidified the plan for taking data and creating intermediate and aggregate features from the mice data. We also shared preliminary feature ideas and clarified the setup/organization of the dataset.

## Setup
- Clarified the organization of the dataset: every folder is one mouse trial. Specifically, the *.tar* file in the *analysis* folder contains **bounding area points** for every frame of the video. Each trial is split into multiple *.list* files in batches of 100.
- We got access to the python script used to convert bounding box data to the desired baseline features

## Data
- A lot of the data provided is not necessary for the analysis we want to perform. In an effort to reduce the size of the drive folder (so that it can be more easily mounted with Google Colab), we will be cleaning the data and looking into **setting up a new google account** with just the analysis folders for the given trial.

## TODO
- **Clean up the data** into manageable format
- **Create new google account** to house the dataset
- Look into **features from bounding box data**, as discussed
- **Run python script** on data

We will be meeting next week as a team to solidify the GDrive setup and discuss next steps in feature engineering.
