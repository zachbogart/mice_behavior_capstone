# mice_behavior_capstone

This repo is for a capstone project on the genetic basis of exploratory behaviors in mice.

## Goal
- Explore genetic basis of exploratory behaviors in two sister species of mice.

![Pipeline](https://github.com/zachbogart/mice_behavior_capstone/blob/master/all_submissions/pipeline.png)

## Pipeline
- Raw video data of mice exploring elevated plus maze
- Feature Engineering: create new features to split up trials and isolate behaviors of mice
- Process video data and extract results for desired features to create CSV of results for all trials
- Visualize differences in features between pure species with violin plots
- Perform significance testing to extract significant features across different dimensions
- Run quantitative trait loci (QTL) analyses for genetic hybrid mice (F2 crosses) to explore genetic basis of behaviors

Below is an explanation of the directories:
- **all_submissions**: deliverables for the course including written reports, the poster, and the ethics audit. The project proposal is also provided.
- **eda**: Scripts to run exploratory data analysis (violin plots to see distributions of features). Resulting plots are provided (`PosterPlots` has final plots used for the poster, etc. as individuals and grids).
- **features**: Scripts to read in the raw data and create CSV of all mice with feature results for each mouse. `process_mice.py` is executed and outputs the CSV to the `results/` folder as `all_the_data.csv`. `feature_engineering.py` does all of the calculations/work and is imported by the `process_mice.py` script. Other package dependencies are in the `resources/` folder. Additionally, `Stat_significance.ipynb` is provided, which used the finished CSV to output the significant features. **Notes**: the processing script expects the data in a folder called `EPM_data` (not included) in the same working directory. Directories used from the dataset are provided in `results_files_used_for_PO_BW_BWPOF1_BWPOF2_analyses_20181129.txt` (located in `resources/`). To only run select inner mouse directories, uncomment lines 39-51 of `process_mice.py`.
- **genetics**: Scripts to create quantitative trait loci (QTL) plots. Vanilla scripts (QTL.ipynb and copies; done to reduce hassle with refreshing cache of RData) and cross-feature scripts (`QTL -- Species*Sex.ipynb`) are provided, as well as starter code (`EPM qtl analysis new chromosome names 2017.ipynb`). Outputs in `results/` folder.
- **meeting-summaries**: Markdown notes of selected biweekly meetings.

## Contributors
- [Zach Bogart](https://github.com/zachbogart)
- [Cynthia Clement](https://github.com/Cynthia3992)
- [Josh Feldman](https://github.com/JoshFeldman777)
- [Kewei Liu](https://github.com/Kewei-Liu)
- [Srinidhi Murthy](https://github.com/Srinidhi-kv)

<img src="https://github.com/zachbogart/mice_behavior_capstone/blob/master/all_submissions/mice_footer.png" width="193" height="52" />
