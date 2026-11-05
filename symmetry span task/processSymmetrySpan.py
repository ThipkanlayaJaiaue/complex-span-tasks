import pandas as pd
import numpy as np
import glob

## PART 1: SYMMETRY SPAN RAW DATA PROCESSING

# set folder path
task_folder = r"D:\PhD\study3\data_raw\Tasks"

# find all Symmetry Span CSV files
task_files = glob.glob(task_folder + r"\*.csv")

symmetryS_files = [file for file in task_files
    if "symmetry" in file.split("\\")[-1].lower()]

# create an empty list for storing data
symmetryS_dataList = []

# create the for loop to process each participant's data
for file in symmetryS_files:

    task = pd.read_csv(file)

    # extract participant ID from filename
    participant_id = file.split("\\")[-1].split("_")[0]

    # group by block and size set
    eachBlockAndSizeSet = (
        task.groupby(
            ['ThreeBlocks.thisN', 'AllExperimentalBlocks.thisN']
        ).agg(
            size_set=('myExp.Accuracy_red', 'count'),  # size set
            recall_correct=('myExp.Accuracy_red', 'sum'),  # total number of correct responses
            distract_correct=('myExp.Accuracy_sym', 'sum'),  # total number of correct distraction trial responses
            distract_incorrect=('myExp.error_sym', 'sum'),  # total number of incorrect distraction trials responses
            distract_tooLate=('myExp.error_speed', 'sum')  # total number of too-late responses on distraction trials
        ).reset_index()
    )

    # calculate percentages
    eachBlockAndSizeSet['accuracyRecall(%)'] = (eachBlockAndSizeSet['recall_correct'] / eachBlockAndSizeSet['size_set']) * 100
    eachBlockAndSizeSet['accuracyDistraction(%)'] = (eachBlockAndSizeSet['distract_correct'] / eachBlockAndSizeSet['size_set']) * 100
    eachBlockAndSizeSet['errorDistraction(%)'] = (eachBlockAndSizeSet['distract_incorrect'] / eachBlockAndSizeSet['size_set']) * 100  # incorrect response during distraction
    eachBlockAndSizeSet['lateDistraction(%)'] = (eachBlockAndSizeSet['distract_tooLate'] / eachBlockAndSizeSet['size_set']) * 100  # no response within the limited time

    # add participant ID
    eachBlockAndSizeSet['participant'] = int(participant_id)

    # append to list
    symmetryS_dataList.append(eachBlockAndSizeSet)

# combine all participants into one DataFrame
rawSymmetryS = pd.concat(symmetryS_dataList, ignore_index=True)

# move participant to the first column
rawSymmetryS = rawSymmetryS[['participant'] + [col for col in rawSymmetryS.columns if col != 'participant']]

# save processed raw data as .csv
rawSymmetryS.to_csv('rawSymmetrySpan.csv', index=False)

# ------------------------------------------------------ #

# PART 2: SYMMETRY SPAN SCORING

# create all-or-nothing load per size set
rawSymmetryS['all_or_nothing_load'] = np.where(rawSymmetryS['recall_correct'] == rawSymmetryS['size_set'], rawSymmetryS['recall_correct'], 0)

# calculate four scoring procedures per participant
# based on (Conway, A. R. A., Kane, M. J., Hambrick, D. Z., Wilhelm, O., & Engle, R. W. (2005).
# Working memory span tasks: A methodological review and user’s guide. Psychonomic Bulletin & Review, 12(5), 769–786.
# https://doi.org/https://doi.org/10.3758/BF03196772)
symmetryS = (rawSymmetryS.groupby('participant').agg(

        ## partial-credit unit score
        # sum of (all correct recalls in each size set == 1
        # otherwise calculate as a proportion [20% == 20/100 == 0.2])
        # and divided by all items (3 blocks of 4 size sets = 12)
        partial_unit=('accuracyRecall(%)', lambda x: (x / 100).sum() / 12),

        ## all-or-nothing unit score
        # sum of (all correct recalls in each size set == 1, otherwise == 0) / all items (3 blocks of 4 size sets = 12)
        all_or_nothing_unit=('accuracyRecall(%)', lambda x: (x == 100).sum() / 12),

        ## partial-credit load score
        # sum of all correct recalls in each size set / total recalls (3 blocks of 14 recalls [size set of 2,3,4,5] = 42)
        partial_load=('recall_correct', lambda x: x.sum() / 42),

        ## all-or-nothing load score
        # sum of all-or-nothing load in each size set / total recalls (3 blocks of 14 recalls [size set of 2,3,4,5] = 42)
        all_or_nothing_load=('all_or_nothing_load', lambda x: x.sum() / 42)

    ).reset_index()
)

# save scores as .csv
symmetryS.to_csv('scoreSymmetrySpan.csv', index=False)