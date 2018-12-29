import csv

# from feature_engineering import *
from logging import exception

from mice_behavior_capstone.features.feature_engineering import *


import sys
import os
import json
import pandas as pd
from datetime import datetime

import plotly.plotly as py
import plotly.tools as tls
import matplotlib.pyplot as plt
import numpy as np


class NoDataError(Exception):
    pass


directoriesToUse = set()



def process_directory(parentDirectory, mouseDirectory):
    populateDirectoriesToUse()
    setMouseDirectory(mouseDirectory)

    print('')
    print('Loading data for {}'.format(mouseDirectory))

    #Remove me  --  Use this to stop at only select mice

    #if mouseDirectory not in [
    #     '20121130_163816_EPM_BWPOF1_784_M',
    #     '20121121_151056_EPM_BWPOF2_767_F',
    #     '20121121_160958_EPM_PO_758_F',
    #     '20121217_165902_EPM_BWPOF1_823_F',
    #     '20130129_162312_EPM_BW_1368_F',
    #     '20130213_165903_EPM_BWPOF2_1464_M',
    #     '20130123_112930_EPM_BWPOF2_1316_M',
    #     '20130407_180759_EPM_BWPOF2_2390_F'
    # ]:
    #     return {}
    # if "_861" not in mouseDirectory:
    #     return {}
    # turnOnHistograms()

    conditions_folder_path, innerDirectory = extractContentDirectory(mouseDirectory, parentDirectory)
    mouseFeatures = extractMouseFeatures(mouseDirectory)

    print('Finding positions over time')
    start_frame, end_frame = getStartEndFrames(conditions_folder_path)
    boundaries, zones_masks, shape = load_data(conditions_folder_path, start_frame, end_frame)
    boundaries, results_array, zones_order = cleanBoundaries(boundaries, shape, zones_masks)
    centroids = calculateCentroids(boundaries, shape)
    centroidsByArm = calculateCentroidsByArm(centroids, zones_order, results_array)

    if not boundaries:  # If we have no tracking data, stop now
        return {}
    testDistance(centroids)  # Testing only

    print('Finding arm entry features')
    fraction_in_arms, totalArmEntries, frames_in_arms, arm_entries = calculateArmEntries(
        zones_order, results_array, start_frame, end_frame, conditions_folder_path
    )
    turningPreferences = calculateTurningPreference(arm_entries)

    print('Finding velocity features')
    distancesPerArm, directionsPerArm, totalDistancePerArm, distances = calculateDistanceFeatures(centroidsByArm, centroids)
    velocity_features = calculateVelocityFeatures(distancesPerArm, directionsPerArm)

    print('Finding mouse size')
    mouseSizeFeatures = calculateMouseLength(boundaries, distances)
    mouseLength = mouseSizeFeatures['mouseLength']

    # Some mice don't move around enough to get a good reading. If this happens we want to exclude the mouse
    mouseLengthValid = mouseLengthIsValid(mouseLength)
    if mouseLengthIsValid:
        mouseSizeFeatures = {}

    print('Finding miscellaneous features')
    restFractionPerArm = calculateRestFeatures(distancesPerArm)
    backtrackCounts = calculateBacktrackCounts(arm_entries)
    if mouseLengthValid: # If the mouse length is not valid, neither are these these features
        safetyFractionsPerArm = calculateSafetyFeatures(centroidsByArm, mouseLength)
        safetyAndRestFractionsPerArm = calculateSafeAndRestingFeatures(centroidsByArm, distancesPerArm, mouseLength)
        peakingFeatures = calculatePeekingFeatures(centroidsByArm, distancesPerArm, mouseLength)
    else:
        safetyFractionsPerArm = {}
        safetyAndRestFractionsPerArm = {}
        peakingFeatures = {}

    print('Convert to cm')
    
    velocity_features = convertToCM(velocity_features)
    totalDistancePerArm = convertToCM(totalDistancePerArm)
    mouseSizeFeatures = convertToCM(mouseSizeFeatures)

    print('Results found for {}'.format(mouseDirectory))
    results = {
        'inner_directory': innerDirectory,
        'mouse_details': mouseFeatures,
        'mouse_dimensions': mouseSizeFeatures,
        'turning_preferences': turningPreferences,
        'fraction_in_arms': fraction_in_arms,
        'tot_arm_entries': totalArmEntries,
        'frames_in_arms': frames_in_arms,
        'total_distance': totalDistancePerArm,
        'velocity': velocity_features,
        'rest_fraction': restFractionPerArm,
        'safety_fraction': safetyFractionsPerArm,
        'safety_and_rest_fraction': safetyAndRestFractionsPerArm,
        'peeking': peakingFeatures,
        'backtrack_counts': backtrackCounts,
    }
    return results


def populateDirectoriesToUse():
    with open('results_files_used_for_PO_BW_BWPOF1_BWPOF2_analyses_20181129.txt') as f:
        global directoriesToUse
        # spamReader = csv.reader(csvfile, delimiter=',')
        for row in f:
            cleanPath = row.rstrip('\n')
            cleanPath = cleanPath[5:]
            lastSlash = cleanPath.rfind("/")
            # secondLastSlash = cleanPath[:lastSlash].rfind("/")
            # actualDirectory = cleanPath[secondLastSlash + 1:lastSlash]
            actualDirectory = cleanPath[:lastSlash]
            directoriesToUse.add(actualDirectory)


def getStartEndFrames(conditions_folder_path):
    start_frame = 0
    end_frame = None

    # Retrieve timeframe data
    file_main_dir = os.path.dirname(os.path.dirname(conditions_folder_path))
    file_path = os.path.join(file_main_dir, os.path.basename(file_main_dir) + '.timeframe.tuple')
    if os.path.isfile(file_path):
        with open(file_path, 'r') as f:
            start_frame, end_frame = eval(f.read())
            start_frame = start_frame * 30  # Substract 15 because Brant tracking starts tracking videos 15 seconds in. Multiply by 30 fps
            if end_frame:
                end_frame = end_frame * 30  # Substract 15 because Brant tracking starts tracking videos 15 seconds in. Multiply by 30 fps
    return start_frame, end_frame


def extractContentDirectory(mouseDirectory, parentDirectory):
    if not os.path.isfile(os.path.join(parentDirectory, mouseDirectory, '{}.zones.dict'.format(mouseDirectory))):
        raise NoDataError('No .zones.dict inside conditions directory for: {}'.format(mouseDirectory))

    conditions_folder_path = None
    actualInnerDirectory = ''
    analysisDirectory = os.path.join(parentDirectory, mouseDirectory, 'analysis')
    if os.path.isdir(analysisDirectory):
        innerDirectories = os.listdir(analysisDirectory)
        if innerDirectories:
            directoryFound = False
            for innerDirectory in innerDirectories:
                if os.path.isdir(os.path.join(analysisDirectory, innerDirectory)):
                    fullPath = mouseDirectory + '/analysis/' + innerDirectory
                    if fullPath in directoriesToUse:
                        conditions_folder_path = os.path.join(analysisDirectory, innerDirectory)
                        if not os.path.isfile(os.path.join(conditions_folder_path, 'miceols.tar')):
                            raise NoDataError(
                                'No miceols.tar inside conditions directory for: {}'.format(mouseDirectory))
                        else:
                            directoryFound = True
                            actualInnerDirectory = innerDirectory
            if not directoryFound:
                raise NoDataError('No inner folders found for: {}'.format(mouseDirectory))
        else:
            raise NoDataError('No directories found inside eda directory for: {}'.format(mouseDirectory))
    else:
        raise NoDataError('No analysis directory found for: {}'.format(mouseDirectory))
    return conditions_folder_path, actualInnerDirectory


def getMouseDirectories():
    currentDirectory = os.getcwd()
    parentDirectory = os.path.join(currentDirectory, "EPM_data")
    mouseDirectories = os.listdir(parentDirectory)
    mouseDirectories.sort()
    return parentDirectory, mouseDirectories


def flattenDict(d):
    def items():
        for key, value in d.items():
            if isinstance(value, dict):
                for subKey, subValue in flattenDict(value).items():
                    yield key + "_" + subKey, subValue
            else:
                yield key, value

    return dict(items())


def main():
    startTime = datetime.now()
    parentDirectory, mouseDirectories = getMouseDirectories()
    aggregateResults = []
    for mouseDirectory in mouseDirectories:
        try:
            mouseResults = process_directory(parentDirectory, mouseDirectory)
            if mouseResults:
                flatResults = flattenDict(mouseResults)
                aggregateResults.append(flatResults)
        except (NoDataError, exception):  # TODO Index Error fix?
            print('Exception: {}'.format(exception))
    # saveResultsAsJson(aggregateResults)
    saveResultsAsCSV(aggregateResults)
    print("Time to execute: {}".format(datetime.now() - startTime))


def saveResultsAsCSV(aggregateResults):
    pd.DataFrame(aggregateResults).to_csv('results/all_the_data.csv')


def saveResultsAsJson(aggregateResults):
    with open('all_the_data.json', 'w') as fp:
        json.dump(aggregateResults, fp)


main()
