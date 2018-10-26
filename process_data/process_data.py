import csv

from EPM_analysis import *

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

    # TODO: Remove me  --  Use this to stop at only 1 mouse
    # turnOnHistograms()
    # if mouseDirectory not in [
    #     '20121130_163816_EPM_BWPOF1_784_M',
    #     '20121121_151056_EPM_BWPOF2_767_F',
    #     '20121121_160958_EPM_PO_758_F',
    #     '20121217_165902_EPM_BWPOF1_823_F',
    #     '20130129_162312_EPM_BW_1368_F',
    #     '20130213_165903_EPM_BWPOF2_1464_M',
    #     '20130123_112930_EPM_BWPOF2_1316_M',
    # ]:
    #     return {}

    conditions_folder_path, innerDirectory = extractContentDirectory(mouseDirectory, parentDirectory)
    mouseFeatures = extractMouseFeatures(mouseDirectory)

    print('Finding positions over time')
    start_frame, end_frame = getStartEndFrames()
    boundaries, zones_masks, shape = load_data(conditions_folder_path, start_frame, end_frame)
    boundaries, results_array, zones_order = cleanBoundaries(boundaries, shape, zones_masks)
    centroids = calculateCentroids(boundaries, shape)
    centroidsByArm = calculateCentroidsByArm(centroids, zones_order, results_array)

    print('Finding arm entry features')
    fraction_in_arms, totalArmEntries, frames_in_arms, arm_entries = calculateArmEntries(
        zones_order, results_array, start_frame, end_frame, conditions_folder_path
    )
    turningPreferences = calculateTurningPreference(arm_entries)

    print('Finding mouse size')
    mouseLength = calculateMouseSize(boundaries)

    print('Finding velocity features')
    distancesPerArm, directionsPerArm, totalDistancePerArm = calculateDistanceFeatures(centroidsByArm)
    velocity_features = calculateVelocityFeatures(distancesPerArm, directionsPerArm)

    print('Finding miscellaneous features')
    restFractionPerArm = calculateRestFeatures(distancesPerArm)
    safetyFractionsPerArm = calculateSafetyFeatures(centroidsByArm, mouseLength)
    safetyAndRestFractionsPerArm = calculateSafeAndRestingFeatures(centroidsByArm, distancesPerArm, mouseLength)
    peakingFeatures = calculatePeekingFeatures(centroidsByArm, distancesPerArm, mouseLength)
    backtrackCounts = calculateBacktrackCounts(arm_entries)

    print('Results found for {}'.format(mouseDirectory))
    results = {
        'inner_directory': innerDirectory,
        'mouse_details': mouseFeatures,
        'mouse_length': mouseLength,
        'turning_preferences': turningPreferences,
        'fraction_in_arms': fraction_in_arms,
        'arm_entries': arm_entries,
        'tot_arm_entries': totalArmEntries,
        'frames_in_arms': frames_in_arms,
        'total_distance': totalDistancePerArm,
        'velocity': velocity_features,
        'active_fraction': restFractionPerArm,
        'safety_fraction': safetyFractionsPerArm,
        'safety_and_rest_fraction': safetyAndRestFractionsPerArm,
        'peeking': peakingFeatures,
        'backtrack_counts': backtrackCounts,
    }
    return results


def populateDirectoriesToUse():
    with open('results_files.txt') as csvfile:
        global directoriesToUse
        spamReader = csv.reader(csvfile, delimiter=',')
        for row in spamReader:
            for item in row:
                cleanPath = item.replace("'", "").strip()
                lastSlash = cleanPath.rfind("/")
                secondLastSlash = cleanPath[:lastSlash].rfind("/")
                actualDirectory = cleanPath[secondLastSlash + 1:lastSlash]
                directoriesToUse.add(actualDirectory)


def getStartEndFrames():
    start_frame = 0
    end_frame = None
    # TODO: What is this argument? Do we need to use this?
    # if len(sys.argv) > 2:
    #     start_frame = (sys.argv[
    #         2]) * 30  # Substract 15 because Brant tracking starts tracking videos 15 seconds in. Multiply by 30 fps
    #     if len(sys.argv) > 3:
    #         end_frame = (sys.argv[
    #             3]) * 30  # Substract 15 because Brant tracking starts tracking videos 15 seconds in. Multiply by 30 fps
    return start_frame, end_frame


def extractContentDirectory(mouseDirectory, parentDirectory):
    if not os.path.isfile(os.path.join(parentDirectory, mouseDirectory, '{}.zones.dict'.format(mouseDirectory))):
        raise NoDataError('No .zones.dict inside conditions directory for: {}'.format(mouseDirectory))

    conditions_folder_path = None
    innerDirectory = None
    analysisDirectory = os.path.join(parentDirectory, mouseDirectory, 'analysis')
    if os.path.isdir(analysisDirectory):
        innerDirectories = os.listdir(analysisDirectory)
        if innerDirectories:
            directoryFound = False
            for innerDirectory in innerDirectories:
                if os.path.isdir(os.path.join(analysisDirectory, innerDirectory)):
                    if innerDirectory in directoriesToUse:
                        conditions_folder_path = os.path.join(analysisDirectory, innerDirectory)
                        if not os.path.isfile(os.path.join(conditions_folder_path, 'miceols.tar')):
                            raise NoDataError(
                                'No miceols.tar inside conditions directory for: {}'.format(mouseDirectory))
                        else:
                            directoryFound = True
            if not directoryFound:
                raise NoDataError('No inner folders found for: {}'.format(mouseDirectory))
        else:
            raise NoDataError('No directories found inside analysis directory for: {}'.format(mouseDirectory))
    else:
        raise NoDataError('No analysis directory found for: {}'.format(mouseDirectory))
    return conditions_folder_path, innerDirectory


def getMouseDirectories():
    currentDirectory = os.getcwd()
    parentDirectory = os.path.join(currentDirectory, "Mice_Capstone_data_files")
    # parentDirectory = os.path.join(currentDirectory, "EPM_data")
    mouseDirectories = os.listdir(parentDirectory)
    mouseDirectories.sort()
    return parentDirectory, mouseDirectories


def flattenDict(dictionary):
    result = {}
    for key in dictionary.keys():
        if key != 'arm_entries':
            if isinstance(dictionary[key], dict):
                for key2 in dictionary[key].keys():
                    result[key + '_' + key2] = dictionary[key][key2]
            else:
                result[key] = dictionary[key]
    return result


def main():
    startTime = datetime.now()
    parentDirectory, mouseDirectories = getMouseDirectories()
    aggregateResults = []
    for mouseDirectory in mouseDirectories:
        try:
            mouseResults = process_directory(parentDirectory, mouseDirectory)
            flatResults = flattenDict(mouseResults)
            aggregateResults.append(flatResults)
        except (NoDataError, IndexError), exception:  # TODO Index Error fix?
            print('Exception: {}'.format(exception))
    # saveResultsAsJson(aggregateResults)
    saveResultsAsCSV(aggregateResults)
    print("Time to execute: {}".format(datetime.now() - startTime))


def saveResultsAsCSV(aggregateResults):
    pd.DataFrame(aggregateResults).to_csv('all_the_data.csv')


def saveResultsAsJson(aggregateResults):
    with open('all_the_data.json', 'w') as fp:
        json.dump(aggregateResults, fp)


main()
