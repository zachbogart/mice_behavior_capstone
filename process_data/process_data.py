import csv

from EPM_analysis import *

import sys
import os
import json
import pandas as pd
from datetime import datetime


class NoDataError(Exception):
    pass


directoriesToUse = set()


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


def process_directory(parentDirectory, mouseDirectory):
    populateDirectoriesToUse()

    print('')
    print('Loading data for {}'.format(mouseDirectory))

    if not os.path.isfile(os.path.join(parentDirectory, mouseDirectory, '{}.zones.dict'.format(mouseDirectory))):
        raise NoDataError('No .zones.dict inside conditions directory for: {}'.format(mouseDirectory))

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
                            raise NoDataError('No miceols.tar inside conditions directory for: {}'.format(mouseDirectory))
                        else:
                            directoryFound = True
                    # else:
                    #     raise NoDataError('Directory not in results_files.txt: {}'.format(mouseDirectory))
            if not directoryFound:
                raise NoDataError('No inner folders found for: {}'.format(mouseDirectory))
        else:
            raise NoDataError('No directories found inside analysis directory for: {}'.format(mouseDirectory))
    else:
        raise NoDataError('No analysis directory found for: {}'.format(mouseDirectory))
    start_frame = 0
    end_frame = None

    # TODO: What is this argument? Do we need to use this?
    # if len(sys.argv) > 2:
    #     start_frame = (sys.argv[
    #         2]) * 30  # Substract 15 because Brant tracking starts tracking videos 15 seconds in. Multiply by 30 fps
    #     if len(sys.argv) > 3:
    #         end_frame = (sys.argv[
    #             3]) * 30  # Substract 15 because Brant tracking starts tracking videos 15 seconds in. Multiply by 30 fps

    mice_ols, zones_masks, shape = load_data(conditions_folder_path, start_frame, end_frame)
    # Make a mask of all the EPM zones (those that don't start with 'F', for floor)
    print >> sys.stdout, 'Calculating residency in arms'
    EPMzones = reduce(lambda x, y: x + y, [zones_masks[k] for k in zones_masks if not k.startswith('F')])
    #
    mice_ols_in_EPMzones = filter_mice_ols(mice_ols, EPMzones)
    filled_in_mice_ols = fill_in_missing_outlines(mice_ols_in_EPMzones)
    zones_order, results_array = initialize_results_array(zones_masks, filled_in_mice_ols)
    mouse_position_in_zones(filled_in_mice_ols, shape, zones_order, zones_masks, results_array)
    frac_in_arms, tot_arm_entries, frames_in_arms, arm_entries = calculateResults(zones_order, results_array,
                                                                                  start_frame,
                                                                                  end_frame, conditions_folder_path)
    arcs_in_arms = calcPosition(shape, filled_in_mice_ols, zones_order, results_array)
    print >> sys.stdout, 'Calculating distance travelled'
    distance, total_distance = calcDistance(arcs_in_arms)  # In pixels
    smoothed_distance = {k: smooth(np.array(v)) for k, v in distance.items()}
    total_smoothed_distance = {k: np.sum(v) for k, v in smoothed_distance.items()}
    # Speed in pixels per second
    print >> sys.stdout, 'Calculating speed'
    fps = 30.083  # Median frames per seconds of >2000 EPM videos
    median_speed = {k: np.median(v) * fps for k, v in distance.items()}
    smoothed_median_speed = {k: np.median(v) * fps for k, v in smoothed_distance.items()}
    print >> sys.stdout, 'Results found'
    turningPreferences = calculateTurningPreference(arm_entries)
    mouseData = getMouseData(mouseDirectory)
    results = {
        'turning_preferences': turningPreferences,
        'mouse_details': mouseData,
        'inner_directory': innerDirectory,
        'frac_in_arms': frac_in_arms,
        'arm_entries': arm_entries,
        'tot_arm_entries': tot_arm_entries,
        'frames_in_arms': frames_in_arms,
        'total_distance': total_distance,
        'total_smoothed_distance': total_smoothed_distance,
        'median_speed': median_speed,
        'smoothed_median_speed': smoothed_median_speed
    }
    return results


def getMouseDirectories():
    currentDirectory = os.getcwd()
    parentDirectory = os.path.join(currentDirectory, "EPM_data")
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
    i = 0
    aggregateResults = []
    for mouseDirectory in mouseDirectories:
        # i += 1
        # if i >= 15:
        #     break
        try:
            mouseResults = process_directory(parentDirectory, mouseDirectory)
            flatResults = flattenDict(mouseResults)
            aggregateResults.append(flatResults)
        except NoDataError, e:
            print('Exception: {}'.format(e))
    saveResultsAsJson(aggregateResults)
    saveResultsAsCSV(aggregateResults)
    print("Time to execute: {}".format(datetime.now() - startTime))


def saveResultsAsCSV(aggregateResults):
    pd.DataFrame(aggregateResults).to_csv('all_the_data1.csv')


def saveResultsAsJson(aggregateResults):
    with open('all_the_data1.json', 'w') as fp:
        json.dump(aggregateResults, fp)


def getMouseData(outer_directory):
    mouse_char = outer_directory.split('/')[0].split('_')
    return {
        'date': mouse_char[0],
        'time': mouse_char[1],
        'EPM': mouse_char[2],
        'strain': mouse_char[3],
        'mouseID': mouse_char[4],
        'sex': mouse_char[5] if len(mouse_char) > 5 else ''
    }


main()
