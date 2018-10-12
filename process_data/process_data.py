from EPM_analysis import *

import sys
import os
import json
import pandas as pd

class NoDataError(Exception):
    pass


def process_directory(parentDirectory, mouseDirectory):
    print('')
    print('Loading data for {}'.format(mouseDirectory))
    analysisDirectory = os.path.join(parentDirectory, mouseDirectory, 'analysis')
    if os.path.isdir(analysisDirectory):
        innerDirectories = os.listdir(analysisDirectory)
        if innerDirectories:
            innerDirectory = innerDirectories[0]  # TODO: How do we know which one to use?
            if os.path.isdir(os.path.join(analysisDirectory, innerDirectory)):
                conditions_folder_path = os.path.join(analysisDirectory, innerDirectory)
                if not os.path.isfile(os.path.join(conditions_folder_path, 'miceols.tar')):
                    raise NoDataError('No miceols.tar inside conditions directory for: {}'.format(mouseDirectory))
            else:
                raise NoDataError('No directories found inside analysis directory for: {}'.format(mouseDirectory))
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

    # print >> sys.stdout, 'Loading data for', conditions_folder
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
    results = {
        'outer_directory': mouseDirectory,
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
    parentDirectory = os.path.join(currentDirectory, "Mice_Capstone_data_files")
    mouseDirectories = os.listdir(parentDirectory)
    mouseDirectories.sort()
    return parentDirectory, mouseDirectories


def unnestDict(dictionary):
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
    parentDirectory, mouseDirectories = getMouseDirectories()
    i = 0
    aggregateResults = []
    for mouseDirectory in mouseDirectories:
        i += 1
        if i >= 15:
            break
        try:
            mouseResults = process_directory(parentDirectory, mouseDirectory)
            flatResults = unnestDict(mouseResults)
            aggregateResults.append(flatResults)
        except NoDataError, e:
            print('Exception: {}'.format(e))
    # with open('aggregate_results.json', 'w') as fp:
    #     #     json.dump(aggregateResults, fp)
    pd.DataFrame(aggregateResults).to_csv('aggregate_results.csv')


main()

# process_folder()
