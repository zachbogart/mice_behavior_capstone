from EPM_analysis import *

# Supply folder to analyze and, optional, analysis start in seconds, analysis end in seconds.
import sys
import os

def process_folder(conditions_folder):
    start_frame = 0
    end_frame = None
    if len(sys.argv) > 2:
        start_frame = (sys.argv[
                           2]) * 30  # Substract 15 because Brant tracking starts tracking videos 15 seconds in. Multiply by 30 fps
        if len(sys.argv) > 3:
            end_frame = (sys.argv[
                             3]) * 30  # Substract 15 because Brant tracking starts tracking videos 15 seconds in. Multiply by 30 fps
    print >> sys.stderr, 'Loading data for', conditions_folder
    mice_ols, zones_masks, shape = load_data(conditions_folder, start_frame, end_frame)
    # Make a mask of all the EPM zones (those that don't start with 'F', for floor)
    print >> sys.stderr, 'Calculating residency in arms'
    EPMzones = reduce(lambda x, y: x + y, [zones_masks[k] for k in zones_masks if not k.startswith('F')])
    #
    mice_ols_in_EPMzones = filter_mice_ols(mice_ols)
    filled_in_mice_ols = fill_in_missing_outlines(mice_ols_in_EPMzones)
    zones_order, results_array = initialize_results_array(zones_masks, filled_in_mice_ols)
    mouse_position_in_zones(filled_in_mice_ols, shape, zones_order, zones_masks, results_array)
    frac_in_arms, tot_arm_entries, frames_in_arms, arm_entries = results(zones_order, results_array, start_frame,
                                                                         end_frame, conditions_folder)
    arcs_in_arms = calcPosition(shape, filled_in_mice_ols)
    print >> sys.stderr, 'Calculating distance travelled'
    distance, total_distance = calcDistance(arcs_in_arms)  # In pixels
    smoothed_distance = {k: smooth(np.array(v)) for k, v in distance.items()}
    total_smoothed_distance = {k: np.sum(v) for k, v in smoothed_distance.items()}
    # Speed in pixels per second
    print >> sys.stderr, 'Calculating speed'
    fps = 30.083  # Median frames per seconds of >2000 EPM videos
    median_speed = {k: np.median(v) * fps for k, v in distance.items()}
    smoothed_median_speed = {k: np.median(v) * fps for k, v in smoothed_distance.items()}
    print >> sys.stderr, 'Saving results'
    saveResults(conditions_folder, results_array, frac_in_arms, arm_entries, tot_arm_entries, frames_in_arms, \
                total_distance, total_smoothed_distance, median_speed, smoothed_median_speed)

current_dir = os.getcwd()
data_dir = os.path.join(current_dir, "Mice_Capstone_data_files")
for data_folder in os.walk(data_dir):
    print(data_folder)

# process_folder()