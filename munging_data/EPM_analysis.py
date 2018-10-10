#!/usr/bin/env python
# Takes one main argument, the folder containing the conditions to analyze (e.g. '20121116_150819_EPM_BW_229/analysis/4975-7475_man-thresh-0.250/')
# and two optional arguments: a starting frame to analyze (or 'None') and an end frame (or 'None')
# Version control
# 0.0.2 2014-11-24
# Analyze distance travelled and speed
# 0.0.3 2016-01-11
# Fixed path issues in load_data and chnaged opening to 'with open'
# Fixed problems with calcPosition

import Util, tarfile, os, sys
import numpy as np
import cPickle as pickle
import vidtools
from pylab import *


def load_data(conditions_folder, start_frame=None, end_frame=None, ext='.zones.dict'):
    '''
    INPUT: This function takes a conditions_folder (string), e.g. '20130202_163641_EPM_BWPOF2_1403_F/analysis/900-7725_man-thresh-0.250',
    and an extension for the zones dictionary (string), which by default is '.zones.dict'
    OUTPUT: Returns miceols_list, a comprehensive list of mice outlines coordinates, frame by frame; 
    zones_masks, a dictionary with EPM zones masks; and
    shape, the shape of the video. 
    '''

    an_dir = os.path.abspath(conditions_folder)

    # Make a single list containing mice outlines in all tracked frames
    mice_ols = []
    tarh = tarfile.open(os.path.join(an_dir, 'miceols.tar'), 'r')
    for m in sorted(tarh.getmembers(), key=lambda x: x.name):
        ols = Util.tar2obj(m.name, tarh)
        mice_ols.extend(ols)

    # Retrieve shape data
    file_path = os.path.join(an_dir, 'SHAPE')
    with open(file_path, 'r') as f:
        shape = eval(f.read())

    # Retrieve zones data
    file_main_dir = os.path.dirname(os.path.dirname(an_dir))
    file_path = os.path.join(file_main_dir, os.path.basename(file_main_dir) + ext)
    with open(file_path, 'r') as f:
        zones = eval(f.read())

    # Covert zones to masks
    zones_masks = dict(
        [(k, vidtools.mask_from_outline(vidtools.outline_from_polygon(v), shape)) for k, v in zones.items()])

    # Set video boundaries
    if not end_frame:
        end_frame = start_frame + 9000  # Return only 9000 miceols (9000 frames which equal 5 min at 30 fps)
    return mice_ols[start_frame:end_frame], zones_masks, shape


def in_EPM(ol, EPMzones):
    '''
    Takes a frame with outlines and returns the outlines within the EPMzones
    '''
    y, x = zip(*ol)  # y,x because coordinates are reversed in the outlines tuples
    return (EPMzones[x, y]).any()


def filter_mice_ols(mice_ols, EPMzones):
    '''
    Takes a list of frames with outlines and filters it, returning only the frames in EPMzones
    '''
    mice_ols_in_EPMzones = []
    for ol in mice_ols:
        mice_ols_in_EPMzones.append(filter(in_EPM, (ol, EPMzones)))
    return mice_ols_in_EPMzones


def initialize_results_array(zones_masks, mice_ols):
    zones_order = zones_masks.keys()
    results_array = np.zeros((len(mice_ols), len(zones_order)), dtype=int)
    return zones_order, results_array


def mouse_position_in_zones(mice_ols, shape, zones_order, zones_masks, results_array):
    for i, m_ol in enumerate(mice_ols):
        mask = reduce(lambda x, y: x + y, [vidtools.mask_from_outline(p, shape) for p in m_ol + [[]]])
        for j, z in enumerate(zones_order):
            results_array[i, j] = mask[zones_masks[z]].sum()
        if i % 100 == 0:
            print >> sys.stderr, "\r%s" % i,


def arm_entry(results_array, zones_order):
    # Arm entries version from 20130829 10:10 am. Doesn't capture extra events anymore. Seems accurate now.
    arm_entries = [[], [], [], [], [], [], [], [], []]
    for frame in range(len(results_array) - 1):
        # Compare each arm, one by one to the middle 
        for zone in [4, 6, 7, 8]:  # 4,6,7,8 are the arms
            if results_array[
                frame, 5] > 0:  # 5 is the middle area. Arm antries can only occur if a mouse was present in the middle.
                # Arm entry: If in a given frame there is less of a mouse in an arm than in the middle and in the next frame most of the mouse is in an arm
                if results_array[frame, zone] <= results_array[frame, 5] and np.argmax(
                        results_array[frame + 1]) == zone:
                    arm_entries[zone].append(frame + 1)
    tot_arm_entries = dict(zip(zones_order, [len(arm) for arm in arm_entries]))
    return arm_entries, tot_arm_entries


def time_in_arms(results_array, zones_order):
    arm_residence = [0, 0, 0, 0, 0]
    for frame in range(len(results_array) - 1):
        for i, zone in enumerate(range(4, 9)):
            if np.argmax(results_array[
                             frame]) == zone:  # Mouse is counted to spend time in the area where most of it's outline is present.
                arm_residence[i] += 1
    arms = zones_order[4:9]
    return dict(zip(arms, arm_residence))


def fill_in_missing_outlines(
        mice_ols):  # fills in position of mouse that disappeared due to immobility based on where he was before he disapeared
    # Run this loop twice. On the first run, fill empty frames with the outline in the preceding frame.
    # Since that doesn't fill missing outlines at the beginning of the tracked video, reverse mice_ols and do again,
    # this time filling missing outlines with the outline in the following frame.
    # At the end, reverse mice_ols again.
    filled_in_mice_ols = mice_ols
    for h in [0, 1]:
        for i in range(len(filled_in_mice_ols)):
            if len(filled_in_mice_ols[i]) == 0:
                filled_in_mice_ols[i] = filled_in_mice_ols[i - 1]
        filled_in_mice_ols.reverse()
    return filled_in_mice_ols


def calcPosition(shape, filled_in_mice_ols, zones_order, results_array):
    '''
    Finds the position of the largest outline in each frame
    '''
    # Find centroids of filled_in_mice_ols
    centroids = []
    cm = vidtools.calc_coordMat(shape)
    for fr in filled_in_mice_ols:
        # If there are more than one outlines in a frame, chose the largest outline
        if len(fr) > 1:
            ol_size = [vidtools.size_of_polygon(ol) for ol in fr]
            largest_ol = fr[np.argmax(ol_size)]
            ol = largest_ol
        else:
            ol = fr[0]
        centroids.append(vidtools.centroid(ol, shape, cm))

    # Split centroids into arcs of consecutive presence in an arm
    # Make dictionary that will hold the position of the mice
    arcs_in_arms = {}
    for z in zones_order:
        arcs_in_arms[z] = [[]]
    for i, frame in enumerate(results_array):
        position = centroids[i]
        arm_thisFrame = zones_order[np.argmax(frame)]
        arm_lastFrame = None
        if arm_thisFrame == arm_lastFrame or arm_lastFrame == None:
            arcs_in_arms[arm_thisFrame][-1].append(position)
        else:
            arcs_in_arms[arm_thisFrame].append([])
            arcs_in_arms[arm_thisFrame][-1].append(position)
        arm_lastFrame = arm_thisFrame
    return arcs_in_arms


def calcDistance(arcs_in_arms):
    # Calculate distance frame by frame
    distance = {k: [] for k in arcs_in_arms.keys()}
    for arm, arcs in arcs_in_arms.items():
        for arc in arcs:
            for frame in range(len(arc) - 1):
                distance[arm].append(
                    np.sqrt(((arc[frame + 1][0] - arc[frame][0]) ** 2) + (arc[frame + 1][1] - arc[frame][1]) ** 2))
    total_distance = {k: np.sum(v) for k, v in distance.items()}
    return distance, total_distance


def smooth(x, window_len=10, window='flat'):
    '''
    from http://wiki.scipy.org/Cookbook/SignalSmooth
    smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal, unless x.size < window_len, in which case returns the raw data
    '''

    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."
    if x.size < window_len:
        return x
        # raise ValueError, "Input vector needs to be bigger than window size."
    if window_len < 3:
        return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
    s = np.r_[x[window_len - 1:0:-1], x, x[-1:-window_len:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')
    y = np.convolve(w / w.sum(), s, mode='valid')
    smoothed = y[(window_len / 2 - 1):-(window_len / 2)]
    return smoothed


def results(zones_order, results_array, start_frame, end_frame, conditions_folder):
    tot = float(sum([a.sum() for a in results_array[start_frame:end_frame].transpose()]))
    # pixels_in_zones can assign 'residence time' to multiple zones if mouse is in multiple zones at a time
    pixels_in_zones = dict(zip(zones_order, [a.sum() / tot for a in results_array[start_frame:end_frame].transpose()]))
    frac_in_openArms = pixels_in_zones['OB'] + pixels_in_zones['OT']
    frac_in_closedArms = pixels_in_zones['CL'] + pixels_in_zones['CR']
    frac_in_middle = pixels_in_zones['M']
    frac_in_closedAndMiddle = frac_in_closedArms + frac_in_middle
    frac_in_arms = {'frac_in_openArms': frac_in_openArms, 'frac_in_closedArms': frac_in_closedArms,
                    'frac_in_closedAndMiddle': frac_in_closedAndMiddle, 'frac_in_middle': frac_in_middle}
    arm_entries, tot_arm_entries = arm_entry(results_array, zones_order)
    # frames_in_arms assigns position of mouse to where the majority of his area is located.
    frames_in_arms = time_in_arms(results_array, zones_order)

    # Plot arm residence time and arm entries
    figure(1, figsize=(12, 8))
    plot(results_array[:, 4:9])
    if arm_entries[4]:
        scatter(arm_entries[4], 200 * ones(len(arm_entries[4])), c='b', s=40)
    if arm_entries[6]:
        scatter(arm_entries[6], 200 * ones(len(arm_entries[6])), c='r', s=40)
    if arm_entries[7]:
        scatter(arm_entries[7], 200 * ones(len(arm_entries[7])), c='c', s=40)
    if arm_entries[8]:
        scatter(arm_entries[8], 200 * ones(len(arm_entries[8])), c='m', s=40)
    legend(zones_order[4:9])
    title(conditions_folder, size=10)
    xlim(0, results_array.shape[0])
    ylim(0, np.max(results_array) * 1.05)
    # Set ticks every minute (1800 frames is 1 minute at 30 fps)
    xticks(range(1800, results_array.shape[0], 1800),
           [str(y) for y in range(1, int(floor(results_array.shape[0] / float(30 * 60))) + 1)])
    xlabel('minutes')
    ylabel('number of mouse pixels')
    # savefig
    savefig(conditions_folder + '/arm_residence_entries.pdf')
    close(1)

    return frac_in_arms, tot_arm_entries, frames_in_arms, arm_entries  # ,xplor_frac


def saveResults(conditions_folder, results_array, frac_in_arms, arm_entries, tot_arm_entries, frames_in_arms,
                total_distance, total_smoothed_distance, median_speed, smoothed_median_speed):
    np.save(os.path.join(conditions_folder, 'results_array.npy'), results_array)
    # np.save(conditions_folder+'/results_array.npy', results_array)

    results = {'frac_in_arms': frac_in_arms,
               'arm_entries': arm_entries,
               'tot_arm_entries': tot_arm_entries,
               'frames_in_arms': frames_in_arms,
               'total_distance': total_distance,
               'total_smoothed_distance': total_smoothed_distance,
               'median_speed': median_speed,
               'smoothed_median_speed': smoothed_median_speed
               }
    with open(os.path.join(conditions_folder, 'results.pkl'), 'w') as file_path:
        pickle.dump(results, file_path)
    # open(conditions_folder+'/fracInArms.dict', 'w').write(frac_in_arms.__repr__())
    # # Filter arm_entries so only contains arms
    # [tot_arm_entries.pop(key) for key in tot_arm_entries.keys() if 'F' in key or 'M' in key]
    # open(conditions_folder+'/armEntries.list', 'w').write(arm_entries.__repr__())
    # open(conditions_folder+'/totArmEntries.dict', 'w').write(tot_arm_entries.__repr__())
    # open(conditions_folder+'/framesInArms.dict', 'w').write(frames_in_arms.__repr__())
    # open(conditions_folder+'/distance.dict', 'w').write(total_distance.__repr__())
    # open(conditions_folder+'/smoothedDistance.dict', 'w').write(smoothed_distance.__repr__())
    # open(conditions_folder+'/medianSpeed.dict', 'w').write(median_speed.__repr__())
    # open(conditions_folder+'/smoothedMedianSpeed.dict', 'w').write(smoothed_median_speed.__repr__())

# if __name__ == "__main__":
#     # Supply folder to analyze and, optional, analysis start in seconds, analysis end in seconds.
#     conditions_folder = sys.argv[1]
#     start_frame = 0
#     end_frame = None
#     if len(sys.argv)>2:
#         start_frame = (sys.argv[2])*30 #Substract 15 because Brant tracking starts tracking videos 15 seconds in. Multiply by 30 fps
#         if len(sys.argv)>3:
#             end_frame = (sys.argv[3])*30 #Substract 15 because Brant tracking starts tracking videos 15 seconds in. Multiply by 30 fps
#     print >> sys.stderr, 'Loading data for', conditions_folder
#     mice_ols,zones_masks,shape = load_data(conditions_folder,start_frame,end_frame)
#     # Make a mask of all the EPM zones (those that don't start with 'F', for floor)
#     print >> sys.stderr, 'Calculating residency in arms'
#     EPMzones = reduce(lambda x,y: x+y, [zones_masks[k] for k in zones_masks if not k.startswith('F')])
#     #
#     mice_ols_in_EPMzones = filter_mice_ols(mice_ols)
#     filled_in_mice_ols = fill_in_missing_outlines(mice_ols_in_EPMzones)
#     zones_order, results_array = initialize_results_array(zones_masks,filled_in_mice_ols)
#     mouse_position_in_zones(filled_in_mice_ols,shape,zones_order,zones_masks,results_array)
#     frac_in_arms,tot_arm_entries,frames_in_arms,arm_entries = results(zones_order,results_array,start_frame,end_frame,conditions_folder)
#     arcs_in_arms = calcPosition(shape,filled_in_mice_ols)
#
#     print >> sys.stderr, 'Calculating distance travelled'
#     distance,total_distance = calcDistance(arcs_in_arms) # In pixels
#     smoothed_distance = {k:smooth(np.array(v)) for k,v in distance.items()}
#     total_smoothed_distance = {k:np.sum(v) for k,v in smoothed_distance.items()}
#
#     # Speed in pixels per second
#
#     print >> sys.stderr, 'Calculating speed'
#     fps = 30.083 # Median frames per seconds of >2000 EPM videos
#     median_speed = {k:np.median(v)*fps for k,v in distance.items()}
#     smoothed_median_speed = {k:np.median(v)*fps for k,v in smoothed_distance.items()}
#
#     print >> sys.stderr, 'Saving results'
#     saveResults(conditions_folder,results_array,frac_in_arms,arm_entries,tot_arm_entries,frames_in_arms, \
#         total_distance,total_smoothed_distance,median_speed,smoothed_median_speed)
