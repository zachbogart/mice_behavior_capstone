from collections import defaultdict

import Util, tarfile, os, sys
import numpy as np
import cPickle as pickle
import vidtools
from pylab import *
import pandas as pd

GOOD_ARMS = {'CR', 'CL', 'OT', 'OB', 'M'}
CLOSED_ARMS = {'CR', 'CL'}
OPEN_ARMS = {'OT', 'OB'}
ARM_ORDER = ['F1', 'F2', 'F3', 'F4', 'CL', 'M', 'OB', 'CR', 'OT']

TESTING_RUN__SAVE_GRAPHS = False  # If this is set to true, we will save histograms for analysis
ZONES = None
MOUSE_DIRECTORY = None

# Speed in pixels per second
# Median frames per seconds of >2000 EPM videos
FPS = 30.083


def turnOnHistograms():
    global TESTING_RUN__SAVE_GRAPHS
    TESTING_RUN__SAVE_GRAPHS = True


def setMouseDirectory(mouseDirectory):
    global MOUSE_DIRECTORY
    MOUSE_DIRECTORY = mouseDirectory


def load_data(conditions_folder_path, start_frame=None, end_frame=None, ext='.zones.dict'):
    """
    INPUT: This function takes a conditions_folder (string), e.g. '20130202_163641_EPM_BWPOF2_1403_F/analysis/900-7725_man-thresh-0.250',
    and an extension for the zones dictionary (string), which by default is '.zones.dict'
    OUTPUT: Returns miceols_list, a comprehensive list of mice outlines coordinates, frame by frame;
    zones_masks, a dictionary with EPM zones masks; and
    shape, the shape of the video.
    """

    an_dir = conditions_folder_path

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

    # Set ZONES as a global variable for use later
    global ZONES
    ZONES = zones

    # Set video boundaries
    if not end_frame:
        end_frame = start_frame + 9000  # Return only 9000 miceols (9000 frames which equal 5 min at 30 fps)
    return mice_ols[start_frame:end_frame], zones_masks, shape


def filterBoundariesByZone(mice_ols, zone_masks):
    """
    Takes a list of frames with outlines and filters it, returning only the frames in EPMZones
    """

    def in_EPM(ol):
        '''
        Takes a frame with outlines and returns the outlines within the EPMZones
        '''
        y, x = zip(*ol)  # y,x because coordinates are reversed in the outlines tuples
        return (EPMZones[x, y]).any()

    EPMZones = calculateEPMZones(zone_masks)
    boundariesInEPMZones = []
    for ol in mice_ols:
        boundariesInEPMZones.append(filter(in_EPM, ol))
    return boundariesInEPMZones


def cleanBoundaries(boundaries, shape, zones_masks):
    boundariesInEPMZones = filterBoundariesByZone(boundaries, zones_masks)
    boundariesClean, results_array, zones_order = fillGapsInBoundaries(boundariesInEPMZones, shape, zones_masks)
    return boundariesClean, results_array, zones_order


def fillGapsInBoundaries(boundariesInEPMZones, shape, zones_masks):
    # Fill in missing outlines twice, once running forwards and once backwards
    boundariesFilledIn = fill_in_missing_outlines(boundariesInEPMZones)
    zones_order, results_array = initialize_results_array(zones_masks, boundariesFilledIn)
    mouse_position_in_zones(boundariesFilledIn, shape, zones_order, zones_masks, results_array)
    return boundariesFilledIn, results_array, zones_order


def calculateEPMZones(zones_masks):
    # Make a mask of all the EPM zones (those that don't start with 'F', for floor)
    EPMzones = reduce(lambda x, y: x + y, [zones_masks[k] for k in zones_masks if not k.startswith('F')])
    return EPMzones


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
            print >> sys.stdout, "\r%s" % i,


def arm_entry(results_array, zones_order):
    # Arm entries version from 20130829 10:10 am. Doesn't capture extra events anymore. Seems accurate now.
    arm_entries = [[], [], [], [], [], [], [], [], []]
    for frame in range(len(results_array) - 1):
        # Compare each arm, one by one to the middle 
        for zone in [4, 6, 7, 8]:  # 4,6,7,8 are the arms
            if results_array[frame, 5] > 0:
                # 5 is the middle area. Arm antries can only occur if a mouse was present in the middle.
                # Arm entry: If in a given frame there is less of a mouse in an arm than in the middle and in the next
                # frame most of the mouse is in an arm
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


def fill_in_missing_outlines(mice_ols):
    # fills in position of mouse that disappeared due to immobility based on where he was before he disapeared
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


def calculateCentroidsByArm(centroids, zones_order, results_array):
    """
    Finds the position of the largest outline in each frame
    """

    # Split centroids into arcs of consecutive presence in an arm
    # Make dictionary that will hold the position of the mice

    centroidsByArm = {}
    arm_lastFrame = None
    for z in zones_order:
        centroidsByArm[z] = [[]]
    for i, frame in enumerate(results_array):
        position = centroids[i]
        arm_thisFrame = zones_order[np.argmax(frame)]
        arm_lastFrame = None
        if arm_thisFrame == arm_lastFrame or arm_lastFrame is None:
            centroidsByArm[arm_thisFrame][-1].append(position)
        else:
            centroidsByArm[arm_thisFrame].append([])
            centroidsByArm[arm_thisFrame][-1].append(position)
        arm_lastFrame = arm_thisFrame
    return centroidsByArm


def calculateCentroids(filled_in_mice_ols, shape):
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
            if fr:
                ol = fr[0]
            else:
                ol = []
        centroids.append(vidtools.centroid(ol, shape, cm))

    # centroidDF = pd.DataFrame(centroids)
    # centroidDF.to_csv('centroids.csv')
    return centroids


def calculateVelocitySummaryStatistics(distances):
    """
    Given distances, compute all summary statistics that we want from both that list and
    a list filtered for only times when the mouse is not 'at rest'
    :param distances: a list of distances (or speeds) over time
    """
    distancesActive = [distance for distance in distances if not isResting(distance)]
    return {
        'median_speed': np.median(distances) * FPS,
        'average_speed': np.average(distances) * FPS,
        'median_speed_active': np.median(distancesActive) * FPS,
        'average_speed_active': np.average(distancesActive) * FPS,
    }


def calculateVelocityFeatures(distancesPerArm, directionsPerArm):
    velocityFeatures = {}
    for arm in GOOD_ARMS:
        velocityFeatures[arm] = {}

    # EX: distancesClosedTowardsMiddle means a list of distances while in a closed arm and
    #     moving towards the middle arm
    distancesClosedTowardsMiddle = []
    distancesClosedTowardsOutside = []
    distancesOpenTowardsMiddle = []
    distancesOpenTowardsOutside = []

    # Map from an arm and a cardinal direction to one of our distance lists above
    armCardinalCombinationMap = {
        'CLright': distancesClosedTowardsMiddle,
        'CLleft': distancesClosedTowardsOutside,
        'CRright': distancesClosedTowardsOutside,
        'CRleft': distancesClosedTowardsMiddle,
        'OTup': distancesOpenTowardsOutside,
        'OTdown': distancesOpenTowardsMiddle,
        'OBup': distancesOpenTowardsMiddle,
        'OBdown': distancesOpenTowardsOutside,
    }

    for arm in GOOD_ARMS:
        distances = distancesPerArm[arm]
        directionVectors = directionsPerArm[arm]

        # Find the basic statistics for each arm
        velocityFeatures[arm]['total'] = calculateVelocitySummaryStatistics(distances)

        cardinalDirections = ['right', 'left', 'up', 'down']
        for cardinal in cardinalDirections:
            distancesCardinal = [
                distance for distance, directionVector
                in zip(distances, directionVectors)
                if isMovingInCardinalDirection(cardinal, directionVector)
            ]
            # Find summary statistics for each cardinal direction
            velocityFeatures[arm][cardinal] = calculateVelocitySummaryStatistics(distancesCardinal)

            # Add distances to one of the lists above based on armCardinalCombinationMap
            armAndCardinal = arm + cardinal
            if armAndCardinal in armCardinalCombinationMap:
                armCardinalCombinationMap[armAndCardinal].extend(distancesCardinal)

    # Find summary statistics for each of the following options
    featureToList = {
        'closed_towards_middle': distancesClosedTowardsMiddle,
        'closed_towards_outside': distancesClosedTowardsOutside,
        'open_towards_middle': distancesOpenTowardsMiddle,
        'open_towards_outside': distancesOpenTowardsOutside,
    }
    for featureName, distances in featureToList.items():
        velocityFeatures[featureName] = calculateVelocitySummaryStatistics(distances)

    return velocityFeatures


def isMovingInCardinalDirection(cardinalDirection, directionVector):
    if cardinalDirection == 'up':
        return directionVector[1] > 0
    elif cardinalDirection == 'down':
        return directionVector[1] < 0
    elif cardinalDirection == 'right':
        return directionVector[0] > 0
    elif cardinalDirection == 'left':
        return directionVector[0] < 0


def calculateSafetyFeatures(centroidsByArm, mouseLength):
    timeSafetyTotal = 0
    timeTotal = 0
    safetyFeatures = {}
    for arm in CLOSED_ARMS:
        centroids = centroidsByArm[arm][0]
        timeSafetyArm = reduce(lambda count, centroid: count + isSafe(arm, centroid, mouseLength), centroids, 0)
        timeArm = len(centroids)

        if timeArm:
            safetyFeatures[arm] = timeSafetyArm / float(timeArm)
            timeSafetyTotal += timeSafetyArm
            timeTotal += timeArm
        else:  # Mouse never in this arm
            safetyFeatures[arm] = None

        # For analysis we graph how far the mouse is from the end. We want to find the
        # cutoff for the mouse to be in safety
        distancesFromEndOfArm = map(lambda centroid: distanceFromEndOfArm(arm, centroid), centroids)
        makeHistogram(distancesFromEndOfArm, title='Distances From Safety in arm {}'.format(arm),
                      verticalLines=[mouseLength], percentiles=[])

    if timeTotal != 0:
        safetyFeatures['closed_arms'] = timeSafetyTotal / float(timeTotal)
    return safetyFeatures


def distanceFromEndOfArm(arm, point):
    distanceFromEnd = 0
    if arm == 'CL':
        leftEdgeOfArm = ZONES['CL'][0][0]
        distanceFromEnd = point[0] - leftEdgeOfArm
    elif arm == 'CR':
        rightEdgeOfArm = ZONES['CR'][1][0]
        distanceFromEnd = rightEdgeOfArm - point[0]
    return max(distanceFromEnd, 0)


def isSafe(arm, mouseLocation, mouseLength):
    if arm == 'CL':
        leftEdgeOfGoodZone = ZONES['CL'][0][0]
        return mouseLocation[0] < leftEdgeOfGoodZone + mouseLength
    elif arm == 'CR':
        rightEdgeOfGoodZone = ZONES['CR'][1][0]
        return mouseLocation[0] > rightEdgeOfGoodZone - mouseLength
    return False


def calculateRestFeatures(distancesPerArm):
    timeRestTotalOpen = 0
    timeTotalOpen = 0
    timeRestTotalClosed = 0
    timeTotalClosed = 0
    timeRestTotalMiddle = 0
    timeTotalMiddle = 0

    restFeatures = {}
    for arm in GOOD_ARMS:
        distances = distancesPerArm[arm]
        timeRestArm = reduce(lambda count, distance: count + isResting(distance), distances, 0)
        timeArm = len(distances)
        if timeArm:
            restFeatures[arm] = timeRestArm / float(timeArm)
            if arm in OPEN_ARMS:
                timeRestTotalOpen += timeRestArm
                timeTotalOpen += timeArm
            elif arm in CLOSED_ARMS:
                timeRestTotalClosed += timeRestArm
                timeTotalClosed += timeArm
            else:
                timeRestTotalMiddle += timeRestArm
                timeTotalMiddle += timeArm
        else:  # Mouse never in this arm
            restFeatures[arm] = None

        # For analysis we graph the speeds of the mouse to see where cutoff should be for 'at rest'
        makeHistogram(distances, title='Speeds in arm {}'.format(arm), percentiles=[10, 25, 50])

    if timeTotalOpen != 0:
        restFeatures['open_arms'] = timeRestTotalOpen / float(timeTotalOpen)
    if timeTotalClosed != 0:
        restFeatures['closed_arms'] = timeRestTotalClosed / float(timeTotalClosed)
    totalTime = timeTotalClosed + timeTotalOpen + timeTotalMiddle
    if totalTime != 0:
        restFeatures['all_arms'] = (timeRestTotalOpen + timeRestTotalClosed + timeRestTotalMiddle) / \
                                   float(totalTime)
    return restFeatures


def isResting(speed):
    threshold = 1
    return speed < threshold


def isPeeking(arm, mouseLocation, mouseLength):
    if arm == 'CL':
        rightEndOfArm = ZONES['CL'][1][0]
        return mouseLocation[0] > rightEndOfArm - mouseLength
    elif arm == 'CR':
        leftEndOfArm = ZONES['CR'][0][0]
        return mouseLocation[0] < leftEndOfArm + mouseLength
    elif arm == 'M':
        return True
    return False


def isRestingAndPeeking(speed, arm, mouseLocation, mouseLength):
    safe = isPeeking(arm, mouseLocation, mouseLength)
    resting = isResting(speed)
    return safe and resting


def calculatePeekingFeatures(centroidsByArm, distancesPerArm, mouseLength):
    timePeekingTotal = 0
    timeTotal = 0
    peekLengthsTotal = []
    peekingFeatures = {}
    for arm in CLOSED_ARMS:
        centroids = centroidsByArm[arm][0]
        centroids = centroids[:-1]  # Remove last centroid that doesn't correspond to a distance
        distances = distancesPerArm[arm]

        minFramesBetweenPeeks = 10

        numPreviousPeeking = 0
        numPreviousNotPeeking = minFramesBetweenPeeks + 1
        peekLengths = []

        for frame, (centroid, distance) in enumerate(zip(centroids, distances)):
            if isRestingAndPeeking(distance, arm, centroid, mouseLength):
                if numPreviousNotPeeking <= minFramesBetweenPeeks and numPreviousNotPeeking and peekLengths:
                    # This hits if the last peek was not far enough away, so we continue that peek
                    numPreviousPeeking = peekLengths.pop() + numPreviousNotPeeking
                numPreviousNotPeeking = 0
                numPreviousPeeking += 1
            else:
                if numPreviousPeeking:
                    peekLengths.append(numPreviousPeeking)
                numPreviousNotPeeking += 1
                numPreviousPeeking = 0
        if numPreviousPeeking:
            peekLengths.append(numPreviousPeeking)

        minPeekLength = 5
        peekLengths = filter(lambda peekLength: peekLength > minPeekLength, peekLengths)

        timePeekingArm = np.sum(peekLengths)
        timeArm = len(centroids)
        if timeArm:
            peekingFeatures['count_{}'.format(arm)] = len(peekLengths)
            peekingFeatures['fraction_{}'.format(arm)] = timePeekingArm / float(timeArm)
            peekingFeatures['average_length_{}'.format(arm)] = np.average(peekLengths) * FPS
            peekingFeatures['median_length_{}'.format(arm)] = np.median(peekLengths) * FPS
            timePeekingTotal += timePeekingArm
            timeTotal += timeArm
            peekLengthsTotal.extend(peekLengths)
        else:  # Mouse never in this arm
            peekingFeatures['fraction_{}'.format(arm)] = None

    peekingFeatures['count_total'] = len(peekLengthsTotal)
    if timeTotal != 0:
        peekingFeatures['fraction_total'] = timePeekingTotal / float(timeTotal)
    peekingFeatures['average_length_total'] = np.average(peekLengthsTotal) * FPS
    peekingFeatures['median_length_total'] = np.median(peekLengthsTotal) * FPS

    return peekingFeatures


def isRestingAndSafe(speed, arm, mouseLocation, mouseLength):
    safe = isSafe(arm, mouseLocation, mouseLength)
    resting = isResting(speed)
    return safe and resting


def calculateSafeAndRestingFeatures(centroidsByArm, distancesPerArm, mouseLength):
    timeSafeAndRestTotal = 0
    timeTotal = 0
    safeAndRestingFeatures = {}
    for arm in CLOSED_ARMS:
        centroids = centroidsByArm[arm][0]
        centroids = centroids[:-1]  # Remove last centroid that doesn't correspond to a distance
        distances = distancesPerArm[arm]

        timeSafeAndRestArm = reduce(
            lambda count, item: count + isRestingAndSafe(item[1], arm, item[0], mouseLength),
            zip(centroids, distances),
            0
        )
        timeArm = len(centroids)
        if timeArm:
            safeAndRestingFeatures[arm] = timeSafeAndRestArm / float(timeArm)
            timeSafeAndRestTotal += timeSafeAndRestArm
            timeTotal += timeArm
        else:  # Mouse never in this arm
            safeAndRestingFeatures[arm] = 0

    if timeTotal != 0:
        safeAndRestingFeatures['closed_arms'] = timeSafeAndRestTotal / float(timeTotal)
    return safeAndRestingFeatures


def testDistance(centroids):
    if not TESTING_RUN__SAVE_GRAPHS:
        return

    distances = []
    directions = []

    for frame in range(len(centroids) - 1):
        point1 = centroids[frame]
        point2 = centroids[frame + 1]
        vectorLength, vectorDirection = calculateVectorStatistics(point1, point2)

        distances.append(vectorLength)
        directions.append(vectorDirection)

    distances = np.array(distances)
    distancesSmoothed = smooth(np.array(distances))
    lineListWithLabels = [(distancesSmoothed, "Smoothed Speed")]
    makeLineGraph(lineListWithLabels, title="Speed Over Time", xLabel="Seconds", yLabel="Speed")
    # print('done')


def calculateDistanceFeatures(centroidsByArm):
    distancesPerArm, directionsPerArm = calculateDistancesByArm(centroidsByArm)  # In pixels
    # totalDistancePerArm = calculateTotalDistancePerArm(distancesPerArm)

    distancesPerArmSmoothed = smoothDistancesByArm(distancesPerArm)
    totalDistancePerArmSmoothed = calculateTotalDistancePerArm(distancesPerArmSmoothed)

    return distancesPerArmSmoothed, directionsPerArm, totalDistancePerArmSmoothed


def smoothDistancesByArm(distancesPerArm):
    return {arm: smooth(np.array(distances)) for arm, distances in distancesPerArm.items()}


def calculateTotalDistancePerArm(distancesPerArm):
    totalDistancePerArm = {k: np.sum(v) for k, v in distancesPerArm.items()}

    closeArmsTotal = 0
    for arm in CLOSED_ARMS:
        closeArmsTotal += totalDistancePerArm[arm]
    totalDistancePerArm["closed"] = closeArmsTotal

    openArmsTotal = 0
    for arm in OPEN_ARMS:
        openArmsTotal += totalDistancePerArm[arm]
    totalDistancePerArm["open"] = openArmsTotal

    return totalDistancePerArm


def calculateDistancesByArm(centroidsByArm):
    distancesPerArm = {k: [] for k in centroidsByArm.keys()}
    directionsPerArm = {k: [] for k in centroidsByArm.keys()}
    for arm, arcs in centroidsByArm.items():
        for arc in arcs:
            for frame in range(len(arc) - 1):
                point1 = arc[frame]
                point2 = arc[frame + 1]
                if point1 and point2:
                    vectorLength, vectorDirection = calculateVectorStatistics(point1, point2)

                    distancesPerArm[arm].append(vectorLength)
                    directionsPerArm[arm].append(vectorDirection)
    return distancesPerArm, directionsPerArm


def calculateVectorStatistics(point1, point2):
    vectorLength = np.sqrt(((point2[0] - point1[0]) ** 2) + (point2[1] - point1[1]) ** 2)
    directionVector = (point2[0] - point1[0], point2[1] - point1[1])
    directionVectorNormalized = [coordinate / vectorLength for coordinate in directionVector]
    return vectorLength, directionVectorNormalized


def smooth(x, window_len=10, window='flat'):
    """
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
    """

    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."
    if x.size < window_len:
        return x
        # raise ValueError, "Input vector needs to be bigger than window size."
    if window_len < 3:
        return x
    if window not in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
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


def calculateArmEntries(zones_order, results_array, start_frame, end_frame, conditions_folder):
    # pixelsInZones can assign 'residence time' to multiple zones if mouse is in multiple zones at a time
    pixelsInZones = dict(zip(zones_order, [a.sum() for a in results_array[start_frame:end_frame].transpose()]))

    total = 0.
    for arm in GOOD_ARMS:
        total += pixelsInZones[arm]

    fractionInZones = {}
    for arm in GOOD_ARMS:
        fractionInZones[arm] = pixelsInZones[arm] / total

    fraction_in_openArms = fractionInZones['OB'] + fractionInZones['OT']
    fraction_in_closedArms = fractionInZones['CL'] + fractionInZones['CR']
    fraction_in_middle = fractionInZones['M']
    fraction_in_closedAndMiddle = fraction_in_closedArms + fraction_in_middle
    fraction_in_left = fractionInZones['CL']
    fraction_in_right = fractionInZones['CR']
    fraction_in_top = fractionInZones['OT']
    fraction_in_bottom = fractionInZones['OB']
    fraction_in_arms = {
        'open-arms': fraction_in_openArms,
        'closed': fraction_in_closedArms,
        'closed_and_middle': fraction_in_closedAndMiddle,
        'middle': fraction_in_middle,
        'left': fraction_in_left,
        'right': fraction_in_right,
        'top': fraction_in_top,
        'bottom': fraction_in_bottom,
    }
    arm_entries, tot_arm_entries = arm_entry(results_array, zones_order)
    # frames_in_arms assigns position of mouse to where the majority of his area is located.
    frames_in_arms = time_in_arms(results_array, zones_order)

    # Plot arm residence time and arm entries
    # figure(1, figsize=(12, 8))
    # plot(results_array[:, 4:9])
    # if arm_entries[4]:
    #     scatter(arm_entries[4], 200 * ones(len(arm_entries[4])), c='b', s=40)
    # if arm_entries[6]:
    #     scatter(arm_entries[6], 200 * ones(len(arm_entries[6])), c='r', s=40)
    # if arm_entries[7]:
    #     scatter(arm_entries[7], 200 * ones(len(arm_entries[7])), c='c', s=40)
    # if arm_entries[8]:
    #     scatter(arm_entries[8], 200 * ones(len(arm_entries[8])), c='m', s=40)
    # legend(zones_order[4:9])
    # title(conditions_folder, size=10)
    # xlim(0, results_array.shape[0])
    # ylim(0, np.max(results_array) * 1.05)
    # # Set ticks every minute (1800 frames is 1 minute at 30 fps)
    # xticks(range(1800, results_array.shape[0], 1800),
    #        [str(y) for y in range(1, int(floor(results_array.shape[0] / float(30 * 60))) + 1)])
    # xlabel('minutes')
    # ylabel('number of mouse pixels')
    # savefig
    # savefig(conditions_folder + '/arm_residence_entries.pdf')
    # close(1)

    return fraction_in_arms, tot_arm_entries, frames_in_arms, arm_entries  # ,xplor_frac


def calculateTurningPreference(arm_entries):
    entries = sorted([entry for zones in arm_entries for entry in zones])
    arm_entries_tuple = [(entry, zone) for zone, subentries in enumerate(arm_entries) for entry in subentries]
    entries_dict = dict(arm_entries_tuple)

    # Collect number of turns in each category
    num_left_into_open = 0
    num_right_into_open = 0
    num_straight_into_open = 0
    num_back_into_open = 0
    num_left_into_closed = 0
    num_right_into_closed = 0
    num_straight_into_closed = 0
    num_back_into_closed = 0

    turn_left_into_open = [(4, 8), (7, 6)]
    turn_right_into_open = [(7, 8), (4, 6)]
    turn_straight_into_open = [(6, 8), (8, 6)]
    turn_back_into_open = [(6, 6), (8, 8)]
    turn_left_into_closed = [(8, 7), (6, 4)]
    turn_right_into_closed = [(8, 4), (6, 7)]
    turn_straight_into_closed = [(4, 7), (7, 4)]
    turn_back_into_closed = [(4, 4), (7, 7)]

    for i in range(len(entries) - 1):
        turn = (entries_dict[entries[i]], entries_dict[entries[i + 1]])
        if turn in turn_left_into_open:
            num_left_into_open += 1
        elif turn in turn_right_into_open:
            num_right_into_open += 1
        elif turn in turn_straight_into_open:
            num_straight_into_open += 1
        elif turn in turn_back_into_open:
            num_back_into_open += 1
        elif turn in turn_left_into_closed:
            num_left_into_closed += 1
        elif turn in turn_right_into_closed:
            num_right_into_closed += 1
        elif turn in turn_straight_into_closed:
            num_straight_into_closed += 1
        elif turn in turn_back_into_closed:
            num_back_into_closed += 1

    num_left = num_left_into_open + num_left_into_closed
    num_right = num_right_into_open + num_right_into_closed
    num_straight = num_straight_into_open + num_straight_into_closed
    num_back = num_back_into_open + num_back_into_closed

    num_total = num_right + num_back + num_left + num_straight
    num_total_into_open = num_left_into_open + num_right_into_open + num_straight_into_open + num_back_into_open
    num_total_into_closed = num_left_into_closed + num_right_into_closed + num_straight_into_closed + num_back_into_closed

    turningFeatures = {}
    # 'num_left': num_left,
    # 'num_right': num_right,
    # 'num_straight': num_straight,
    # 'num_back': num_back,
    turningFeatures['num_total'] = num_total
    turningFeatures['num_total_into_open'] = num_total_into_open
    turningFeatures['num_total_into_closed'] = num_total_into_closed
    if num_total != 0:
        turningFeatures['fraction_right'] = float(num_right) / num_total
        turningFeatures['fraction_left'] = float(num_left) / num_total
        turningFeatures['fraction_straight'] = float(num_straight) / num_total
        turningFeatures['fraction_back'] = float(num_back) / num_total
    if (num_right + num_left) != 0:
        turningFeatures['fraction_right_only_right_left'] = float(num_right) / (num_right + num_left)
        turningFeatures['fraction_left_only_right_left'] = float(num_left) / (num_right + num_left)
    if (num_straight + num_back) != 0:
        turningFeatures['fraction_straight_only_straight_back'] = float(num_straight) / (num_straight + num_back)
        turningFeatures['fraction_back_only_straight_back'] = float(num_back) / (num_straight + num_back)
    if num_total_into_open != 0:
        turningFeatures['fraction_right_into_open'] = float(num_right_into_open) / num_total_into_open
        turningFeatures['fraction_left_into_open'] = float(num_left_into_open) / num_total_into_open
        turningFeatures['fraction_straight_into_open'] = float(num_straight_into_open) / num_total_into_open
        turningFeatures['fraction_back_into_open'] = float(num_back_into_open) / num_total_into_open
    if num_total_into_closed != 0:
        turningFeatures['fraction_right_into_closed'] = float(num_right_into_closed) / num_total_into_closed
        turningFeatures['fraction_left_into_closed'] = float(num_left_into_closed) / num_total_into_closed
        turningFeatures['fraction_straight_into_closed'] = float(num_straight_into_closed) / num_total_into_closed
        turningFeatures['fraction_back_into_closed'] = float(num_back_into_closed) / num_total_into_closed
    if (num_right_into_open + num_left_into_open) != 0:
        turningFeatures['fraction_right_only_right_left_into_open'] = float(num_right_into_open) / (
                num_right_into_open + num_left_into_open)
        turningFeatures['fraction_left_only_right_left_into_open'] = float(num_right_into_open) / (
                num_right_into_open + num_left_into_open)
    if (num_straight_into_open + num_back_into_open) != 0:
        turningFeatures['fraction_straight_only_straight_back_into_open'] = float(num_straight_into_open) / (
                num_straight_into_open + num_back_into_open)
        turningFeatures['fraction_back_only_straight_back_into_open'] = float(num_back_into_open) / (
                num_straight_into_open + num_back_into_open)
    if (num_right_into_closed + num_left_into_closed) != 0:
        turningFeatures['fraction_right_only_right_left_into_closed'] = float(num_right_into_closed) / (
                num_right_into_closed + num_left_into_closed)
        turningFeatures['fraction_left_only_right_left_into_closed'] = float(num_left_into_closed) / (
                num_right_into_closed + num_left_into_closed)
    if (num_straight_into_closed + num_back_into_closed) != 0:
        turningFeatures['fraction_straight_only_straight_back_into_closed'] = float(num_straight_into_closed) / (
                num_straight_into_closed + num_back_into_closed)
        turningFeatures['fraction_back_only_straight_back_into_closed'] = float(num_back_into_closed) / (
                num_straight_into_closed + num_back_into_closed)

    return turningFeatures


def calculateBacktrackCounts(arm_entries, threshold=150):
    # if a mouse leaves an arm then goes back without entering any other arm in
    # a threshold of frames(e.g., threshold = 150(5 seconds))
    # then count it as one time of peeking
    times = dict()
    arms = [4, 6, 7, 8]
    for arm in arms:
        times[arm] = 0

    entries = sorted([entry for zones in arm_entries for entry in zones])
    arm_entries_tuple = [(entry, zone) for zone, subentries in enumerate(arm_entries) for entry in subentries]
    entries_dict = dict(arm_entries_tuple)

    for k in range(len(entries) - 1):
        if entries[k + 1] - entries[k] < threshold and entries_dict[entries[k]] == entries_dict[entries[k + 1]]:
            times[entries_dict[entries[k]]] += 1

    numBacktracksByArm = {}
    totalPeeksOpen = 0
    totalPeeksClosed = 0
    for arm in arms:
        armName = ARM_ORDER[arm]
        numBacktracksByArm[armName] = times[arm]
        if armName in OPEN_ARMS:
            totalPeeksOpen += times[arm]
        elif armName in CLOSED_ARMS:
            totalPeeksClosed += times[arm]
    numBacktracksByArm['total_open'] = totalPeeksOpen
    numBacktracksByArm['total_closed'] = totalPeeksClosed
    numBacktracksByArm['total'] = totalPeeksOpen + totalPeeksClosed

    return numBacktracksByArm


def RegionToRegionFreq(arm_entries):
    # get the freq of a mouse move from one region to another.
    freq = defaultdict(int)
    entries = sorted([entry for zones in arm_entries for entry in zones])
    arm_entries_tuple = [(entry, zone) for zone, subentries in enumerate(arm_entries) for entry in subentries]
    entries_dict = dict(arm_entries_tuple)

    arms = [4, 6, 7, 8]
    for i in range(len(arms) - 1):
        for j in range(i, len(arms)):
            for k in range(len(entries) - 1):
                if entries_dict[entries[k]] == arms[i] and entries_dict[entries[k + 1]] == arms[j]:
                    freq[(arms[i], arms[j])] += 1

    return dict(freq)


def calculateMouseLength(boundaries):
    mouseLengthOverTime = []
    for boundaryPoints in boundaries:
        if boundaryPoints:
            boundaryPoints = boundaryPoints[0]
            xMax, xMin, yMax, yMin = findBoundaryBox(boundaryPoints)
            mouseWidth = xMax - xMin
            mouseHeight = yMax - yMin
            mouseLength = max(mouseWidth, mouseHeight)
            mouseLengthOverTime.append(mouseLength)

    # This is for analysis to figure out which percentile to use
    makeHistogram(mouseLengthOverTime, 'MouseLength', percentiles=[50, 75, 90])

    # We estimate the length of the mouse to be the 90th percentile of measurements we took
    if not mouseLengthOverTime:
        return 0
    mouseLength = np.percentile(mouseLengthOverTime, 90)
    return mouseLength


def findBoundaryBox(boundaryPoints):
    xMin = boundaryPoints[0][0]
    xMax = boundaryPoints[0][0]
    yMin = boundaryPoints[0][1]
    yMax = boundaryPoints[0][1]
    for point in boundaryPoints:
        x = point[0]
        y = point[1]
        if x < xMin:
            xMin = x
        elif x > xMax:
            xMax = x
        if y < yMin:
            yMin = y
        elif y > yMax:
            yMax = y
    return xMax, xMin, yMax, yMin


def extractMouseFeatures(outer_directory):
    mouse_char = outer_directory.split('/')[0].split('_')
    return {
        'date': mouse_char[0],
        'time': mouse_char[1],
        'EPM': mouse_char[2],
        'strain': mouse_char[3],
        'mouseID': mouse_char[4],
        'sex': mouse_char[5] if len(mouse_char) > 5 else ''
    }


def makeHistogram(dataList, title='Hist', verticalLines=[], percentiles=[]):
    if not TESTING_RUN__SAVE_GRAPHS:
        return
    fig = plt.figure(figsize=(20, 10))

    percentileLines = []
    for percentile in percentiles:
        percentileValue = np.percentile(dataList, percentile)
        percentileLine = plt.axvline(x=percentileValue, color='k', linestyle='dashed', linewidth=1)
        percentileLines.append(percentileLine)
        # if 'Speed' in title:
        # print('{} : Percentile: {}, value: {}'.format(title, percentile, percentileValue))

    for verticalLine in verticalLines:
        plt.axvline(x=verticalLine, color='red', linestyle='dashed', linewidth=1)

    binwidth = 1.

    if 'Speed' in title:
        plt.xticks(np.concatenate((np.arange(0., 5., 0.25), np.arange(5., 20., 1.))), rotation=70)
        plt.xlim(xmin=0., xmax=2.5)
        binwidth = 0.05

    plt.hist(
        dataList,
        bins=np.arange(min(dataList), max(dataList) + binwidth, binwidth),
        # range=[0., 2.]
    )
    # plt.hist(dataList, bins='auto')
    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.show()

    ax = plt.axes()

    # _,  plt.subplots()
    ax.legend(percentileLines, percentiles)

    mouseDirectoryPath = 'Basic Data/{}'.format(MOUSE_DIRECTORY)
    if not os.path.isdir(mouseDirectoryPath):
        os.makedirs(mouseDirectoryPath)
    fig.savefig('{}/{}.png'.format(mouseDirectoryPath, title))
    # with open('mouseSizes.csv', 'wb') as myfile:
    #     wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    #     wr.writerow(mouseSizeOverTime)

    plt.close()


def makeLineGraph(dataListsWithLabel, title='Line', xLabel="Frames", yLabel="Value"):
    if not TESTING_RUN__SAVE_GRAPHS:
        return
    fig = plt.figure(figsize=(25, 10))

    for dataList, label in dataListsWithLabel:
        plt.plot(dataList, label=label)
    plt.title(title)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    # plt.show()

    tickLocationsX = np.arange(0, len(dataListsWithLabel[0][0]), 60)
    tickLabelsX = map(lambda x: x / 30, tickLocationsX)
    plt.xticks(tickLocationsX, tickLabelsX, rotation=70)

    tickLocationsY = np.arange(0, max(dataListsWithLabel[0][0]), 1)
    plt.yticks(tickLocationsY)

    plt.grid(True)
    plt.legend()

    mouseDirectoryPath = 'Basic Data/{}'.format(MOUSE_DIRECTORY)
    if not os.path.isdir(mouseDirectoryPath):
        os.makedirs(mouseDirectoryPath)
    fig.savefig('{}/{}.png'.format(mouseDirectoryPath, title))
    # with open('mouseSizes.csv', 'wb') as myfile:
    #     wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    #     wr.writerow(mouseSizeOverTime)

    plt.close()


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
