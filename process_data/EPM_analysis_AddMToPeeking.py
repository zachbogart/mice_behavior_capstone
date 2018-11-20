


def isPeeking(arm, mouseLocation, mouseLength):
    if arm == 'CL':
        rightEndOfArm = ZONES['CL'][1][0]
        return mouseLocation[0] > rightEndOfArm - mouseLength
    elif arm == 'CR':
        leftEndOfArm = ZONES['CR'][0][0]
        return mouseLocation[0] < leftEndOfArm + mouseLength
    elif arm == 'M':
        LeftEndOfMiddle = ZONES['CL'][1][0]
        RightEndOfMiddle = ZONES['CR'][0][0]
        return mouseLocation[0] > LeftEndOfMiddle and mouseLocation[0] < RightEndOfMiddle 
    return False

def calculatePeekingFeatures(centroidsByArm, distancesPerArm, mouseLength):
    timePeekingTotal = 0
    timeTotal = 0
    peekLengthsTotal = []
    peekingFeatures = {}
    considered_zones = ['CR', 'CL', 'M']
    for arm in considered_zones:
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
