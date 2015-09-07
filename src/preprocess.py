from __future__ import print_function
from collections import defaultdict
import numpy as np
import datetime
import csv
from operator import itemgetter
import sys

species_map = {'CULEX RESTUANS' : "10000000",
              'CULEX TERRITANS' : "01000000",
              'CULEX PIPIENS'   : "00100000",
              'CULEX PIPIENS/RESTUANS' : "00010000",
              'CULEX ERRATICUS' : "00001000",
              'CULEX SALINARIUS': "00000100",
              'CULEX TARSALIS' :  "00000010",
              'UNSPECIFIED CULEX': "00000001"}

def date(text):
    return datetime.datetime.strptime(text, "%Y-%m-%d").date()

def entries_to_remove(entries, training):
    for record in training:
        for key in entries:
            if key in record:
                del record[key]

def getVectDuplicateInfo(ind):
    duplicateInfo = np.zeros(1000)
    y = np.bincount(ind)[2:]
    duplicateInfo[0:y.shape[0]] = np.divide(y,np.arange(y.shape[0])+2)
    duplicateInfoFinal = np.append(duplicateInfo[0:4],[sum(duplicateInfo[4:])], axis=1)
    duplicateInfoFinal = np.append(duplicateInfoFinal,[sum(duplicateInfo)], axis=1)
    duplicateInfoFinal = np.append(duplicateInfoFinal,[sum(y)], axis=1)
    return duplicateInfoFinal

def closeststation(stationlist, station, nStationToReturn):

    indCurrentTrap = (stationlist[:,0] == station).nonzero()
    coordCurrentTrap = stationlist[indCurrentTrap,1:3][0][0,:]
    distToCurrentTrap = (stationlist[:,1:3]-coordCurrentTrap)*(stationlist[:,1:3]-coordCurrentTrap)
    distToCurrentTrap = distToCurrentTrap.sum(axis=1)
    return stationlist[np.argsort(distToCurrentTrap)[0:nStationToReturn],0]

def makeNewFeaturesByStation(data, nbTrap):

    b = data[:,[2,4,5]]
    stationlist = np.vstack({tuple(row) for row in b})

    retFeatures = np.zeros((data.shape[0],8))
    yearlist = np.unique(data[:,1])
    for year in yearlist:
        print("Preprocessing year " + str(year))
        traplist = np.unique(data[(data[:,1] == year).nonzero(),2])
        for trap in traplist:
            inddaylist = np.nonzero((data[:,1] == year) & (data[:,2] == trap))
            daylist = np.unique(data[inddaylist,3])
            trapAllList = closeststation(stationlist, trap, nbTrap)
            for day in daylist:
                ind = np.nonzero((data[:,1] == year) & (np.in1d(data[:,2], trapAllList)) & (data[:,3] == day))
                ind2 = np.nonzero((data[:,1] == year) & (np.in1d(data[:,2], trapAllList)) & (data[:,3] <= day))
                ind3 = np.nonzero((data[:,1] == year) & (data[:,2] == trap) & (data[:,3] == day))
                k = data[ind,7].astype(int)
                res = getVectDuplicateInfo(k[0])
                k2 = data[ind2,7].astype(int)
                res2 = getVectDuplicateInfo(k2[0])
                res = np.append(res2,[res[-1]],axis=1)
                retFeatures[ind3,:] = res

    return retFeatures

def makeNewFeaturesAllStation(data):
    retFeatures = np.zeros((data.shape[0],8))
    yearlist = np.unique(data[:,1])
    for year in yearlist:
        daylist = np.unique(data[(data[:,1] == year).nonzero(),3])
        for day in daylist:
            ind = np.nonzero((data[:,1] == year) & (data[:,3] == day))
            ind2 = np.nonzero((data[:,1] == year) & (data[:,3] <= day))
            k = data[ind,7].astype(int)
            res = getVectDuplicateInfo(k[0])
            k2 = data[ind2,7].astype(int)
            res2 = getVectDuplicateInfo(k2[0])
            res = np.append(res2,[res[-1]],axis=1)
            retFeatures[ind,:] = res

    return retFeatures

def createnDuplicatedFeature(dataset):
    species = []

    training = sorted(dataset, key=itemgetter('year', 'day_of_year'))
    hashmap = defaultdict(int)
    traphashmap = defaultdict(int)

    i = 1
    for key in training:
        hashmap[key['Trap'] + '_' + str(key['day_of_year']) + '_' + str(key['year']) + '_' + str(key['Latitude']) + '_' + str(key['Longitude']) + '_' + key['Species']] += 1
        if key['Trap'] not in traphashmap:
            traphashmap[key['Trap']] = i
            i += 1

    for key in training:
        key['nDuplicated'] = hashmap[key['Trap'] + '_' + str(key['day_of_year']) + '_' + str(key['year']) + '_' + str(key['Latitude']) + '_' + str(key['Longitude']) + '_' + key['Species']]
        dict = { 'Species' : key['Species'] }
        species.append(dict)
        key['TrapId'] = traphashmap[key['Trap']]

    return species

def createNpArray(dataset, species):

    tab_species = np.zeros((len(dataset), 8))
    i = 0
    for b in species:
        species_vector = [float(x) for x in species_map[b["Species"]]]
        tab_species[i,:] =  np.array(species_vector)
        i += 1

    tab = np.zeros((len(dataset), len(dataset[0])))
    i = 0
    for line in dataset:
        tab[i,:] =  np.array(line.values())
        i += 1

    return np.concatenate((tab,tab_species),axis=1)

def createData(data, entriesToDelete, isTestData):
    species = createnDuplicatedFeature(data)
    entries_to_remove(entriesToDelete, data)

    data = createNpArray(data, species)

    if isTestData:
        data = np.concatenate((np.zeros((data.shape[0],1)), data), axis=1)

    finalData = data
    finalData = np.concatenate((finalData,makeNewFeaturesByStation(data,60)), axis=1)
    finalData = np.concatenate((finalData,makeNewFeaturesAllStation(data)), axis=1)
    finalData = np.concatenate((finalData,makeNewFeaturesByStation(data,1)[:,7:8]), axis=1)

    return finalData

def load_training(trainfile):
    training = []
    yearlist = set()
    for line in csv.DictReader(open(trainfile)):
        for name, converter in {"Date" : date,
                                "Latitude" : float, "Longitude" : float,
                                "NumMosquitos" : int, "WnvPresent" : int}.items():
            line[name] = converter(line[name])
        line['day_of_year'] = line['Date'].timetuple().tm_yday
        line['year'] = line['Date'].timetuple().tm_year
        yearlist.add(line['year'])
        line['week_of_year'] = line['Date'].isocalendar()[1]
        training.append(line)

    entriesToDelete = ('Date', 'NumMosquitos', 'Street', 'Trap', 'Block', 'Address', 'AddressAccuracy', 'AddressNumberAndStreet', 'Species')
    return createData(training, entriesToDelete, False)

def load_testing(testfile):
    test = []
    yearlist = set()
    for line in csv.DictReader(open(testfile)):
        for name, converter in {"Date" : date,
                                "Latitude" : float, "Longitude" : float}.items():
            line[name] = converter(line[name])
        line['day_of_year'] = line['Date'].timetuple().tm_yday
        line['year'] = line['Date'].timetuple().tm_year
        yearlist.add(line['year'])
        line['week_of_year'] = line['Date'].isocalendar()[1]
        test.append(line)

    entriesToDelete = ('Date', 'Street', 'Trap', 'Block', 'Address', 'AddressAccuracy', 'AddressNumberAndStreet', 'Species','Id')
    return createData(test, entriesToDelete, True)

if __name__ == "__main__":
    if len(sys.argv) == 3:
        if (sys.argv[2] == "train"):
            training = load_training(sys.argv[1])
            np.savetxt("preprocesstrain.csv", training, delimiter=",", fmt='%10.5f')
        elif (sys.argv[2] == "test"):
            test = load_testing(sys.argv[1])
            np.savetxt("preprocesstest.csv", test, delimiter=",", fmt='%10.5f')
        else:
            print("The second argument must be the flag train or test")
    else:
        print("The script needs 2 arguments : \n1: Train or test file \n2: flag train or test\n"
              "Example: python preprocess.py ./input/train.csv train")



