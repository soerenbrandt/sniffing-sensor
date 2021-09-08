"""
Creadted Mar 8 2019 by Soeren Brandt

This file contains all the functions used in my python scripts for HANDLING OF DATA
"""

# import our modules
import fra_expt

#import some generally useful modules
import numpy as np
import matplotlib.pyplot as plt
import scipy

#------------------------------------------------------------#
# Load datasets
#------------------------------------------------------------#

import time
import sys
import warnings
from collections import OrderedDict 

def load_labeled_set(exp_set): # originally load_exps
    #sets up dictionaries to hold the derivs for different compounds and corresponding number/chemical
    exp_derivs = {}
    exp_num_name = {}
    
    # create report if chem labels don't match
    report = ""

    # start toolbar
    sys.stdout.write("Loading experimental data \n")
    sys.stdout.flush()
    loaded = 0

    for count, chem in enumerate(exp_set):
        nums = exp_set[chem]
        
        t = time.time() # start timer
        #loads experiments then smoothes and normalizes data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            exps = []#[fra_expt.get_expt(n) for n in nums]
            for b, n in enumerate(nums):
                exp = fra_expt.get_expt(n)
                exps.append(exp)
                
                # check if chemical is actually labeled correctly
                if sum([exp.sample_composition[string] == 1 for n, string in enumerate(exp.sample_composition.keys()) if string.lower() == chem.lower()]) != 1:
                    report = report + "\nExp. " + str(n) + " does not match. Labeled " + chem + " but lists " + str([key+": "+str(value) for key, value in exp.sample_composition.items()])                    
            
                # update the bar
                sys.stdout.write('\r')
                bar = ((b+1)*100)/len(nums)
                # the exact output you're looking for:
                sys.stdout.write(str(chem) + ": [%-20s] %d%%" % ('='*(bar/5-1) + str(bar % 10), bar))
                sys.stdout.flush()
                loaded += 1
                
        t1 = int(time.time()-t) # time loading
        sys.stdout.write(" " + str(t1/60) + ":" + "%02.d" % (t1 % 60) + "min loading")
        sys.stdout.flush()
            
        datas = [exp.main_spec_data.set_times(600) for exp in exps]
        datas = [da.lower_data_freq(1) for da in datas] #necessary because some experiments sampled at 4 Hz or 1 Hz
    #    freqs = [da.get_data_frequency() for da in datas]
    #    print freqs

        #calculate the phase derivative, UNNORMALIZED TO SHOW BAD LINEAR PCA WEAKNESSES
        phase_derivs = [da.getPhaseDerivative('spectral',1,smoothing=True,normalize=False) for da in datas]
        phase_derivs = [deriv for deriv in phase_derivs]
        
        #saves the derivs in the dictionary
        for n,num in enumerate(nums):
            if np.size(phase_derivs[n]) < 600:
                phase_derivs[n]= np.append(phase_derivs[n],[0])
            exp_derivs[num] = phase_derivs[n]
            exp_num_name[num] = chem
        
        t2 = int(time.time()-t)-t1 # time derivatives
        sys.stdout.write("  " + str(t2/60) + ":" + "%02.d" % (t2 % 60) + "min derivatives")
        sys.stdout.flush()
        
        # update the bar
        sys.stdout.write("   " + str(count+1) + "/" + str(len(exp_set)) + " complete\n")
        sys.stdout.flush()
        
    exp_set_size = len(np.concatenate(exp_set.values(),axis=None))
    print("Length of experimental set loaded: " + str(exp_set_size))

    return (exp_derivs, exp_num_name, exp_set_size)


def load_unlabeled_set(exp_set):
    # setup dictionary to hold datasets
    dataset = OrderedDict()

    # start toolbar
    sys.stdout.write("Loading experimental data \n")
    sys.stdout.flush()
    loaded = 0

    for name, nums in exp_set.items:
        #sets up dictionaries to hold the derivs for different compounds and corresponding number/chemical
        exp_derivs = {}
        exp_concentrations = {}
        
        for num in nums: #enumerate(nums):
            t = time.time() # start timer
            #loads experiment then smoothes and normalizes data
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                exp = fra_expt.get_expt(num)

            data = exp.main_spec_data.set_times(600) # might have to set to 500 for regression
            data = data.lower_data_freq(1)

            #calculate the phase derivative
            phase_deriv = data.getPhaseDerivative('spectral',1,smoothing=True,normalize=False)
            exp_derivs[num] = phase_deriv/np.sqrt(np.sum(phase_deriv**2)) # normalize

            #get concentration
            exp_concs[num] = exp.sample_composition

            # update the bar
            sys.stdout.write('\r')
            bar = ((count+1)*100)/len(exp_set)
            # the exact output you're looking for:
            sys.stdout.write("[%-20s] %d%%" % ('='*(bar/5-1) + str(bar % 10), bar))
            sys.stdout.flush()
            loaded += 1
            
        # clean up concentrations dictionary
        keys = set([key for composition in exp_concentrations for key in composition])
        exp_concs = {}
        for num in nums:
            exp_concs[num] = {}
            for chem in keys:
                try:
                    exp_concs[num][chem.capitalize()] = A[num][chem]
                except:
                    exp_concs[num][chem.capitalize()] = 0
        
        # add dictionaries to dataset
        dataset[name] = [exp_derivs, exp_concs]

        # update the bar
        sys.stdout.write("   " + " complete\n")
        sys.stdout.flush()

    exp_set_size = count+1
    print(str(exp_set_size) + " experiments loaded in " + len(exp_set) + " datasets.")
    
    return (exp_derivs, exp_concentrations, exp_set_size)


def load_unlabeled_list(exp_set,chem): # previously load_list
    #sets up dictionaries to hold the derivs for different compounds and corresponding number/chemical
    exp_derivs = {}
    exp_concentrations = {}

    # start toolbar
    sys.stdout.write("Loading experimental data \n")
    sys.stdout.flush()
    loaded = 0

    for count, num in enumerate(exp_set):
        t = time.time() # start timer
        #loads experiment then smoothes and normalizes data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            exp = fra_expt.get_expt(num)

        data = exp.main_spec_data.set_times(600) # might have to set to 500 for regression
        data = data.lower_data_freq(1)

        #calculate the phase derivative
        phase_deriv = data.getPhaseDerivative('spectral',1,smoothing=True,normalize=False)
        exp_derivs[num] = phase_deriv/np.sqrt(np.sum(phase_deriv**2)) # normalize

        #get concentration
        exp_concentrations[num] = exp.sample_composition[chem]

        # update the bar
        sys.stdout.write('\r')
        bar = ((count+1)*100)/len(exp_set)
        # the exact output you're looking for:
        sys.stdout.write("[%-20s] %d%%" % ('='*(bar/5-1) + str(bar % 10), bar))
        sys.stdout.flush()
        loaded += 1

    # update the bar
    sys.stdout.write("   " + " complete\n")
    sys.stdout.flush()

    exp_set_size = count+1
    print("Length of experimental set loaded: " + str(exp_set_size))
    
    return (exp_derivs, exp_concentrations, exp_set_size)


#------------------------------------------------------------#
# Load datasets from CSV
#------------------------------------------------------------#

import sys
import pandas as pd
from PIL import Image

def load_images_from_CSV(exp_set):
    folder = '../fra_exp_csv'

    # prepare dictionaries to reference data
    exp_images = {}
    exp_data = {}
    exp_num_chem = {}

    # prepare arrays to story data
    train_data = []
    train_im = []
    labels = []
    exp_num = []

    # start toolbar
    sys.stdout.write("Loading experimental data \n")
    sys.stdout.flush()
    loaded = 0

    for count, chem in enumerate(exp_set):
        nums = exp_set[chem]

        #loads image then normalizes data
        for b, num in enumerate(nums):
            data = pd.read_csv(folder + "/%06d.csv" % (num)) # load data from file
            data_im = (data.values[:,1:]-np.min(data.values[:,1:])).transpose()
            data_im = data_im*255/np.max(data_im)

            img = Image.fromarray(data_im)
            img = img.convert('RGB')
            img_resized = img.resize((299,299)) # necessary for Inception module

            exp_images[num] = data_im # keep unconverted data for reference
            exp_num_chem[num] = chem # keep reference of chemical
            exp_data[num] = np.array(img_resized)*255/np.max(img_resized)
            exp_num.append(num)
            train_im.append(img_resized)
            train_data.append(np.array(img_resized)*255/np.max(img_resized))
            labels.append(chem) # create labels

            # update the bar
            sys.stdout.write('\r')
            bar = ((b+1)*100)/len(nums)
            # the exact output you're looking for:
            sys.stdout.write(chem + ": [%-20s] %d%%" % ('='*int(bar/5-1) + str(int(bar % 10)), bar))
            sys.stdout.flush()
            loaded += 1

        # update the bar
        sys.stdout.write("   " + str(count+1) + "/" + str(len(exp_set)) + " complete\n")
        sys.stdout.flush()

    train_data = np.array(train_data)

    exp_set_size = loaded
    print("Length of experimental set loaded: " + str(exp_set_size))
    
    return


def load_set_from_CSV(exp_set):
    folder = '../fra_exp_csv'

    # prepare dictionaries to reference data
    exp_derivs = {}
    exp_num_chem = {}

    # start toolbar
    sys.stdout.write("Loading experimental data \n")
    sys.stdout.flush()
    loaded = 0

    for count, chem in enumerate(exp_set):
        nums = exp_set[chem]

        #loads image then normalizes data
        datas = []
        for b, num in enumerate(nums):
            data = pd.read_csv(folder + "/%06d.csv" % (num)) # load data from file
            data_spectra = (data.values[:,1:]-np.min(data.values[:,1:])).transpose()
            datas.append(data_spectra)

            # update the bar
            sys.stdout.write('\r')
            bar = ((b+1)*100)/len(nums)
            # the exact output you're looking for:
            sys.stdout.write(str(chem) + ": [%-20s] %d%%" % ('='*int(bar/5-1) + str(int(bar % 10)), bar))
            sys.stdout.flush()
            loaded += 1
        
        phase_derivs = []
        for da in datas:
            # performing manually: phase_derivs = [da.getPhaseDerivative('spectral',1,smoothing=True,normalize=False) for da in datas]
            # Step 1: get phase derivative data at each time step
            ft_data = np.fft.fft(da)
            # Step 2: Calculates the phase
            R = np.real(ft_data[:, 1])
            I = np.imag(ft_data[:, 1])
            phi = I / (R ** 2 + I ** 2) ** 0.5
            # Step 3: Calculate phase derivative and perform smoothing with default parameters
            phi_deriv = np.diff(phi)
            phi_deriv = scipy.signal.savgol_filter(phi_deriv, window_length=31, polyorder=2)
            
            phase_derivs.append(phi_deriv)
            
        #phase_derivs = [deriv for deriv in phase_derivs]
        
        #saves the derivs in the dictionary
        for n,num in enumerate(nums):
            if np.size(phase_derivs[n]) < 600:
                phase_derivs[n]= np.append(phase_derivs[n],[0])
            exp_derivs[num] = phase_derivs[n]
            exp_num_chem[num] = chem
            
        # update the bar
        sys.stdout.write("   " + str(count+1) + "/" + str(len(exp_set)) + " complete\n")
        sys.stdout.flush()

    exp_set_size = loaded
    print("Length of experimental set loaded: " + str(exp_set_size))

    return (exp_derivs, exp_num_chem, exp_set_size)