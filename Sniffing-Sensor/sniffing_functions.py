import time

# River plot
def plot_river(exp, plt, np):
    # Plot river plot
    fig, ax = plt.subplots()
    #fig = plt.figure(frameon=False)
    #ax = plt.Axes(fig, [0., 0., 1., 1.])
    #fig.add_axes(ax)
    #fig.set_size_inches(5,5)

    # Plot figure (adapted from: fra_plotting.image_plot_2d(std_data,subtle=True, cmap = 'jet') )
    if type(exp) == np.ndarray:
        d_array = exp
        extent = [0, 1, 1, 0]
    else:
        d_array = exp.spectra
        extent = [np.min(exp.wavelengths), np.max(exp.wavelengths), np.max(exp.times), np.min(exp.times)]

    vmin = np.amin(d_array)
    vmax = np.amax(d_array)
    plt.imshow(d_array, vmin=vmin, vmax=vmax, interpolation='none', cmap='jet')
    
    # Define image labels
    plt.ylabel('Time (s)', fontsize=16)
    plt.xlabel('Wavelength (nm)', fontsize=16)
    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')
    ax.tick_params(axis='x',colors='black')
    ax.tick_params(axis='y',colors='black')
    
    # Image aspect ratio
    xext, yext = plt.gca().axes.get_xlim(), plt.gca().axes.get_ylim()
    xrange = xext[1] - xext[0]
    yrange = yext[1] - yext[0]
    plt.gca().set_aspect(1 * abs(xrange / yrange)) # This is the line that causes the warnings about unicode
    
    plt.tight_layout()

##### ================= PROCEDURES =================

## Procedure 2
def short_sniff(proc, valves, compound, ser, intensities, spec, delay_time, elapse): 
    """ PROCEDURE 2"""
    
    print("Performing short sniff of {} injection".format(compound))

    for i in range(int(proc['Repeat'])):

        toggle_valve(valves[compound], float(proc['Odorant']), ser, intensities, spec, delay_time, elapse) # Change the function based on what odorant you are testing
        Sleep(float(proc['Break']), ser, intensities, spec, delay_time, elapse)

    Nitrogen(float(proc['Nitrogen']), ser, intensities, spec, delay_time, elapse)

## Procedure 3
def deep_sniff(proc, valves, compound, ser, intensities, spec, delay_time, elapse): 
    """ PROCEDURE 3 """
    
    print("Performing deep sniff of {} injection".format(compound))

    toggle_valve(valves[compound], float(proc['Odorant']), ser, intensities, spec, delay_time, elapse) # Change the function based on what odorant you are testing
    Sleep(float(proc['Break']), ser, intensities, spec, delay_time, elapse)

    Nitrogen(float(proc['Nitrogen']), ser, intensities, spec, delay_time, elapse)

## Procedure 4
def short_held_sniff(proc, valves, compound, ser, intensities, spec, delay_time, elapse):
    """ PROCEDURE 4 """
    
    print("Performing short held sniff of {} injection".format(compound))

    toggle_valve(valves[compound], float(proc['Odorant']), ser, intensities, spec, delay_time, elapse)  
    Hold(valves[compound], float(proc['Hold']), ser, intensities, spec, delay_time, elapse)
    toggle_valve(valves[compound], float(proc['Odorant']), ser, intensities, spec, delay_time, elapse)
    Close(float(proc['Close']), ser, intensities, spec, delay_time, elapse)
    Nitrogen(float(proc['Nitrogen']), ser, intensities, spec, delay_time, elapse)
    
## Procedure 5
def short_sniff_exhale(proc, valves, compound, ser, intensities, spec, delay_time, elapse): 
    """ PROCEDURE 5 """
    
    print("Performing short sniff exhale of {} injection".format(compound))

    toggle_valve(valves[compound], float(proc['Odorant']), ser, intensities, spec, delay_time, elapse) # Change the function based on what odorant you are testing

    Nitrogen(float(proc['Nitrogen']), ser, intensities, spec, delay_time, elapse)

### ====================== SCAN FUNCTIONS =========================
def Sleep(sleep_time, ser, intensities, spec, delay_time, elapse): 
    """ Captures the scans at a rate of 20fps with no gas injection. """
    
    last_time = elapse[-1]
    i = 0
    start_time = time.time()

    while i < sleep_time:
        intensities.append(spec.intensities())
        i = time.time() - start_time
        elapse.append(i+last_time)           
        time.sleep(1/delay_time)

def Purge(purge_time, ser, intensities_broad, spec, elapse_broad): 
    """ Captures scans at a rate of 1fps with Nitrogen gas injection. """
    
    ser.write(b'A')

    last_time = elapse_broad[-1]
    i = 0
    start_time = time.time()

    while i < purge_time:
        intensities_broad.append(spec.intensities())
        i = time.time() - start_time
        elapse_broad.append(i+last_time)
        time.sleep(1)
    
    ser.write(b'a')

def Hold(letter, hold_time, ser, intensities, spec, delay_time, elapse):
    """ Injects odorant while outlet valve is closed """
    ser.write(b'E')
    ser.write(str.encode(letter.upper()))
    
    last_time = elapse[-1]
    i = 0
    start_time = time.time()
    
    while i < hold_time:
        intensities.append(spec.intensities())
        i = time.time() - start_time
        elapse.append(i+last_time)
        time.sleep(1/delay_time)
        
    ser.write(b'e')
    ser.write(str.encode(letter.lower()))
    
def Close(close_time, ser, intensities, spec, delay_time, elapse):
    """ Closes outlet """
    ser.write(b'E')
    
    last_time = elapse[-1]
    i = 0
    start_time = time.time()
    
    while i < close_time:
        intensities.append(spec.intensities())
        i = time.time() - start_time
        elapse.append(i+last_time)
        time.sleep(1/delay_time)
        
    ser.write(b'e')
    
def Nitrogen(scan_time, ser, intensities, spec, delay_time, elapse):
    """ Captures scans at a rate of 20fps with Nitrogen gas injection. """
    ser.write(b'A')
    last_time = elapse[-1]
    i = 0
    start_time = time.time()
    
    while i < scan_time:
        intensities.append(spec.intensities())
        i = time.time() - start_time
        elapse.append(i+last_time)
        time.sleep(1/delay_time)
        
    ser.write(b'a')

def toggle_valve(letter, scan_time, ser, intensities, spec, delay_time, elapse):
    """ Toggles odorant valve based on coded letter in excel sheet. Captures scans at a rate of 20fps. """
    ser.write(str.encode(letter.upper()))
    last_time = elapse[-1]
    i = 0
    start_time = time.time()
    
    while i < scan_time:
        intensities.append(spec.intensities())
        i = time.time() - start_time
        elapse.append(i+last_time)
        time.sleep(1/delay_time)
        
    ser.write(str.encode(letter.lower()))


    
## ====================== PHASE COMPARISON SCRIPT FUNCTIONS ========================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

fig = plt.figure(figsize=(12,12))
ax = fig.add_subplot(111)

def graph(start, end, graphs, skip=[], ax=ax):
    """ A function used to graph the phase of the selected chemical. 
    
    Takes the following parameters: 
         start:   The experiment ID of the first replicate sample of the experiment
         end:     The experiment ID of the last replicate sample of the experiment
         graphs:  an empty list to be populated with the phase derivative values
         skip:    For botched samples (i.e. noise or comm error), you can choose to skip the experiment ID's of these samples
         ax:      the axis upon which the values will be plotted.
    
    Returns a list with the phase values of the chemical to be plotted.
    """
    output = []
    num = (end-start)//3+1 # The function assumes that your chemical occurs every 3 experiment ID's between start and end
    
    for i in map(int, np.linspace(start, end, num)):
        if i not in skip:
            df = pd.read_csv("../Scan CSV Files/Phase/{}.csv".format(i))
            df.drop('Unnamed: 0', axis=1, inplace=True)
            df = (df-df.mean())/df.std() # Normalization
            output.append(list(df["0"]))
            
    plot_df = pd.DataFrame(output)
    plot_df = [np.mean(plot_df.loc[:,i]) for i in plot_df.columns]
    graphs.append(plot_df)
    
def graph_pd(start, end, graphs, skip=[], ax=ax):
    """ A function used to graph the phase derivative of the selected chemical. 
    
    Takes the following parameters: 
         start:   The experiment ID of the first replicate sample of the experiment
         end:     The experiment ID of the last replicate sample of the experiment
         graphs:  an empty list to be populated with the phase derivative values
         skip:    For botched samples (i.e. noise or comm error), you can choose to skip the experiment ID's of these samples
         ax:      the axis upon which the values will be plotted.
    
    Returns a list with the phase derivative values of the chemical to be plotted.
    """
    
    output = []
    num = (end-start)//3+1 # The function assumes that your chemical occurs every 3 experiment ID's between start and end
    
    for i in map(int, np.linspace(start, end, num)):
        if i not in skip:
            df = pd.read_csv("../Scan CSV Files/Phase Derivative/{}.csv".format(i))
            df.drop('Unnamed: 0', axis=1, inplace=True)
            output.append(list(df["0"]))
            
    plot_df = pd.DataFrame(output)
    plot_df = [np.mean(plot_df.loc[:,i]) for i in plot_df.columns]
    graphs.append(plot_df)
    

## ====================== MACHINE LEARNING SCRIPT FUNCTIONS ========================

# Cursors for sqlite database
import pandas as pd
import numpy as np
import sqlite3
from scipy.interpolate import interp1d as interp
from sklearn.metrics import confusion_matrix
conn = sqlite3.connect("Database/Sniffing-Sensor.db")
c = conn.cursor()

def classification_data(proc_num=2):
    """ This function returns an array of phase values for specified chemicals, as well as the numeric labels (the OdorID). The function prompts the user for which chemicals they want to classify, and the user shall input comma separated values of the chemical names specified in the 'Odors' table in the sniffing-sensor database file. Else, the user can input 'all' and all of the chemicals under the chosen procedure will be trained and classified against one another.

    The function takes an optional parameter: 
        proc_num: the default value is 2, but the user can specify the procedure they want to classify.
        
    The function returns: 
        phase_datasets: the array of x-values.
        labels:         a list of the numeric labels
        compounds:      a list of the string labels

    """
    # x-data for interpolation (240 timesteps within 12 seconds)
    if proc_num == 2:
        xdata = np.linspace(0, 11.5, 230) # Procedure 2 is 11.5 seconds long
    elif proc_num == 3: 
        xdata = np.linspace(0, 12.5, 250) # Procedure 3 is 12.5 seconds long
    elif proc_num == 4: 
        xdata = np.linspace(0, 17, 340) # Procedure 4 is 17 seconds long
    elif proc_num == 5:
        xdata = np.linspace(0, 11, 220) # Procedure 5 is 11 seconds long

    # Compound dictionary
    compound_dict = {"WATER":    ["WATER"], 
                     "ETHANOL":  ["ETHANOL"], 
                     "IPA":      ["IPA"], 
                     "PENTANE":  ["PENTANE"], 
                     "HEXANE":   ["HEXANE"], 
                     "HEPTANE":  ["HEPTANE"], 
                     "OCTANE":   ["OCTANE"],
                     "NONANE":   ["NONANE"], 
                     "DECANE":   ["DECANE"] 
                    }
    
    compounds = input("What compounds do you want to classify? (i.e. 'All'): ").strip()
    compounds = compounds.split(", ")
    compounds = [name.upper().strip() for name in compounds]
    
    datasets = []
    labels = []
    
    if "ALL" in compounds:
        compounds = list(compound_dict)
        
    for compound in compounds:
        for compound in compound_dict[compound]:
                temp = c.execute("SELECT OdorID FROM Odors WHERE Name='%s'" %str(compound.upper())) # All database entries are uppercase
                Odor = c.fetchall()[0][0]
                c.execute(""" SELECT ExperimentID FROM Experiments WHERE OdorID='%i' 
                                                                     AND ProcedureID='%s' 
                                                                     AND NOTES != 'Skip'
                                                                     AND NOTES != 'Validation' """ % (Odor, proc_num))
                data = c.fetchall()
                data = [i[0] for i in data] # Turns list of tuples into list of integers
                datasets.extend(data)
                if '-' in compound: 
                    labels.extend([int(compound[0])/10 for i in range(len(data))])
                else: 
                    labels.extend([Odor for i in range(len(data))])
                
    phase_datasets = []
    
    for i in datasets:

        # Load phase data
        df = pd.read_csv("../Scan CSV Files/Phase/{}.csv".format(i))
        df.drop('Unnamed: 0', axis=1, inplace=True)
        df.drop(0, inplace=True)
        ydata = list(df["0"])

        # Load time data
        df2 = pd.read_csv("../Scan CSV Files/{}.csv".format(i))
        df2.drop(['Unnamed: 0', 'Wavelengths', '0'], axis=1, inplace=True)
        time = [float(i) for i in list(df2.columns)]

        # Interpolate
        f = interp(time, ydata)
        phase = f(xdata)

        # Normalize
        norm_phase = (phase-phase.mean())/phase.std()

        phase_datasets.append(norm_phase)
        
    return np.asarray(phase_datasets), labels, compounds

def regression_data(proc_num=2):
    """ This function returns an array of phase values for specified mixtures, as well as the numeric labels (the OdorID). The function prompts the user for which chemicals they want to classify, and the user shall input comma separated values of the chemical names specified in the 'Odors' table in the sniffing-sensor database file. Else, the user can input 'all' and all of the chemicals under the chosen procedure will be trained and classified against one another.

    The function takes an optional parameter: 
        proc_num: the default value is 2, but the user can specify the procedure they want to classify.
        
    The function returns: 
        phase_datasets: the array of x-values.
        labels:         a list of the numeric labels
    """
    if proc_num == 2:
        xdata = np.linspace(0, 11.5, 230) # Procedure 2 is 11.5 seconds long
    elif proc_num == 3: 
        xdata = np.linspace(0, 12.5, 250) # Procedure 3 is 12.5 seconds long
    elif proc_num == 4: 
        xdata = np.linspace(0, 17, 340) # Procedure 4 is 17 seconds long
    elif proc_num == 5:
        xdata = np.linspace(0, 11, 220) # Procedure 5 is 11 seconds long
        
    # Compound dictionary
    compound_dict = {"PENT+HEX": ['1C5-9C6', '2C5-8C6', '3C5-7C6', '4C5-6C6', '5C5-5C6', '6C5-4C6', '7C5-3C6', '8C5-2C6', '9C5-1C6'],
                     "PENT+OCT": ['1C5-9C8', '2C5-8C8', '3C5-7C8', '4C5-6C8', '5C5-5C8', '6C5-4C8', '7C5-3C8', '8C5-2C8', '9C5-1C8'],
                     "WATER+ETOH":  ['1W-9ET', '2W-8ET', '3W-7ET', '4W-6ET', '5W-5ET', '6W-4ET', '7W-3ET', '8W-2ET', '9W-1ET'],
                     }
    
    compounds = input("What mixtures do you want to classify? (i.e. water+etoh): ").strip()
    compounds = compounds.split(", ")
    compounds = [name.upper().strip() for name in compounds]
    
    datasets = []
    labels = []
    
    if "ALL" in compounds:
        compounds = list(compound_dict)
        
    for compound in compounds:
        for compound in compound_dict[compound]:
                temp = c.execute("SELECT OdorID FROM Odors WHERE Name='%s'" %str(compound.upper())) # All database entries are uppercase
                Odor = c.fetchall()[0][0]
                c.execute(""" SELECT ExperimentID FROM Experiments WHERE OdorID='%i' 
                                                                     AND ProcedureID='%s' 
                                                                     AND NOTES != 'Skip' 
                                                                     AND NOTES != 'Validation' """ % (Odor, proc_num))
                data = c.fetchall()
                data = [i[0] for i in data] # Turns list of tuples into list of integers
                datasets.extend(data)
                if '-' in compound: 
                    labels.extend([int(compound[0])/10 for i in range(len(data))])
                else: 
                    labels.extend([Odor for i in range(len(data))])
                
    phase_datasets = []
    
    for i in datasets:

        # Load phase data
        df = pd.read_csv("../Scan CSV Files/Phase/{}.csv".format(i))
        df.drop('Unnamed: 0', axis=1, inplace=True)
        df.drop(0, inplace=True)
        ydata = list(df["0"])

        # Load time data
        df2 = pd.read_csv("../Scan CSV Files/{}.csv".format(i))
        df2.drop(['Unnamed: 0', 'Wavelengths', '0'], axis=1, inplace=True)
        time = [float(i) for i in list(df2.columns)]

        # Interpolate
        f = interp(time, ydata, fill_value="extrapolate")
        
        try: 
            phase = f(xdata)
        except: 
            print("Problem with {}.csv",format(i))
                
        # Normalize
        norm_phase = (phase-phase.mean())/phase.std()

        phase_datasets.append(norm_phase)
        
    return np.asarray(phase_datasets), labels

def plot_confusion_matrix(y_true, y_pred, classes=None, title=None, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred,labels=classes)
    
    fig, ax = plt.subplots(figsize=(12,12))
    
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    
    plt.tick_params(labelsize=15)
    ax.title.set_fontsize(23)
    ax.xaxis.label.set_fontsize(20)
    ax.yaxis.label.set_fontsize(20)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

def calculate_R2(actual,predicted):
    """ Function calculates the coefficient of determination of the scatter plot. Essentially, the accuracy of the model. """
    # convert to array
    actual = np.array(actual).reshape(1,-1)
    predicted = np.array(predicted).reshape(1,-1)
    
    # calculate errors
    res_ss = np.sum((predicted-actual)**2)
    total_ss = np.sum((np.mean(actual)-actual)**2)
    
    return 1-res_ss/total_ss