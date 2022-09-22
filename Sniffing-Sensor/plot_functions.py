"""
Creadted Mar 8 2019 by Soeren Brandt

This file contains all the functions used in my python scripts for VISUALIZATION of machine learning results
"""

#import some generally useful modules
import numpy as np
import matplotlib.pyplot as plt

#------------------------------------------------------------#
# Define color sets
#------------------------------------------------------------#

import seaborn as sns

def get_colors(n, palette = 'bright'):
    # gives you a color range
    colors = sns.color_palette(palette, n) #plt.cm.hsv(np.linspace(0,1,n))
    return colors


def get_markers(n = None):
    # gives a list of markers
    markers = ['o', 'v', '^', '<', '>', 's', 'p', 'P', '*', 'X', 'D', 'd', '8', 'H', 'h', 'o', 'v', '^', '<', '>']
    if n is None:
        return markers
    else:
        return markers[0:n]
    
def get_contour_order(dataset):
    order = {}
    for count, key in enumerate(dataset.keys()):
        order[key] = count
    
    return order

from colour import Color

def get_gradient_colors(n, start, end):
    # gives you a color range between start and end
    if type(start) == str:
        firstColor = Color(start)
    elif type(start) == tuple:
        firstColor = Color(rgb=start)
        
    if type(end) == str:
        finalColor = Color(end)
    elif type(end) == tuple:
        finalColor = Color(rgb=end)
        
    colors = list(firstColor.range_to(finalColor,n))
    colors = [color.rgb for color in colors]

    return colors


#------------------------------------------------------------#
# Basic Plots
#------------------------------------------------------------#

def plot_data(x,y, marker = 'o', face = [0, 0, 0, 1], edge = [0, 0, 0, 1], name = '', ax = None):
    if ax == None:
        #initialize figure and set size
        fig, ax = plt.subplots()
        fig.set_size_inches(6,6)
        
    ax.scatter(x,y,marker = marker,c=face,edgecolors=edge,s=20,label=name)
    return


def create_plot(bounds):
    if type(bounds) == np.ndarray:
        #determine the bounds for plotting
        plot_x_min = np.min(bounds[:, 0]) - 0.1*abs(np.min(bounds[:, 0]))
        plot_x_max = np.max(bounds[:, 0]) + 0.75*abs(np.max(bounds[:, 0]))
        plot_y_min = np.min(bounds[:, 1]) - 0.1*abs(np.min(bounds[:, 1]))
        plot_y_max = np.max(bounds[:, 1]) + 0.1*abs(np.max(bounds[:, 1]))
        bounds = [plot_x_min, plot_x_max, plot_y_min, plot_y_max]
    
    #initialize figure and set size
    fig, ax = plt.subplots()
    fig.set_size_inches(6,6)
    
    #all the nice plotting things
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.tick_params(axis='both', which='major')
    plt.tight_layout()
    ax.tick_params(direction='in', length=8, width=1, colors='k')

    #sets limits since scatter plots are too dumb to automatically do it
    plt.xlim(bounds[0],bounds[1])
    plt.ylim(bounds[2],bounds[3])

    #shows legend
    plt.legend()
    
    return (fig, ax)


#------------------------------------------------------------#
# SVC Plots
#------------------------------------------------------------#

def create_scatter_plot(exp_set,set_key,plot_PCA,face = [0, 0, 0, 1],edge = [0, 0, 0, 1],marker = 'o',ax = None):
    if ax == None:
        #determine the bounds for plotting
        plot_x_min = np.min(plot_PCA[:, 0]) - 0.1*abs(np.min(plot_PCA[:, 0]))
        plot_x_max = np.max(plot_PCA[:, 0]) + 0.75*abs(np.max(plot_PCA[:, 0]))
        plot_y_min = np.min(plot_PCA[:, 1]) - 0.1*abs(np.min(plot_PCA[:, 1]))
        plot_y_max = np.max(plot_PCA[:, 1]) + 0.1*abs(np.max(plot_PCA[:, 1]))

        # create plot with bounds from PCA
        fig, ax = create_plot([plot_x_min, plot_x_max, plot_y_min, plot_y_max])
    else:
        fig = ax.figure
        
    # list whole dataset
    #wholeset = list(np.sort(np.concatenate(exp_set.values(),axis=None)))

    # add all chemical data points to the plot
    for chem, values in exp_set.items():
        # getting indices for chemical and procedure used for the experiments
        chemNum = exp_set.keys().index(chem)
        # procNum = [group_dict[exp] for exp in values]

        # get PC1 and PC2 from b3_deriv_pca_fit for the set of chem
        indices = [set_key.index(exp) for exp in values]  # old: [exp_numbers.index(exp) for exp in exp_set[chem]]
        pca_PC1 = plot_PCA[indices,0] #PCA_fit[indices,0]
        pca_PC2 = plot_PCA[indices,1] #PCA_fit[indices,1]

        plot_data(pca_PC1, pca_PC2, marker = marker[chemNum % len(marker)], face = face[chemNum % len(face)], edge = edge[chemNum % len(edge)], name = chem, ax = ax)
    plt.legend()
    return (fig, ax)


def plot_prediction(exp_set,plot_PCA,SVC,PCA_transform=None,grid_size = 200,face = [0, 0, 0, 1],edge = [0, 0, 0, 1],marker = 'o', line = None, ax = None):
    #determine the bounds for plotting
    plot_x_min = np.min(plot_PCA[:, 0]) - 0.1*abs(np.min(plot_PCA[:, 0]))
    plot_x_max = np.max(plot_PCA[:, 0]) + 0.75*abs(np.max(plot_PCA[:, 0]))
    plot_y_min = np.min(plot_PCA[:, 1]) - 0.1*abs(np.min(plot_PCA[:, 1]))
    plot_y_max = np.max(plot_PCA[:, 1]) + 0.1*abs(np.max(plot_PCA[:, 1]))
    
    if ax == None:
        # create plot with bounds from PCA
        fig, ax = create_plot([plot_x_min, plot_x_max, plot_y_min, plot_y_max])
    else:
        fig = ax.figure
    
    ## PLOT PREDICTION REGIONS
    # create grid to draw prediction boundaries
    plot_x_range = np.linspace(plot_x_min,plot_x_max,grid_size)
    plot_y_range = np.linspace(plot_y_min,plot_y_max,grid_size)
    X,Y = np.meshgrid(plot_x_range, plot_y_range)
    if PCA_transform == None:
        SVC_predictions = SVC.predict(np.array([X.ravel(),Y.ravel()]).T)
    else:
        deriv_at_point = PCA_transform(np.array([X.ravel(),Y.ravel()]).T)
        SVC_predictions = SVC.predict(deriv_at_point)

    # set color scheme
    color_dict = {}
    for chem, color in zip(exp_set, face):
        color_dict[chem] = [c for c in color]
        color_dict[chem].append(0.25)

    # sort color scheme
    cont_dict = get_contour_order(exp_set)
    cont_order = cont_dict.values()
    res = dict((v,k) for k,v in get_contour_order(exp_set).items()) # necessary to order colors properly
    colors = [color_dict[res[chem]] for chem in np.sort(cont_order)] # ordered colors
    colors.append([0,0,0,1]) # adding background color

    # reshape the matrix, then perform edge detection to find where to draw the boundaries
    contour = []
    for chem in SVC_predictions:
        contour.append(cont_dict[chem])
    contour = np.array(contour).reshape(X.shape)
    
    # plot regions of predictions
    cont_order.append(-1)
    plt.contourf(plot_x_range,plot_y_range,contour,np.sort(cont_order),colors=colors) #['r','g','b','k']) #
    
    # add contours
    if line != None:
        C_x = scipy.signal.convolve2d(contour,np.array([[1,0],[0,-1]]))
        C_y = scipy.signal.convolve2d(contour,np.array([[0,1],[-1,0]]))
        C = C_x**2 + C_y**2
        C = np.logical_not(C>0)
        
        plt.contour(plot_x_range[:-1],plot_y_range[:-1],C[1:-1,1:-1],colors=contour,linewidths=0.5)
    
    return (fig, ax)


#------------------------------------------------------------#
# SVR Plots
#------------------------------------------------------------#

def calculate_R2(actual,predicted):
    # convert to array
    actual = np.array(actual).reshape(1,-1)
    predicted = np.array(predicted).reshape(1,-1)
    
    # calculate errors
    res_ss = np.sum((predicted-actual)**2)
    total_ss = np.sum((np.mean(actual)-actual)**2)
    
    return 1-res_ss/total_ss


def plot_regression(actual, predicted, c='k'):
    # plots the real vs predicted concentrations
    plt.figure(figsize = (3,5))
    plt.xlabel('Actual Pentane Concentration\n(Mole Fraction)', fontsize = 12)
    plt.ylabel('Predicted Pentane Concentration (Mole Fraction)', fontsize = 12)
    ax = plt.gca()
    ax.tick_params(direction='in', top = True, bottom = True, left = True, right = True, labelsize = 11)
    
    # plot line representing 1:1 correspondence
    plt.plot(np.linspace(0,1,100),np.linspace(0,1,100),'--',c=c,linewidth=1)
    
    # scatter predictions
    plt.scatter(np.array(actual),np.array(predicted),c='k',s=15)
    

#------------------------------------------------------------#
# Plot learning curve for SVC and SVR
#------------------------------------------------------------#

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

#title = r"Learning Curves (SVM, RBF kernel, $\gamma=0.001$)"
## SVC is more expensive so we do a lower number of CV iterations:
#cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
#estimator = SVC(gamma=0.001)
#plot_learning_curve(estimator, title, X, y, (0.7, 1.01), cv=cv, n_jobs=4)

#plt.show()


from sklearn.model_selection import train_test_split

def plot_my_learning_curve(estimator, exp_derivs, exp_concentrations, ylim=(0,1), random_seeds = np.arange(0,100), M = None, test_size = 1/2.5):
    if M == None:
        M = np.arange(1,int(len(exp_derivs)*(1-test_size)))

    train_mean = []
    train_std = []
    test_mean = []
    test_std = []

    for m in M:
        train_score = []
        test_score = []
        for state in random_seeds:
            train_derivs, test_derivs, train_lbl, test_lbl = train_test_split(exp_derivs, exp_concentrations, test_size=test_size,random_state=state)
            derivs = train_derivs[0:m]
            lbls = train_lbl[0:m]
            if len(set(lbls)) < 2:
                continue
            estimator.fit(derivs, lbls)
            predicted = estimator.predict(test_derivs)
        
            train_score.append(estimator.score(derivs, lbls))
            test_score.append(estimator.score(test_derivs, test_lbl))
        train_mean.append(np.mean(train_score, axis=0))
        train_std.append(np.std(train_score, axis=0))
        test_mean.append(np.mean(test_score, axis=0))
        test_std.append(np.std(test_score, axis=0))

    plt.figure()
    plt.ylim(ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.fill_between(M , np.array(train_mean) - np.array(train_std),np.array(train_mean) + np.array(train_std), alpha=0.1,color="r")
    plt.fill_between(M, np.array(test_mean) - np.array(test_std),np.array(test_mean) + np.array(test_std), alpha=0.1, color="g")
    plt.plot(M, train_mean, 'o-', color="r",label="Training score")
    plt.plot(M, test_mean, 'o-', color="g",label="Cross-validation score")
    plt.grid()
    plt.legend(loc="best")

    print("Training accuracy: " + str(train_mean[-1]) + " (" + str(train_std[-1]) + ")")
    print("Validation accuracy: " + str(test_mean[-1]) + " (" + str(test_std[-1]) + ")")
    
    return test_mean, test_std, M
    

#------------------------------------------------------------#
# Plot confusion matrix
#------------------------------------------------------------#

from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred, classes=None, normalize=False, title=None, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred,labels=classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
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

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


#------------------------------------------------------------#
# Phase derivative plots
#------------------------------------------------------------#

import inspect

def plot_deriv(exps, labels = None, norm = True, ax = None, color = None):
    if ax == None:
        fig, ax = plt.subplots()
    
    # convert input to list
    if not type(exps) == list:
        exps = [exps]
    if not type(labels) == list:
        labels = [labels]*len(exps)
       
    for exp,label in zip(exps,labels):
        # determine input type and handle apropriately
        if type(exp) == int: # get experiment, load data, calculate and plot
            exp_file = fra_expt.get_expt(exp)
            data = exp_file.main_spec_data.set_times(600)
            data = data.lower_data_freq(1)

            phase_deriv = data.getPhaseDerivative('spectral',1,smoothing=True,normalize=False)
        elif inspect.isclass(exp): # expect experiment loaded data, calculate and plot
            data = exp.main_spec_data.set_times(600)
            data = data.lower_data_freq(1)

            phase_deriv = data.getPhaseDerivative('spectral',1,smoothing=True,normalize=False)
        else: # expect phase derivative data, just plot
            phase_deriv = exp
        
        # normalize data
        if norm:
            phase_deriv = phase_deriv/np.sqrt(np.sum(phase_deriv**2))
    
        ax.plot(phase_deriv, label = label, color = color)
        
    return ax


#------------------------------------------------------------#
# Raw data plots
#------------------------------------------------------------#

import fra_expt


def plot_river(exp):
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
    plt.imshow(d_array, vmin=vmin, vmax=vmax, interpolation='none', extent=extent, cmap='jet')
    
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
    
    
def plot_spectrum(exp,n):
    # Plot river plot
    fig, ax = plt.subplots()
    #fig = plt.figure(frameon=False)
    #ax = plt.Axes(fig, [0., 0., 1., 1.])
    #fig.add_axes(ax)
    #fig.set_size_inches(5,5)

    # Plot figure (adapted from: fra_plotting.image_plot_2d(std_data,subtle=True, cmap = 'jet') )
    d_array = exp.spectra[n,:]
    plt.xlim((np.min(exp.wavelengths), np.max(exp.wavelengths)))
    plt.ylim((0, 1.1*np.max(d_array)))
    
    plt.plot(exp.wavelengths, d_array)
    
    # Define image labels
    plt.ylabel('Intensity (a.u.)', fontsize=16)
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
  

def plot_FT(exp,f):
    # Plot river plot
    fig, ax = plt.subplots()
    #fig = plt.figure(frameon=False)
    #ax = plt.Axes(fig, [0., 0., 1., 1.])
    #fig.add_axes(ax)
    #fig.set_size_inches(5,5)

    # get data
    FT = exp.get_ft_data('spectral')
    FT_real = np.real(FT[:, f])
    FT_imag = np.imag(FT[:, f])

    # Plot figure (adapted from: fra_plotting.image_plot_2d(std_data,subtle=True, cmap = 'jet') )
    plt.xlim((np.min(exp.times), np.max(exp.times)))
    plt.margins(0, 0.05)
    #plt.ylim((np.min(FT_real)-0.1*abs(np.min(FT_real)), np.max(FT_real)+0.1*abs(np.max(FT_real))))
    
    plt.plot(exp.times, FT_real, color='b')
    plt.plot(exp.times, FT_imag, color='orange')
    
    # Define image labels
    plt.ylabel('Intensity (a.u.)', fontsize=16)
    plt.xlabel('Time (s)', fontsize=16)
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


def plot_phase(exps, labels = None, norm = True, ax = None, color = None):
    if ax == None:
        fig, ax = plt.subplots()
    
    # convert input to list
    if not type(exps) == list:
        exps = [exps]
    if not type(labels) == list:
        labels = [labels]*len(exps)
    
    maximum = 0
    for exp,label in zip(exps,labels):
        # determine input type and handle apropriately
        if type(exp) == int: # get experiment, load data, calculate and plot
            exp_file = fra_expt.get_expt(exp)
            data = exp_file.main_spec_data.set_times(600)
            data = data.lower_data_freq(1)

            phase = data.getPhase('spectral',1,normalize=False)
        elif inspect.isclass(exp): # expect experiment loaded data, calculate and plot
            data = exp.main_spec_data.set_times(600)
            data = data.lower_data_freq(1)

            phase = data.getPhase('spectral',1,normalize=False)
        else: # expect phase derivative data, just plot
            phase = exp
        
        # normalize data
        if norm:
            phase = phase/np.sqrt(np.sum(phase**2))
        
        maximum = np.max([maximum, np.max(phase)])
        plt.plot(phase, label = label, color = color)
    
    if type(exps[0]) == int or inspect.isclass(exps[0]):
        exp = data
    
    # Clean up plot
    ax.margins(y=0)
    plt.xlim((np.min(exp.times), np.max(exp.times)))
    plt.ylim((ax.get_ylim()[0], ax.get_ylim()[1] + 0.05*np.diff(ax.get_ylim())))
    
    # Define image labels
    plt.ylabel('Phase (a.u.)', fontsize=16)
    plt.xlabel('Time (s)', fontsize=16)
    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')
    ax.tick_params(axis='x',colors='black')
    ax.tick_params(axis='y',colors='black')
    
    # Image aspect ratio
    xext, yext = plt.gca().axes.get_xlim(), plt.gca().axes.get_ylim()
    xrange = xext[1] - xext[0]
    yrange = yext[1] - yext[0]
    plt.gca().set_aspect(1 * abs(xrange / yrange)) # This is the line that causes the warnings about unicode

        
def plot_deriv(exps, labels = None, norm = True, ax = None, color = None):
    if ax == None:
        fig, ax = plt.subplots()
    
    # convert input to list
    if not type(exps) == list:
        exps = [exps]
    if not type(labels) == list:
        labels = [labels]*len(exps)
    
    maximum = 0
    for exp,label in zip(exps,labels):
        # determine input type and handle apropriately
        if type(exp) == int: # get experiment, load data, calculate and plot
            exp_file = fra_expt.get_expt(exp)
            data = exp_file.main_spec_data.set_times(600)
            data = data.lower_data_freq(1)

            phase_deriv = data.getPhaseDerivative('spectral',1,smoothing=True,normalize=False)
        elif inspect.isclass(exp): # expect experiment loaded data, calculate and plot
            data = exp.main_spec_data.set_times(600)
            data = data.lower_data_freq(1)

            phase_deriv = data.getPhaseDerivative('spectral',1,smoothing=True,normalize=False)
        else: # expect phase derivative data, just plot
            phase_deriv = exp
        
        # normalize data
        if norm:
            phase_deriv = phase_deriv/np.sqrt(np.sum(phase_deriv**2))
    
        maximum = np.max([maximum, np.max(phase_deriv)])
        plt.plot(phase_deriv, label = label, color = color)
    
    if type(exps[0]) == int or inspect.isclass(exps[0]):
        exp = data
            
        # Clean up plot
        plt.xlim((np.min(exp.times), np.max(exp.times)))
        plt.ylim((0, 1.1*maximum))

        # Define image labels
        plt.ylabel('Phase derivative (a.u.)', fontsize=16)
        plt.xlabel('Time (s)', fontsize=16)
        ax.xaxis.label.set_color('black')
        ax.yaxis.label.set_color('black')
        ax.tick_params(axis='x',colors='black')
        ax.tick_params(axis='y',colors='black')

        # Image aspect ratio
        xext, yext = plt.gca().axes.get_xlim(), plt.gca().axes.get_ylim()
        xrange = xext[1] - xext[0]
        yrange = yext[1] - yext[0]
        plt.gca().set_aspect(1 * abs(xrange / yrange)) # This is the line that causes the warnings about unicode
        
    return ax


