The Sniffing-Sensor contains the following folders and files: 

	- Archived Notebooks: 
		Contains notebook and .py files which were used in testing, or in developing certain functionality for the main 
		scripts in the parent folder.

	- Database
		Folder containing the sqlite database which records the relevant information for each experiment, the procedures, 
		and the chemicals used in testing.

	- GUI
		Folder containing files used to make a graphical user interface for the python script. A more user-friendly 
		experience for those who would rather not interact with a python notebook. Not completed, and some functionality 
		is missing.

	- Important Data Images
		Folder containing the most important analysis images for each procedure. The phase comparison, confusion matrix,
		and regression plots can be found. The plots are saved directly from the [MAIN] Phase Comparison script.

	- [MAIN] Machine Learning.ipynb
		Python Notebook used to train SVC and SVR models. More specific information can be found in the notebook itself.

	- [MAIN] Phase Comparison.ipynb
		Python Notebook used to generate plots which compare the phases of different chemicals averaged out over multiple
		runs. More specific information can be found in the notebook, itself.

	- [MAIN] Sniffing Procedures.ipynb
		Python Notebook used to run experiments; the script controls the spectrometer, the teensy board, and writes 
		directly into the sqlite database. It takes an excel file as an input to tell it what sniffing procedures 
		to run and with what chemicals. More specific information can be found in the notebook, itself.

	- sniffing_functions.ipynb
		Python file containing the necessary functions used by the notebooks in this folder. It contains functions to
		control the spectrometer, control valves, generate plots, load data, and process data. The functions can be modified
		for a chosen application.