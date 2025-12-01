# Dionne Group Renishaw Raman Spectrometer Software

This repository is used for working on and collaborating on shared software infrastructure for using, automating and analyzing data from the Dlab renishaw instrument. In this package you will find a

- Library of functions for interfacing with the instrument
- PyQT5 based interface as a supplementary method for operating the Renishaw instrument
- Example scripts for interfacing with the Renishaw instrument
- General documentation for use of the instrument
- Data analysis scripts for performing


Ongoing Projects:
1. Automated Liquid Well Mapping
2. Surface fitting of best focus
    - Track plane of best visual focus and Raman focus for various samples of interest
3. Instructions for calibrating new objectives


Projects in Queue:
1. Addition of fluorescent control with automated lamp and camera control, manual cube movement and control
2. Volumetric Raman measurements
3. Addition of tunable laser controls and calibration routine

Goals of State control:
1. In session storage of 
    - Surface profile
    - Global measurement information



### Installation of WDF into Python Local Environment
1. Navigate to the wdf_python directory
2. Activate your virtual environment using mamba or venv
3. run, pip install -e .
    - This should install the wdf python package into your environment for use with any of the additional packages in the future


This package can then be used in conjungtion with the Utility folder files to extract from WDF files all of the metadata. 
There is also a convenient GUI that can be accessed by running 

python -m wdf.browser 

This can let you take a look at all of the attributes available to you
