import sys
import os
import argparse
import time
from ecm import ECMConnection
from wdf import Wdf


def getZHeightMap():
    # Input 3 points from the edges of the microscope stage
    # Return function params used for calculating and adjusting the z height before measurement
    
    pass

def calculateDrift():
    pass

def selectMeasurementRegions():
    # Use an image or montage to select regions of interest to add to the measurement queue
    pass

def generateMontage():
    pass

def captureImage(location=[],):
    pass

def seriesMeasurement(settings, xyList, measurement_time, timeout_multiple=1.4, print_process = False):
    
    # Open a connection to the WiRE system
    ecm = ECMConnection(settings.connection_url)
    if print_process: print("ECM connection opened")

    # If called for enable a debug mode which displays requests/responses to WiRE
    ecm.debug = settings.debug

    # Create a new measurement on the remote system in the paused state
    handle = ecm.call("Queue.Add", paused=True, monitor=False, remoteString=settings.template)
    if print_process: print("Measurement queued with handle = " + str(handle))

    try:
        # Set the data filename on the remote measurement
        if settings.filename is not None:
            filename = ecm.call("Measurement.SetFilename", handle=handle, filename=settings.filename)
            if print_process: print(f"File name set to '{filename}'")

        # Configure the measurement into a series measurement
        _ = ecm.call("Measurement.SetMap", handle=handle, mapXYPoints={'xy_values':xyList})
        if print_process: print("Series measurement options set")

        # Release the measurement to run on the remote system
        _ = ecm.call("Queue.Continue", handle=handle)
        if print_process: print("Begin data collection")

        # Wait for the measurement to complete
        timeout_time=measurement_time*len(xyList)*timeout_multiple
        status = ecm.wait(handle=handle, timeout=timeout_time)

        # If the trigger handling loop has exited and the measurement is not "COMPLETE"
        # then a measurement has timed out or status is "IDLE" due to aborting"
        if status != "COMPLETE":
            print(f"Timeout after {timeout_time}ms with status {status}. Use --timeout to adjust. Aborting.", file=sys.stderr)
            ecm.call("Queue.Abort", handle=handle)
            time.sleep(0.500)
        else:
            print("Measurement complete")

    finally:
        # Retrieve the currently queued measurement handles to
        # check if the measurement is still in the queue
        handlesPresent = ecm.call("Queue.GetHandles")
        if handlesPresent is not None:
            for queuedHandle in handlesPresent:
                # if the measurement is present remove it from the queue
                if queuedHandle == handle:
                    # Remove the measurement
                    ecm.call("Queue.Remove", handle=handle)
                    print("Measurement with handle = " + str(handle) + " removed")

def generate_grid_params(center, xy_spacing, grid_size, row_major=True, snake=False):
    """ This function is used to generate the inputs for rectangleMap fuction in the Renishaw Wire API
    it takes in the below arguments and returns the array of params for passing through the API call

    Args:
        center (list, tuple)(2): List or Tuple of the center of the rectangle array
        xy_spacing (int/float or list, tuple): Spacing between points either as int/float or pair in list/tuple format if different along x and y
        grid_size (int/float or list, tuple): Num of pots either as int or pair in list/tuple format if different along x and y
        row_major (bool, optional): Input for Renishaw Software on whether to scan in rows or columns. Defaults to True.
        snake (bool, optional): Param for Renishaw software to snake or raster scan. Defaults to False.

    Raises:
        ValueError: Errors to center, xy_spacing or grid_size params

    Returns:
        list: Input list for rectangleMap API call in Renishaw Wire Software
    """
    # Ensure center is a list or tuple
    if not isinstance(center, (list, tuple)) or len(center) != 2:
        raise ValueError("center must be a list or tuple of length 2")

    # Handle xy_spacing: can be a single number or a list/tuple
    if isinstance(xy_spacing, (int, float)):
        x_spacing = y_spacing = xy_spacing
    elif isinstance(xy_spacing, (list, tuple)) and len(xy_spacing) == 2:
        x_spacing, y_spacing = xy_spacing
    else:
        raise ValueError("xy_spacing must be a number or a list/tuple of length 2")

    # Handle grid_size: can be a single number or a list/tuple
    if isinstance(grid_size, int):
        nx = ny = grid_size
    elif isinstance(grid_size, (list, tuple)) and len(grid_size) == 2:
        nx, ny = grid_size
    else:
        raise ValueError("grid_size must be an integer or a list/tuple of length 2")

    center_x, center_y = center

    # Calculate the start points
    x_start = center_x - (x_spacing * (nx - 1) / 2)
    y_start = center_y - (y_spacing * (ny - 1) / 2)

    # Create the input array
    params = [
        x_start,    # double xStart
        y_start,    # double yStart
        x_spacing,  # double xStep
        y_spacing,  # double yStep
        nx,         # number nX
        ny,         # number nY
        row_major,  # boolean row_major
        snake       # boolean snake
    ]

    return params


def mapMeasurement(settings, center, xy_spacing, grid_size, measurement_time, timeout_multiple=1.4, print_process = False, snake=False):
    
    map_settings = generate_grid_params(center, xy_spacing, grid_size)
    
    # Open a connection to the WiRE system
    ecm = ECMConnection(settings.connection_url)
    if print_process: print("ECM connection opened")

    # If called for enable a debug mode which displays requests/responses to WiRE
    ecm.debug = settings.debug

    # Create a new measurement on the remote system in the paused state
    handle = ecm.call("Queue.Add", paused=True, monitor=False, remoteString=settings.template)
    if print_process: print("Measurement queued with handle = " + str(handle))

    try:

        # Set the data filename on the remote measurement
        if settings.filename is not None:
            filename = ecm.call("Measurement.SetFilename", handle=handle, filename=settings.filename)
            if print_process: print(f"File name set to '{filename}'")

        # Configure the measurement into a series measurement
        _ = ecm.call("Measurement.SetMap", handle=handle, rectangleParam=map_settings)
        if print_process: print("Series measurement options set")

        # Release the measurement to run on the remote system
        _ = ecm.call("Queue.Continue", handle=handle)
        if print_process: print("Begin data collection")

        # Wait for the measurement to complete
        timeout_time=measurement_time*map_settings[5]*map_settings[6]*timeout_multiple
        status = ecm.wait(handle=handle, timeout=timeout_time)

        # If the trigger handling loop has exited and the measurement is not "COMPLETE"
        # then a measurement has timed out or status is "IDLE" due to aborting"
        if status != "COMPLETE":
            print(f"Timeout after {timeout_time}ms with status {status}. Use --timeout to adjust. Aborting.", file=sys.stderr)
            ecm.call("Queue.Abort", handle=handle)
            time.sleep(0.500)
        else:
            print("Measurement complete")

    finally:
        # Retrieve the currently queued measurement handles to
        # check if the measurement is still in the queue
        handlesPresent = ecm.call("Queue.GetHandles")
        if handlesPresent is not None:
            for queuedHandle in handlesPresent:
                # if the measurement is present remove it from the queue
                if queuedHandle == handle:
                    # Remove the measurement
                    ecm.call("Queue.Remove", handle=handle)
                    print("Measurement with handle = " + str(handle) + " removed")