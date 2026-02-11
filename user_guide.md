# Intended workflow

As we are designing this package to be usable with KSAT systems, specifically relying on the dataframes given by Azure. 
What is the effect of this? At point of initialization we know the measurements, timestamps, and pass indexes.
What do we need to set up? 
The jacobian of all our measurements, for now just focus on Doppler, Azimuth and Elevation.

For all measurements, we need tags, like what station it was collected from, and what the measurement type was.

The definition of the state vector will be rather tricky. The dynamic StateDefinition describes what orbital elements to solve for. It might also make sense here to define our "sensor parameter budget" This is the number of elements in the state vector which are assigned to the sensor. Ideally these parameters should be gotten automatically when we define our sensor types.
For instance for the doppler measurements, we want to define N parameters for N passes, and optionally +1 if we want to fit the center frequency. 
This defines our "local state vector", for each sensor, these are then concatinated with the orbital elements to consruct the full state vector for the measurement system. This should describe how we expect the measurements to look for different state parameters.

What might be useful:

Use class structures for defining the state vector and sensors. The program flow can however be better handled by functions.

What is currently easy to do in dSGP4: 

* Loading TLEs
* Propagate
* batch-propagate
* Covariance propagation.
* MLdSGP4

What is currently hard to do in dSGP4:

* Defining the state vector
* Getting station TEME coordinates
* Train/Tune neural networks with dSGP4 as a layer
* Keeping the computational graph intact when updating TLE

What should we implement?

* Batched ground station propagation: takes the timestamps and coordinates for the different stations
* dataclass for ground station: holds WGS84 coordinates
* Better update function: takes the previous TLE object, and has parameters for all 9 orbital elements. if left None we simply leave it static

What would be our idea?

* Each Sensor is defined simply as a function, it.
* For doppler, we take take as argument the tsincex3x2 tensor from dSGP4, alongside the center frequency, and biases and bias selection matrix.

For instance the workflow would be 
'''python

import dSGP4
import torch
import diffod

TLE = dsgp4.tle.TLE(...)
dsgp4.initialize(TLE)

stations = diffod.gse.groundStation(list of stations)

TEME_satellite = dsgp4.propagate(TLE, torch.tensor(1000))
TEME_stations = diffod.gse.propagate(stations, list of times) # returns a single Nx3x2 tensor of stations listed according to the different timestamps

def DopplerEffect(TEME_sat, TEME_station, center_freq):

  return # Nx1 tensor of all doppler measurements

def LinearBias(measurements, statevector, bias_selection_matrix):

  We then simply return: M + A @ x, here M are the measurements, x is the state vector, and A is the sparse selection matrix

# When using the package, it might be important to have a "StateVector" object. Basic usage might look like this:

statevector = diffod.stateVector(mean_anomaly: true, mean_motion: true, argpo: true)

# Then we can optionally define extra parameters:

biasMap = torch.zeros(1000)
biasMap[500:] = 1 # the first 500 measurements have index 0, the next 500 have index 1. this describes how the biases relate to the measurements.

statevector.addLinearBias(map=biasMap)
# What should happen internally now is we perform a one-hot encoding of the biasmap, as this may be strings. We then count how many paramters to add.
# In this case it would be 2, the state vector is then increased to have 2 more parameters. We then construct a sparse and boolean selection matrix.
# How this would be used might be:

x = statevector.flatten() # Converts from a statevector object to a torch tensor.
biasSelection = statevector.LinearBiasSelector # shape is 1000x5

Then to compute the Jacobian, all we do is to define a system transfer function

def systemFunction(x, timestamps):

  # We need to define some simple update function that takes the previous TLE, our state vector, and some relation map to know what elements to update
  # We should probably do this with a dictionary that stores "mean_anomaly":0, "mean_motion":1, i.e a map of the state vector to the correct state vector element.

  TLE = diffod.tle.update(TLE, statevector, x)

  TEME_satellite = dsgp4.propagate(TLE, timestamps)
  
  doppler_measurements = DopplerEffect(TEME_satellite, TEME_station, center freq)
  
  biased_doppler = LinearBias(doppler_measurements, x, biasSelection)
  
  return biased_doppler

We should now be able to compute the jacobian using the simple jacfwd method.

'''
