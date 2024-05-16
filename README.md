# EuXFEL-HED-VISAR

SaveVISAR_Images.ipynb and SaveVISAR_downsampleImages.ipynb are jupyter notebooks to be run on the Maxwell cluster at EuXFEL to save the VISAR images corrected for the curvature. The second jupyter notebook saves a 1024x1024 image. The original image is 4096x4096.

The "genLVpair" function looks first for an X-ray train ID when the DiPOLE shutter was OPEN (corresponding to a laser shot) and a train ID  when the DiPOLE shutter was CLOSE (corresponding to the reference image). The function also corrects for the image distortion of the Sydor cameras using the calibration files in the "VISAR_DewarpingCalibration/" folder. 

The "processshot" function extracts automatically some useful values (etalon thicknesses, VPFs, ...). A figure of the reference and shot images for both VISARs is also displayed and the time axis is constructed using the values extracted from the calibration of the sweep window ("dts", "pixDipole_0ns_KEPLER1", "pixDipole_0ns_KEPLER2", "pixXray_KEPLER1", "pixXray_KEPLER2"). These values can be found in "VISAR_DewarpingCalibration/CalibrationLog.txt" for the different sweep windows.

When running the notebooks, a new folder VISAR_Images/ will be automatically created. For each run, the two VISAR images will be saved in the folder folder VISAR_Images/runXXX.
To use the notebook, change the "experimentID" variable to the ID corresponding to your experiment at HED and the "runnumber" variable with the run number you would like to extract. 

NOTE: There are calibration files for each of VISAR window. In practice, the calibration file for the 20 ns window works well for the 50 ns and 10 ns VISAR windows as well. The calibration file for the 20 ns window also corrects for the X-ray fiducial on VISAR 2 ("VISAR_DewarpingCalibration/VISAR_KEPLER2_20nsWindow_p2740_run1222_evt1_polynomial_AdditionalPointForXrayFiducial.cal").
