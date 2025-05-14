# EuXFEL-HED-VISAR

Please cite XX if you are using the VISAR system at the HED-HiBEF instrument at the EuXFEL. 

"visar_V3.py" contains a serie of classes used to extract and correct the curvature of the VISAR images. On a general level, the VISAR images corresponding to a laser shot are filtered using a PPU signal from the DiPOLE-100X laser defined in the class "DipolePPU". A "CalibrationData" class is defined to read the information in the "visar_calibration_values.toml" file. The ".toml" contains information about the calibration files to use for the curvature correction. These files are found in the folder "DewarpingCalibrationFiles". The ".toml" file also includes values used to construct the time axis and the spatial axis. NOTE: Here t = 0 ns is defined as the rising edge of the DiPOLE-100X laser pulse. The class "VISAR" extracts useful information for the data acquisition system (DAQ), corrects for the curvature of the images and plot the results with a spatial and temporal axis. The class "SOP" is similar to the "VISAR" class but targets the Streaked Optical Pyrometry system.

"visar_V3.py" automatically pulls out useful information regarding the long pulse laser (DiPOLE-100X) and the VISAR system. It creates three folders in the root directory called "./VISAR_TIFF_pXXXX", "./VISAR_Summary_pXXXX", "./VISAR_HDF5_pXXXX" with XXXX the proposal number. When running "visar_V3.py", a folder "/r_YYYY" (with YYYY the run number) is created in "./VISAR_TIFF_pXXXX" and "./VISAR_Summary_pXXXX". In "./VISAR_TIFF_pXXXX/r_YYYY", the VISAR images corrected for the curvature for each of the streak cameras are saved. Both the 2048x2048 images and 4096x4096 images. In "./VISAR_Summary_pXXXX/r_YYYY", a .png image of the result of the curvature correction for each streak cameras are saved. The image includes a calibrated temporal and spatial axis. The timing of the X-ray pulse is shown with a vertical purple line. In "./VISAR_HDF5_pXXXX", the script will generate a .hdf5 file which gathers information regarding DiPOLE and all the streak cameras (time axis, spatial axis, etalon thicknesses, etalon sensitivities, correct images, ...). NOTE: the script assumes that the hdf5 does not already exist. If it does, the existing hdf5 file should be deleted first. 


VISAR_Save.ipynb is a jupyter notebook to be run on the Maxwell cluster at the EuXFEL to correct for the curvature and save the VISAR images. It creates objects defined in "visar_V3.py" file.

The "Calibration" folder contains jupyter notebooks used to generate the temporal and spatial axis.

The "DewarpingCalibration" folder contains jupyter notebooks used to generate the file used for the correction of the curvature. 
