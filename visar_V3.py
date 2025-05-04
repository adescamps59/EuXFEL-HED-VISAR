from enum import Enum
from functools import cached_property, lru_cache
from pathlib import Path
import os
import tifffile as tif
import cv2
import numpy as np
import toml
from extra_data import by_id
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import h5py

import extra_data as ed


VISAR_DEVICES = {
    'KEPLER1': {
        'arm': 'COMP_HED_VISAR/MDL/VISAR_SENSITIVITY_ARM_1',
        'trigger': 'HED_EXP_VISAR/TSYS/ARM_1_TRIG',
        'detector': 'HED_SYDOR_TEST/CAM/KEPLER_1:daqOutput',
        'ctrl': 'HED_SYDOR_TEST/CTRL/CONTROL_UNIT_1',
    },
    'KEPLER2': {
        'arm': 'COMP_HED_VISAR/MDL/VISAR_SENSITIVITY_ARM_2',
        'trigger': 'HED_EXP_VISAR/TSYS/ARM_2_TRIG',
        'detector': 'HED_SYDOR_TEST/CAM/KEPLER_2:daqOutput',
        'ctrl': 'HED_SYDOR_TEST/CTRL/CONTROL_UNIT_2',
    },
    'VISAR_1w': {
        'arm': 'COMP_HED_VISAR/MDL/VISAR_SENSITIVITY_ARM_3',
        'trigger': 'HED_EXP_VISAR/TSYS/ARM_3_TRIG',
        'detector': 'HED_EXP_VISAR/EXP/ARM_3_STREAK:daqOutput',
        'ctrl': 'HED_EXP_VISAR/EXP/ARM_3_STREAK',
    },
}


class DipolePPU(Enum):
    OPEN = np.uint32(1)
    CLOSED = np.uint(0)


def remap(image, source, target):
    return cv2.remap(image, source, target, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)


class CalibrationData:
    def __init__(self, visar, file_path=None):
        self.visar = visar

        if file_path is not None:
            config = toml.load(file_path)
            self.config = config[self.visar.name]
            self.config.update(config['global'])
        else:
            self.config = {}

        
    @cached_property
    def dx(self):
        """Length per pixel in µm
        """
        return self.config['dx']

    @cached_property
    def dipole_zero(self):
        """Dipole position at 0 ns delay, 0 ns sweep delay
        """
        pixel_offset = self.config['pixDipole_0ns'][f'{self.visar.sweep_time}ns']
        return pixel_offset

    @cached_property
    def fel_zero(self):
        """Xray position at 0 ns delay, 0 ns sweep delay
        """
        return self.config['pixXray']

    @cached_property
    def time_polynomial(self):
        """Time per pixel in ns
        """
        """Pad with leading 0 because there is no intercept for the time axis
        """

        return np.hstack(([0], self.config['timeAxisPolynomial'][f'{self.visar.sweep_time}ns']))

    @cached_property
    def reference_trigger_delay(self):
        """Trigger time in ns for DiPOLE at 0 ns
        """
        ref = self.config['positionTrigger_ref'][f'{self.visar.sweep_time}ns']
        return ref

    def map(self) -> tuple[np.ndarray, np.ndarray]:
        """Return input and output transformation maps
        """

        tr_map_file = self.config['transformationMaps'][f'{self.visar.sweep_time}ns']
        file_path = Path(self.config['dirTransformationMaps']) / tr_map_file
        coords = np.loadtxt(file_path, delimiter=',')
        target = coords[..., 2:]
        source = coords[..., :2]

        y, x = self.visar.data.entry_shape
        grid_1, grid_2 = np.mgrid[:y, :x]
        grid_z = griddata(target, source, (grid_1, grid_2), method='linear')
        map_1 = grid_z[..., 1].astype(np.float32)
        map_2 = grid_z[..., 0].astype(np.float32)

        return map_1, map_2

            


class VISAR:

    SWEEP_SPEED = {1: 50, 2: 20, 3: 10, 4: 5, 5: 1, 6: 100}
    

    def __init__(self, run, name='KEPLER1', cal_file=None, proposalNumber=None, runNumber=None):
        self.name = name
        visar = VISAR_DEVICES[name]

        self.run = run
        self.arm = run[visar['arm']]
        self.trigger = run[visar['trigger']]
        self.detector = run[visar['detector']]
        self.ctrl = run[visar['ctrl']]

        self.cal = CalibrationData(self, cal_file)
        self.proposalNumber = proposalNumber
        self.runNumber = runNumber

    def __repr__(self):
        return f'<{type(self).__name__} {self.name}>'

    # def _as_single_value(self, kd):
    #     value = kd.as_single_value()
    #     if value.is_integer():
    #         value = int(value)
    #     return Quantity(value, kd.units)

    def as_dict(self):
        ...

    def info(self):
        """Print information about the VISAR component
        """
        print(self.format())

    def format(self, compact=False):
        """Format information about the VISAR component.
        """
        self.run_str = f'p{self.proposalNumber}, r{self.runNumber}'
        info_str = f'{self.name} properties for {self.run_str}:\n'

        if compact:
            return f'{self.name}, {self.run_str}'


    @cached_property
    def etalon_thickness(self):
        """Etalon thickness in mm
        """
        return self.arm['etalonThickness'][by_id[[self.shot()[0]]]].ndarray()[0]

    @cached_property
    def motor_displacement(self):
        """Motor displacement interferometer etalon arm in mm
        """
        return self.arm['motorDisplacement'][by_id[[self.shot()[0]]]].ndarray()[0]

    @cached_property
    def sensitivity(self):
        """Sensitivity VISAR in m/s
        """
        return self.arm['sensitivity'][by_id[[self.shot()[0]]]].ndarray()[0]
    
    @cached_property
    def temporal_delay(self):
        """Temporal delay of etalon in ns
        """
        return self.arm['temporalDelay'][by_id[[self.shot()[0]]]].ndarray()[0]

    @cached_property
    def sweep_delay(self):
        """Sweep window delay in ns, positive looks at later time
        """
        for key in ['actualDelay', 'actualPosition']:
            if key in self.trigger:
                return self.trigger[key][by_id[self.shot()[1]]].ndarray() - self.cal.reference_trigger_delay

    @cached_property
    def sweep_time(self):
        """Sweep window time in ns for the refernce shot. This assumes the sweep window is not changing during the run!
        """
        if self.name == 'VISAR_1w':
            ss_string = self.run.get_run_value("HED_EXP_VISAR/EXP/ARM_3_STREAK", "timeRange")
            return ss_string.split(' ')[0]
        else:
            ss = self.ctrl['sweepSpeed.value']
            return self.SWEEP_SPEED[ss[by_id[[self.shot()[0]]]].ndarray()[0]]

    @cached_property
    def dipole_energy(self):
        """DiPOLE energy at 2w in J
        """
        energy = self.run['APP_DIPOLE/MDL/DIPOLE_DIAGNOSTIC', 'energy2W']
        
        return energy[by_id[self.shot()[1]]].ndarray()

    @cached_property
    def dipole_delay(self):
        """DiPOLE delay in ns
        """
        delay = self.run['APP_DIPOLE/MDL/DIPOLE_TIMING', 'actualPosition']
        return delay[by_id[self.shot()[1]]].ndarray()
        
    # @cached_property
    def dipole_trace(self):
        """DiPOLE temporal profile
        """
        scope_trace = self.run['HED_PLAYGROUND/SCOPE/TEXTRONIX_TEST:output', 'ch2.corrected']
        print('scope_trace', scope_trace.drop_empty_trains().train_id_coordinates())
        # scope_trace = scope_trace[by_id[self.shot()[1]-1]].ndarray()
        scope_trace = [scope_trace[5, :]]
        # scope_trace = scope_trace[by_id[scope_trace.drop_empty_trains().train_id_coordinates()]].ndarray()
        # print('tt', self.shot())
        # print('ttt', scope_trace.shape)

        
        # return scope_trace
        digitizer_max = np.max(np.abs(np.unique(scope_trace[:, :10000], axis = 1)), axis = 1)
        print('digitizer_max', digitizer_max)
        dipole_trace_idx = [np.where(st > dm)[0] for (st, dm) in zip(scope_trace, digitizer_max)]

        print('dipole_trace_idx', dipole_trace_idx)
        dt = 0.1 # 200 ps/sample
        time_axis_trace = [(np.arange(dti[0]-50, dti[-1]+50) - dti[0])*dt for dti in dipole_trace_idx]
    
        dipole_duration = [(dti[-1] - dti[0])*dt*1e-9 for dti in dipole_trace_idx]
        energyTest = np.array([self.dipole_energy, self.dipole_energy])
        power_sacling = [energy / (np.sum(st[dti[0]:dti[-1]]) * dd) for (energy, st, dti, dd) in zip(energyTest, scope_trace, dipole_trace_idx, dipole_duration)]
        
        power_trace = [st[dti[0]-50:dti[-1]+50]*ps for (st, dti, ps) in zip(scope_trace, dipole_trace_idx, power_sacling)]
        return time_axis_trace, power_trace
        # return scope_trace


    @property
    def data(self):
        return self.detector['data.image.pixels']

    @lru_cache()
    def shot(self):
        """Get train ID of data with open PPU.

        If reference is True, return the first data with closed PPU instead.
        """
        # train ID with data in the run
        tids = self.data.drop_empty_trains().train_id_coordinates()
        print('tids', tids)
        # ------------------------------------------------------------ ##
        # To account for offset of 2 in trainIDs in KEPLER2 
        # THIS VALUE CAN CHANGE AND IS NOT PREDICTABLE
        # Try different values until you find the image with DiPOLE ON
        if self.name == 'KEPLER1':
            tids -= 2
        # ----------------------------------------------------------- ##
        
        # ------------------------------------------------------------ ##
        # To account for offset of 2 in trainIDs in KEPLER2 
        # THIS VALUE CAN CHANGE AND IS NOT PREDICTABLE
        # Try different values until you find the image with DiPOLE ON
        if self.name == 'KEPLER2':
            tids -= 2
        # ----------------------------------------------------------- ##

        # ------------------------------------------------------------ ##
        # To account for offset of 2 in trainIDs in KEPLER2 
        # THIS VALUE CAN CHANGE AND IS NOT PREDICTABLE
        # Try different values until you find the image with DiPOLE ON
        if self.name == 'VISAR_1w':
            tids -= 2
        # ----------------------------------------------------------- ##
        
        if tids.size == 0:
            return  # there's not data in this run
        
        ppu_open = self.run['HED_HPLAS_HET/SHUTTER/DIPOLE_PPU', 'isOpened.value'].xarray().where(lambda x: x == DipolePPU.OPEN.value, drop=True)
        shot_ids = np.intersect1d(ppu_open.trainId, tids)
    
        
        print('ppu_open', ppu_open.trainId)
        print('tids', tids)
        # Reference -- return the first data point with closed ppu
        for tid in tids:
            if tid not in shot_ids:
                ref_id = int(tid)
                
                break
            else:
                return  # no data with closed ppu


        if shot_ids.size == 0:
            return  # no data with open ppu in this run
        # ref_id = shot_ids[0]
        print(ref_id+2, shot_ids+2)

        # ------------------------------------------------------------ ##
        # The offset in trainID introduced earlier needs to be corrected 
        # when accessing the images -- the offset is find the trainID of the 
        # streak camera when DiPOLE is ON
        # e.g.
        # Above we have an offset of -2
        # We need to remove this offset when accessing the VISAR images
        # We add an offset of +2 below
        # ----------------------------------------------------------- ##

        if self.name == 'KEPLER1':
            return ref_id+2, shot_ids+2
        
        if self.name == 'KEPLER2':
            return ref_id+2, shot_ids+2
        else:
            return ref_id+2, shot_ids+2

    @lru_cache()
    def frame(self):
        
        if self.name != 'VISAR_1w':
            source, target = self.cal.map()
            ref_id, shot_ids = self.shot()
            if ref_id == None or len(shot_ids) == 0:
                return
            ref_frame = self.data[by_id[[ref_id]]].ndarray().squeeze()

            ref_frame = cv2.rotate(ref_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            ref_corrected = np.fliplr(remap(ref_frame, source, target))


            shot_frame = self.data[by_id[shot_ids]].ndarray()
            print('shot_frame', shot_frame.shape)
            shot_corrected = np.array([np.fliplr(remap(cv2.rotate(f, cv2.ROTATE_90_COUNTERCLOCKWISE), source, target)) for f in shot_frame])

            return ref_corrected, shot_corrected
        
        else:
            # 1w VISAR is 2x2 binned -- need to upscale it full size to use the time calibration
            ref_id, shot_ids = self.shot()
            print(ref_id, shot_ids)
            if ref_id == None or len(shot_ids) == 0:
                return
            ref_frame = self.data[by_id[[ref_id]]].ndarray().squeeze()
            print(ref_frame.shape)
            ref_corrected = np.flipud(cv2.rotate(cv2.resize(ref_frame,
                                                 (ref_frame.shape[0]*2, ref_frame.shape[1]*2),
                                                  interpolation=cv2.INTER_CUBIC),
                                                  cv2.ROTATE_90_COUNTERCLOCKWISE))

            
            shot_frame = self.data[by_id[shot_ids]].ndarray()
                        
            shot_corrected = np.array([np.flipud(cv2.rotate(cv2.resize(f,
                                                 (ref_frame.shape[0]*2, ref_frame.shape[1]*2),
                                                  interpolation=cv2.INTER_CUBIC),
                                                  cv2.ROTATE_90_COUNTERCLOCKWISE)) for f in shot_frame])
            return ref_corrected, shot_corrected
        


    def plot(self, vmin = None, vmax = None, ax=None):
        print(f'VISAR_TIFF_p{self.proposalNumber}')
        if f'VISAR_TIFF_p{self.proposalNumber}' not in os.listdir('./'):
            os.mkdir(f'VISAR_TIFF_p{self.proposalNumber}')

        if os.path.isdir(f'./VISAR_TIFF_p{self.proposalNumber}/r_{self.runNumber}'):
            pass
        else:
            os.mkdir(f'./VISAR_TIFF_p{self.proposalNumber}/r_{self.runNumber}')
            
        
        if f'VISAR_Summary_p{self.proposalNumber}' not in os.listdir('./'):
            os.mkdir(f'./VISAR_Summary_p{self.proposalNumber}/')
            
        if os.path.isdir(f'./VISAR_Summary_p{self.proposalNumber}/r_{self.runNumber}'):
            pass
        else:
            os.mkdir(f'./VISAR_Summary_p{self.proposalNumber}/r_{self.runNumber}')

        print(self.shot())
        ref_id, shot_ids = self.shot()
        
        ref_data, shot_data = self.frame()

        ## Save reference tif image
        tif.imwrite(f'./VISAR_TIFF_p{self.proposalNumber}/r_{self.runNumber}/{self.name}_r{self.runNumber}_ref.tif', ref_data)
        if self.name == 'KEPLER1' or self.name == 'KEPLER2':
            print(ref_data.shape)
            tif.imwrite(f'./VISAR_TIFF_p{self.proposalNumber}/r_{self.runNumber}/{self.name}_r{self.runNumber}_ref_downsample.tif', cv2.resize(ref_data, (int(ref_data.shape[0]/2), int(ref_data.shape[1]/2)), interpolation=cv2.INTER_CUBIC))
        

        
        self.time_axis_shot = []
        self.space_axis_shot = []
        self.Xdelay_shot = []
        
        for idx, (shot_id, data, dipole_delay, dipole_energy, sweep_delay) in enumerate(zip(shot_ids, shot_data, self.dipole_delay, self.dipole_energy, self.sweep_delay)):


            time_conversion = np.poly1d(self.cal.time_polynomial[::-1])
            time_axis = time_conversion(np.arange(data.shape[1]))

            offset = time_conversion(self.cal.dipole_zero) + dipole_delay - sweep_delay
            # print('offset', offset)
            time_axis -= offset
            self.time_axis_shot.append(time_axis)          

        
            space_axis = np.arange(ref_data.shape[0]) * self.cal.dx
            space_axis -= space_axis.mean()
            self.space_axis_shot.append(space_axis)
            
            # Xdelay = time_conversion(self.cal.fel_zero) - time_conversion(self.cal.dipole_zero) - dipole_delay - sweep_delay
            Xdelay = time_conversion(self.cal.fel_zero) - time_conversion(self.cal.dipole_zero) - dipole_delay

            # print('tt', time_conversion(self.cal.fel_zero) - time_conversion(self.cal.dipole_zero))
            # print('test', time_axis[int(self.cal.fel_zero)])
            self.Xdelay_shot.append(Xdelay)
            plt.figure()
            fig, ax = plt.subplots(figsize=(9, 5))

            tid_str = f'{"SHOT"}, tid:{shot_id}'
            ax.set_title(f'{self.format(compact=True)}, {tid_str}')
            ax.set_ylabel(f'Distance [µm]')
            ax.set_xlabel(f'Time [ns]')


            im = ax.imshow(data, extent=[time_axis[0], time_axis[-1], space_axis[0], space_axis[-1]], cmap='binary', vmin=vmin, vmax=vmax)
            ax.vlines(
                Xdelay,
                ymin=space_axis[0],
                ymax=space_axis[-1],
                linestyles='-',
                lw=2,
                color='purple',
                alpha=1,
            )

            ys, xs = np.where(data > 0)
            ax.set_xlim(xmin=time_axis[xs.min()], xmax=time_axis[xs.max()])
            ax.set_ylim(ymin=-space_axis[ys.max()], ymax=-space_axis[ys.min()])
            ax.set_aspect('auto')

            from matplotlib.ticker import MultipleLocator
            ax.xaxis.set_major_locator(MultipleLocator(5))
            ax.xaxis.set_minor_locator(MultipleLocator(1))
            ax.yaxis.set_major_locator(MultipleLocator(200))
            ax.yaxis.set_minor_locator(MultipleLocator(100))
            ax.grid(which='major', color='k', linestyle = '--', linewidth=2, alpha = 0.5)
            ax.grid(which='minor', color='k', linestyle=':', linewidth=1, alpha = 1)

            fig.colorbar(im)
            fig.tight_layout()
            plt.savefig(f'./VISAR_Summary_p{self.proposalNumber}/r_{self.runNumber}/{self.name}_{idx}.png')
            plt.show()

            ## Save ref shots tif image
            tif.imwrite(f'./VISAR_TIFF_p{self.proposalNumber}/r_{self.runNumber}/{self.name}_r{self.runNumber}_shot_{idx}.tif', data)
            if self.name == 'KEPLER1' or self.name == 'KEPLER2':
                tif.imwrite(f'./VISAR_TIFF_p{self.proposalNumber}/r_{self.runNumber}/{self.name}_r{self.runNumber}_shot_{idx}_downsample.tif', cv2.resize(data, (int(data.shape[0]/2), int(data.shape[1]/2)), interpolation=cv2.INTER_CUBIC))
            plt.close(fig)

        return ax

    def save(self):


            
        ref_id, shot_ids = self.shot()
        ref_data, shot_data = self.frame()
        


        if f'VISAR_HDF5_p{self.proposalNumber}' not in os.listdir('./'):
            os.mkdir(f'./VISAR_HDF5_p{self.proposalNumber}/')

        # if f'VISAR_p{self.proposal}_r{self.runNumber}.h5' in os.listdir('./VISAR_HDF5_p8040'):
        #     os.remove(f'./VISAR_HDF5/VISAR_p{self.proposal}_r{self.runNumber}.h5')
        
        with h5py.File(f'./VISAR_HDF5_p{self.proposalNumber}/VISAR_p{self.proposalNumber}_r{self.runNumber}.h5', 'a') as hdf:
            if 'DiPOLE' not in hdf:
                # DiPOLE_trace_timeAxis, DiPOLE_trace = self.dipole_trace()

                D = hdf.create_group('DiPOLE')
                D.create_dataset('DiPOLE energy [J]', data = self.dipole_energy)      
                D.create_dataset('DiPOLE delay [ns]', data = self.dipole_delay)   
                # D.create_dataset('Time axis DiPOLE profile [ns]', data = DiPOLE_trace_timeAxis)  
                # D.create_dataset('DiPOLE profile [W]', data = DiPOLE_trace)  


            VISAR = hdf.create_group(f'{self.name}')
            
            ref = VISAR.create_group('Reference')
            ref.create_dataset('Reference', data=ref_data)
            ref.create_dataset('Train ID', data=ref_id)
            
            shot = VISAR.create_group('Shot')
            shot.create_dataset('Corrected images', data = shot_data)
            shot.create_dataset('Time axis', data = self.time_axis_shot)
            shot.create_dataset('Space axis', data = self.space_axis_shot)
            shot.create_dataset('Drive pixel t0', data = self.cal.dipole_zero)  
            shot.create_dataset('Sensitivity [km.s^-1]', data = self.sensitivity/1e3)
            shot.create_dataset('Etalon thickness [mm]', data = self.etalon_thickness)
            shot.create_dataset('Sweep window [ns]', data = self.sweep_time)                  
            shot.create_dataset('Sweep delay [ns]', data = self.sweep_delay)           
            shot.create_dataset('Difference X-Drive [ns]', data = self.Xdelay_shot)
            shot.create_dataset('Train ID', data = shot_ids)
            shot.create_dataset('Etalon delay [ns]', data=self.temporal_delay)
            
            # shot.create_dataset('Size X-ray [µm]', data = ...)
            # shot.create_dataset('Shock breakout time [ns]', data = self.sweep_delay)        
            
            
SOP_DEVICES = {
    'SOP': {
        'trigger': 'HED_EXP_VISAR/TSYS/SOP_TRIG',
        'detector': 'HED_EXP_VISAR/EXP/SOP_STREAK:daqOutput',
        'ctrl': 'HED_EXP_VISAR/EXP/SOP_STREAK',
    },
}
            
class SOP:
    

    def __init__(self, run, cal_file=None, proposalNumber=None, runNumber=None):
        self.name = 'SOP'
        sop = SOP_DEVICES[self.name]

        self.run = run
        self.trigger = run[sop['trigger']]
        self.detector = run[sop['detector']]
        self.ctrl = run[sop['ctrl']]

        self.cal = CalibrationData(self, cal_file)
        self.proposalNumber = proposalNumber
        self.runNumber = runNumber

    def __repr__(self):
        return f'<{type(self).__name__} {self.name}>'

    # def _as_single_value(self, kd):
    #     value = kd.as_single_value()
    #     if value.is_integer():
    #         value = int(value)
    #     return Quantity(value, kd.units)

    def as_dict(self):
        ...

    def info(self):
        """Print information about the VISAR component
        """
        print(self.format())

    def format(self, compact=False):
        """Format information about the VISAR component.
        """
        self.meta = self.run.run_metadata()
        self.run_str = f'p{self.meta.get("proposalNumber", "?"):06}, r{self.meta.get("runNumber", "?"):04}'
        info_str = f'{self.name} properties for {self.run_str}:\n'

        if compact:
            return f'{self.name}, {self.run_str}'


    @cached_property
    def sweep_delay(self):
        """Sweep window delay in ns, positive looks at later time
        """
        for key in ['actualDelay', 'actualPosition']:
            if key in self.trigger:
                return self.trigger[key][by_id[self.shot()[1]]].ndarray() - self.cal.reference_trigger_delay

    @cached_property
    def sweep_time(self):
        """Sweep window time in ns for the refernce shot. This assumes the sweep window is not changing during the run!
        """
        ss_string = self.run.get_run_value("HED_EXP_VISAR/EXP/ARM_3_STREAK", "timeRange")
        return ss_string.split(' ')[0]

    @cached_property
    def dipole_energy(self):
        """DiPOLE energy at 2w in J
        """
        energy = self.run['APP_DIPOLE/MDL/DIPOLE_DIAGNOSTIC', 'energy2W']
        return energy[by_id[self.shot()[1]]].ndarray()

    @cached_property
    def dipole_delay(self):
        """DiPOLE delay in ns
        """
        delay = self.run['APP_DIPOLE/MDL/DIPOLE_TIMING', 'actualPosition']
        return delay[by_id[self.shot()[1]]].ndarray()


    @property
    def data(self):
        return self.detector['data.image.pixels']

    @lru_cache()
    def shot(self):
        """Get train ID of data with open PPU.

        If reference is True, return the first data with closed PPU instead.
        """
        # train ID with data in the run
        tids = self.data.drop_empty_trains().train_id_coordinates()

        
        if tids.size == 0:
            return  # there's not data in this run
        
        ppu_open = self.run['HED_HPLAS_HET/SHUTTER/DIPOLE_PPU', 'isOpened.value'].xarray().where(lambda x: x == DipolePPU.OPEN.value, drop=True)
        shot_ids = np.intersect1d(ppu_open.trainId, tids)
    
        
        # Reference -- return the first data point with closed ppu
        for tid in tids:
            if tid not in shot_ids:
                ref_id = int(tid)
                
                break
            else:
                return  # no data with closed ppu

        if shot_ids.size == 0:
            return  # no data with open ppu in this run
        # ref_id = shot_ids[0]
        
        if self.name == 'KEPLER1':
            return ref_id+2, shot_ids+2
        
        if self.name == 'KEPLER2':
            return ref_id+2, shot_ids+2
        else:
            return ref_id, shot_ids

    @lru_cache()
    def frame(self):

        # SOP is 2x2 binned -- need to upscale it full size to use the time calibration
        ref_id, shot_ids = self.shot()
        if ref_id == None or len(shot_ids) == 0:
            return
        ref_frame = self.data[by_id[[ref_id]]].ndarray().squeeze()
        ref_corrected = cv2.rotate(cv2.resize(ref_frame,
                                             (ref_frame.shape[1]*2, ref_frame.shape[0]*2),
                                              interpolation=cv2.INTER_CUBIC),
                                              cv2.ROTATE_90_COUNTERCLOCKWISE)


        shot_frame = self.data[by_id[shot_ids]].ndarray()

        shot_corrected = np.array([cv2.rotate(cv2.resize(f,
                                             (ref_frame.shape[1]*2, ref_frame.shape[0]*2),
                                              interpolation=cv2.INTER_CUBIC),
                                              cv2.ROTATE_90_COUNTERCLOCKWISE) for f in shot_frame])
        return ref_corrected, shot_corrected
        


    def plot(self, vmin = None, vmax = None, ax=None):

        if f'SOP_TIFF_p{self.proposalNumber}' not in os.listdir('./'):
            os.mkdir(f'SOP_TIFF_p{self.proposalNumber}')

        if os.path.isdir(f'./SOP_TIFF_p{self.proposalNumber}/r_{self.runNumber}'):
            pass
        else:
            os.mkdir(f'./SOP_TIFF_p{self.proposalNumber}/r_{self.runNumber}')
            
        
        if f'SOP_Summary_p{self.proposalNumber}' not in os.listdir('./'):
            os.mkdir(f'./SOP_Summary_p{self.proposalNumber}/')
            
        if os.path.isdir(f'./SOP_Summary_p{self.proposalNumber}/r_{self.runNumber}'):
            pass
        else:
            os.mkdir(f'./SOP_Summary_p{self.proposalNumber}/r_{self.runNumber}')
            
        ref_id, shot_ids = self.shot()
        ref_data, shot_data = self.frame()

        ## Save reference tif image
        tif.imwrite(f'./SOP_TIFF_p{self.proposalNumber}/r_{self.runNumber}/{self.name}_r{self.runNumber}_ref.tif', ref_data)
            
        
        self.time_axis_shot = []
        self.space_axis_shot = []
        self.Xdelay_shot = []
        
        for idx, (shot_id, data, dipole_delay, dipole_energy, sweep_delay) in enumerate(zip(shot_ids, shot_data, self.dipole_delay, self.dipole_energy, self.sweep_delay)):


            time_conversion = np.poly1d(self.cal.time_polynomial[::-1])
            time_axis = time_conversion(np.arange(data.shape[1]))

            offset = time_conversion(self.cal.dipole_zero) + dipole_delay - 0*sweep_delay
            time_axis -= offset
            self.time_axis_shot.append(time_axis)          

            
            space_axis = np.arange(ref_data.shape[0]) * self.cal.dx
            space_axis -= space_axis.mean()
            self.space_axis_shot.append(space_axis)
            
            Xdelay = time_conversion(self.cal.fel_zero) - time_conversion(self.cal.dipole_zero) - dipole_delay - sweep_delay
            self.Xdelay_shot.append(Xdelay)
            
            fig, ax = plt.subplots(figsize=(9, 5))

            tid_str = f'{"SHOT"}, tid:{shot_id}'
            ax.set_title(f'{self.format(compact=True)}, {tid_str}')
            ax.set_ylabel(f'Distance [µm]')
            ax.set_xlabel(f'Time [ns]')


            im = ax.imshow(data, extent=[time_axis[0], time_axis[-1], space_axis[0], space_axis[-1]], cmap='jet', vmin=vmin, vmax=vmax)
            ax.vlines(
                Xdelay,
                ymin=space_axis[0],
                ymax=space_axis[-1],
                linestyles='-',
                lw=2,
                color='purple',
                alpha=1,
            )

            ys, xs = np.where(data > 0)
            ax.set_xlim(xmin=time_axis[xs.min()], xmax=time_axis[xs.max()])
            ax.set_ylim(ymin=-space_axis[ys.max()], ymax=-space_axis[ys.min()])
            ax.set_aspect('auto')

            from matplotlib.ticker import MultipleLocator
            ax.xaxis.set_major_locator(MultipleLocator(5))
            ax.xaxis.set_minor_locator(MultipleLocator(1))
            ax.yaxis.set_major_locator(MultipleLocator(400))
            ax.yaxis.set_minor_locator(MultipleLocator(100))
            ax.grid(which='major', color='k', linestyle = '--', linewidth=2, alpha = 0.5)
            ax.grid(which='minor', color='k', linestyle=':', linewidth=1, alpha = 1)
            
            fig.colorbar(im)
            fig.tight_layout()
            plt.savefig(f'./SOP_Summary_p{self.proposalNumber}/r_{self.runNumber}/{self.name}_{idx}.png')
            plt.show()

            ## Save ref shots tif image
            tif.imwrite(f'./SOP_TIFF_p{self.proposalNumber}/r_{self.runNumber}/{self.name}_r{self.runNumber}_shot_{idx}.tif', data)

        return ax

    def save(self):
        ref_id, shot_ids = self.shot()
        ref_data, shot_data = self.frame()

        with h5py.File(f'./VISAR_HDF5_p{self.proposalNumber}/VISAR_p{self.proposalNumber}_r{self.runNumber}.h5', 'a') as hdf:

            SOP = hdf.create_group(f'{self.name}')
            
            ref = SOP.create_group('Reference')
            ref.create_dataset('Reference', data=ref_data)
            ref.create_dataset('Train ID', data=ref_id)
            
            shot = SOP.create_group('Shot')
            shot.create_dataset('Corrected images', data = shot_data)
            shot.create_dataset('Time axis', data = self.time_axis_shot)
            shot.create_dataset('Space axis', data = self.space_axis_shot)
            shot.create_dataset('Drive pixel t0', data = self.cal.dipole_zero)  
            shot.create_dataset('Sweep window [ns]', data = self.sweep_time)                  
            shot.create_dataset('Sweep delay [ns]', data = self.sweep_delay)           
            shot.create_dataset('Difference X-Drive [ns]', data = self.Xdelay_shot)
            shot.create_dataset('Train ID', data = shot_ids)
            
            # shot.create_dataset('Size X-ray [µm]', data = ...)
            # shot.create_dataset('Shock breakout time [ns]', data = self.sweep_delay)        
            