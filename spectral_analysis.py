# The module created and developed by 
# Ksawery Mielczarek
# Faculty of Physics, University of Warsaw
# Quantum Photonics Lab
# 2023-2024

#                      -------
#                      MODULES
#                      -------



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import floor, log10, ceil
from scipy.fft import fft, fftfreq, fftshift, ifft, ifftshift
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit
from scipy.special import hermite as hermite_gen
import pyvisa
from pyvisa.constants import VI_FALSE, VI_ATTR_SUPPRESS_END_EN, VI_ATTR_SEND_END_EN
from IPython.display import display, clear_output
import time
import warnings
warnings.filterwarnings('ignore') # they made me do it


#                   --------------
#                   SPECTRUM CLASS
#                   --------------



class spectrum:
    '''
    ### Class of an optical spectrum. 
    
    It stores values of the \"X\" axis (wavelength, frequency or time) and the \"Y\" axis
    (intensity, phase or complex amplitude) as 1D numpy arrays. Several standard transforms and basic statistics are implemented as class methods.
    '''

    def __init__(self, X, Y, x_type = "THz", y_type = "intensity"):

        if x_type not in ["time", "THz", "nm", "kHz", "Hz"]:
            raise Exception("x_type must be either \"time\" or \"THz\", \"kHz\", \"Hz\" or \"nm\".")
        if y_type not in ["phase", "intensity", "e-field", "current"]:
            raise Exception("y_type must be either \"phase\", \"intensity\", \"e-field\" or \"current\".")
        if len(X) != len(Y):
            raise Exception("X and Y axis must be of the same length.")
        
        if type(X) == list:
            self.X = np.array(X)
        elif type(X) == np.ndarray:
            self.X = X
        else:
            raise Exception("\"X\" must be either a list or an array.")
        
        if type(Y) == list:
            self.Y = np.array(Y)
        elif type(Y) == np.ndarray:
            self.Y = Y
        else:
            raise Exception("\"Y\" must be either a list or an array.")
        
        self.x_type = x_type
        self.y_type = y_type
        self.spacing = self.calc_spacing()
        self.power = self.comp_power()


    def __len__(self):
        '''
        ### Number of point, that spectrum consists of.
        '''
        return len(self.X)
    

    def calc_spacing(self):
        '''
        ### Calculate the mean (hopefully constant) spacing between subsequent points on X-axis.
        '''
        return np.mean(np.diff(np.real(self.X)))
    

    def copy(self):
        X_copied = self.X.copy()
        Y_copied = self.Y.copy()
        return (spectrum(X_copied, Y_copied, self.x_type, self.y_type))
    

    def save(self, title):
        '''
        ### Save spectrum to .csv file. 
        
        The \"title\" must be a string.
        '''
        data = pd.DataFrame(np.transpose(np.vstack([self.X, self.Y])))
        data.to_csv(title, index = False)


    def wl_to_freq(self, inplace = True):
        '''
        ### Transformation from wavelength domain (nm) to frequency domain (THz).
        '''

        c = 299792458 # light speed
        freq = np.flip(c / self.X / 1e3) # output in THz
        intensity = np.flip(self.Y)

        if inplace == True:
            self.X = freq
            self.Y = intensity
            self.x_type = "THz"
            self.spacing = self.calc_spacing()
        
        else:
            return spectrum(freq, intensity, "THz", self.y_type)
        

    def hz_to_khz(self, inplace = True):
        if inplace:
            self.X /= 1000
            self.x_type = "kHz"
            self.spacing = self.calc_spacing()

        else:
            return spectrum(self.X/1000, self.Y, "kHz", self.y_type)

        
    def constant_spacing(self, inplace = True):
        '''
        ### Transformation of a spectrum to have constant spacing on X-axis by linearly interpolating two nearest values on Y-axis.
        '''

        def linear_inter(a, b, c, f_a, f_b):
            if a == b:
                return (f_a + f_b)/2
            
            else:
                f_c =  f_a * np.abs(b-c) / np.abs(b-a) + f_b * np.abs(a-c) / np.abs(b-a)
                return f_c

        length = self.__len__()

        freq = self.X.copy()
        intens = self.Y.copy()
        new_freq = np.linspace(start = freq[0], stop = freq[-1], num = length, endpoint = True)
        new_intens = []

        j = 1
        for f in range(length):         # each new interpolated intensity
            interpolated_int = 0
            
            if f == 0:
                new_intens.append(intens[0])
                continue

            if f == length - 1:
                new_intens.append(intens[length - 1])
                continue

            for i in range(j, length):     # we compare with "each" measured intensity

                if new_freq[f] <= freq[i]: # so for small i that's false. That's true for the first freq[i] greater than new_freq[f]
                    
                    interpolated_int = linear_inter(freq[i - 1], freq[i], new_freq[f], intens[i-1], intens[i])
                    break

                else:
                    j += 1 # j is chosen in such a way, because for every new "freq" it will be greater than previous

            new_intens.append(interpolated_int)

        new_intens = np.array(new_intens)

        if inplace == True:
            self.X = new_freq
            self.Y = new_intens
            self.spacing = self.calc_spacing()
        
        else:
            return spectrum(new_freq, new_intens, self.x_type, self.y_type)
        

    def cut(self, start = None, end = None, how = "units", inplace = True):
        '''
        ### Limit spectrum to a segment [\"start\", \"end\"].

        ARGUMENTS:

        start - start of the segment, to which the spectrum is limited.

        end - end of the segment, to which the spectrum is limited.

        how - defines meaning of \"start\" and \"end\". If \"units\", then those are values on X-axis. 
        If \"fraction\", then the fraction of length of X-axis. If \"index\", then corresponding indices of border observations.
        '''

        import numpy as np
        from math import floor

        if start != None:
            if how == "units":
                s = np.searchsorted(self.X, start)
            elif how == "fraction":
                s = floor(start*self.__len__())
            elif how == "index":
                s = start
            else:
                raise Exception("\"how\" must be either \"units\", \"fraction\" or \"index\".")

        if end != None:
            if how == "units":
                e = np.searchsorted(self.X, end)
            elif how == "fraction":
                e = floor(end*self.__len__())
            elif how == "index":
                e = end
            else:
                raise Exception("\"how\" must be either \"units\", \"fraction\" or \"index\".")

        new_X = self.X.copy()
        new_Y = self.Y.copy()
        
        if start != None and end != None:
            new_X = new_X[s:e]
            new_Y = new_Y[s:e]
        elif start == None and end != None:
            new_X = new_X[:e]
            new_Y = new_Y[:e]
        elif start != None and end == None:
            new_X = new_X[s:]
            new_Y = new_Y[s:]
        else: pass

        self.power = self.comp_power()

        if inplace == True:
            self.X = new_X
            self.Y = new_Y
        
        else:
            return spectrum(new_X, new_Y, self.x_type, self.y_type)
        

    def fourier(self, inplace = True, force = False, abs = False, constant_power = False):
        '''
        ### Performs Fourier Transform from \"frequency\" to \"time\" domain. 
        
        The \"force\" argument allows you to skip the warnings.
        '''
        
        # Exceptions

        if not force:
            if self.x_type == "nm":
                raise Exception("Before applying Fourier transform, transform spectrum from wavelength to frequency domain.")
            if self.x_type == "time":
                print("WARNING: Sticking to the convention you are encouraged to use the inverse Fourier transform.")

        # Fourier Transform
    
        if constant_power:
            FT_intens = np.sqrt(np.abs(self.Y))*np.exp(1j*np.angle(self.Y)) # to ensure that the powers in time and frequency pictures are the same
        else:
            FT_intens = self.Y.copy()
        FT_intens = fftshift(FT_intens)
        FT_intens = fft(FT_intens, norm = "ortho")
        FT_intens = fftshift(FT_intens)
        if constant_power:
            FT_intens = np.abs(FT_intens)**2*np.exp(1j*np.angle(FT_intens))
        else:
            pass

        if abs:
            FT_intens = np.abs(FT_intens)
        time = fftfreq(self.__len__(), self.spacing)
        time = fftshift(time)

        if inplace:
            self.X = time
            self.Y = FT_intens
            self.x_type = "time"
            self.spacing = self.calc_spacing()
        else:
            return spectrum(time, FT_intens, "time", self.y_type)
            

    def inv_fourier(self, inplace = True, force = False, abs = False, constant_power = False):
        '''
        ### Performs inverse Fourier transform from \"time\" to \"frequency\" domain.

        The \"force\" argument allows you to skip the warnings.
        '''

        if not force:
            if self.x_type != "time":
                print("WARNING: Sticking to the convention you are encouraged to use the Fourier transform (instead for the inverse one).")

        # prepare input

        time = self.X.copy()
        intens = self.Y.copy()


        time = ifftshift(time)
        if constant_power:
            intens = np.sqrt(np.abs(intens))*np.exp(1j*np.angle(intens)) # to ensure that the powers in time and frequency pictures are the same
        else:
            pass
        # Fourier Transform

        FT_intens = ifftshift(intens)
        FT_intens = ifft(FT_intens, norm = "ortho")
        FT_intens = ifftshift(FT_intens)
        if constant_power:
            FT_intens = np.abs(FT_intens)**2*np.exp(1j*np.angle(FT_intens))
        else:
            pass

        if abs:
            FT_intens = np.abs(FT_intens)
        freq = fftfreq(self.__len__(), self.spacing)
        freq = fftshift(freq)

        if inplace == True:
            self.X = freq
            self.Y = FT_intens
            self.x_type = "THz"
            self.spacing = self.calc_spacing()
        else:
            return spectrum(freq, FT_intens, "THz", self.y_type)
        

    def find_period(self, height = 0.5, hist = False):
        '''
        ### Function finds period in interference fringes by looking for wavelengths, where intensity is around given height and is decreasing. 

        ARGUMENTS:

        height - height, at which we to look for negative slope. Height is the fraction of maximum intensity.

        hist - if to plot the histogram of all found periods.

        RETURNS:

        (mean, std) - mean and standard deviation of all found periods. 
        '''

        wl = self.X
        intens = self.Y
        h = height * np.max(intens)
        nodes = []

        for i in range(2, len(intens) - 2):
            # decreasing
            if intens[i-2] > intens[i-1] > h > intens[i] > intens[i+1] > intens[i+2]:
                nodes.append(wl[i])

        diff = np.diff(np.array(nodes))

        if hist:
            plt.hist(diff, color = "orange")
            plt.xlabel("Period length (nm)")
            plt.ylabel("Counts")
            plt.show()

        mean = np.mean(diff)
        std = np.std(diff)

        if len(diff) > 4:
            diff_cut = [d for d in diff if np.abs(d - mean) < std]
        else:
            diff_cut = diff 

        return np.mean(diff_cut), np.std(diff_cut)
    

    def remove_comb(self, period = 0.00063, inplace = True): 
        '''
        ### Remove the \"frequency comb\" from the spectrum. 
        
        Applicable if the spectrometer has a very fine resolution. \"period\" denotes the period in which the local maximum is searched.
        '''
        if self.x_type != "nm":
            raise Exception("Make sure, that the x_type = \"nm\".")
        
        new_X = []
        new_Y = []

        idx_period = ceil(period/self.spacing)
        num_of_periods = ceil(len(self) / idx_period)
        
        # first estimation of peaks

        for n in range(num_of_periods - 1): # don't care very much about the last - not full - period
            start = n*idx_period
            end = (n+1)*idx_period
            new_X.append(self.X[start + np.argmax(self.Y[start: end])])
            new_Y.append(np.max(self.Y[start: end]))

        new_X = np.array(new_X)
        new_Y = np.array(new_Y)
        
        # correction for under-sampling (missing some peaks)
        '''
        x_spacing = np.mean(np.diff(new_X))
        undersampl_indc = np.array(range(len(new_X)-1))[np.diff(new_X) > x_spacing*1.33]     # we choose left indices of the periods where we missed a peak

        for idx in undersampl_indc:
            central_idx = ceil((idx + 0.5)*idx_period)
            x_value = self.X[central_idx]
            y_value = self.Y[central_idx]
            new_X = np.insert(new_X, idx + 1, x_value)
            new_Y = np.insert(new_Y, idx + 1, y_value)
        '''
        # correction for oversampling (choosing a single peak twice or finding a peak, where there's none)
        '''
        oversampl_indc = np.array(range(len(new_X)-1))[np.diff(new_X) < x_spacing*0.66]     # we choose left indices of the periods where we missed a peak           

        for idx in np.flip(oversampl_indc): # flip is crucial because we wan other indices to have sense further on
            new_X = np.delete(new_X, idx)
            new_Y = np.delete(new_Y, idx)
        '''
        # return stuff

        self.power = self.comp_power() # because the presence of the comb decreases measured intensity
            
        if inplace:
            self.X = new_X
            self.Y = new_Y
            self.spacing = self.calc_spacing()
        else:
            return spectrum(new_X, new_Y, x_type = self.x_type, y_type = self.y_type)
    

    def naive_visibility(self):
        '''        
        ### Function returns float in range (0, 1) which is the fraction of maximum intensity, at which the greatest number of fringes is observable.
        ''' 

        maximum = np.max(self.Y)

        # Omg, this loop is so smart. Sometimes I impress myself.

        heights = []
        sample_levels = 1000
        for height in np.linspace(0, maximum, sample_levels, endpoint = True):
            if height == 0: 
                continue
            safe_Y = self.Y.copy()
            safe_Y[safe_Y < height] = 0
            safe_Y[safe_Y > 0] = 1
            safe_Y += np.roll(safe_Y, shift = 1)
            safe_Y[safe_Y == 2] = 0
            heights.append(np.sum(safe_Y))

        heights = np.array(heights) # height is array of numbers of fringes on given level (times 2)
        fring_num = np.max(heights)
        indices = np.array([i for i in range(sample_levels-1) if heights[i] == fring_num])
        the_index = indices[0] #[floor(len(indices)/2)] 

        level = the_index/sample_levels
        print(1-level)
        return 1-level
    

    def visibility(self, show_plot = False):
        '''
        ### Find visibility of interference fringes. 
        
        First maxima and minima of fringes are found, then they are interpolated 
        and corresponding sums of intensities are calculated. Visibility is then (max_int-min_int)/max_int

        Optionally you can use \"show_plot\" to plot interpolated maxima and minima. It looks very cool.
        '''

        safe_X = self.X.copy()
        minima = find_minima(self)
        maxima = find_maxima(self)
        inter_minima = interpolate(minima, safe_X, how = "spline")
        inter_maxima = interpolate(maxima, safe_X, how = "spline")
        max_sum = np.sum(inter_maxima.Y)
        min_sum = np.sum(inter_minima.Y)

        left = self.comp_quantile(0.2)
        right = self.comp_quantile(0.8)
        delta = right-left

        if show_plot:
            compare_plots([self, inter_maxima, inter_minima], 
                             colors = ["deepskyblue", "green", "red"], 
                             title = "Visibility of {}".format(round((max_sum-min_sum)/max_sum, 3)),
                             start = left-delta, end = right+delta)
        
        return (max_sum-min_sum)/max_sum


    def replace_with_zeros(self, start = None, end = None, inplace = True):
        '''
        ### Replace numbers on Y axis from "start" to "end" with zeroes. 
        
        "start" and "end" are in X axis' units. 
        If \"start\" and \"end\" are \"None\", then beginning and ending of whole spectrum are used as the borders.
        '''

        if start == None:
            s = 0
        else:
            s = np.searchsorted(self.X, start)

        if end == None:
            e = self.__len__() - 1
        else:
            e = np.searchsorted(self.X, end)

        new_Y = self.Y.copy()
        new_Y[s:e] = np.zeros(e-s)

        self.power = self.comp_power()

        if inplace:
            self.Y = new_Y
        else:
            return spectrum(self.X, new_Y, self.x_type, self.y_type)
        

    def shift(self, shift, inplace = True):
        '''
        ### Shifts the spectrum by X axis. 
        
        Warning: only values on X axis are modified.
        '''

        new_X = self.X.copy()
        new_X = new_X + shift

        if inplace == True:
            self.X = new_X
        if inplace == False:
            return spectrum(new_X, self.Y, self.x_type, self.y_type)

    
    def smart_shift(self, shift = None, inplace = True):
        '''
        ### Shift spectrum by rolling Y-axis. 
        
        Value of shift is to be given in X axis units. If shift = \"None\",
        the spectrum is shifted so, that 1/2-order quantile for Y axis is reached for x = 0.
        '''

        shift2 = shift

        if shift == None:
            shift2 = -self.comp_quantile(1/2)

        index_shift = floor(np.real(shift2)/np.real(self.spacing))

        Y_new = self.Y.copy()
        Y_new = np.roll(Y_new, index_shift)
            
        if inplace == True:
            self.Y = Y_new
        if inplace == False:
            return spectrum(self.X, Y_new, self.x_type, self.y_type)
        
    def very_smart_shift(self, shift, inplace = True):
        '''
        ### Shift spectrum by applying FT, multiplying by linear temporal phase and applying IFT. 
        
        Value of shift is to be given in X axis units.
        '''

        X2 = self.X.copy()
        Y2 = self.Y.copy()
        spectrum2 = spectrum(X2, Y2, self.x_type, self.y_type)
        spectrum2.fourier()
        spectrum2.Y *= np.exp(2j*np.pi*shift*spectrum2.X)
        spectrum2.inv_fourier()
        if inplace == True:
            self.Y = spectrum2.Y
        if inplace == False:
            return spectrum2


    def insert_zeroes(self, how_much, inplace = True):
        '''
        ### Add zeros on Y-axis to the left and right of data with constant (mean) spacing on X-axis.

        \"how_much\" specifies number of added zeroes on left and right side of spectrum as a fraction of spectrums length.
        '''

        length = floor(how_much*self.__len__())
        
        left_start = self.X[0] - self.spacing*length
        left_X = np.linspace(left_start, self.X[0], endpoint = False, num = length - 1)
        left_Y = np.zeros(length - 1)

        right_end = self.X[-1] + self.spacing*length
        right_X = np.linspace(self.X[-1] + self.spacing, right_end, endpoint = True, num = length - 1)
        right_Y = np.zeros(length-1)

        new_X = np.concatenate([left_X, self.X.copy(), right_X])
        new_Y = np.concatenate([left_Y, self.Y.copy(), right_Y])

        if inplace == True:
            self.X = new_X
            self.Y = new_Y
        if inplace == False:
            return spectrum(new_X, new_Y, self.x_type, self.y_type)
        
    
    def increase_resolution(self, times, inplace = True):
        '''
        ### Interpolate by padding zeroes in Fourier domain. 
        
        \"times\" tells, how many times do you want the resolution increased.
        '''
        if times < 1:
            raise Exception("You want to increase and not decrease the resolution, right? Keep \"times\" above 1.")
        
        spectrum_safe = self.copy()
        initial_mean = np.mean(spectrum_safe.X)

        if times > 1:   # just a quick fix for times = 1 case
            spectrum_safe.fourier(inplace = True, force = True)
            spectrum_safe.insert_zeroes(how_much = 1/2*(times-1))
            spectrum_safe.inv_fourier(inplace = True, force = True)

        spectrum_safe.X -= (np.mean(spectrum_safe.X) - initial_mean)

        spectrum_safe.Y = spectrum_safe.Y/np.max(spectrum_safe.Y)   # thats not that smart fix
        spectrum_safe.Y = spectrum_safe.Y*np.max(self.Y)

        if inplace:
            self.X = spectrum_safe.X
            self.Y = spectrum_safe.Y
            self.spacing = self.calc_spacing()
        else:
            return spectrum_safe


    def comp_quantile(self, q, norm = "L1"):
        '''
        ### Finds x in X axis such that the integral of the intensity to the \"x\" is fraction of value \"q\" of whole intensity. 
        The\"norm\" is either \"L1\" or \"L2\".
        '''
        if norm not in ["L1", "L2"]:
            raise Exception("The norm must be either \"L1\" or \"L2\".")
        
        integral = 0
        if norm == "L1":
            integral_infinite = np.sum(np.abs(self.Y))
        if norm == "L2":
            integral_infinite = np.sum(self.Y*np.conjugate(self.Y))

        for i in range(self.__len__()):
            if norm == "L1": integral += np.abs(self.Y[i])
            if norm == "L2": integral += self.Y[i]*np.conjugate(self.Y[i])

            if integral >= integral_infinite*q:
                x = self.X[i]
                break

        return x
    

    def comp_centre(self):
        '''
        ### Return center of mass of the spectrum in X-axis units.
        '''
        return np.sum(self.X*np.abs(self.Y))/np.sum(np.abs(self.Y))
    

    def normalize(self, norm = "highest", shift_to_zero = True, inplace = True):
        '''
        ### Normalize spectrum by a linear scaling of the values of intensity and eventually shifting spectrum to zero.

        ARGUMENTS:

        by - way of normalizing the spectrum. If \"highest\", then the spectrum is scaled to 1, so that its greatest value is 1. 
        If \"intensity\", then spectrum is scaled, so that its integral is equal to 1.
        
        shift_to_zero - if \"True\", then spectrum is shifted by X axis, by simply np.rolling it.
        '''

        if norm not in ["sup", "L1", "L2"]:
            raise Exception("\"norm\" parameter must be either \"sup\", \"L1\" or \"L2\".")

        X_safe = self.X.copy()
        Y_safe = self.Y.copy()
        safe_spectrum = spectrum(X_safe, Y_safe, self.x_type, self.y_type)

        if norm == "sup":
            max = np.max(np.abs(Y_safe))
            max_idx = np.argmax(np.abs(Y_safe))
            Y_safe /= max
            if shift_to_zero:
                zero_idx = np.searchsorted(X_safe, 0)
                shift_idx = max_idx - zero_idx
                X_safe = np.roll(X_safe, shift_idx)

        if norm == "L1":
            integral = np.sum(np.abs(Y_safe))/(X_safe[-1]-X_safe[0])
            median = safe_spectrum.comp_quantile(1/2)
            max_idx = np.searchsorted(np.abs(Y_safe), median)
            Y_safe /= integral
            if shift_to_zero:
                zero_idx = np.searchsorted(X_safe, 0)
                shift_idx = max_idx - zero_idx
                X_safe = np.roll(X_safe, shift_idx)

        if norm == "L2":
            integral = np.sum(Y_safe**2)/(X_safe[-1]-X_safe[0])
            median = safe_spectrum.comp_quantile(1/2)
            max_idx = np.searchsorted(np.abs(Y_safe), median)
            Y_safe /= np.sqrt(integral)
            if shift_to_zero:
                zero_idx = np.searchsorted(X_safe, 0)
                shift_idx = max_idx - zero_idx
                X_safe = np.roll(X_safe, shift_idx)

        if inplace == True:
            self.X = X_safe
            self.Y = Y_safe
        
        if inplace == False:
            return spectrum(X_safe, Y_safe, self.x_type, self.y_type)


    def remove_temporal_phase(self, magnitude = 0):
        '''
        ### Remove temporal (linear) phase from the phase spectrum. 
        
        \"magnitude\" denotes the number that is subtracted from the phase in the left-most
        part of spectrum. The global phase is chosen so that the value of phase in the precise middle of the spectrum is not modified.
        '''
        if self.y_type != "phase":
            raise Exception("The spectrum, you are trying to remove temporal phase from, is NOT a phase spectrum.")

        X = self.X.copy() - np.mean(self.X)
        X /= X[-1]
        Y = magnitude*X.copy()
        self.Y = self.Y + Y


    def comp_power(self):
        '''
        ### Power of a spectrum. 
        
        In other words - integral of absolute value of the spectrum. Arbitrary units are used.
        '''

        # simple function to round to significant digits
        def round_to_dig(x, n):
            if x == 0:
                return 0
            return round(x, -int(floor(log10(abs(x)))) + n - 1)
    
        return np.abs(round_to_dig(np.sum(np.sqrt(self.Y*np.conjugate(self.Y))), 5))
        

    def comp_FWHM(self):
        '''
        ### Calculate Full Width at Half Maximum. 
        
        If multiple peaks are present in spectrum, the function might not work properly.
        '''

        left = None
        right = None
        peak = np.max(np.abs(self.Y))

        for idx, y in enumerate(self.Y):
            if np.abs(y) >= peak/2:
                left = idx
                break

        for idx, y in enumerate(np.flip(self.Y)):
            if np.abs(y) >= peak/2:
                right = self.__len__() - idx
                break

        if self.__len__() == 0:
            raise Exception("Failed to calculate FWHM, because spectrum is empty.")
        if self.__len__() < 5:
            raise Exception("The spectrum consists of too little data points to calculate FWHM.")
        if left == None or right == None:
            return 0
        width = right-left
        width *= self.spacing

        return np.abs(width)


    def remove_offset(self, period, inplace = True, negative_noise = True, norm = "L1"):
        '''
        ### The function improves visibility of interference fringes 
        by subtracting a moving minimum i. e. minimum of a segment of length \"period\", centered at given point.
        If \"negative_noise\" = True, the local mean of the noise is subtracted, which results in the noise being partially negative.
        \"norm\" can be equal to "L1" or "L2", in the latter case the spectrum is squared before removing the offset.
        '''

        if norm not in ["L1", "L2"]:
            raise Exception("\"norm\" must be either \"L1\" or \"L2\".")

        idx_period = floor(period/self.spacing)

        squared_spectrum = self.copy()
        if norm == "L2":
            squared_spectrum.Y = squared_spectrum.Y**2

        # in the first loop we remove local minima

        new_Y = []
        for i in range(squared_spectrum.__len__()):
            left = max([i - floor(idx_period/2), 0])
            right = min([i + floor(idx_period/2), squared_spectrum.__len__() - 1])
            new_Y.append(squared_spectrum.Y[i] - np.min(squared_spectrum.Y[left:right]))

        squared_spectrum.Y = np.array(new_Y)

        # and in the second the local mean

        new_Y = []
        if negative_noise:
            for i in range(squared_spectrum.__len__()):
                
                # firstly we choose the appropriate surrounding of the point
                left = max([i - floor(idx_period/2), 0])
                right = min([i + floor(idx_period/2), self.__len__() - 1])
                low_bamplitude_copy = squared_spectrum.Y[left:right].copy()
                for j in range(5):

                    # secondly we kill extreme values
                    stan_dev = np.std(low_amplitude_copy)
                    low_amplitude_copy = low_amplitude_copy[np.abs(low_amplitude_copy-np.mean(low_amplitude_copy))<3*stan_dev]

                # and we subtract the mean
                new_Y.append(squared_spectrum.Y[i] - np.mean(low_amplitude_copy))
                
            squared_spectrum.Y = np.array(new_Y)

        if norm == "L2":
            if negative_noise:
                squared_spectrum.Y = np.array([np.sqrt(y) if y > 0 else 0 for y in squared_spectrum.Y])
            else:
                squared_spectrum.Y = np.sqrt(squared_spectrum.Y)

        if inplace:
            self.X = squared_spectrum.X
            self.Y = squared_spectrum.Y
        else:
            return squared_spectrum
        

    def noise_level(self):
        '''
        Applicable for a peak-like spectrum AND under condition that the function \"remove_offset\" has already been applied.
        20 times the extreme values are removed, which should remove all the peaks, but hardly influence the noise.
        '''

        low_amplitude_copy = self.Y.copy()
        for i in range(20):
            low_amplitude_copy = low_amplitude_copy[np.abs(low_amplitude_copy - 0) < 3*np.std(low_amplitude_copy)]
        return np.max(low_amplitude_copy)


    def moving_average(self, period, inplace = True):
        '''
        ### Smooth spectrum 
        by taking moving average with \"period\" in X axis units.
        
        On the beginning and ending of spectrum shorter segments are used.
        '''

        idx_period = floor(period/self.spacing)

        new_Y = []
        for i in range(self.__len__()):
            left = np.max([i - floor(idx_period/2), 0])
            right = np.min([i + floor(idx_period/2), self.__len__() - 1])
            new_Y.append(np.mean(self.Y[left:right]))
        new_Y = np.array(new_Y)

        if inplace == True:
            self.Y = new_Y

        if inplace == False:
            return spectrum(self.X, new_Y, self.x_type, self.y_type)
        

    def moving_average_interior(self, period, inplace = True):
        '''
        ### Smooth the INTERIOR of the spectrum 
        
        by taking moving average with \"period\" in X axis units.

        On the beginning and ending of spectrum shorter segments are used.
        '''

        sup = np.max(self.Y)
        support = self.X[self.Y >= sup/2]
        left_border = support[0]
        right_border = support[-1]
        left_border_idx = np.searchsorted(self.X, left_border)
        right_border_idx = np.searchsorted(self.X, right_border)

        idx_period = floor(period/self.spacing)

        new_Y = []
        for i in range(self.__len__()):
            if i < left_border_idx + 1 or i > right_border_idx - 1:
                new_Y.append(self.Y[i])
                continue
            smooth_range = np.min([idx_period, 2*np.abs(left_border_idx - i), 2*np.abs(right_border_idx - i)])
            left = np.max([i - floor(smooth_range/2), 0])
            right = np.min([i + floor(smooth_range/2), self.__len__() - 1])
            new_Y.append(np.mean(self.Y[left:right]))
        new_Y = np.array(new_Y)

        if inplace == True:
            self.Y = new_Y

        if inplace == False:
            return spectrum(self.X, new_Y, self.x_type, self.y_type)



#                   ------------------------
#                   SPECTRUM CLASS FUNCTIONS
#                   ------------------------



def load_csv(filename, x_type = "nm", y_type = "intensity", rows_to_skip = 2):
    '''
    ### Load CSV file to a spectrum class.
     
    Spectrum has on default wavelengths on X axis and intensity on Y axis.
    '''
    spectr = pd.read_csv(filename, skiprows = rows_to_skip)
    return spectrum(spectr.values[:, 0], spectr.values[:, 1], x_type = x_type, y_type = y_type)


def load_tsv(filename, x_type = "nm", y_type = "intensity", source = "OSA"):
    '''
    ### Load TSV file to a spectrum class. 
    
    Spectrum has on default wavelengths on X axis and intensity on Y axis.
    '''
    if source == "APEX":
        spectr = pd.read_table(filename, skiprows = 3)
        spectr.columns = ["X", "Y", "trash"]
        spectr.drop_duplicates(inplace = True, subset = ["X"])

    elif source == "OSA":
        spectr = pd.read_table(filename, skiprows = 2)

    else:
        raise Exception("\"source\" must be either \"OSA\" or \"APEX\"")

    return spectrum(spectr.values[:, 0], spectr.values[:, 1], x_type = x_type, y_type = y_type)


def interpolate(old_spectrum, new_X, how):
    '''
    ### Interpolate rarely sampled spectrum for values in new X-axis. 
    
    Interpolation is performed with cubic functions. If y-values to be interpolated are complex, they are casted to reals.

    ARGUMENTS:

    old_spectrum - spectrum that is the basis for interpolation.

    new_X - new X axis (i. e. frequencies or wavelengths). Interpolated Y-values are calculated for values from new_X.

    RETURNS:

    Interpolated spectrum.
    '''

    X = np.real(old_spectrum.X.copy())
    Y = np.real(old_spectrum.Y.copy())

    if how == "spline":
        model = CubicSpline(X, Y)
        new_Y = model(np.real(new_X))

    elif how == "linear":
        new_Y = []
        for x in np.real(new_X):
            idx = np.searchsorted(X, x)
            segment = X[idx] - X[idx-1]
            y = (X[idx] - x)*Y[idx-1]/segment + (x - X[idx-1])*Y[idx]/segment
            new_Y.append(y)
        new_Y = np.array(new_Y)

    else:
        raise Exception("The \"how\" argument must be equal either to \"spline\" or to \"linear\".")

    return spectrum(new_X, new_Y, old_spectrum.x_type, old_spectrum.y_type)


def create_complex_spectrum(intensity_spectrum, phase_spectrum, extrapolate = False):
    '''
    ### Given the intensity spectrum and the phase spectrum compute the complex amplitude spectrum. 
    
    With \"extrapolate = False\" the spectral phase of the spectrum
    is zeroed outside the area with the well-known phase.
    '''

    if intensity_spectrum.y_type != "intensity":
        print("WARNING: Are you sure that the \"intensity_spectrum\" carries the information about intensity?")
    if phase_spectrum.y_type != "phase":
        print("WARNING: Are you sure that the \"phase_spectrum\" carries the information about phase?")

    support_left = np.searchsorted(intensity_spectrum.X, phase_spectrum.X[0])
    support_right = np.searchsorted(intensity_spectrum.X, phase_spectrum.X[-1])
    new_phase = interpolate(phase_spectrum, intensity_spectrum.X, how = "spline")

    if not extrapolate:
        for i in range(len(intensity_spectrum)):
            if i < support_left or i > support_right:
                new_phase.Y[i] = 0

    amplitude = intensity_spectrum.Y * np.exp(1j*new_phase.Y)
    
    return spectrum(intensity_spectrum.X, amplitude, intensity_spectrum.x_type, "intensity")


def fit_fiber_length(phase_spectrum, show_plot = False, guessed_length = 80):
    '''
    ### Fit parabolic spectral phase 
    
    to the given spectrum and return the length of chirping fiber corresponding to that phase.
    '''

    def chirp_phase(frequency, centre, fiber_length):
        c = 299792458 
        l_0 = c/(centre*1e3)
        D_l = 20
        omega = frequency*2*np.pi
        omega_mean = centre*2*np.pi
        return l_0**2*fiber_length*D_l/(4*np.pi*c)*(omega-omega_mean)**2

    param, cov = curve_fit(chirp_phase, phase_spectrum.X, phase_spectrum.Y, [192, guessed_length], bounds = [[150, 10],[300, 300]])

    if show_plot:
        new_X = phase_spectrum.X.copy()
        fit_phase = spectrum(new_X, chirp_phase(new_X, param[0], param[1]), x_type = "THz", y_type = "phase")
        compare_plots([phase_spectrum, fit_phase], legend = ["Original spectrum", "Phase corresponding to fiber of {}m".format(round(param[1]))])

    return np.abs(param[1])
    

def chirp_r2(phase_spectrum, fiber_length, show_plot = False):
    '''
    Given spectral phase spectrum and the length of fiber employed in experiment, 
    fit corresponding spectral phase to experimental data (only fitting center) and 
    return the R^2 statistics judging quality of that fit.

    Additionally you can plot both phases on a single plot.
    '''

    def R2(data_real, data_pred):
        TSS = np.sum(np.power(data_real - np.mean(data_real), 2))
        RSS = np.sum(np.power(data_real - data_pred, 2))
        return (TSS - RSS)/TSS
    
    if np.mean(phase_spectrum.Y) < 0:
        sign = -1
    else:
        sign = 1

    def chirp_phase(frequency, centre):
        c = 299792458 
        l_0 = c/(centre*1e3)
        D_l = 20
        omega = frequency*2*np.pi
        omega_mean = centre*2*np.pi
        return sign*l_0**2*fiber_length*D_l/(4*np.pi*c)*(omega-omega_mean)**2
    
    param, cov = curve_fit(chirp_phase, phase_spectrum.X, phase_spectrum.Y, p0 = 192)
    centrum = param[0]

    Y_real = phase_spectrum.Y.copy()
    Y_pred = chirp_phase(phase_spectrum.X, centrum)
    score = R2(Y_real, Y_pred)

    if show_plot:
        reality = spectrum(phase_spectrum.X.copy(), Y_real, x_type = "THz", y_type = "phase")
        prediction = spectrum(phase_spectrum.X.copy(), Y_pred, x_type = "THz", y_type = "phase")

        if np.mean(reality.Y) < 0:
            reality.Y *= -1
            prediction.Y *= -1

        compare_plots([reality, prediction], 
                         title = "Experimental and model chirp phase for fiber of {}m\nR-squared value equal to {}".format(fiber_length, round(score, 3)),
                         legend = ["Experiment", "Model"],
                         colors = ["darkorange", "darkgreen"])
    return score


def fit_rect(spectr, output = "params", fixed_area = True, slope = False, slope_factor = 0.1):
    '''
    ### Fit a rectangle to the given spectrum. 
    If output = \"params\" then tuple (start, end, height) characterizing the shape of rectangle is returned.
    If output = \"spectrum\" returns fitted rectangle as a spectrum class object. If fixed_area = True, than fitted spectrum has the same area as the initial.
    If slope = True, than not a rectangle but a trapeze is fitted. The slope of its sides is the greater, the greater slope_factor is.
    '''

    def rect(x, start, end, height):

        if isinstance(x, type(np.array([]))):
            y = x.copy()
            delta = (end-start)*slope_factor
            for i in range(len(x)):
                if x[i] <= start:
                    y[i] = 0
                if x[i] >= end:
                    y[i] = 0
                if start + delta <= x[i] <= end - delta:
                    y[i] = height
                if start < x[i] < start + delta:
                    if slope:
                        y[i] = (0*(np.abs(x[i] - start - delta)) + height*(np.abs(x[i] - start)))/delta
                    else:
                        y[i] = height
                if end - delta < x[i] < end:
                    if slope:
                        y[i] = (height*(np.abs(x[i] - end)) + 0*(np.abs(x[i] - end + delta)))/delta
                    else:
                        y[i] = height
            return y
        
        else:
            print("You gave me a scalar. Is it good?")
            if x < start or x > end:
                return 0
            else:
                return height
            
    sum = np.sum(spectr.Y)/len(spectr)*(spectr.X[-1]-spectr.X[0])
            
    def rect_fixed(x, start, end):
        return rect(x = x, start = start, end = end, height = sum/(end-start))

    if fixed_area:
        params, cov = curve_fit(f = rect_fixed, 
                        xdata = spectr.X, 
                        ydata = np.abs(spectr.Y),
                        p0 = [spectr.quantile(0.1), spectr.quantile(0.9)],
                        bounds = [[spectr.quantile(0.001), spectr.quantile(0.5)], [spectr.quantile(0.5), spectr.quantile(0.999)]])
        
        if output == "params":
            return params[0], params[1], sum/(params[1]-params[0])
        elif output == "spectrum":
            new_X = spectr.X.copy()
            new_Y = rect_fixed(new_X, params[0], params[1])
            return spectrum(new_X, new_Y, spectr.x_type, spectr.y_type)
        else:
            raise Exception("output must be either \"params\" or \"spectrum\"")
    else:
        params, cov = curve_fit(f = rect, 
                            xdata = spectr.X, 
                            ydata = np.abs(spectr.Y),
                            p0 = [spectr.quantile(0.1), spectr.quantile(0.9), 0.8*np.max(spectr.Y)],
                            bounds = [[spectr.quantile(0.001), spectr.quantile(0.5), 0.2*np.max(spectr.Y)], [spectr.quantile(0.5), spectr.quantile(0.999), np.max(spectr.Y)]])
        
        if output == "params":
            return params[0], params[1], params[2]
        elif output == "spectrum":
            new_X = spectr.X.copy()
            new_Y = rect(new_X, params[0], params[1], params[2])
            return spectrum(new_X, new_Y, spectr.x_type, spectr.y_type)
        else:
            raise Exception("output must be either \"params\" or \"spectrum\"")


def fit_rect_smart(spectr, output = "params"):
    '''
    ### Fit a rectangle to the given spectrum in a very fucking smart way.

    output must be either \"params\" - then a tuple (a, b, height) is returned -
    or \"spectrum\", resulting in returning a fitted rectangle spectrum
    '''

    # we limit ourselves to the significant part of spectrum

    start = spectr.quantile(0.01)
    end = spectr.quantile(0.99)
    mid = spectr.quantile(0.5)

    start_idx = np.searchsorted(spectr.X, start)
    end_idx = np.searchsorted(spectr.X, end)
    mid_idx = np.searchsorted(spectr.X, mid)

    s_safe = spectr.copy()
    left_axis = s_safe.Y[start_idx: mid_idx].copy()
    right_axis = s_safe.Y[mid_idx: end_idx].copy()

    # we will compute the left and right side od rectangle in a smart wat, but we need to use a brute force to compute the height
    
    height_sample_rate = 1000
    delta = np.mean(spectr.cut(start, end, inplace = False).Y)/height_sample_rate

    losses_l = []
    losses_r = []
    loss_idx_l = []
    loss_idx_r = []

    # the loop over heights

    for i in range(height_sample_rate-2):
        height = np.mean(spectr.cut(start, end, inplace = False).Y) -height_sample_rate/2*delta + i*delta

        # the absolute error if we fit at given place the rectangle

        left_axis_if_big = np.abs(left_axis.copy()-height)
        right_axis_if_big = np.abs(right_axis.copy()-height)

        # the absolute error if we fit at given place the zeroes

        left_axis_if_small = np.abs(left_axis.copy())
        right_axis_if_small = np.abs(right_axis.copy())

        # basically we integrate the losses

        left_axis_if_big = np.cumsum(np.flip(left_axis_if_big))
        right_axis_if_big = np.cumsum(right_axis_if_big)
        left_axis_if_small = np.cumsum(left_axis_if_small)
        right_axis_if_small = np.cumsum(np.flip(right_axis_if_small))

        # and sum them up so that at the n-th index there is a loss of setting the rectangle side there
        # left and right sides are computed independently

        left_axis_loss = np.flip(left_axis_if_big) + left_axis_if_small
        right_axis_loss = right_axis_if_big + np.flip(right_axis_if_small)

        # cool looking loss spectrum

        Y = np.concatenate([left_axis_loss, right_axis_loss])
        loss_spectrum = spectrum(spectr.X[start_idx:end_idx], Y, spectr.x_type, spectr.y_type)

        # now, for each side, we find height minimizing the loss at best

        losses_l.append(np.min(left_axis_loss))
        losses_r.append(np.min(right_axis_loss))
        loss_idx_l.append(np.argmin(left_axis_loss))
        loss_idx_r.append(np.argmin(right_axis_loss))

    left_iter_min = np.argmin(losses_l)
    right_iter_min = np.argmin(losses_r)

    # height estimation

    final_height = np.mean([left_iter_min, right_iter_min])*delta + np.mean(spectr.cut(start, end, inplace = False).Y) +(-height_sample_rate/2*delta)

    # and sides calculation

    left_idx = loss_idx_l[left_iter_min] + start_idx
    right_idx = mid_idx + loss_idx_r[right_iter_min]
    left_side = spectr.X[left_idx]
    right_side = spectr.X[right_idx]

    def rect(x, start, end, height):
        y = x.copy()
        for i in range(len(x)):
            if x[i] < start:
                y[i] = 0
            elif x[i] > end:
                y[i] = 0
            else:
                y[i] = height
        return y

    # returns

    if output == "params":
        return left_side, right_side, final_height
    
    elif output == "spectrum":
        new_X = spectr.X.copy()
        new_Y = rect(new_X, left_side, right_side, final_height)
        return spectrum(new_X, new_Y, spectr.x_type, spectr.y_type)
    
    else:
        raise Exception("output must be either \"params\" or \"spectrum\"")


def find_minima(fringes_spectrum):
    '''
    ### Find minima of interference fringes 
    by looking at nearest neighbors. Spectrum with minima is returned.
    '''
    X = []
    Y = []
    for i in range(len(fringes_spectrum)):
        if i in [0, 1, len(fringes_spectrum) - 2, len(fringes_spectrum) - 1]:
            continue
        if fringes_spectrum.Y[i-2] >= fringes_spectrum.Y[i-1] >= fringes_spectrum.Y[i] <= fringes_spectrum.Y[i+1] <= fringes_spectrum.Y[i+2]:
            X.append(fringes_spectrum.X[i])
            Y.append(fringes_spectrum.Y[i])
    
    return spectrum(np.array(X), np.array(Y), fringes_spectrum.x_type, fringes_spectrum.y_type)


def find_maxima(fringes_spectrum):
    '''
    ### Find maxima of interference fringes 
    by looking at nearest neighbors. Spectrum with maxima is returned.
    '''

    X = []
    Y = []
    for i in range(len(fringes_spectrum)):
        if i in [0, 1, len(fringes_spectrum) - 2, len(fringes_spectrum) - 1]:
            continue
        if fringes_spectrum.Y[i-2] <= fringes_spectrum.Y[i-1] <= fringes_spectrum.Y[i] >= fringes_spectrum.Y[i+1] >= fringes_spectrum.Y[i+2]:
            X.append(fringes_spectrum.X[i])
            Y.append(fringes_spectrum.Y[i])
    
    return spectrum(np.array(X), np.array(Y), fringes_spectrum.x_type, fringes_spectrum.y_type)


def plot(spectrum, 
         color = "darkviolet", 
         title = "Spectrum", 
         what_to_plot = "trigonometric", 
         start = None, end = None, 
         save = False,
         show_grid = False):
    '''
    ### Fast spectrum plotting using matplotlib.pyplot library.

    ARGUMENTS:

    spectrum - the spectrum object class to be plotted.

    color - color of the plot.

    title - title of the plot.

    what_to_plot - either \"abs\", \"imag\", \"real\", \"complex\" or \"trigonometric\". In the last two cases two curves are plotted.

    start - starting point (in X-axis units) of a area to be shown on plot. If \"min\", then plot starts with lowest X-value in all spectra.

    end - ending point (in X-axis units) of a area to be shown on plot. If \"max\", then plot ends with highest X-value in all spectra.

    save - if to save the plot in the program's directory. The title of the plot will be by default used as the filename.
    '''

    spectrum_safe = spectrum.copy()
    
    # simple function to round to significant digits

    def round_to_dig(x, n):
        if x == 0:
            return 0
        return round(x, -int(floor(log10(abs(x)))) + n - 1)

    # invalid arguments

    if what_to_plot not in ("abs", "imag", "real", "complex", "trigonometric"):
        raise Exception("Argument \"what_to_plot\" must be either \"abs\", \"imag\", \"real\", \"complex\" or \"trigonometric\".")

    # if we dont want to plot whole spectrum

    n_points = len(spectrum_safe)

    to_cut = False
    inf = round(np.real(spectrum_safe.X[0]))
    sup = round(np.real(spectrum_safe.X[-1]))

    s = 0
    e = -1

    if start != None:
        s = np.searchsorted(spectrum_safe.X, start)
        to_cut = True

    if end != None:        
        e = np.searchsorted(spectrum_safe.X, end)
        to_cut = True

    spectrum_safe.cut(s, e, how = "index") 

    # what do we want to have on Y axis

    if what_to_plot == "abs":
        spectrum_safe.Y = np.abs(spectrum_safe.Y)
    if what_to_plot == "real":
        spectrum_safe.Y = np.real(spectrum_safe.Y)
    if what_to_plot == "imag":
        spectrum_safe.Y = np.imag(spectrum_safe.Y)

    # start to plot

    fig, ax = plt.subplots()

    if what_to_plot == "complex":
        ax.plot(spectrum_safe.X, np.real(spectrum_safe.Y), color = "darkviolet")
        ax.plot(spectrum_safe.X, np.imag(spectrum_safe.Y), color = "green")
        ax.legend(["Real part", "Imaginary part"], bbox_to_anchor = [1, 0.17])

    elif what_to_plot == "trigonometric":
        phase = np.angle(spectrum_safe.Y)
        intensity = np.abs(spectrum_safe.Y)

        start_idx_phase = 0
        end_idx_phase = len(phase) -1
        for i in range(len(phase)): # we don't plot the phase in the area, where it is equal to zero
            if np.abs(phase[i]) > 1e-5:
                start_idx  = i
                break
        for i in reversed(range(len(phase))):
            if np.abs(phase[i]) > 1e-5:
                end_idx  = i
                break

        start_idx_intens = np.searchsorted(spectrum_safe.X, spectrum_safe.comp_quantile(0.001))
        end_idx_intens = np.searchsorted(spectrum_safe.X, spectrum_safe.comp_quantile(0.999))
        start_idx = np.max([start_idx_phase, start_idx_intens])
        end_idx = np.min([end_idx_phase, end_idx_intens])

        phase = np.unwrap(phase)

        ax.fill_between(spectrum_safe.X, intensity, color = "orange", alpha = 0.5, label = "Intensity")
        phase_ax = ax.twinx()
        phase_ax.plot(spectrum_safe.X[start_idx: end_idx], phase[start_idx: end_idx], color = "darkviolet", label = "Phase")
        phase_ax.set_ylabel("Phase (rad)")

        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = phase_ax.get_legend_handles_labels()
        phase_ax.legend(lines + lines2, labels + labels2, loc=0)

    else:
        ax.plot(spectrum_safe.X, spectrum_safe.Y, color = color)
        if spectrum_safe.y_type == "intensity":
            ax.set_ylim([min([0, np.min(spectrum_safe.Y)]), 1.1*np.max(spectrum_safe.Y)])
     
    ax.grid()
    if to_cut:
        ax.set_title(title + " [only part shown]")
    else:
        ax.set_title(title)
    if spectrum_safe.y_type == "phase":
        ax.set_ylabel("Phase (rad)")
    if spectrum_safe.y_type == "intensity":
        ax.set_ylabel("Intensity (a.u.)")
    if spectrum_safe.y_type == "e-field":
        ax.set_ylabel("Electric Field")        
    if spectrum_safe.x_type == "nm":
        ax.set_xlabel("Wavelength (nm)")
        unit = "nm"
    if spectrum_safe.x_type == "THz":
        ax.set_xlabel("Frequency (THz)")
        unit = "THz"
    if spectrum_safe.x_type == "Hz":
        ax.set_xlabel("Frequency (Hz)")
        unit = "Hz"
    if spectrum_safe.x_type == "kHz":
        ax.set_xlabel("Frequency (kHz)")
        unit = "kHz"
    if spectrum_safe.x_type == "time":
        ax.set_xlabel("Time (ps)")
        unit = "ps"

    # grid

    if show_grid and spectrum_safe.y_type != "phase":

        left_idx = None
        right_idx = None
        peak = np.max(np.abs(spectrum_safe.Y))

        for idx, y in enumerate(spectrum_safe.Y):
            if np.abs(y) >= peak/2:
                left_idx = idx
                break

        for idx, y in enumerate(np.flip(spectrum_safe.Y)):
            if np.abs(y) >= peak/2:
                right_idx = len(spectrum_safe) - idx
                break

        ax.axhline(y = peak, linestyle = "dashed", lw = 1, color = "grey")
        ax.axhline(y = peak/2, linestyle = "dashed", lw = 1, color = "grey")
        ax.axvline(x = spectrum_safe.comp_centre(), linestyle = "dashed", lw = 1, color = "black")
        if left_idx != None: 
            ax.axvline(x = spectrum_safe.X[left_idx], linestyle = "dashed", lw = 1, color = "grey")
        if right_idx != None: 
            ax.axvline(x = spectrum_safe.X[right_idx], linestyle = "dashed", lw = 1, color = "grey")

    # quick stats
    
    spacing = round_to_dig(spectrum_safe.spacing, 3)
    p_per_unit = floor(1/spectrum_safe.spacing)
    if p_per_unit == 0:
        p_per_unit = "< 1"
    else:
        p_per_unit = str(p_per_unit)

    space = 0
    if what_to_plot == "trigonometric":
        space = 0.15

    points_txt = "Number of points: {}".format(n_points)
    spacing_txt = "\nX-axis spacing: {} ".format(spacing) + unit
    per_1_txt = "\nPoints per 1 " + unit +": " + p_per_unit
    full_range_txt = "\nFull X-axis range: {} : {} ".format(inf, sup) + unit
    centr_txt = "\nCentroid: {} ".format(round_to_dig(spectrum_safe.comp_quantile(0.5), 5)) + unit
    fwhm_txt =  "\nFWHM: {} ".format(round_to_dig(spectrum_safe.comp_FWHM(), 5)) + unit
    power_txt = "\nPower: {}".format(round_to_dig(spectrum_safe.comp_power(), 5))

    if to_cut:
        ax.text(1.05 + space, 0.65, 
                points_txt + spacing_txt + per_1_txt + full_range_txt + centr_txt + fwhm_txt + power_txt, 
                transform = ax.transAxes)
    else:
        ax.text(1.05 + space, 0.7, 
                points_txt + spacing_txt + per_1_txt + centr_txt + fwhm_txt + power_txt, 
                transform = ax.transAxes)


    # additional info, what do you actually see

    views = {
        "abs" : r"$\it{Absolute value}$",
        "real" : r"$\it{Real}$",
        "imag" : r"$\it{Imaginary}$",
        "complex" : r"$\it{Complex}$",
        "trigonometric" : r"$\it{Trigonometric}$"
    }

    ax.text(1.05 + space, 0.0, views[what_to_plot] + " view", transform = ax.transAxes)

    if save:
        fig.savefig("{}.jpg".format(title))

    fig.show()


def compare_plots(spectra, title = "Spectra", legend = None, colors = None, start = None, end = None, abs = False):
    '''
    ### Show several spectra on single plot.

    ARGUMENTS:

    spectra - list with spectra to be show on plot.

    title - title of the plot.

    legend - list with names of subsequent spectra. If \"None\", then no legend is shown.

    start - starting point (in X-axis units) of a area to be shown on plot. If \"min\", then plot starts with lowest X-value in all spectra.

    end - ending point (in X-axis units) of a area to be shown on plot. If \"max\", then plot ends with highest X-value in all spectra.

    abs - if \"True\", then absolute values of spectra is plotted.
    '''
    
    # invalid input

    dummy = []
    for i in range(len(spectra)):
        dummy.append(spectra[i].x_type == spectra[0].x_type)

    if not np.all(np.array(dummy)):
        raise Exception("The X axes of spectra are not of unique type.")
    
    dummy = []
    for i in range(len(spectra)):
        dummy.append(spectra[i].y_type == spectra[0].y_type)

    if not np.all(np.array(dummy)):
        raise Exception("The Y axes of spectra are not of unique type.")
    
    # and plotting

    if colors == None:
        colours = ["violet", "blue", "green", "yellow", "orange", "red", "brown", "black"]
    else:
        colours = colors

    for c, spectrum in enumerate(spectra):
        safe_spectrum = spectrum.copy()
        if abs:
            safe_spectrum.Y = np.abs(safe_spectrum.Y)
        if start == None:
            s = 0
        else:
            s = np.searchsorted(safe_spectrum.X, start)
        if end == None:
            e = len(safe_spectrum) - 1
        else:
            e = np.searchsorted(safe_spectrum.X, end)

        plt.plot(safe_spectrum.X[s:e], safe_spectrum.Y[s:e], color = colours[c])

    plt.grid()
    plt.title(title)
    if spectra[0].y_type == "int":
        plt.ylabel("Intensity")
    if spectra[0].y_type == "phase":
        plt.ylabel("Spectral phase (rad)")    
    if spectra[0].x_type == "nm":
        plt.xlabel("Wavelength (nm)")
    if spectra[0].x_type == "THz":
        plt.xlabel("Frequency (THz)")
    if spectra[0].x_type == "Hz":
        plt.xlabel("Frequency (Hz)")
    if spectra[0].x_type == "kHz":
        plt.xlabel("Frequency (kHz)")
    if spectra[0].x_type == "time":
        plt.xlabel("Time (ps)")
    if isinstance(legend, list):
        plt.legend(legend, bbox_to_anchor = [1, 1])
    plt.show()       


def recover_pulse(phase_spectrum, intensity_spectrum):
    '''
    ### Reconstructs the pulse 
    (or any type of spectrum in time), given spectrum with spectral phase and spectrum with spectral intensity.
    '''

    if len(phase_spectrum) != len(intensity_spectrum):
        raise Exception("Frequency axes of phase and intensity are not of equal length.")
    
    complex_Y = intensity_spectrum.Y.copy()
    complex_Y = complex_Y.astype(complex) # surprisingly that line is necessary
    for i in range(len(complex_Y)):
        complex_Y[i] *= np.exp(1j*phase_spectrum.Y[i])

    pulse_spectrum = spectrum(phase_spectrum.X.copy(), complex_Y, "THz", "e-field")
    pulse_spectrum.fourier()
    pulse_spectrum.Y = 2*np.real(pulse_spectrum.Y)
    return pulse_spectrum


def find_shift(spectrum_1, spectrum_2):
    '''
    ### Returns translation between two spectra in THz. 
    Spectra in wavelength domain are at first transformed into frequency domain.
    Least squares is loss function to be minimized. Shift is found by brute force: 
    checking number of shifts equal to number of points on X axis.
    '''

    spectrum1 = spectrum_1.copy()
    spectrum2 = spectrum_2.copy()

    if len(spectrum1) != len(spectrum2):
        raise Exception("Spectra are of different length.")

    if spectrum1.x_type == "nm":

        spectrum1.wl_to_freq()
        spectrum1.constant_spacing()

    if spectrum2.x_type == "nm":

        spectrum2.wl_to_freq()
        spectrum2.constant_spacing()

    def error(v_1, v_2):
        return np.sum(np.abs(v_1 - v_2)**2)
    
    minimum = np.sum(np.abs(spectrum1.Y)**2)
    idx = 0
    width = np.max(np.array([spectrum1.comp_FWHM(), spectrum2.comp_FWHM()]))
    index_width = floor(width/spectrum1.spacing)
    for i in range(-index_width, index_width):
         if minimum > error(spectrum1.Y, np.roll(a = spectrum2.Y, shift = i)):
             minimum = error(spectrum1.Y, np.roll(a = spectrum2.Y, shift = i))
             idx = i
    
    return spectrum1.spacing*idx


def ratio(vis_value):
    '''
    Given visibility of interference fringes in spectrum, the function returns ratio of intensities of two pulses that 
    are responsible for the interference pattern. Ratio of lower intensity to the greater is calculated.
    '''

    def vis(x):
        '''Inverse function of ratio.'''
        return 4*np.sqrt(x)/(1+np.sqrt(x))**2
    
    vis = np.vectorize(vis)
    X = np.linspace(0, 1, 20000)
    Y = vis(X)
    r = np.searchsorted(Y, vis_value)

    return X[r]


def gaussian_pulse(bandwidth, centre, FWHM, x_type = "THz", num = 1000):
    '''
    ### Creates spectrum with gaussian intensity. 
    "bandwidth" is a tuple with start and the end of the entire spectrum. 
    "centre" and "FWHM" characterize the pulse itself. The spectrum is composed of \"num\" = 1000 points on default.
    '''

    def gauss(x, mu, std):
        return 1/(std*np.sqrt(2*np.pi))*np.exp(-(x-mu)**2/(2*std**2))
    gauss = np.vectorize(gauss)

    sd = FWHM/2.355
    X = np.linspace(bandwidth[0], bandwidth[1], num = num)
    Y = gauss(X, centre, sd)

    return spectrum(X, Y, x_type, "intensity")


def hermitian_pulse(pol_num, bandwidth, centre, FWHM, num = 1000, x_type = "THz"):
    '''
    ### Creates spectrum with \"pol-num\"-th Hermit-Gauss intensity mode. 
    "bandwidth" is a tuple with start and the end of the entire spectrum. 
    "centre" and "FWHM" characterize the pulse itself. The spectrum is composed of \"num\" = 1000 points on default.
    '''

    # exceptions

    if x_type not in ["THz", "kHz", "Hz", "nm", "time"]:
        raise Exception("x_type must be either \"THz\", \"kHz\", \"Hz\", \"nm\" or \"time\"")

    # and calculations

    hermite_pol = hermite_gen(pol_num)
    def gauss(x, mu, std):
        return 1/(std*np.sqrt(2*np.pi))*np.exp(-(x-mu)**2/(2*std**2))
    
    X = np.linspace(bandwidth[0], bandwidth[1], num = num)
    sd = FWHM/2.355
    Y_gauss = gauss(X, centre, sd)
    Y_hermite = hermite_pol(2*(X-centre)/FWHM)
    Y_out = Y_hermite*Y_gauss

    spectrum_out = spectrum(X, Y_out, x_type, "intensity")
    spectrum_out.normalize(norm = "L2", shift_to_zero = False)

    return spectrum_out


def pkin_pulse(centre, width, num, x_type = "THz"):
    X = np.linspace(-1, 1, 2400)
    Y = np.ones(1000)

    # first floor

    Y[200:335] *= 456
    Y[200:235] += np.flip(np.linspace(0, 50, 35))
    Y[245:280] += np.linspace(0, 50, 35)
    Y[280:315] += np.flip(np.linspace(0, 50, 35))
    Y[325:335] += np.linspace(0, 14, 10)

    # second floor

    Y[335:400] *= 1164
    Y[340:342] += 70
    Y[342:380] += 10
    Y[342:361] += np.linspace(0, 35, 19)
    Y[361:380] += np.flip(np.linspace(0, 35, 19))
    Y[380:383] += 70

    # third floor

    Y[400:430] *= 1445
    Y[403:406] += 50

    # fourth floor

    Y[430:480] *= 1624
    Y[430:480] += np.linspace(0, 70, 50)
    Y[430] += 30

    # spire

    Y[480:500] *= 2164
    Y[480:490] = 1820
    Y[490:495] = 2000
    Y[495:500] += np.linspace(-164, 0, 5)

    Y[500:1000] = np.flip(Y[0:500])
    Y = np.hstack([np.zeros(700), Y, np.zeros(700)])

    # create spectrum

    signal = spectrum(X, Y, x_type = x_type, y_type = "intensity")

    # interpolation

    X_true = np.linspace(centre-width*2, centre+width*2, num)

    def linear_transform(old_spectrum, X_target):
        start = old_spectrum.X[0]
        end = old_spectrum.X[-1]
        new_spectrum = old_spectrum.copy()
        new_spectrum.X = new_spectrum.X - start
        new_spectrum.X = new_spectrum.X/(end-start)*(X_target[-1]-X_target[0])
        new_spectrum.X = new_spectrum.X + X_target[0]
        return new_spectrum
    
    signal = linear_transform(signal, X_true)
    signal = interpolate(signal, X_true, how = "linear")

    return signal


def chirp_phase(bandwidth, centre, fiber_length, num):
    '''
    ### Creates spectrum class object with spectral phase corresponding to propagation of a pulse through "fiber_length" of PM fiber.
    "bandwidth" is a tuple with start and the end of the entire spectrum OR already created X-axis of the spectrum. 
    "centre" determines the minimum of spectral phase.
    '''
    c = 299792458 
    l_0 = c/(centre*1e3)
    D_l = 17
    if len(bandwidth) == 2:
        X = np.linspace(bandwidth[0], bandwidth[1], num)
    else:
        X = bandwidth.copy()
    omega = X*2*np.pi
    omega_mean = centre*2*np.pi
    return l_0**2*fiber_length*D_l/(4*np.pi*c)*(omega-omega_mean)**2


def find_slope_shift(sheared_spectrum, not_sheared_spectrum, low = 0.1, high = 0.5, sampl_num = 500):
    '''
    ### Find relative shift between the spectra by computing the shift of the slope. 
    Precisely, one computes the difference between x-arguments, when spectrum reaches for the very first time 
    value y, where \"low\"*max <= y <= \"high\"*max, where max denotes maximum value of the intensity. The number of y-values used is equal to \"sampl_num\". 
    The shear returned is the mean of such differences for the left and the right slope.
    '''

    sup_sheared = np.max(sheared_spectrum.Y)
    sup_not_sheared = np.max(not_sheared_spectrum.Y)

    # left slope, sheared spectrum

    left_s = []
    for i in np.linspace(low, high, sampl_num):
        for k in range(len(sheared_spectrum)):
            if sheared_spectrum.Y[k] < i*sup_sheared:
                continue
            else: 
                left_s.append(sheared_spectrum.X[k])
                break

    # left slope, not sheared spectrum

    left_ns = []
    for i in np.linspace(low, high, sampl_num):
        for k in range(len(not_sheared_spectrum)):
            if not_sheared_spectrum.Y[k] < i*sup_not_sheared:
                continue
            else: 
                left_ns.append(not_sheared_spectrum.X[k])
                break

    # right slope, sheared spectrum

    right_s = []
    for i in np.linspace(low, high, sampl_num):
        for k in reversed(range(len(sheared_spectrum))):
            if sheared_spectrum.Y[k] < i*sup_sheared:
                continue
            else: 
                right_s.append(sheared_spectrum.X[k])
                break

    # right slope, not sheared spectrum

    right_ns = []
    for i in np.linspace(low, high, sampl_num):
        for k in reversed(range(len(not_sheared_spectrum))):
            if not_sheared_spectrum.Y[k] < i*sup_not_sheared:
                continue
            else: 
                right_ns.append(not_sheared_spectrum.X[k])
                break

    left_shift = np.mean(np.array(left_s)-np.array(left_ns))
    right_shift = np.mean(np.array(right_s)-np.array(right_ns))

    return (left_shift+right_shift)/2


def find_shear(sheared_spectrum, not_sheared_spectrum, smoothing_period = None, normalize = False, how = "slope", show_plot = False, improve_resolution = 1):
    '''
    ## Find shear between two spectra.

    ### ARGUMENTS:

    sheared_spectrum - path to .csv file containing measured sheared spectrum.

    not_sheared_spectrum - path to .csv file containing measured not-sheared spectrum.

    smoothing_period - apply \"moving average\" to locally smooth the spectra.

    normalize - normalize both spectra, i.e. divide their intensities by the maximum intensity value.

    how - method of computing the shift between two spectra. You can find the shift by fitting it (how = \"fit\"), 
    by calculating the shift between centers of mass (how = \"com\"), by computing the shift of the slope (how = \"slope\") 
    or by calculating the shift of fitted rectangle (how = \"rect\").

    show_plot - show both spectra on the first plot and on the second plot show again both spectra, 
    while the sheared spectrum has been numerically \"desheared\". This may help estimate if the computed shear is accurate.

    improve_resolution - use the zero padding method for interpolation and increasing number of points constituting the spectra.
    Writing \"improve_resolution = 5\" will cause number of points being increased 5 times.
    '''

    if how not in ["com", "fit", "slope", "rect"]:
        raise Exception("\"how\" must be equal either to \"com\",\"fit\", \"slope\" or \"rect\".")

    if isinstance(sheared_spectrum, str):
        sheared = load_csv(sheared_spectrum)
    elif isinstance(sheared_spectrum, spectrum):
        sheared = sheared_spectrum.copy()
    else:
        raise Exception("\"sheared_spectrum\" must be either a path of .csv file with spectrum or a \"spectrum\" class object.")
    
    if isinstance(not_sheared_spectrum, str):
        not_sheared = load_csv(not_sheared_spectrum)
    elif isinstance(not_sheared_spectrum, spectrum):
        not_sheared = not_sheared_spectrum.copy()
    else:
        raise Exception("\"sheared_spectrum\" must be either a path of .csv file with spectrum or a \"spectrum\" class object.")
    
    if sheared.x_type == "nm":
        sheared.wl_to_freq()
        sheared.constant_spacing()
    if not_sheared.x_type == "nm":
        not_sheared.wl_to_freq()
        not_sheared.constant_spacing()

    sheared.increase_resolution(improve_resolution)
    not_sheared.increase_resolution(improve_resolution)

    if smoothing_period != None:
        sheared.moving_average(smoothing_period)
        not_sheared.moving_average(smoothing_period)
        
    if normalize:
        sheared.normalize("highest", False)
        not_sheared.normalize("highest", False)

    if how == "com":
        shear = sheared.comp_centre() - not_sheared.comp_centre()

    elif how == "fit":
        shear = find_shift(sheared, not_sheared)

    elif how == "slope":
        shear = find_slope_shift(sheared, not_sheared)

    elif how == "rect":
        params_s = fit_rect_smart(sheared, output = "params")
        params_ns = fit_rect_smart(not_sheared, output = "params")
        shear = np.mean([params_s[0]-params_ns[0], params_s[1]-params_ns[1]])

    if show_plot:    
        compare_plots([sheared, not_sheared], 
                         legend = ["Sheared spectrum", "Not sheared spectrum"], 
                         title = "Shear = {} THz".format(round(shear, 5)))
        compare_plots([sheared.shift(-shear, inplace = False), not_sheared], 
                         legend = ["Resheared sheared spectrum", "Not sheared spectrum"], 
                         title = "Shifting sheared spectrum to the \"zero\" position".format(round(shear, 5)),
                         colors = ["green", "orange"])
    
    return shear



#                   ---------
#                   BEAM CLASS
#                   ---------



class beam:
    '''
    Suitable for performing basic simulations with the experimental setup.
    '''

    def __init__(self, vertical_polarization, horizontal_polarization):
        self.ver = vertical_polarization
        self.hor = horizontal_polarization


    def copy(self):
        return beam(self.ver.copy(), self.hor.copy())


    def mix(self, transmission_matrix):
        ver = transmission_matrix[0, 0]*self.ver.Y + transmission_matrix[0,1]*self.hor.Y
        hor = transmission_matrix[0, 1]*self.ver.Y + transmission_matrix[1,1]*self.hor.Y
        self.ver.Y = ver
        self.hor.Y = hor


    def loss(self, fraction):
        self.ver.Y *= (1-fraction)
        self.hor.Y *= (1-fraction)


    def delay(self, polarization, delay):
        
        if polarization not in ["ver", "hor"]:
            raise Exception("Polarization must be either \"ver\" or \"hor\".")

        if polarization == "ver":
            self.ver.Y = self.ver.Y * np.exp(np.pi*2j*delay*self.ver.X)

        elif polarization == "hor":
            self.hor.Y = self.hor.Y * np.exp(np.pi*2j*delay*self.hor.X)


    def rotate(self, angle):    # there is small problem because after rotation interference pattern are the same and should be negatives
        
        hor = np.cos(angle)*self.hor.Y + np.sin(angle)*self.ver.Y
        ver = np.sin(angle)*self.hor.Y + np.cos(angle)*self.ver.Y

        self.hor.Y = hor
        self.ver.Y = ver


    def chirp(self, polarization, fiber_length):

        if polarization not in ["ver", "hor"]:
            raise Exception("Polarization must be either \"ver\" or \"hor\".")
        
        if polarization == "ver": spectr = self.hor.copy()
        if polarization == "hor": spectr = self.ver.copy()
        l_0 = 1560
        c = 3*1e8
        D_l = 20
        omega = spectr.X*2*np.pi
        omega_mean = spectr.quantile(1/2)
        phase = l_0**2*fiber_length*D_l/(4*np.pi*c)*(omega-omega_mean)**2
        if polarization == "ver": self.ver.Y = self.ver.Y*np.exp(1j*phase)
        if polarization == "hor": self.hor.Y = self.hor.Y*np.exp(1j*phase)


    def shear(self, polarization, shift):

        if polarization not in ["ver", "hor"]:
            raise Exception("Polarization must be either \"ver\" or \"hor\".")

        if polarization == "hor":
            self.hor.smart_shift(shift, inplace = True)
        if polarization == "ver":
            self.ver.smart_shift(shift, inplace = True)


    def polarizer(self, transmission_polar):

        if transmission_polar not in ["ver", "hor"]:
            raise Exception("Polarization must be either \"ver\" or \"hor\".")
        if transmission_polar == "ver":
            self.hor.Y *= 0
        if transmission_polar == "hor":
            self.ver.Y *= 0

    def powermeter(self):

        h = np.sum(self.hor.Y*np.conjugate(self.hor.Y))
        v = np.sum(self.ver.Y*np.conjugate(self.ver.Y))
        h = np.abs(h)
        v = np.abs(v)
        h = round(h)
        v = round(v)
        print(u"Total power measured: {} \u03bcW\nPower on horizontal polarization: {} \u03bcW\nPower on vertical polarization: {} \u03bcW".format(h+v, h, v))

    def OSA(self, start = None, end = None, show_plot = True):

        Y = self.hor.Y*np.conjugate(self.hor.Y) + self.ver.Y*np.conjugate(self.ver.Y)
        spectr = spectrum(self.hor.X, Y, "THz", "intensity")
        spectr.spacing = np.abs(spectr.spacing)
        if show_plot:
            plot(spectr, title = "OSA", start = start, end = end, color = "green")
        
        return spectr



#                   ----------------
#                   OSA measurements
#                   ----------------



def measurement(centre = 1550, span = 10):
    '''
    Performs single measurement with OSA and returns it as \"spectrum\" class object. \"centre\" and \"span\" are in nm. 
    '''

    rm = pyvisa.ResourceManager()
    osa = rm.open_resource("TCPIP0::10.33.8.140::1045::SOCKET")
    osa.set_visa_attribute(VI_ATTR_SUPPRESS_END_EN, VI_FALSE)
    osa.set_visa_attribute(VI_ATTR_SEND_END_EN, VI_FALSE)
    print(osa.query('open "anonymous"'))
    print(osa.query(" "))
    osa.write(":sens:wav:cent %fnm" % centre)
    osa.write(":sens:wav:span %fnm" % span)
    #osa.write(":trac:Y1:SCAL:UNIT DBM")

    osa.write(":init:smode:1")
    osa.write(":init")

    osa.write(":trac:Y1:SCAL:UNIT DBM")
    wait_for_scan = "0"
    while "0" in wait_for_scan:
        wait_for_scan = osa.query(":stat:oper:even?")

    Lambda = osa.query(":trac:X? trA").split(",")
    intensity = osa.query(":trac:Y? trA").split(",")
    osa.close()

    Lambda = np.asarray(Lambda)
    Lambda = [float(i)/1e-9 for i in Lambda]
    intensity = np.asarray(intensity)
    intensity = [float(i) for i in intensity]

    osa_spectrum = spectrum(Lambda, intensity, "nm", "intensity")
    return osa_spectrum


def OSA(centre = 1550, span = 10, plot_size = [10, 10]):
    '''
    Plots continuous in time spectrum from OSA. \"plot_size\" is tuple of dimensions of image to be shown in cm
    '''

    fig, ax = plt.subplots(fig_size = [plot_size[0]/2.54, plot_size[1]/2.54])

    start_time = time.time()
    while True:
        current_time = time.time()
        if current_time - start_time > 1800:
            raise RuntimeWarning("Measurement lasted longer than 0.5h.")
        pure_spectrum = measurement(centre = centre, span = span)
        ax.clear()
        ax.plot(pure_spectrum.X, pure_spectrum.Y, color = "red")
        ax.set_title("OSA")
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Intensity")
        display(fig)
        clear_output(wait=True)



#                   ----------------
#                   SPIDER ALGORITHM
#                   ----------------



def spider(phase_spectrum, 
           temporal_spectrum, 
           shear = None, 
           intensity_spectrum = None,
           phase_borders = None,
           what_to_return = None,
           smoothing_period = None,
           improve_resolution = 1,
           find_shear = "slope",
           sheared_is_bigger = True,
           fiber_length = 0,
           temp_phase_param = 0,
           cut_here = None,
           plot_steps = False, 
           plot_phase_inter = False,
           plot_phase = True,
           plot_shear = False, 
           plot_pulse = False):
    '''
    ## Performs SPIDER algorithm.

    ### ARGUMENTS:

    phase_spectrum - spectrum with interference pattern created in a SPIDER setup. 
        It should be either spectrum object with wavelength or frequency in X-column OR path of .csv file with that data.

    temporal_spectrum - analogous to phase_spectrum, but measured with EOPM off.

    shear - spectral shear applied by EOPM given in frequency units (default THz). 
        If \"None\", then shear is estimated using fourier filtering - which will be probably highly inaccurate.

    intensity_spectrum - amplitude of initial not interfered pulse. Similar as "phase_spectrum" it might be either a spectrum object or pathname. 
        If "None", then its approximation derived from SPIDER algorithm is used.  
    
    phase_borders - specify range of frequencies (in THz), where you want to find spectral phase. It should be a tuple looking like (start, end).
        If to big, boundary effects may appear. If "None", the borders are estimated by calculating quantiles of intensity.

    what_to_return - if None, then RETURNS nothing. If "pulse", then RETURNS spectrum with reconstructed pulse. 
        If "phase", then RETURNS the interpolated phase. If "phase_diff", then RETURNS the spectrum with
        spectral phase difference. If "FT", then RETURNS the spectrum with Fourier transform of the OSA spectrum.
        If "IFT", then RETURNS the spectrum with inverse Fourier transform of the filtered spectrum.

    smoothing_period - if None, nothing happens. Otherwise the phase difference is smoothed by taking 
        a moving average within an interval of smoothing_period.
    
    improve_resolution - at the very beginning of the code, how many times do you want the resolution increased. This is achieved by zero-padding.     
        
    find_shear - method of finding shear. NOTE: This argument refers to part of code, which is NOT used, unless shear = None.
        If "center of mass", then shift of center of mass between sheared and non-sheared spectrum is computed. 
        If "least_squares", then shear is found as value of shift minimizing squared difference between spectra. 
        If "slope", then finds the first argument, where spectrum reaches 0.1 of maximum, 0.11 of maximum, etc. Shear is estimated as mean of shifts of these points. 
        If "rect", then rectangles are fitted to the both spectra, and the shear is computed as the mean of shifts of both its sides.

    sheared_is_bigger - if the intensity of the sheared polarization in phase_spectrum (being the mixture of polarization) is bigger
        than the intensity of the non-sheared polarization. NOTE: This argument refers to part of code, which is NOT used, unless shear = None.    

    fiber_length - the total length [in meters] of all silica fibers used in the setup BEFORE the electro-optic modulator

    temp_phase_param - governs the size of temporal (linear) phase to be subtracted from the phase spectrum. 
        \"temp_phase_param\" denotes the number that is subtracted from the phase in the left-most part of spectrum. 

    cut_here - if None, then the delay between both pulses is estimated and - while filtering - only the part of the spectrum within range (delay/2, infinity)
        is being kept. By writing "cut_here = (a, b)" you force that this range is exactly (a, b).

    plot_steps - if to plot all intermediate steps of the SPIDER algorithm.

    plot_phase_inter - if to plot the intermediate phase plots.

    plot_phase - if to plot the found spectral phase.

    plot_shear - if to plot the spectra used to find the shear.

    plot_pulse - if to plot the reconstructed pulse.
    '''

    # load data - spider

    if isinstance(phase_spectrum, spectrum):
        p_spectrum = phase_spectrum.copy()

    elif isinstance(phase_spectrum, str):
        p_spectrum = load_csv(phase_spectrum)    

    else:
        raise Exception("Wrong \"phase_spectrum\" format.")
    
    # load data - temporal phase

    if isinstance(temporal_spectrum, spectrum):
        t_spectrum = temporal_spectrum.copy()

    elif isinstance(temporal_spectrum, str):
        t_spectrum = load_csv(temporal_spectrum)    

    else:
        raise Exception("Wrong \"temporal_spectrum\" format.")

    # zero padding

    t_spectrum.insert_zeroes(3)
    p_spectrum.insert_zeroes(3)
    t_spectrum.increase_resolution(improve_resolution)
    p_spectrum.increase_resolution(improve_resolution)

    # plot OSA

    min_wl = p_spectrum.comp_quantile(0.1)
    max_wl = p_spectrum.comp_quantile(0.9)
    delta = (max_wl - min_wl)
    min_wl -= delta
    max_wl += delta 

    if plot_steps:
        plot(p_spectrum, "orange", title = "Data from OSA", start = min_wl, end = max_wl)

    # transform X-axis to frequency

    # spider

    if p_spectrum.x_type == "nm":
        s_freq = p_spectrum.wl_to_freq(inplace = False)
        s_freq.constant_spacing()
        min_freq = s_freq.comp_quantile(0.1)
        max_freq = s_freq.comp_quantile(0.9)
        delta = (max_freq - min_freq)
        min_freq -= delta
        max_freq += delta
        if plot_steps: 
            plot(s_freq,"orange", title = "Wavelength to frequency", start = min_freq, end = max_freq)

    elif p_spectrum.x_type == "THz":
        s_freq = p_spectrum
        # we need following lines, because we want in every case have min_freq and max_freq defined for later
        min_freq = s_freq.comp_quantile(0.1)
        max_freq = s_freq.comp_quantile(0.9)
        delta = (max_freq - min_freq)
        min_freq -= delta
        max_freq += delta

    s_freq_for_later = s_freq.copy()

    # temporal

    if t_spectrum.x_type == "nm":
        s_freq_t = t_spectrum.wl_to_freq(inplace = False)
        s_freq_t.constant_spacing()

    elif t_spectrum.x_type == "THz":
        s_freq_t = t_spectrum

    # fourier transform
        
    s_ft = s_freq.fourier(inplace = False)         # spider
    s_ft_t = s_freq_t.fourier(inplace = False)     # temporal

    s_shear = s_ft.copy()       # SPOILER: we will use it later to find the shear
    s_shear_t = s_ft_t.copy()
    s_intens = s_ft_t.copy()    # and this will be used to reconstruct the pulse

    min_time = s_ft.comp_quantile(0.1)
    max_time = s_ft.comp_quantile(0.9)
    delta = (max_time-min_time)/3
    min_time -= delta
    max_time += delta

    if max_time > 250:  # fix if quantiles don't work properly
        max_time = 250
        min_time = -250

    if plot_steps: 
        plot(s_ft, title = "Fourier transformed", start = min_time, end = max_time) 

    # estimate time delay

    period = s_freq_t.find_period(1/3+2/3*(1-s_freq_t.visibility()))[0] # look for the fringes in 1/3 distance between visibility level and max intensity
    delay = 1/period

    # find exact value of time delay

    zero_right_site = False

    s_ft2 = s_ft.copy()
    s_ft_return = s_ft.copy()        # if you wish it to return it later
    s_ft2.replace_with_zeros(end = delay*0.5)
    if zero_right_site: s_ft2.replace_with_zeros(start = delay*1.5)
    idx = s_ft2.Y.argmax()

    if isinstance(idx, np.ndarray): 
        idx = idx[0]
    delay2 = s_ft.X[idx]

    if delay2 < 10: # if it failed to found automatically a big enough delay
        delay2 = 100

    # and filter the spectrum to keep only one of site peaks

    if cut_here == None:
        s_ft.replace_with_zeros(end = delay2*0.5)           # spider
        s_ft_t.replace_with_zeros(end = delay2*0.5)         # temporal
    else:
        if cut_here[0] != None:
            s_ft.replace_with_zeros(end = cut_here[0])           # spider
            s_ft_t.replace_with_zeros(end = cut_here[0])         # temporal
        if cut_here[1] != None:
            s_ft.replace_with_zeros(start = cut_here[1])           # spider
            s_ft_t.replace_with_zeros(start = cut_here[1])         # temporal

    if plot_steps: 
        plot(s_ft, title = "Filtered (absolute value)", start = -2*delay2, end = 2*delay2, what_to_plot = "abs")

    # let's find the shear
    
    if shear == None:

        s_shear.replace_with_zeros(start = None, end = -delay2*0.5)         # spider
        if zero_right_site: s_shear.replace_with_zeros(start = delay2*0.5, end = None)
        
        s_shear_t.replace_with_zeros(start = None, end = -delay2*0.5)       # temporal
        if zero_right_site: s_shear_t.replace_with_zeros(start = delay2*0.5, end = None)

        s_shear.inv_fourier()
        s_shear_t.inv_fourier()

        s_shear.Y = np.abs(s_shear.Y)
        s_shear_t.Y = np.abs(s_shear_t.Y)

        mu = ratio(t_spectrum.visibility())
        if sheared_is_bigger:
            s_shear_t.Y /= (1+mu)
            s_shear.Y -= (mu*s_shear_t.Y)
        else:
            s_shear_t.Y /= (1+mu)
            s_shear.Y -= (s_shear_t.Y)
            s_shear.Y /= mu
        
        if find_shear == "least squares":
            shear = find_shift(s_shear, s_shear_t)
        elif find_shear == "center of mass":
            shear = s_shear.comp_quantile(1/2) - s_shear_t.comp_quantile(1/2)
        elif find_shear == "slope":
            shear = find_slope_shift(s_shear, s_shear_t)
        elif find_shear == "rect":
            params_s = fit_rect_smart(s_shear, output = "params")
            params_ns = fit_rect_smart(s_shear_t, output = "params")
            shear = np.mean([params_s[0]-params_ns[0], params_s[1]-params_ns[1]])        
        else:
            raise Exception("\"find_shear\" must be either \"least squares\", \"center of mass\", \"slope\" or \"rect\".")
        
        shear = np.abs(shear)
        
        if shear == 0:
            raise Exception("Failed to find non zero shear.")
        if plot_shear:
            compare_plots([s_shear, s_shear_t], 
                             start = 1.5*s_shear.comp_quantile(0.05), 
                             end = 1.5*s_shear.comp_quantile(0.95), 
                             abs = True, 
                             title = "Shear of {} THz".format(round(shear,5)),
                             legend = ["Sheared", "Not sheared"])

    integrate_interval = ceil(np.abs(shear)/(s_freq_for_later.spacing))
    scale_correction = 1/((np.abs(shear)/(integrate_interval*s_freq_for_later.spacing)))
    mean = np.mean(s_freq_for_later.X)

    # inverse fourier

    s_ift = s_ft.inv_fourier(inplace = False)        # spider
    s_ift_t = s_ft_t.inv_fourier(inplace = False)    # temporal
    if plot_steps:
        s_ift2 = s_ift.copy()
        s_ift2.X += np.real(mean) 
        plot(s_ift2, title = "Inverse Fourier transformed", start = min_freq, end = max_freq, what_to_plot = "abs")

    # cut spectrum to area of significant phase

    if phase_borders == None:
        min_phase = s_ift.comp_quantile(0.01)
        max_phase = s_ift.comp_quantile(0.99)

    else:
        min_phase = phase_borders[0] - mean
        max_phase = phase_borders[1] - mean

    s_cut = s_ift.cut(start = min_phase, end = max_phase, inplace = False)
    s_cut_t = s_ift_t.cut(start = min_phase, end = max_phase, inplace = False)

    # extract phase differences

    phase_values = s_cut.Y.copy()
    temporal_phase = s_cut_t.Y.copy()
    X_sampled = s_cut.X.copy()

    phase_values = np.angle(phase_values)
    temporal_phase = np.angle(temporal_phase)
    
    # extract phase

    values = phase_values - temporal_phase
    plt.show()
    values = ((values + np.pi)%(2*np.pi) -np.pi)
    values -= np.mean(values)
    plt.show()

    if smoothing_period != None:
        V = spectrum(X_sampled, values, "THz", "phase")
        V.moving_average(smoothing_period)
        values = V.Y

    # prepare data to plot
    
    X_sampled += mean
    X_continuous = X_sampled.copy()

    # plot phase difference

    diff_spectrum = spectrum(X_sampled, values, "THz", "phase")    # if you wish to return it later
    if plot_phase_inter:

        plt.scatter(X_sampled, np.real(values), color = "orange", s = 1)
        if smoothing_period == None:
            plt.title("Spectral phase difference between pulse and its sheared copy")
        else:
            plt.title("Spectral phase difference between pulse and its sheared copy\n[smoothed with period of {} THz].".format(smoothing_period))

        plt.xlabel("Frequency (THz)")
        plt.ylabel("Spectral phase")
        plt.grid()
        plt.show()

    # recover discrete phase
    
    integration_start = floor((len(X_sampled) % integrate_interval)/2) # we want to extrapolate equally on both sides

    X_sampled = X_sampled[integration_start::integrate_interval]

    values = values[integration_start::integrate_interval]
    values = values*scale_correction # correction if the shear is not a multiple of spacing
    Y_sampled = np.cumsum(values)
    if shear < 0: 
        X_sampled -= shear
        X_continuous -= shear
        Y_sampled *= -1

    phase_spectrum_first = spectrum(X_sampled, Y_sampled, "THz", "phase")

    # recover intensity spectrum

    if intensity_spectrum == None:
        s_intens.replace_with_zeros(end = -0.5*delay2)      # DC filtering
        s_intens.replace_with_zeros(start = 0.5*delay2)

        intensity = s_intens.inv_fourier(inplace = False)
        mu = ratio(t_spectrum.visibility())
        intensity.Y /= (1+mu)
        intensity.X += mean

        start = np.searchsorted(intensity.X, X_continuous[0]) # this must be done with "index" method to ensure equal length of "intensity" and "X_continuous"
        end = start + len(X_continuous)
        intensity.cut(start = start, end = end, how = "index")

    else:
        if isinstance(intensity_spectrum, spectrum):
            intensity = intensity_spectrum
        elif isinstance(intensity_spectrum, str):
            intensity = load_csv(intensity_spectrum)

        if intensity.x_type == "nm":
            intensity.wl_to_freq()
            intensity.constant_spacing()

        intensity.increase_resolution(improve_resolution)

        start = np.searchsorted(intensity.X, X_continuous[0]) # this must be done with "index" method to ensure equal length of "intensity" and "X_continuous"
        end = start + len(X_continuous)
        intensity.cut(start = start, end = end, how = "index")

    # firstly initial interpolation to translate spectrum to X-axis (global phase standarization)

    interpolated_phase_first = interpolate(phase_spectrum_first, X_continuous, how = "spline")
    Y_continuous = interpolated_phase_first.Y

    if np.mean(Y_continuous) > Y_continuous[floor(len(Y_continuous)/2)]:
        Y_sampled -= np.min(Y_continuous)
    else:
        Y_sampled -= np.max(Y_continuous)

    phase_spectrum = spectrum(X_sampled, Y_sampled, "THz", "phase")

    # proper interpolation

    interpolated_phase = interpolate(phase_spectrum, X_continuous, how = "spline")
    interpolated_phase_zeros = interpolated_phase.insert_zeroes(3, inplace = False)

    # plot intermediate phase plots

    if plot_phase_inter:
        plt.scatter(X_sampled, Y_sampled, color = "orange", s = 20)
        plt.plot(interpolated_phase.X, interpolated_phase.Y, color = "green")
        plt.title("Phase with offset introduced by fiber")
        plt.legend(["Phase reconstructed from SPIDER", "Interpolated phase"], bbox_to_anchor = [1, 1])
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Spectral phase (rad)")
        plt.grid()
        plt.show()

    # plot phase

    interpolated_phase.Y = interpolated_phase.Y - chirp_phase(bandwidth = [interpolated_phase.X[0], interpolated_phase.X[-1]],
                                                              centre = interpolated_phase.X[floor(len(interpolated_phase)/2)],
                                                              fiber_length = fiber_length,
                                                              num = len(interpolated_phase))
    
    interpolated_phase.remove_temporal_phase(temp_phase_param)

    if plot_phase:

        discrete_Y = np.array([interpolated_phase.Y[np.searchsorted(interpolated_phase.X, x)] for x in X_sampled])

        plt.plot(interpolated_phase.X, interpolated_phase.Y - np.min(interpolated_phase.Y), color = "darkorange", zorder = 5)
        plt.scatter(X_sampled, discrete_Y - np.min(interpolated_phase.Y), color = "green", zorder = 10, s = 9)
        plt.fill_between(intensity.X, np.abs(intensity.Y/np.max(np.abs(intensity.Y))*np.max(np.abs(interpolated_phase.Y))), 
            color = "darkviolet", 
            alpha = 0.5, 
            zorder = 0)
        plt.title("Phase reconstruction")
        plt.legend(["Interpolated spectral phase", "Reconstructed spectral phase", "Spectral intensity"], bbox_to_anchor = [1,1])
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Spectral phase (rad)")
        plt.grid()
        plt.show()

    # extract the pulse

    intensity.insert_zeroes(3)
    pulse = recover_pulse(interpolated_phase_zeros, intensity)

    if plot_pulse:
        min_pulse = pulse.comp_quantile(0.25)
        max_pulse = pulse.comp_quantile(0.75)
        delta = (max_pulse - min_pulse)*3

        plot(pulse, color = "red", title = "Recovered pulse", start = min_pulse - delta, end = max_pulse + delta, what_to_plot = "real")

    # return what's needed

    if what_to_return == "pulse":
        return pulse
    elif what_to_return == "phase":
        return interpolated_phase
    elif what_to_return == "phase_diff":
        return diff_spectrum
    elif what_to_return == "FT":
        return s_ft_return
    elif what_to_return == "IFT":
        return s_ift2
    elif what_to_return == None:
        pass
    else:
        raise Exception("\"what to return\" must be \"pulse\", \"phase\", \"phase_diff\", \"FT\", \"IFT\" or None.")