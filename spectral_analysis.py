import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



#                   --------------
#                   SPECTRUM CLASS
#                   --------------


class spectrum:
    '''
    Class of an optical spectrum. It stores values of the \"X\" axis (wavelength, frequency or time) and the \"Y\" axis
    (intensity, phase or complex amplitude) as 1D numpy arrays. Several standard transforms and basic statistics are implemented as class methods.
    '''

    def __init__(self, X, Y, x_type = "freq", y_type = "intensity"):

        if x_type not in ["time", "freq", "wl"]:
            raise Exception("x_type must be either \"time\" or \"freq\" or \"wl\".")
        if y_type not in ["phase", "intensity", "complex_ampl"]:
            raise Exception("y_type must be either \"phase\", \"intensity\" or \"complex_ampl\".")
        if len(X) != len(Y):
            raise Exception("X and Y axis must be of the same size.")

        self.X = X
        self.Y = Y
        self.x_type = x_type
        self.y_type = y_type
        self.spacing = self.calc_spacing()


    def __len__(self):
        return len(self.X)
    

    def calc_spacing(self):
        return np.mean(np.diff(np.real(self.X)))
    

    def copy(self):
        X_copied = self.X.copy()
        Y_copied = self.Y.copy()
        return (spectrum(X_copied, Y_copied, self.x_type, self.y_type))
    

    def save(self, title):
        data = pd.DataFrame(np.transpose(np.vstack([self.X, self.Y])))
        data.to_csv(title, index = False)


    def wl_to_freq(self, inplace = True):
        '''
        Transformation from wavelength domain [nm] to frequency domain [THz].
        '''

        c = 299792458 # light speed
        freq = np.flip(c / self.X / 1e3) # output in THz
        intensity = np.flip(self.Y)

        if inplace == True:
            self.X = freq
            self.Y = intensity
            self.x_type = "freq"
            self.spacing = self.calc_spacing()
        
        else:
            return spectrum(freq, intensity, "freq", self.y_type)
        
        
    def constant_spacing(self, inplace = True):
        '''
        Transformation of a spectrum to have constant spacing on X-axis by linearly interpolating two nearest values on 
        Y-axis.
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

        if inplace == True:
            self.X = new_freq
            self.Y = new_intens
            self.spacing = self.calc_spacing()
        
        else:
            return spectrum(new_freq, new_intens, self.x_type, self.y_type)
        

    def cut(self, start = None, end = None, how = "units", inplace = True):
        '''
        Limit spectrum to a segment [\"start\", \"end\"].

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

        if inplace == True:
            self.X = new_X
            self.Y = new_Y
        
        else:
            return spectrum(new_X, new_Y, self.x_type, self.y_type)
        

    def fourier(self, inplace = True):
        '''
        Performs Fourier Transform from \"frequency\" to \"time\" domain.
        '''

        from scipy.fft import fft, fftfreq, fftshift
        
        # Exceptions

        if self.x_type == "wl":
            raise Exception("Before applying Fourier transform, transform spectrum from wavelength to frequency domain.")
        if self.x_type == "time":
            print("WARNING: Sticking to the convention you are encouraged to use the inverse Fourier transform.")

        # Fourier Transform

        FT_intens = fftshift(self.Y.copy())
        FT_intens = fft(FT_intens, norm = "ortho")
        FT_intens = fftshift(FT_intens)
        time = fftfreq(self.__len__(), self.spacing)
        time = fftshift(time)

        if inplace:
            self.X = time
            self.Y = FT_intens
            self.x_type = "time"
            self.spacing = self.calc_spacing()
        else:
            return spectrum(time, FT_intens, "time", self.y_type)
            

    def inv_fourier(self, inplace = True):
        '''
        Performs inverse Fourier transform from \"time\" to \"frequency\" domain.
        '''

        from scipy.fft import ifft, fftfreq, ifftshift, fftshift

        if self.x_type != "time":
            print("WARNING: Sticking to the convention you are encouraged to use the Fourier transform (instead for the inverse one).")

        # prepare input

        time = self.X.copy()
        intens = self.Y.copy()

        time = ifftshift(time)
        intens = ifftshift(intens)

        # Fourier Transform

        FT_intens = ifft(intens, norm = "ortho")
        FT_intens = ifftshift(FT_intens)
        freq = fftfreq(self.__len__(), self.spacing)
        freq = fftshift(freq)

        if inplace == True:
            self.X = freq
            self.Y = FT_intens
            self.x_type = "freq"
            self.spacing = self.calc_spacing()
        else:
            return spectrum(freq, FT_intens, "freq", self.y_type)
        

    def find_period(self, height = 0.5, hist = False):
        '''
        Function finds period in interference fringes by looking for wavelengths, where intensity is around given height and is decreasing. 

        ARGUMENTS:

        height - height, at which we to look for negative slope. Height is the fraction of maximum intensity.

        hist - if to plot the histogram of all found periods.

        RETURNS:

        (mean, std) - mean and standard deviation of all found periods. 
        '''

        import numpy as np
        import matplotlib.pyplot as plt

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
            plt.xlabel("Period length [nm]")
            plt.ylabel("Counts")
            plt.show()

        mean = np.mean(diff)
        std = np.std(diff)

        if len(diff) > 4:
            diff_cut = [d for d in diff if np.abs(d - mean) < std]
        else:
            diff_cut = diff 

        return np.mean(diff_cut), np.std(diff_cut)
    

    def naive_visibility(self):
        '''        
        Function returns float in range (0, 1) which is the fraction of maximum intensity, 
        at which the greatest number of fringes is observable.
        '''

        import numpy as np
        from math import floor

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
    

    def visibility(self, to_plot = False):
        '''
        Find visibility of interference fringes. First maxima and minima of fringes are found, then they are interpolated 
        and corresponding sums of intensities are calculated. Visibility is then (max_int-min_int)/max_int

        Optionally you can plot plot interpolated maxima and minima. It looks very nice.
        '''
        import numpy as np
        import spectral_analysis as sa

        safe_X = self.X.copy()
        minima = find_minima(self)
        maxima = find_maxima(self)
        inter_minima = sa.interpolate(minima, safe_X)
        inter_maxima = sa.interpolate(maxima, safe_X)
        max_sum = np.sum(inter_maxima.Y)
        min_sum = np.sum(inter_minima.Y)

        left = self.quantile(0.2)
        right = self.quantile(0.8)
        delta = right-left

        if to_plot:
            sa.compare_plots([self, inter_maxima, inter_minima], 
                             colors = ["deepskyblue", "green", "red"], 
                             title = "Visibility of {}".format(round((max_sum-min_sum)/max_sum, 3)),
                             start = left-delta, end = right+delta)
        
        return (max_sum-min_sum)/max_sum


    def replace_with_zeros(self, start = None, end = None, inplace = True):
        '''
        Replace numbers on Y axis from "start" to "end" with zeroes. "start" and "end" are in X axis' units. 
        If \"start\" and \"end\" are \"None\", then beginning and ending of whole spectrum are used as the borders.
        '''

        import numpy as np

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

        if inplace == True:
            self.Y = new_Y
        if inplace == False:
            return spectrum(self.X, new_Y, self.x_type, self.y_type)
        

    def shift(self, shift, inplace = True):
        '''
        Shifts the spectrum by X axis. Warning: only values on X axis are modified.
        '''
        import numpy as np
        from math import floor

        new_X = self.X.copy()
        new_X = new_X + shift

        if inplace == True:
            self.X = new_X
        if inplace == False:
            return spectrum(new_X, self.Y, self.x_type, self.y_type)

    
    def smart_shift(self, shift = None, inplace = True):
        '''
        Shift spectrum by rolling Y-axis. Value of shift is to be given in X axis units. If shift = \"None\",
        the spectrum is shifted so, that 1/2-order quantile for Y axis is reached for x = 0.
        '''

        import numpy as np
        import pandas as pd
        from math import floor
        import spectral_analysis as sa

        shift2 = shift

        if shift == None:
            shift2 = -self.quantile(1/2)

        index_shift = floor(np.real(shift2)/np.real(self.spacing))

        Y_new = self.Y.copy()
        Y_new = np.roll(Y_new, index_shift)
            
        if inplace == True:
            self.Y = Y_new
        if inplace == False:
            return spectrum(self.X, Y_new, self.x_type, self.y_type)
        
    def very_smart_shift(self, shift, inplace = True):
        '''
        Shift spectrum by applying FT, multiplying by linear temporal phase and applying IFT. Value of shift is to be given in X axis units.
        '''
        import numpy as np

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

    def zero_padding(self, how_much, inplace = True):
        '''
        Add zeros on Y-axis to the left and right of data with constant (mean) spacing on X-axis. 
        \"how_much\" specifies number of added zeroes on left and right side of spectrum as a fraction of spectrums length.
        '''
        import numpy as np
        from math import floor

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


    def quantile(self, q):
        '''
        Finds x in X axis such that integral of intensity to x is fraction of value q of whole intensity.
        '''
        import numpy as np

        sum = 0
        all = np.sum(np.abs(self.Y))
        for i in range(self.__len__()):
            sum += np.abs(self.Y[i])
            if sum >= all*q:
                x = self.X[i]
                break

        return x
    

    def normalize(self, norm = "highest", shift_to_zero = True, inplace = True):
        '''
        Normalize spectrum by linearly changing values of intensity and eventually shifting spectrum to zero.

        ARGUMENTS:

        by - way of normalizing the spectrum. If \"highest\", then the spectrum is scaled to 1, so that its greatest value is 1. 
        If \"intensity\", then spectrum is scaled, so that its integral is equal to 1.
        
        shift_to_zero - if \"True\", then spectrum is shifted by X axis, by simply np.rolling it.
        '''
        import numpy as np

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
            median = safe_spectrum.quantile(1/2)
            max_idx = np.searchsorted(np.abs(Y_safe), median)
            Y_safe /= integral
            if shift_to_zero:
                zero_idx = np.searchsorted(X_safe, 0)
                shift_idx = max_idx - zero_idx
                X_safe = np.roll(X_safe, shift_idx)

        if norm == "L2":
            integral = np.sum(Y_safe**2)/(X_safe[-1]-X_safe[0])
            median = safe_spectrum.quantile(1/2)
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
        
    def power(self):
        '''
        Power of a spectrum. In other words - integral of absolute value of it.
        '''
        import numpy as np
        from math import floor, log10

        # simple function to round to significant digits
        def round_to_dig(x, n):
            return round(x, -int(floor(log10(abs(x)))) + n - 1)
    
        return  round_to_dig(np.sum(self.X*np.conjugate(self.X)), 3)
        
    def FWHM(self):
        '''
        Calculate Full Width at Half Maximum. If multiple peaks are present in spectrum, the function might not work properly.
        '''

        import numpy as np

        left = None
        right = None
        peak = np.max(self.Y)
        for idx, y in enumerate(self.Y):
            if y >= peak/2:
                left = idx
        for idx, y in enumerate(np.flip(self.Y)):
            if y >= peak/2:
                right = self.__len__() - idx
        if self.__len__() == 0:
            raise Exception("Failed to calculate FWHM, because spectrum is empty.")
        if self.__len__() < 5:
            raise Exception("The spectrum consists of too little data points to calculate FWHM.")
        if left == None or right == None:
            raise Exception("Failed to calculate FWHM due to very strange error.")
        width = right-left
        width *= self.spacing

        return np.abs(width)


    def remove_offset(self, period, inplace = True):
        '''
        The function improves visibility of interference fringes by subtracting moving minimum i. e. 
        minimum of a segment of length \"period\", centered at given point.
        '''

        import numpy as np
        from math import floor as flr

        idx_period = flr(period/self.spacing)

        new_Y = []
        for i in range(self.__len__()):
            left = np.max([i - flr(idx_period/2), 0])
            right = np.min([i + flr(idx_period/2), self.__len__() - 1])
            new_Y.append(self.Y - np.min(self.Y[left:right]))
        new_Y = np.array(new_Y)

        if inplace == True:
            self.Y = new_Y

        if inplace == False:
            return spectrum(self.X, new_Y, self.x_type, self.y_type)


    def moving_average(self, period, inplace = True):
        '''
        Smooth spectrum by taking moving average with \"period\" in X axis units.
        On the beginning and ending of spectrum shorter segments are used.
        '''

        from math import floor as flr
        import numpy as np

        idx_period = flr(period/self.spacing)

        new_Y = []
        for i in range(self.__len__()):
            left = np.max([i - flr(idx_period/2), 0])
            right = np.min([i + flr(idx_period/2), self.__len__() - 1])
            new_Y.append(np.mean(self.Y[left:right]))
        new_Y = np.array(new_Y)

        if inplace == True:
            self.Y = new_Y

        if inplace == False:
            return spectrum(self.X, new_Y, self.x_type, self.y_type)
        

    def moving_average_interior(self, period, inplace = True):
        '''
        Smooth the INTERIOR of the spectrum by taking moving average with \"period\" in X axis units.
        On the beginning and ending of spectrum shorter segments are used.
        '''

        from math import floor as flr
        import numpy as np

        sup = np.max(self.Y)
        support = self.X[self.Y >= sup/2]
        left_border = support[0]
        right_border = support[-1]
        left_border_idx = np.searchsorted(self.X, left_border)
        right_border_idx = np.searchsorted(self.X, right_border)

        idx_period = flr(period/self.spacing)

        new_Y = []
        for i in range(self.__len__()):
            if i < left_border_idx + 1 or i > right_border_idx - 1:
                new_Y.append(self.Y[i])
                continue
            smooth_range = np.min([idx_period, 2*np.abs(left_border_idx - i), 2*np.abs(right_border_idx - i)])
            left = np.max([i - flr(smooth_range/2), 0])
            right = np.min([i + flr(smooth_range/2), self.__len__() - 1])
            new_Y.append(np.mean(self.Y[left:right]))
        new_Y = np.array(new_Y)

        if inplace == True:
            self.Y = new_Y

        if inplace == False:
            return spectrum(self.X, new_Y, self.x_type, self.y_type)



#                   ------------------------
#                   SPECTRUM CLASS FUNCTIONS
#                   ------------------------



def load_csv(filename, x_type = "wl", y_type = "intensity", rows_to_skip = 2):
    '''
    Load CSV file to a spectrum class. Spectrum has on default wavelengths on X axis and intensity on Y axis.
    '''
    import pandas as pd
    spectr = pd.read_csv(filename, skiprows = rows_to_skip)
    return spectrum(spectr.values[:, 0], spectr.values[:, 1], x_type = x_type, y_type = y_type)


def load_tsv(filename, x_type = "wl", y_type = "intensity"):
    '''
    Load TSV file to a spectrum class. Spectrum has on default wavelengths on X axis and intensity on Y axis.
    '''
    import pandas as pd
    spectr = pd.read_table(filename, skiprows = 2)
    return spectrum(spectr.values[:, 0], spectr.values[:, 1], x_type = x_type, y_type = y_type)


def interpolate(old_spectrum, new_X):
    '''
    Interpolate rarely sampled spectrum for values in new X-axis. Interpolation is performed with cubic functions. 
    If y-values to be interpolated are complex, they are casted to reals.

    ARGUMENTS:

    old_spectrum - spectrum that is the basis for interpolation.

    new_X - new X axis (i. e. frequencies or wavelengths). Interpolated Y-values are calculated for values from new_X.

    RETURNS:

    Interpolated spectrum.
    '''
    
    import numpy as np
    from scipy.interpolate import CubicSpline

    X = np.real(old_spectrum.X.copy())
    Y = np.real(old_spectrum.Y.copy())

    model = CubicSpline(X, Y)
    new_Y = model(np.real(new_X))

    return spectrum(new_X, new_Y, old_spectrum.x_type, old_spectrum.y_type)


def create_complex_spectrum(intensity_spectrum, phase_spectrum, extrapolate = False):
    '''
    Given the intensity spectrum and the phase spectrum compute the complex amplitude spectrum. With \"extrapolate = False\" the spectral phase of the spectrum
    is zeroed outside the area with the well-known phase.
    '''

    import spectral_analysis as sa
    import numpy as np

    if intensity_spectrum.y_type != "intensity":
        print("WARNING: Are you sure that the \"intensity_spectrum\" carries the information about intensity?")
    if phase_spectrum.y_type != "phase":
        print("WARNING: Are you sure that the \"phase_spectrum\" carries the information about phase?")

    support_left = np.searchsorted(intensity_spectrum.X, phase_spectrum.X[0])
    support_right = np.searchsorted(intensity_spectrum.X, phase_spectrum.X[-1])
    new_phase = sa.interpolate(phase_spectrum, intensity_spectrum.X)

    if not extrapolate:
        for i in range(len(intensity_spectrum)):
            if i < support_left or i > support_right:
                new_phase.Y[i] = 0

    amplitude = intensity_spectrum.Y * np.exp(1j*new_phase.Y)
    
    return sa.spectrum(intensity_spectrum.X, amplitude, intensity_spectrum.x_type, "complex_ampl")


def fit_fiber_length(phase_spectrum, plot = False, guessed_length = 80):
    '''
    Fit parabolic spectral phase to the given spectrum and return the length of chirping fiber corresponding to that phase.
    '''
    from scipy.optimize import curve_fit
    import spectral_analysis as sa
    import numpy as np

    def chirp_phase(frequency, centre, fiber_length):
        c = 299792458 
        l_0 = c/(centre*1e3)
        D_l = 20
        omega = frequency*2*np.pi
        omega_mean = centre*2*np.pi
        return l_0**2*fiber_length*D_l/(4*np.pi*c)*(omega-omega_mean)**2

    param, cov = curve_fit(chirp_phase, phase_spectrum.X, phase_spectrum.Y, [192, guessed_length], bounds = [[150, 30],[300, 300]])

    if plot:
        new_X = phase_spectrum.X.copy()
        fit_phase = sa.spectrum(new_X, chirp_phase(new_X, param[0], param[1]), x_type = "freq", y_type = "phase")
        sa.compare_plots([phase_spectrum, fit_phase], legend = ["Original spectrum", "Phase corresponding to fiber of {}m".format(round(param[1]))])

    return np.abs(param[1])
    

def chirp_r2(phase_spectrum, fiber_length, plot = False):
    '''
    Given spectral phase spectrum and the length of fiber employed in experiment, fit corresponding spectral phase to
    experimental data (only fitting center) and return the R^2 statistics judging quality of that fit.
    Additionally you can plot both phases on a single plot.
    '''
    from scipy.optimize import curve_fit
    import spectral_analysis as sa
    import numpy as np

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

    if plot:
        reality = sa.spectrum(phase_spectrum.X.copy(), Y_real, x_type = "freq", y_type = "phase")
        prediction = sa.spectrum(phase_spectrum.X.copy(), Y_pred, x_type = "freq", y_type = "phase")

        if np.mean(reality.Y) < 0:
            reality.Y *= -1
            prediction.Y *= -1

        sa.compare_plots([reality, prediction], 
                         title = "Experimental and model chirp phase for fiber of {}m\nR-squared value equal to {}".format(fiber_length, round(score, 3)),
                         legend = ["Experiment", "Model"],
                         colors = ["darkorange", "darkgreen"])
    return score


def fit_rect(spectr, output = "params", fixed_area = True, slope = False, slope_factor = 0.1):
    '''
    Fit a rectangle to the given spectrum. If output = \"params\" then tuple (start, end, height) characterizing the shape of rectangle is returned.
    If output = \"spectrum\" returns fitted rectangle as a spectrum class object. If fixed_area = True, than fitted spectrum has the same area as the initial.
    If slope = True, than not a rectangle but a trapeze is fitted. The slope of its sides is the greater, the greater slope_factor is.
    '''
    from scipy.optimize import curve_fit
    from spectral_analysis import spectrum
    import numpy as np

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
    With functions like that I impress myself. output must be either \"params\" - then a tuple (a, b, height) is returned -
    or \"spectrum\", resulting in returning a fitted rectangle spectrum
    '''
    import numpy as np
    from math import floor
    import spectral_analysis as sa

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
        loss_spectrum = sa.spectrum(spectr.X[start_idx:end_idx], Y, spectr.x_type, spectr.y_type)

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
        return sa.spectrum(new_X, new_Y, spectr.x_type, spectr.y_type)
    
    else:
        raise Exception("output must be either \"params\" or \"spectrum\"")


def find_minima(fringes_spectrum):
    '''
    Find minima of interference fringes by looking at nearest neighbors. Spectrum with minima is returned.
    '''
    import spectral_analysis as sa

    X = []
    Y = []
    for i in range(len(fringes_spectrum)):
        if i in [0, 1, len(fringes_spectrum) - 2, len(fringes_spectrum) - 1]:
            continue
        if fringes_spectrum.Y[i-2] >= fringes_spectrum.Y[i-1] >= fringes_spectrum.Y[i] <= fringes_spectrum.Y[i+1] <= fringes_spectrum.Y[i+2]:
            X.append(fringes_spectrum.X[i])
            Y.append(fringes_spectrum.Y[i])
    
    return sa.spectrum(np.array(X), np.array(Y), fringes_spectrum.x_type, fringes_spectrum.y_type)


def find_maxima(fringes_spectrum):
    '''
    Find maxima of interference fringes by looking at nearest neighbors. Spectrum with maxima is returned.
    '''
    import spectral_analysis as sa
    import numpy as np

    X = []
    Y = []
    for i in range(len(fringes_spectrum)):
        if i in [0, 1, len(fringes_spectrum) - 2, len(fringes_spectrum) - 1]:
            continue
        if fringes_spectrum.Y[i-2] <= fringes_spectrum.Y[i-1] <= fringes_spectrum.Y[i] >= fringes_spectrum.Y[i+1] >= fringes_spectrum.Y[i+2]:
            X.append(fringes_spectrum.X[i])
            Y.append(fringes_spectrum.Y[i])
    
    return sa.spectrum(np.array(X), np.array(Y), fringes_spectrum.x_type, fringes_spectrum.y_type)


def plot(spectrum_safe, color = "darkviolet", title = "Spectrum", what_to_plot = "abs", start = None, end = None, save = False):
    '''
    Fast spectrum plotting using matplotlib.pyplot library.

    ARGUMENTS:

    spectrum - the spectrum object class to be plotted.

    color - color of the plot.

    title - title of the plot.

    what_to_plot - either \"abs\", \"imag\", \"real\", \"complex\" or \"trigonometric\". In the last two cases two curves are plotted.

    start - starting point (in X-axis units) of a area to be shown on plot. If \"min\", then plot starts with lowest X-value in all spectra.

    end - ending point (in X-axis units) of a area to be shown on plot. If \"max\", then plot ends with highest X-value in all spectra.

    save - if to save the plot in the program's directory. The title of the plot will be by default used as the filename.
    '''

    import numpy as np
    import matplotlib.pyplot as plt
    from math import floor, log10

    spectrum_safe = spectrum_safe.copy()
    
    # simple function to round to significant digits

    def round_to_dig(x, n):
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

    f, ax = plt.subplots()

    if what_to_plot == "complex":
        plt.plot(spectrum_safe.X, np.real(spectrum_safe.Y), color = "darkviolet")
        plt.plot(spectrum_safe.X, np.imag(spectrum_safe.Y), color = "green")
        plt.legend(["Real part", "Imaginary part"], bbox_to_anchor = [1, 0.17])

    elif what_to_plot == "trigonometric":
        phase = np.angle(spectrum_safe.Y)
        intensity = np.abs(spectrum_safe.Y)

        start_idx = 0
        end_idx = len(phase) -1
        for i in range(len(phase)):
            if phase[i] != 0:
                start_idx  = i
                break
        for i in reversed(range(len(phase))):
            if phase[i] != 0:
                end_idx  = i
                break

        phase = np.unwrap(phase)
        intensity /= np.max(np.abs(intensity))  # normalize intensity with respect to phase
        intensity *= np.max(np.abs(phase[start_idx: end_idx]))

        plt.plot(spectrum_safe.X, intensity, color = "orange", alpha = 0.5)
        plt.plot(spectrum_safe.X[start_idx: end_idx], phase[start_idx: end_idx], color = "darkviolet")
        plt.legend(["Spectral intensity", "Spectral phase"], bbox_to_anchor = [1, 0.17])

    else:
        plt.plot(spectrum_safe.X, spectrum_safe.Y, color = color)
     
    plt.grid()
    if to_cut:
        plt.title(title + " [only part shown]")
    else:
        plt.title(title)
    if spectrum_safe.y_type in ["phase", "complex_ampl"] :
        plt.ylabel("Spectral phase [rad]")
    if spectrum_safe.y_type == "intensity":
        plt.ylabel("Intensity")    
    if spectrum_safe.x_type == "wl":
        plt.xlabel("Wavelength (nm)")
        unit = "nm"
    if spectrum_safe.x_type == "freq":
        plt.xlabel("Frequency (THz)")
        unit = "THz"
    if spectrum_safe.x_type == "time":
        plt.xlabel("Time (ps)")
        unit = "ps"

    # quick stats
    
    spacing = round_to_dig(spectrum_safe.spacing, 3)
    p_per_unit = floor(1/spectrum_safe.spacing)

    if to_cut:
        plt.text(1.05, 0.85, "Number of points: {}\nX-axis spacing: {} ".format(n_points, spacing) + unit + "\nPoints per 1 " + unit +": {}".format(p_per_unit) + "\nFull X-axis range: {} - {} ".format(inf, sup) + unit , transform = ax.transAxes)
    else:
        plt.text(1.05, 0.9, "Number of points: {}\nX-axis spacing: {} ".format(n_points, spacing) + unit + "\nPoints per 1 " + unit +": {}".format(p_per_unit) , transform = ax.transAxes)

    if save:
        plt.savefig("{}.jpg".format(title))

    plt.show()


def compare_plots(spectra, title = "Spectra", legend = None, colors = None, start = None, end = None, abs = False):
    '''
    Show several spectra on single plot.

    ARGUMENTS:

    spectra - list with spectra to be show on plot.

    title - title of the plot.

    legend - list with names of subsequent spectra. If \"None\", then no legend is shown.

    start - starting point (in X-axis units) of a area to be shown on plot. If \"min\", then plot starts with lowest X-value in all spectra.

    end - ending point (in X-axis units) of a area to be shown on plot. If \"max\", then plot ends with highest X-value in all spectra.

    abs - if \"True\", then absolute values of spectra is plotted.
    '''
    
    import matplotlib.pyplot as plt
    import numpy as np

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
    if spectra[0].x_type == "wl":
        plt.xlabel("Wavelength (nm)")
    if spectra[0].x_type == "freq":
        plt.xlabel("Frequency (THz)")
    if spectra[0].x_type == "time":
        plt.xlabel("Time (ps)")
    if isinstance(legend, list):
        plt.legend(legend, bbox_to_anchor = [1, 1])
    plt.show()       


def recover_pulse(phase_spectrum, intensity_spectrum):
    '''
    Reconstructs the pulse (or any type of spectrum in time), given spectrum with spectral phase and spectrum with spectral intensity.
    '''

    import numpy as np

    if len(phase_spectrum) != len(intensity_spectrum):
        raise Exception("Frequency axes of phase and intensity are not of equal length.")
    
    complex_Y = intensity_spectrum.Y.copy()
    complex_Y = complex_Y.astype(complex) # surprisingly that line is necessary
    for i in range(len(complex_Y)):
        complex_Y[i] *= np.exp(1j*phase_spectrum.Y[i])

    pulse_spectrum = spectrum(phase_spectrum.X.copy(), complex_Y, "freq", "intensity")
    pulse_spectrum.fourier()
    pulse_spectrum.Y = 2*np.real(pulse_spectrum.Y)
    return pulse_spectrum


def find_shift(spectrum_1, spectrum_2):
    '''
    Returns translation between two spectra in THz. Spectra in wavelength domain are at first transformed into frequency domain.
    Least squares is loss function to be minimized. Shift is found by brute force: 
    checking number of shifts equal to number of points on X axis.
    '''

    import numpy as np
    from math import floor

    spectrum1 = spectrum_1.copy()
    spectrum2 = spectrum_2.copy()

    if len(spectrum1) != len(spectrum2):
        raise Exception("Spectra are of different length.")

    if spectrum1.x_type == "wl":

        spectrum1.wl_to_freq()
        spectrum1.constant_spacing()

    if spectrum2.x_type == "wl":

        spectrum2.wl_to_freq()
        spectrum2.constant_spacing()

    def error(v_1, v_2):
        return np.sum(np.abs(v_1 - v_2)**2)
    
    minimum = np.sum(np.abs(spectrum1.Y)**2)
    idx = 0
    width = np.max(np.array([spectrum1.FWHM(), spectrum2.FWHM()]))
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
    import numpy as np

    def vis(x):
        '''Inverse function of ratio.'''
        return 4*np.sqrt(x)/(1+np.sqrt(x))**2
    
    vis = np.vectorize(vis)
    X = np.linspace(0, 1, 20000)
    Y = vis(X)
    r = np.searchsorted(Y, vis_value)

    return X[r]


def gaussian_pulse(bandwidth, centre, FWHM, x_type = "freq", num = 1000):
    '''
    Creates spectrum with gaussian intensity. "bandwidth" is a tuple with start and the end of the entire spectrum. 
    "centre" and "FWHM" characterize the pulse itself. The spectrum is composed of \"num\" = 1000 points on default.
    '''

    import spectral_analysis as sa
    X = np.linspace(bandwidth[0], bandwidth[1], num = num)
    sd = FWHM/2.355
    def gauss(x, mu, std):
        return 1/(std*np.sqrt(2*np.pi))*np.exp(-(x-mu)**2/(2*std**2))
    gauss = np.vectorize(gauss)
    Y = gauss(X, centre, sd)
    return sa.spectrum(X, Y, "freq", "intensity")


def hermitian_pulse(pol_num, bandwidth, centre, FWHM, num = 1000, x_type = "freq"):
    '''
    Creates spectrum with \"pol-num\"-th Hermit-Gauss intensity mode. "bandwidth" is a tuple with start and the end of the entire spectrum. 
    "centre" and "FWHM" characterize the pulse itself. The spectrum is composed of \"num\" = 1000 points on default.
    '''

    # exceptions

    if x_type not in ["freq", "wl", "time"]:
        raise Exception("x_type must be either \"freq\", \"nm\" or \"time\"")

    # modules

    import spectral_analysis as sa
    from scipy.special import hermite as hermite_gen

    # and calculations

    hermite_pol = hermite_gen(pol_num)
    def gauss(x, mu, std):
        return 1/(std*np.sqrt(2*np.pi))*np.exp(-(x-mu)**2/(2*std**2))
    
    X = np.linspace(bandwidth[0], bandwidth[1], num = num)
    sd = FWHM/2.355
    Y_gauss = gauss(X, centre, sd)
    Y_hermite = hermite_pol(2*(X-centre)/FWHM)
    Y_out = Y_hermite*Y_gauss

    spectrum_out = sa.spectrum(X, Y_out, "freq", "intensity")
    spectrum_out.normalize(norm = "L2", shift_to_zero = False)

    return spectrum_out


def chirp_phase(bandwidth, centre, fiber_length):
    '''
    Creates spectrum class object with spectral phase corresponding to propagation of a pulse through "fiber_length" of PM fiber.
    "bandwidth" is a tuple with start and the end of the entire spectrum OR already created X-axis of the spectrum. 
    "centre" determines the minimum of spectral phase.
    '''
    c = 299792458 
    l_0 = c/(centre*1e3)
    D_l = 17
    if len(bandwidth) == 2:
        X = np.linspace(bandwidth[0], bandwidth[1], 1000)
    else:
        X = bandwidth.copy()
    omega = X*2*np.pi
    omega_mean = centre*2*np.pi
    return l_0**2*fiber_length*D_l/(4*np.pi*c)*(omega-omega_mean)**2


def find_slope_shift(sheared_spectrum, not_sheared_spectrum, low = 0.1, high = 0.3, sampl_num = 500):

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


def find_shear(sheared_spectrum, not_sheared_spectrum, smoothing_period = None, how = "slope", plot = False):
    '''
    Find shear between two spectra given names of two csv files with those spectral, optional \"smoothing period\". 
    You can find the shift by fitting it (how = \"fit\"), by calculating the shift between centers of mass (how = \"com\"),
    by computing the shift of the slope (how = \"slope\") or by calculating the shift of fitted rectangle (how = \"rect\")
    '''

    import spectral_analysis as sa
    import numpy as np

    if how not in ["com", "fit", "slope", "rect"]:
        raise Exception("\"how\" must be equal either to \"com\",\"fit\", \"slope\" or \"rect\".")

    sheared = sa.load_csv(sheared_spectrum)
    not_sheared = sa.load_csv(not_sheared_spectrum)

    sheared.wl_to_freq()
    not_sheared.wl_to_freq()
    sheared.constant_spacing()
    not_sheared.constant_spacing()

    if smoothing_period != None:
        sheared.moving_average_interior(smoothing_period)
        not_sheared.moving_average_interior(smoothing_period)

    if how == "com":
        shear = sheared.quantile(0.5)-not_sheared.quantile(0.5)

    elif how == "fit":
        shear = sa.find_shift(sheared, not_sheared)

    elif how == "slope":
        shear = sa.find_slope_shift(sheared, not_sheared)

    elif how == "rect":
        params_s = sa.fit_rect_smart(sheared, output = "params")
        params_ns = sa.fit_rect_smart(not_sheared, output = "params")
        shear = np.mean([params_s[0]-params_ns[0], params_s[1]-params_ns[1]])

    if plot:    
        sa.compare_plots([sheared, not_sheared], 
                         legend = ["Sheared spectrum", "Not sheared spectrum"], 
                         title = "Shear = {} THz".format(round(shear, 5)))
        sa.compare_plots([sheared.shift(-shear, inplace = False), not_sheared], 
                         legend = ["Resheared sheared spectrum", "Not sheared spectrum"], 
                         title = "Shifting sheared spectrum to the \"zero\" position".format(round(shear, 5)),
                         colors = ["green", "orange"])
    
    return shear



#                   ---------
#                   BEAM CLASS
#                   ---------



class beam:
    import numpy as np

    def __init__(self, vertical_polarization, horizontal_polarization):
        self.ver = vertical_polarization
        self.hor = horizontal_polarization


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
        import spectral_analysis as sa

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


    def shear(self, shift, polarization):
        import spectral_analysis as sa

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
        h = np.real(h)
        v = np.real(v)
        h = round(h)
        v = round(v)
        print(u"Total power measured: {} \u03bcW\nPower on horizontal polarization: {} \u03bcW\nPower on vertical polarization: {} \u03bcW".format(h+v, h, v))

    def OSA(self, start = None, end = None):
        import spectral_analysis as sa

        Y = self.hor.Y*np.conjugate(self.hor.Y) + self.ver.Y*np.conjugate(self.ver.Y)
        spectr = spectrum(self.hor.X, Y, "freq", "intensity")
        spectr.spacing = np.abs(spectr.spacing)
        sa.plot(spectr, title = "OSA", start = start, end = end, color = "green")
        
        return spectr



#                   ----------------
#                   OSA measurements
#                   ----------------



def measurement(centre = 1550, span = 10):
    '''
    Performs single measurement with OSA and returns it as \"spectrum\" class object. \"centre\" and \"span\" are in nm. 
    '''
    import pyvisa
    import pylab as plt
    import numpy as np
    import spectral_analysis as sa
    from pyvisa.constants import VI_FALSE, VI_ATTR_SUPPRESS_END_EN, VI_ATTR_SEND_END_EN

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

    osa_spectrum = sa.spectrum(Lambda, intensity, "wl", "intensity")
    return osa_spectrum


def OSA(centre = 1550, span = 10, plot_size = [10, 10]):
    '''
    Plots continuous in time spectrum from OSA. \"plot_size\" is tuple of dimensions of image to be shown in cm
    '''
    import matplotlib.pyplot as plt
    import spectral_analysis as sa
    from IPython.display import display, clear_output
    import time

    fig, ax = plt.subplots(fig_size = [plot_size[0]/2.54, plot_size[1]/2.54])

    start_time = time.time()
    while True:
        current_time = time.time()
        if current_time - start_time > 1800:
            raise RuntimeWarning("Measurement lasted longer than 0.5h.")
        pure_spectrum = sa.measurement(centre = centre, span = span)
        ax.clear()
        ax.plot(pure_spectrum.X, pure_spectrum.Y, color = "red")
        ax.set_title("OSA")
        ax.set_xlabel("Wavelength [nm]")
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
           find_shear = "least squares",
           sheared_is_bigger = True,
           plot_steps = False, 
           plot_phase = True,
           plot_shear = False, 
           plot_pulse = False):
    '''
    Performs SPIDER algorithm.

    ARGUMENTS:

    phase_spectrum - spectrum with interference pattern created in a SPIDER setup. 
        It should be either spectrum object with wavelength or frequency in X-column OR path of .csv file with that data.

    temporal_spectrum - analogous to phase_spectrum, but measured with EOPM off.

    shear - spectral shear applied by EOPM given in frequency units (default THz). 
        If \"None\", then shear is estimated using fourier filtering.

    intensity_spectrum - amplitude of initial not interfered pulse. Similar as "phase_spectrum" it might be either a spectrum object or pathname. 
        If "None", then its approximation derived from SPIDER algorithm is used.  
    
    phase_borders - specify range of frequencies (in THz), where you want to find spectral phase. 
        If to big, boundary effects may appear. If "None", the borders are estimated by calculating quantiles of intensity.

    what_to_return - if None, then RETURNS nothing. If "pulse", then RETURNS spectrum with reconstructed pulse. 
        If "phase", then RETURNS tuple with two spectra - phase and interpolated phase. If "phase_diff", then RETURNS the spectrum with
        spectral phase difference. If "FT", then RETURNS the spectrum withFourier transform of the OSA spectrum.

    smoothing_period - if None, nothing happens. Otherwise the phase difference is smoothed by taking 
        a moving average within an interval of smoothing_period.
    
    find_shear - method of finding shear. If "center of mass", then shift of center of mass between sheared and non-sheared spectrum is computed. 
        If "least_squares", then shear is found as value of shift minimizing squared difference between spectra. If "slope", then finds the first argument,
        where spectrum reaches 0.1 of maximum, 0.11 of maximum, etc. Shear is estimated as mean of shifts of these points.

    plot_steps - if to plot all intermediate steps of the SPIDER algorithm.

    plot_phase - if to plot the found spectral phase.

    plot shear - if to plot the spectra used to find the shear.

    plot_pulse - if to plot the reconstructed pulse.
    '''

    import pandas as pd
    import numpy as np
    from math import floor as flr
    import spectral_analysis as sa
    import matplotlib.pyplot as plt

    # load data - spider

    if isinstance(phase_spectrum, spectrum):
        p_spectrum = phase_spectrum.copy()

    elif isinstance(phase_spectrum, str):
        p_spectrum = sa.load_csv(phase_spectrum)    

    else:
        raise Exception("Wrong \"phase_spectrum\" format.")
    
    # load data - temporal phase

    if isinstance(temporal_spectrum, spectrum):
        t_spectrum = temporal_spectrum.copy()

    elif isinstance(temporal_spectrum, str):
        t_spectrum = sa.load_csv(temporal_spectrum)    

    else:
        raise Exception("Wrong \"temporal_spectrum\" format.")

    # zero padding

    t_spectrum.zero_padding(3)
    p_spectrum.zero_padding(3)

    # plot OSA

    min_wl = p_spectrum.quantile(0.1)
    max_wl = p_spectrum.quantile(0.9)
    delta = (max_wl - min_wl)
    min_wl -= delta
    max_wl += delta 

    if plot_steps:
        sa.plot(p_spectrum, "orange", title = "Data from OSA", start = min_wl, end = max_wl)

    # transform X-axis to frequency

    # spider

    if p_spectrum.x_type == "wl":
        s_freq = p_spectrum.wl_to_freq(inplace = False)
        s_freq.constant_spacing()
        min_freq = s_freq.quantile(0.1)
        max_freq = s_freq.quantile(0.9)
        delta = (max_freq - min_freq)
        min_freq -= delta
        max_freq += delta
        if plot_steps: 
            sa.plot(s_freq,"orange", title = "Wavelength to frequency", start = min_freq, end = max_freq)

    elif p_spectrum.x_type == "freq":
        s_freq = p_spectrum
        # we need following lines, because we want in every case have min_freq and max_freq defined for later
        min_freq = s_freq.quantile(0.1)
        max_freq = s_freq.quantile(0.9)
        delta = (max_freq - min_freq)
        min_freq -= delta
        max_freq += delta

    s_freq_for_later = s_freq.copy()

    # temporal

    if t_spectrum.x_type == "wl":
        s_freq_t = t_spectrum.wl_to_freq(inplace = False)
        s_freq_t.constant_spacing()

    elif t_spectrum.x_type == "freq":
        s_freq_t = t_spectrum

    # fourier transform

    s_ft = s_freq.fourier(inplace = False)         # spider
    s_ft_t = s_freq_t.fourier(inplace = False)     # temporal

    s_shear = s_ft.copy()       # SPOILER: we will use it later to find the shear
    s_shear_t = s_ft_t.copy()
    s_intens = s_ft_t.copy()    # and this will be used to reconstruct the pulse

    min_time = s_ft.quantile(0.1)
    max_time = s_ft.quantile(0.9)
    delta = (max_time-min_time)/3
    min_time -= delta
    max_time += delta
   
    if plot_steps: 
        sa.plot(s_ft, title = "Fourier transformed", start = min_time, end = max_time) 

    # estimate time delay
    
    period = s_freq_t.find_period(1/3+2/3*(1-s_freq_t.visibility()))[0] # look for the fringes in 1/3 distance between visibility level and max intensity
    delay = 1/period

    # find exact value of time delay

    s_ft2 = s_ft.copy()
    s_ft_return = s_ft.copy()        # if you wish it to return it later
    s_ft2.replace_with_zeros(end = delay*0.5)
    s_ft2.replace_with_zeros(start = delay*1.5)
    idx = s_ft2.Y.argmax()
    if isinstance(idx, np.ndarray): 
        idx = idx[0]
    delay2 = s_ft.X[idx]

    # and filter the spectrum to keep only one of site peaks
    
    s_ft.replace_with_zeros(end = delay2*0.5)           # spider
    s_ft.replace_with_zeros(start = delay2*1.5)
    s_ft_t.replace_with_zeros(end = delay2*0.5)         # temporal
    s_ft_t.replace_with_zeros(start = delay2*1.5)

    if plot_steps: 
        sa.plot(s_ft, title = "Filtered (absolute value)", start = -2*delay2, end = 2*delay2, what_to_plot = "abs")

    # let's find the shear
    
    if shear == None:

        s_shear.replace_with_zeros(start = None, end = -delay2*0.5)         # spider
        s_shear.replace_with_zeros(start = delay2*0.5, end = None)
        
        s_shear_t.replace_with_zeros(start = None, end = -delay2*0.5)       # temporal
        s_shear_t.replace_with_zeros(start = delay2*0.5, end = None)

        s_shear.inv_fourier()
        s_shear_t.inv_fourier()

        s_shear.Y = np.abs(s_shear.Y)
        s_shear_t.Y = np.abs(s_shear_t.Y)

        mu = sa.ratio(t_spectrum.visibility())
        if sheared_is_bigger:
            s_shear_t.Y /= (1+mu)
            s_shear.Y -= (mu*s_shear_t.Y)
        else:
            s_shear_t.Y /= (1+mu)
            s_shear.Y -= (s_shear_t.Y)
            s_shear.Y /= mu
        
        if find_shear == "least squares":
            shear = sa.find_shift(s_shear, s_shear_t)
        elif find_shear == "center of mass":
            shear = s_shear.quantile(1/2) - s_shear_t.quantile(1/2)
        elif find_shear == "slope":
            shear = sa.find_slope_shift(s_shear, s_shear_t)
        else:
            raise Exception("\"find_shear\" must be either \"least squares\", \"center of mass\" or \"slope\".")
        
        shear = np.abs(shear)
        
        if shear == 0:
            raise Exception("Failed to find non zero shear.")
        if plot_shear:
            sa.compare_plots([s_shear, s_shear_t], 
                             start = 1.5*s_shear.quantile(0.05), 
                             end = 1.5*s_shear.quantile(0.95), 
                             abs = True, 
                             title = "Shear of {} THz".format(round(shear,5)),
                             legend = ["Sheared", "Not sheared"])

    integrate_interval = flr(np.abs(shear)/(s_freq_for_later.spacing))
    mean = np.mean(s_freq_for_later.X)

    # inverse fourier

    s_ift = s_ft.inv_fourier(inplace = False)        # spider
    s_ift_t = s_ft_t.inv_fourier(inplace = False)    # temporal
    if plot_steps:
        s_ift2 = s_ift.copy()
        s_ift2.X += np.real(mean) 
        sa.plot(s_ift2, title = "Inverse Fourier transformed", start = min_freq, end = max_freq, what_to_plot = "abs")

    # cut spectrum to area of significant phase

    if phase_borders == None:
        min_phase = s_ift.quantile(0.05)
        max_phase = s_ift.quantile(0.95)

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
    values = np.unwrap(values)
    values -= np.mean(values)
    #values -= values[0]             # deletes linear aberration in phase

    if smoothing_period != None:
        V = sa.spectrum(X_sampled, values, "freq", "phase")
        V.moving_average(smoothing_period)
        values = V.Y

    #values -= values[flr(len(values)/2)]    # without that line, spectral phase will later always start at zero and grow

    # prepare data to plot
    
    X_sampled += mean
    X_continuous = X_sampled.copy()

    # plot phase difference

    diff_spectrum = spectrum(X_sampled, values, "freq", "phase")    # if you wish to return it later
    if plot_phase:

        plt.scatter(X_sampled, np.real(values), color = "orange", s = 1)
        if smoothing_period == None:
            plt.title("Spectral phase difference between pulse and its sheared copy")
        else:
            plt.title("Spectral phase difference between pulse and its sheared copy\n[smoothed with period of {} THz].".format(smoothing_period))

        plt.xlabel("Frequency [THz]")
        plt.ylabel("Spectral phase")
        plt.grid()
        plt.show()

    # recover intensity spectrum

    if intensity_spectrum == None:
        s_intens.replace_with_zeros(end = -0.5*delay2)      # DC filtering
        s_intens.replace_with_zeros(start = 0.5*delay2)

        intensity = s_intens.inv_fourier(inplace = False)
        mu = sa.ratio(t_spectrum.visibility())
        intensity.Y /= (1+mu)
        intensity.cut(min_phase, max_phase)
        intensity.X += mean + shear
    
    else:
        if isinstance(intensity_spectrum, spectrum):
            intensity = intensity_spectrum
        elif isinstance(intensity_spectrum, str):
            intensity = sa.load_csv(intensity_spectrum)

        if intensity.x_type == "wl":
            intensity.wl_to_freq()
            intensity.constant_spacing()

        start = np.searchsorted(intensity.X, X_continuous[0])
        num = len(X_continuous)
        end = start + num
        intensity.cut(start = start, end = end, how = "index")

    # recover discrete phase
    
    integration_start = flr((len(X_sampled) % integrate_interval)/2) # we want to extrapolate equally on both sides

    X_sampled = X_sampled[integration_start::integrate_interval]

    values = values[integration_start::integrate_interval]
    Y_sampled = np.cumsum(values)

    if shear < 0: 
        X_sampled -= shear
        X_continuous -= shear
        Y_sampled *= -1

    phase_spectrum_first = spectrum(X_sampled, Y_sampled, "freq", "phase")

    # firstly initial interpolation to translate spectrum to X-axis (global phase standarization)

    interpolated_phase_first = sa.interpolate(phase_spectrum_first, X_continuous)
    Y_continuous = interpolated_phase_first.Y

    if np.mean(Y_continuous) > Y_continuous[flr(len(Y_continuous)/2)]:
        Y_sampled -= np.min(Y_continuous)
    else:
        Y_sampled -= np.max(Y_continuous)

    phase_spectrum = spectrum(X_sampled, Y_sampled, "freq", "phase")

    # proper interpolation

    interpolated_phase = sa.interpolate(phase_spectrum, X_continuous)
    interpolated_phase_zeros = interpolated_phase.zero_padding(3, inplace = False)  

    # plot phase

    if plot_phase:
        plt.scatter(X_sampled, Y_sampled, color = "orange", s = 20)
        plt.plot(interpolated_phase.X, interpolated_phase.Y, color = "green")
        plt.title("Spectral phase of original pulse")
        plt.legend(["Phase reconstructed from SPIDER", "Interpolated phase"], bbox_to_anchor = [1, 1])
        plt.xlabel("Frequency [THz]")
        plt.ylabel("Spectral phase [rad]")
        plt.grid()
        plt.show()

    # extract the pulse

    intensity.zero_padding(3)
    pulse = sa.recover_pulse(interpolated_phase_zeros, intensity)

    if plot_pulse:
        min_pulse = pulse.quantile(0.25)
        max_pulse = pulse.quantile(0.75)
        delta = (max_pulse - min_pulse)*3

        sa.plot(pulse, color = "red", title = "Recovered pulse", start = min_pulse - delta, end = max_pulse + delta, what_to_plot = "real")

    # return what's needed

    if what_to_return == "pulse":
        return pulse
    elif what_to_return == "phase":
        return phase_spectrum, interpolated_phase
    elif what_to_return == "phase_diff":
        return diff_spectrum
    elif what_to_return == "FT":
        return s_ft_return
    elif what_to_return == None:
        pass
    else:
        raise Exception("\"what to return\" must be \"pulse\", \"phase\", \"phase_diff\", \"FT\" or None.")