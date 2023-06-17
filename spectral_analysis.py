import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#                   --------------
#                   SPECTRUM CLASS
#                   --------------


class spectrum:
    '''
    Class of an optical spectrum. It stores values of the \"X\" axis (wavelength, frequency or time) and the \"Y\" axis
    (intensity or phase) as 1D numpy arrays. Several standard transforms and basic statistics are implemented as class methods.
    '''

    def __init__(self, X, Y, x_type, y_type):

        if x_type not in ["time", "freq", "wl"]:
            raise Exception("x_type must be either \"time\" or \"freq\" or \"wl\".")
        if y_type not in ["phase", "intensity"]:
            raise Exception("y_type must be either \"phase\" or \"intensity\".")
        if len(X) != len(Y):
            raise Exception("X and Y axis must be of the same size.")

        self.X = X
        self.Y = Y
        self.x_type = x_type
        self.y_type = y_type
        self.spacing = np.mean(np.diff(self.X))


    def __len__(self):
        return len(self.X)
    

    def copy(self):
        X_copied = self.X.copy()
        Y_copied = self.Y.copy()
        return (spectrum(X_copied, Y_copied, self.x_type, self.y_type))


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

        freq = spectrum.X.copy()
        intens = spectrum.Y.copy()
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

        data = [[new_freq[i], new_intens[i]] for i in range(length)]
        df_spectrum = pd.DataFrame(data)

        if inplace == True:
            self.X = new_freq
            self.Y = new_intens
        
        else:
            return spectrum(new_freq, new_intens, self.x_type, self.y_type)
        

    def cut(self, start, end, how = "units", inplace = True):
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

        if how == "units":
            s = np.searchsorted(self.X, start)
            e = np.searchsorted(self.X, end)
            

        elif how == "fraction":
            s = floor(start*self.__len__())
            e = floor(end*self.__len__())

        elif how == "index":
            s = start
            e = end

        else:
            raise Exception("\"how\" must be either \"units\", \"fraction\" or \"index\".")

        new_X = self.X.copy()
        new_Y = self.Y.copy()
        new_X = new_X[s:e]
        new_Y = new_Y[s:e]

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
            raise Exception("Before applying Fourier Transform, transform spectrum from wavelength to frequency domain.")
        if self.x_type == "time":
            raise Exception("Sticking to the convention: use Inverse Fourier Transform.")

        # Fourier Transform

        FT_intens = fft(self.Y, norm = "ortho")
        FT_intens = fftshift(FT_intens)

        time = fftfreq(self.__len__(), self.spacing())
        time = fftshift(time)

        if inplace:
            self.X = time
            self.Y = FT_intens
            self.x_type = "time"
        else:
            return spectrum(time, FT_intens, "time", self.y_type)
            

    def inv_fourier(self, inplace = True):
        '''
        Performs Inverse Fourier Transform from \"time\" to \"frequency\" domain.
        '''

        from scipy.fft import ifft, fftfreq, ifftshift, fftshift

        if self.x_type != "time":
            raise Exception("Sticking to the convention: use Fourier Transform (not Inverse).")

        # prepare input

        time = self.X.copy()
        intens = self.Y.copy()

        time = ifftshift(time)
        intens = ifftshift(intens)

        # Fourier Transform

        FT_intens = ifft(intens, norm = "ortho")

        freq = fftfreq(self.__len__(), self.spacing)
        freq = fftshift(freq)

        if inplace == True:
            self.X = freq
            self.Y = FT_intens
            self.x_type = "freq"
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
    

    def visibility(self):   # TODO the functions returns height where greatest number of fringes is seen, meanwhile we want a plateau
        '''
        BE AWARE: not classical concept of visibility is applied. 
        
        Function returns float in range (0, 1) which is the fraction of maximum intensity, 
        at which the greatest number of fringes is observable.
        '''

        import numpy as np
        from math import floor

        max = np.max(self.Y)

        # Omg, this loop is so smart. Sometimes I impress myself.

        heights = []
        sample_levels = 100
        for height in np.linspace(0, max, sample_levels, endpoint = True):
            if height == 0: 
                continue
            safe_Y = self.Y.copy()
            safe_Y[safe_Y < height] = 0
            safe_Y[safe_Y > 0] = 1
            safe_Y += np.roll(safe_Y, shift = 1)
            safe_Y[safe_Y == 2] = 0
            heights.append(np.sum(safe_Y))

        heights = np.array(heights)
        fring_num = np.max(heights)
        indices = np.array([i for i in range(sample_levels-1) if heights[i] == fring_num])
        the_index = indices[floor(len(indices)/2)] 
        level = the_index/sample_levels*max

        return level/max
    

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


    def zero_padding(self, how_much, inplace = True):
        '''
        Add zeros on Y-axis to the left and right of data with constant (mean) spacing on X-axis. 
        \"how_much\" specifies number of added zeroes on left and right side of spectrum as a fraction of spectrums length.
        '''
        import numpy as np
        from math import floor

        length = floor(how_much*self.__len__())
        
        left_start = self.x[0] - self.spacing*length
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
            sum += np.abs(self.Y)
            if sum >= all*q:
                x = self.X[i]
                break

        return x
    

    def normalize(self, by = "highest", shift_to_zero = True, inplace = True):
        '''
        Normalize spectrum by linearly changing values of intensity and eventually shifting spectrum to zero.

        ARGUMENTS:

        by - way of normalizing the spectrum. If \"highest\", then the spectrum is scaled to 1, so that its greatest value is 1. 
        If \"intensity\", then spectrum is scaled, so that its integral is equal to 1.
        
        shift_to_zero - if \"True\", then spectrum is shifted by X axis, by simply np.rolling it.
        '''
        import numpy as np

        if by not in ["highest", "intensity"]:
            raise Exception("\"by\" parameter must be either \"highest\" or \"intensity\".")

        X_safe = self.X.copy()
        Y_safe = self.Y.copy()
        safe_spectrum = spectrum(X_safe, Y_safe, self.x_type, self.y_type)

        if by == "highest":
            max = np.max(np.abs(Y_safe))
            max_idx = np.argmax(np.abs(Y_safe))
            Y_safe /= max
            if shift_to_zero:
                zero_idx = np.searchsorted(X_safe, 0)
                shift_idx = max_idx - zero_idx
                X_safe = np.roll(X_safe, shift_idx)

        if by == "intensity":
            integral = np.sum(np.abs(Y_safe))
            median = safe_spectrum.quantile(1/2)
            max_idx = np.searchsorted(np.abs(Y_safe), median)
            Y_safe /= integral
            if shift_to_zero:
                zero_idx = np.searchsorted(X_safe, 0)
                shift_idx = max_idx - zero_idx
                X_safe = np.roll(X_safe, shift_idx)

        if inplace == True:
            self.X = X_safe
            self.Y = Y_safe
        
        if inplace == False:
            return spectrum(X_safe, Y_safe, self.x_type, self.y_type)
        
    def FWMH(self):
        '''
        Calculate Full Width at Half Maximum. If multiple peaks are present in spectrum, the function might not work properly.
        '''

        import numpy as np

        peak = np.max(self.Y)
        for idx, y in enumerate(self.Y):
            if y > peak/2:
                left = idx
        for idx, y in enumerate(np.flip(self.Y)):
            if y > peak/2:
                right = self.__len__() - idx

        width = right-left
        width *= (self.X[1]-self.X[2])

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



#                   ------------------------
#                   SPECTRUM CLASS FUNCTIONS
#                   ------------------------



def interpolate(old_spectrum, new_X):
    '''
    Interpolate rarely sampled spectrum for values in new X-axis. Interpolation is performed with cubic functions. If y-values to be interpolated are complex, 
    their absolute value is used.

    ARGUMENTS:

    old_spectrum - spectrum that is the basis for interpolation.

    new_X - new X axis (i. e. frequencies or wavelengths). Interpolated Y-values are calculated for values from new_X.

    RETURNS:

    Interpolated spectrum.
    '''
    
    import numpy as np
    from scipy.interpolate import CubicSpline

    X = np.real(old_spectrum.X)
    Y = np.abs(old_spectrum.Y)

    model = CubicSpline(X, Y)
    new_Y = model(np.real(new_X))

    return spectrum(new_X, new_Y, old_spectrum.x_type, old_spectrum.y_type)


def plot(spectrum, color = "darkviolet", title = "Spectrum", what_to_plot = "abs", start = None, end = None):
    '''
    Fast spectrum plotting using matplotlib.pyplot library.

    ARGUMENTS:

    spectrum - DataFrame with Intensity on Y axis.

    color - color of the plot.

    title - title of the plot.

    start - starting point (in X-axis units) of a area to be shown on plot. If \"min\", then plot starts with lowest X-value in all spectra.

    end - ending point (in X-axis units) of a area to be shown on plot. If \"max\", then plot ends with highest X-value in all spectra.

    what_to_plot - either \"abs\" or \"imag\" or \"real\".
    '''

    import numpy as np
    import matplotlib.pyplot as plt
    from math import floor, log10

    spectrum_safe = spectrum.copy()
    
    # simple function to round to significant digits

    def round_to_dig(x, n):
        return round(x, -int(floor(log10(abs(x)))) + n - 1)

    # invalid arguments
    
    if what_to_plot not in ("abs", "imag", "real"):
        raise Exception("Input \"what_to_plot\" not defined.")

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
    plt.plot(spectrum_safe.X, spectrum_safe.Y, color = color)
    plt.grid()
    if to_cut:
        plt.title(title + " [only part shown]")
    else:
        plt.title(title)
    if spectrum_safe.y_type == "phase":
        plt.ylabel("Spectral phase [rad]")
    if spectrum_safe.y_type == "intensity":
        plt.ylabel("Intensity")    
    if spectrum_safe.x_type == "wl":
        plt.xlabel("Wavelength [nm]")
        unit = "nm"
    if spectrum_safe.x_type == "freq":
        plt.xlabel("Frequency [THz]")
        unit = "THz"
    if spectrum_safe.x_type == "time":
        plt.xlabel("Time [ps]")
        unit = "ps"

    # quick stats
    
    spacing = round_to_dig(spectrum_safe.spacing, 3)
    p_per_unit = floor(1/spectrum_safe.spacing)

    if to_cut:
        plt.text(1.05, 0.85, "Number of points: {}\nX-axis spacing: {} ".format(n_points, spacing) + unit + "\nPoints per 1 " + unit +": {}".format(p_per_unit) + "\nFull X-axis range: {} - {} ".format(inf, sup) + unit , transform = ax.transAxes)
    else:
        plt.text(1.05, 0.9, "Number of points: {}\nX-axis spacing: {} ".format(n_points, spacing) + unit + "\nPoints per 1 " + unit +": {}".format(p_per_unit) , transform = ax.transAxes)

    plt.show()


def compare_plots(spectra, title = "Spectra", legend = None, start = None, end = None, abs = False):
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
    for i in len(spectra):
        dummy.append(spectra[i].x_type == spectra[0].x_type)

    if not np.all(np.array(dummy)):
        raise Exception("The X axes of spectra are not of unique type.")
    
    dummy = []
    for i in len(spectra):
        dummy.append(spectra[i].y_type == spectra[0].y_type)

    if not np.all(np.array(dummy)):
        raise Exception("The Y axes of spectra are not of unique type.")
    
    # and plotting

    colors = ["violet", "blue", "green", "yellow", "orange", "red", "brown", "black"]
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

        plt.plot(safe_spectrum.X[s:e], safe_spectrum.Y[s:e], color = colors[c])

    plt.grid()
    plt.title(title)
    if spectra[0].y_type == "int":
        plt.ylabel("Intensity")
    if spectra[0].y_type == "phase":
        plt.ylabel("Spectral phase [rad]")    
    if spectra[0].x_type == "wl":
        plt.xlabel("Wavelength [nm]")
    if spectra[0].x_type == "freq":
        plt.xlabel("Frequency [THz]")
    if spectra[0].x_type == "time":
        plt.xlabel("Time [ps]")
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
        complex_Y[i] *= np.exp(1j*phase_spectrum.Y)

    pulse_spectrum = spectrum(phase_spectrum.X.copy(), complex_Y, "time", "intensity")
    return pulse_spectrum.fourier(inplace = False)


def find_shift(spectrum_1, spectrum_2):
    '''
    Returns translation between two spectra in THz. Spectra in wavelength domain are at first transformed into frequency domain.
    Least squares is loss function to be minimized. Shift is found by brute force: 
    checking number of shifts equal to number of points on X axis.
    '''

    import numpy as np

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
    for i in range(len(spectrum1)):
         if minimum > error(spectrum1.Y, np.roll(a = spectrum2.Y, shift = i)):
             minimum = error(spectrum1.Y, np.roll(a = spectrum2.Y, shift = i))
             idx = i

    if idx > len(spectrum1.Y)/2:
        idx = len(spectrum1.Y) - idx
    
    return spectrum1.spacing*idx



#                   ---------
#                   RAY CLASS
#                   ---------



class ray:

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
            self.ver.Y *= np.exp(np.pi*2j*delay*self.ver.X)

        elif polarization == "hor":
            self.hor.Y *= np.exp(np.pi*2j*delay*self.hor.X)


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
        D_l = 17
        omega = spectr.X*2*np.pi
        omega_mean = sa.quantile(spectr, 1/2)
        phase = l_0**2*fiber_length*D_l/(4*np.pi*c)*(omega-omega_mean)**2
        if polarization == "ver": self.ver.Y *= np.exp(1j*phase)
        if polarization == "hor": self.hor.Y *= np.exp(1j*phase)


    def shear(self, shift, polarization):
        import spectral_analysis as sa

        if polarization not in ["ver", "hor"]:
            raise Exception("Polarization must be either \"ver\" or \"hor\".")

        if polarization == "hor":
            self.hor = sa.smart_shift(self.hor, shift)
        if polarization == "ver":
            self.ver = sa.smart_shift(self.ver, shift)


    def polarizer(self, transmission_polar):

        if transmission_polar not in ["ver", "hor"]:
            raise Exception("Polarization must be either \"ver\" or \"hor\".")
        if transmission_polar == "ver":
            self.hor.Y *= 0
        if transmission_polar == "hor":
            self.ver.Y *= 0

    def OSA(self, start = "min", end = "max"):
        import spectral_analysis as sa

        Y = self.hor.Y*np.conjugate(self.hor.Y) + self.ver.Y*np.conjugate(self.ver.Y)
        spectr = spectrum(self.hor.X, Y, "freq", "intensity")
        sa.plot(spectr, title = "OSA", x_type = "freq", start = start, end = end, color = "green")
        
        return spectr



#                   ----------------
#                   SPIDER ALGORITHM
#                   ----------------



def spider(phase_spectrum, 
           temporal_spectrum, 
           shear, 
           intensity_spectrum = None,
           phase_borders = None,
           what_to_return = None,
           vis_param = None,
           plot_steps = True, 
           plot_phase = True, 
           plot_pulse = True):
    '''
    Performs SPIDER algorithm.

    ARGUMENTS:

    phase_spectrum - spectrum with interference pattern created in a SPIDER setup. I should be a pandas DataFrame with wavelength or frequency in first column and intensity in the second OR path of .csv file with that data.

    shear - spectral shear applied by EOPM given in frequency units (default THz). If \"None\", then shear is estimated using fourier filtering.

    intensity_spectrum - amplitude of initial not interfered pulse. Similar as "phase_spectrum" it might be either DataFrame of string. If "None", then its approximation derived from SPIDER algorithm is used.  
    
    phase_borders - specify range of frequencies (in THz), where you want to find spectral phase. If to big, boundary effects may appear. If "None", the borders are estimated by calculating quantiles of intensity.

    what_to_return - if None, then RETURNS nothing. If "pulse", then RETURNS DataFrame with reconstructed pulse. If "phase", then RETURNS tuple with two DataFrame - phase and interpolated phase.

    plot_steps - if to plot all intermediate steps of the SPIDER algorithm.

    plot_phase - if to plot found spectral phase.

    plot_pulse - if to plot reconstructed pulse.
    '''
    import pandas as pd
    import numpy as np
    from math import floor as flr
    import spectral_analysis as sa
    import matplotlib.pyplot as plt

    # load data - spider

    if isinstance(phase_spectrum, spectrum):
        p_spectrum = phase_spectrum

    elif isinstance(phase_spectrum, str):    
        spectrum_df = pd.read_csv(phase_spectrum, skiprows = 2)
        p_spectrum = spectrum(spectrum_df.values[:, 0], spectrum_df.values[:, 1], "wl", "intensity")
    
    # load data - temporal phase

    if isinstance(temporal_spectrum, spectrum):
        t_spectrum = temporal_spectrum

    elif isinstance(temporal_spectrum, str):    
        spectrum_df = pd.read_csv(phase_spectrum, skiprows = 2)
        t_spectrum = spectrum(spectrum_df.values[:, 0], spectrum_df.values[:, 1], "wl", "intensity")

    # zero padding

    t_spectrum.zero_padding(1)
    p_spectrum.zero_padding(1)

    # plot OSA

    minimum = p_spectrum.quantile(0.1)
    maximum = p_spectrum.quantile(0.9)
    delta = (maximum - minimum)
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
            sa.plot(s_freq,"orange", title = "Wavelength to frequency", x_type = "freq", start = min_freq, end = max_freq)

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
        s_freq_t = s_freq_t.constant_spacing

    elif t_spectrum.x_type == "freq":
        s_freq_t = t_spectrum

    # fourier transform

    s_ft = s_freq.fourier(inplace = False)         # spider
    s_ft_t = s_freq_t.fourier(inplace = False)     # temporal

    s_shear = s_ft.copy()       # SPOILER: we will use it later to find the shear
    s_shear_t = s_ft_t.copy()

    min_time = s_ft.quantile(0.1)
    max_time = s_ft.quantile(0.9)
    delta = (max_time-min_time)/3
    min_time -= delta
    max_time += delta
   
    if plot_steps: 
        sa.plot(s_ft, title = "Fourier transformed", start = min_time, end = max_time) 

    # estimate time delay
    
    if vis_param == None:
        height = s_freq_t.visibility()
    else:
        height = vis_param

    period = s_freq_t.find_period(height)[0]
    delay = 1/period

    # find exact value of time delay

    s_ft2 = s_ft.copy()
    s_ft2.replace_with_zeros(end = delay*0.5)
    s_ft2.replace_with_zeros(start = delay*1.5)
    
    idx = s_ft2.Y.argmax()
    if isinstance(idx, np.ndarray): 
        idx = idx[0]
    delay2 = s_ft.X[idx]
    
    # and filter the spectrum to keep only one of site peaks
    
    s_ft.replace_with_zeros(end = delay2*0.5)           # spider
    s_ft.replace_with_zeros(s_ft, start = delay2*1.5)

    s_ft_t.replace_with_zeros(end = delay2*0.5)         # temporal
    s_ft_t.replace_with_zeros(start = delay2*1.5)

    if plot_steps: 
        sa.plot(s_ft, title = "Filtered", start = -2*delay2, end = 2*delay2)

    # let's find the shear
    
    if shear == None:

        s_shear.replace_with_zeros(start = None, end = -delay2*0.5)          # spider
        s_shear.replace_with_zeros(start = delay2*0.5, end = None)

        s_shear_t.replace_with_zeros(start = None, end = -delay2*0.5)      # temporal
        s_shear_t.replace_with_zeros(start = delay2*0.5, end = None)
        
        s_shear.inv_fourier()
        s_shear_t.inv_fourier()

        X_shear = np.real(s_shear.X)
        Y_shear = np.abs(s_shear.Y)
        Y_shear_t = np.abs(s_shear_t.Y)

        Y_shear_t /= 2
        Y_shear -= Y_shear_t

        s_shear = spectrum(X_shear, Y_shear, s_shear.x_type, s_shear.y_type)
        s_shear_t = spectrum(X_shear, Y_shear_t, s_shear.x_type, s_shear.y_type)

        s_shear.replace_with_zeros(start = None, end = -0.5) # TODO: maybe automatize values of end and start?
        s_shear.replace_with_zeros(start = 0.5, end = None)

        s_shear_t.replace_with_zeros(start = None, end = -0.5)
        s_shear_t.replace_with_zeros(start = 0.5, end = None)
        
        shear = sa.find_shift(s_shear, s_shear_t)
        if True:
            sa.compare_plots([s_shear, s_shear_t], 
                             start = -0.6, 
                             end = 0.6, 
                             abs = True, 
                             title = "Shear of {} THz".format(round(shear,5)),
                             legend = ["Sheared", "Not sheared"])

    integrate_interval = flr(shear/(s_freq_for_later.spacing))
    mean = np.mean(s_freq_for_later.X)
        
    # inverse fourier

    s_ift = s_ft.inv_fourier(inplace = False)        # spider
    s_ift_t = s_ft_t.inv_fourier(inplace = False)    # temporal
    if plot_steps:
        s_ift2 = s_ift.copy()
        s_ift2.X += mean 
        sa.plot(s_ift2, title = "Inverse Fourier transformed", x_type = "freq", start = min_freq, end = max_freq, what_to_plot = "abs")

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
    X = s_cut.X.copy()

    phase_values = np.angle(phase_values)
    temporal_phase = np.angle(temporal_phase)

    # extract phase

    values = phase_values - temporal_phase
    values -= values[0]                     # deletes linear aberration in phase
    values = np.unwrap(values)
    values -= values[flr(len(values)/2)]    # without that line, spectral phase will later always start at zero and grow

    # prepare data to plot
    
    X = X + mean + shear
    X2 = X.copy()

    # plot phase difference

    if plot_phase:

        plt.scatter(X, np.real(values), color = "orange", s = 1)
        plt.title("Spectral phase difference between pulse and its sheared copy")
        plt.xlabel("Frequency [THz]")
        plt.ylabel("Spectral phase")
        plt.grid()
        plt.show()

    # firstly initial interpolation to estimate vertical translation to be made

    X = X[::integrate_interval]
    values = values[::integrate_interval]

    Y = np.cumsum(values)

    phase_spectrum_old = spectrum(X, Y, "freq", "phase")
    interpolated_phase_old = sa.interpolate(phase_spectrum_old, X2)

    Y2 = interpolated_phase_old.Y
    Y -= Y2[flr(len(Y2)/2)]

    # now proper interpolation

    phase_frame = spectrum(X, Y, "freq", "phase")
    interpolated_phase = sa.interpolate(phase_spectrum, X2)

    #plot phase

    if plot_phase:

        plt.scatter(X, Y, color = "orange", s = 20)
        plt.plot(interpolated_phase.X, interpolated_phase.Y, color = "green")
        plt.title("Spectral phase of original pulse")
        plt.legend(["Phase reconstructed from SPIDER", "Interpolated phase"], bbox_to_anchor = [1, 1])
        plt.xlabel("Frequency [THz]")
        plt.ylabel("Spectral phase [rad]")
        plt.grid()
        plt.show()

    # recover intensity spectrum

    if intensity_spectrum == None:
        intensity = s_cut.copy()
        s_cut.X = np.abs(s_cut.X)
        s_cut.Y = np.abs(s_cut.Y)
    
    else:
        if isinstance(intensity_spectrum, spectrum):
            intensity = intensity_spectrum
        elif isinstance(intensity_spectrum, str):
            intensity_df = pd.read_csv(intensity_spectrum, skiprows = 2)
            intensity = spectrum(intensity_df.X, intensity_df.Y, "wl", "intensity")

        if intensity.x_type == "wl":
            intensity.wl_to_freq()
            intensity.constant_spacing()

        start = np.searchsorted(intensity.X, interpolated_phase.X[0])
        num = len(interpolated_phase)
        end = start + num
        intensity.cut(start = start, end = end, how = "index")

    # extract the pulse

    interpolated_phase2 = interpolated_phase.zero_padding(2, inplace = False)  
    intensity.zero_padding(2) 

    pulse = sa.recover_pulse(interpolated_phase2, intensity)

    if plot_pulse:
        min_pulse = sa.quantile(pulse, 0.25)
        max_pulse = sa.quantile(pulse, 0.75)
        delta = (max_pulse - min_pulse)*3
        sa.plot(pulse, color = "red", title = "Recovered pulse", start = min_pulse - delta, end = max_pulse + delta, what_to_plot = "abs")

    # return what's needed

    if what_to_return == "pulse":
        return pulse
    elif what_to_return == "phase":
        return phase_frame, interpolated_phase