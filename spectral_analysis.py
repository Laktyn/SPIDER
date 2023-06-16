import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#                   --------------
#                   SPECTRUM CLASS
#                   --------------


class spectrum:

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
        import pandas as pd
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
    

    def visibility(self):
        '''
        BE AWARE: not classical concept of visibility is applied. 
        
        Function returns float in range (0, 1) which is the fraction of maximum intensity, at which the greatest number of fringes is observable.
        '''

        import numpy as np
        import pandas as pd
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
    
    def quantile(self, q):
        '''
        Finds x in X axis such that integral of intensity to x is fraction of value q of whole intensity.
        '''
        import pandas as pd
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
        import pandas as pd
        import spectral_analysis as sa

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
        Calculate Full Width at Half Maximum.
        '''

        import numpy as np
        import pandas as pd

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
    
    import pandas as pd
    import numpy as np
    from scipy.interpolate import CubicSpline

    X = np.real(old_spectrum.X)
    Y = np.abs(old_spectrum.Y)

    model = CubicSpline(X, Y)
    new_Y = model(np.real(new_X))

    return spectrum(new_X, new_Y, old_spectrum.x_type, old_spectrum.y_type)


def compare_plots(spectra, title = "Spectra", legend = None, y_type = "int", x_type = "freq", start = "min", end = "max", abs = False):
    '''
    Show several spectra on single plot.

    ARGUMENTS:

    spectra - list with spectra to be show on plot.

    title - title of the plot.

    legend - list with names of subsequent spectra. If \"None\", then no legend is shown.

    y_type - unit of Y-axis.

    x-type - unit of X-axis.

    start - starting point (in X-axis units) of a area to be shown on plot. If \"min\", then plot starts with lowest X-value in all spectra.

    end - ending point (in X-axis units) of a area to be shown on plot. If \"max\", then plot ends with highest X-value in all spectra.

    abs - if \"True\", then absolute values of spectra is plotted.
    '''
    
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    colors = ["violet", "blue", "green", "yellow", "orange", "red", "brown", "black"]
    for c, spectrum in enumerate(spectra):
        safe_spectrum = spectrum.copy()
        if abs:
            safe_spectrum.Y = np.abs(safe_spectrum.Y)
        if start == "min":
            s = 0
        else:
            s = np.searchsorted(safe_spectrum.X, start)
        if end == "max":
            e = len(safe_spectrum) - 1
        else:
            e = np.searchsorted(safe_spectrum.X, end)

        plt.plot(safe_spectrum.X[s:e], safe_spectrum.Y[s:e], color = colors[c])

    plt.grid()
    plt.title(title)
    if y_type == "int":
        plt.ylabel("Intensity")
    if y_type == "phase":
        plt.ylabel("Spectral phase [rad]")    
    if x_type == "wl":
        plt.xlabel("Wavelength [nm]")
    if x_type == "freq":
        plt.xlabel("Frequency [THz]")
    if x_type == "time":
        plt.xlabel("Time [ps]")
    if isinstance(legend, list):
        plt.legend(legend, bbox_to_anchor = [1, 1])
    plt.show()       



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


def make_it_visible(spectrum, segment_length):
    '''
    The function improves visibility of interference fringes by subtracting local minima of spectrum. Each local minimum is a minimum of a
    segment of segment_length points. Last segment might be shorter.

    ARGUMENTS:

    spectrum - DataFrame with Wavelength/Frequency on X axis and Intensity on Y axis.

    segment_length - length of a segment of which we take a local minimum. It needs to be chosen manually and carefully as it highly
    influences quality of visibility.

    RETURNS:

    Spectrum DataFrame spectrum with subtracted offset and increased visibility
    '''

    import numpy as np
    import pandas as pd
    from math import floor as flr

    wl = spectrum.values[:, 0]
    intens = spectrum.values[:, 1]

    minima = []
    samples_num = len(intens) // segment_length + 1

    # find "local" minima

    for i in range(samples_num):
        start = segment_length*i
        end = segment_length*(i+1)

        if start >= len(intens): break

        if end > len(intens) - 1:
            end = len(intens)

        minimum = np.min(intens[start: end])
        minima.append(minimum)

    # subtract the minima

    new_intens = []
    for i in range(len(intens)):
        new =  intens[i] - minima[flr(i/segment_length)]
        new_intens.append(new)

    # and return a nice dataframe

    return pd.DataFrame(np.array([[wl[i], new_intens[i]] for i in range(len(wl))]))


def cut(spectrum, start, end, how = "units"):
    '''
    Returns the \"spectrum\" limited to the borders. 

    ARGUMENTS:

    spectrum - DataFrame with Intensity on Y-axis.

    start - start of the segment, to which the spectrum is limited.

    end - end of the segment, to which the spectrum is limited.

    how - defines meaning of \"start\" and \"end\". If \"units\", then those are values on X-axis. 
    If \"fraction\", then the fraction of length of X-axis. If \"index\", then corresponding indices of border observations.
    
    RETURNS:

    Limited spectrum DataFrame.

    '''

    import pandas as pd
    import numpy as np
    from math import floor

    if how == "units":
        s = np.searchsorted(spectrum.values[:, 0], start)
        e = np.searchsorted(spectrum.values[:, 0], end)
        cut_spectrum = pd.DataFrame(spectrum.values[s: e, :])

    elif how == "fraction":
        s = floor(start*spectrum.shape[0])
        e = floor(end*spectrum.shape[0])
        cut_spectrum = pd.DataFrame(spectrum.values[s:e, :])

    elif how == "index":
        cut_spectrum = pd.DataFrame(spectrum.values[start:end, :]) 

    else:
        raise Exception("Argument not defined.")

    return cut_spectrum


def plot(spectrum, color = "darkviolet", title = "Spectrum", x_type = "wl", y_type = "int", what_to_plot = "abs", start = "min", end = "max"):
    '''
    Fast spectrum plotting using matplotlib.pyplot library.

    ARGUMENTS:

    spectrum - DataFrame with Intensity on Y axis.

    color - color of the plot.

    title - title of the plot.

    x_type - \"wl\" (Wavelength) or \"freq\" (Frequency) or \"time\" (Time). Determines the label of X-axis.

    y_type = \"phase\" (Spectral phase) or \"int\" (Intensity). Determines the label of Y-axis.

    start - starting point (in X-axis units) of a area to be shown on plot. If \"min\", then plot starts with lowest X-value in all spectra.

    end - ending point (in X-axis units) of a area to be shown on plot. If \"max\", then plot ends with highest X-value in all spectra.

    what_to_plot - either \"abs\" or \"imag\" or \"real\".

    RETURNS:

    Continuous plot of the \"spectrum\"/.
    '''

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from math import floor, log10

    spectrum_safe = spectrum.copy()
    
    # simple function to round to significant digits

    def round_to_dig(x, n):
        return round(x, -int(floor(log10(abs(x)))) + n - 1)

    # invalid arguments

    if x_type not in ("wl", "freq", "time"):
        raise Exception("Input \"x_type\" not defined.")
    
    if y_type not in ("int", "phase"):
        raise Exception("Input \"y_type\" not defined.")
    
    if what_to_plot not in ("abs", "imag", "real"):
        raise Exception("Input \"what_to_plot\" not defined.")

    # if we dont want to plot whole spectrum

    n_points = len(spectrum_safe.values[:, 0])

    to_cut = False
    inf = round(np.real(spectrum_safe.values[0, 0].copy()))
    sup = round(np.real(spectrum_safe.values[-1, 0].copy()))

    s = 0
    e = -1

    if start != "min":
        s = np.searchsorted(spectrum_safe.values[:, 0], start)
        to_cut = True

    if end != "max":        
        e = np.searchsorted(spectrum_safe.values[:, 0], end)
        to_cut = True

    spectrum_safe = pd.DataFrame(spectrum_safe.values[s: e, :])   

    values = spectrum_safe.values[:, 1].copy()
    X = spectrum_safe.values[:, 0].copy()
    
    # what do we want to have on Y axis

    if what_to_plot == "abs":
        values = np.abs(values)
    if what_to_plot == "real":
        values = np.real(values)
    if what_to_plot == "imag":
        values = np.imag(values)

    # start to plot

    f, ax = plt.subplots()
    plt.plot(X, values, color = color)
    plt.grid()
    if to_cut:
        plt.title(title + " [only part shown]")
    else:
        plt.title(title)
    if y_type == "phase":
        plt.ylabel("Spectral phase [rad]")
    if y_type == "int":
        plt.ylabel("Intensity")    
    if x_type == "wl":
        plt.xlabel("Wavelength [nm]")
        unit = "nm"
    if x_type == "freq":
        plt.xlabel("Frequency [THz]")
        unit = "THz"
    if x_type == "time":
        plt.xlabel("Time [ps]")
        unit = "ps"

    # quick stats
    
    spaces = np.diff(np.real(X))
    spacing = round_to_dig(np.mean(spaces), 3)
    p_per_unit = floor(1/np.mean(spaces))

    if to_cut:
        plt.text(1.05, 0.85, "Number of points: {}\nX-axis spacing: {} ".format(n_points, spacing) + unit + "\nPoints per 1 " + unit +": {}".format(p_per_unit) + "\nFull X-axis range: {} - {} ".format(inf, sup) + unit , transform = ax.transAxes)
    else:
        plt.text(1.05, 0.9, "Number of points: {}\nX-axis spacing: {} ".format(n_points, spacing) + unit + "\nPoints per 1 " + unit +": {}".format(p_per_unit) , transform = ax.transAxes)

    plt.show()

def shift(spectrum, shift):
    '''
    Shifts the spectrum by X axis. Warning: only values on X axis are modified.

    ARGUMENTS:

    spectrum - DataFrame with Intensity on Y axis.
    
    shift - size of a shift in X axis units.

    RETURNS:

    Shifted spectrum DataFrame.
    '''
    import pandas as pd
    import numpy as np

    data = spectrum.values
    new_data = data.copy()
    new_data[:, 0] = new_data[:, 0] + shift

    return pd.DataFrame(new_data)



def zero_padding(spectrum, how_much):
    '''
    Add zeros on Y-axis to the left and right of data with constant (mean) spacing on X-axis.
    
    ARGUMENTS:

    spectrum - DataFrame with Intensity on Y axis.

    how_much - number of added zeroes on left and right side of spectrum as a fraction of spectrums length.

    RETURNS:
    
    DataFrame spectrum with padded zeroes.
    '''
    import pandas as pd
    import numpy as np
    from math import floor

    length = floor(how_much*spectrum.shape[0])
    spacing = np.mean(np.diff(spectrum.values[:, 0]))
    
    left_start = spectrum.values[0, 0] - spacing*length
    left = np.linspace(left_start, spectrum.values[0, 0], endpoint = False, num = length - 1)
    left_arr = np.transpose(np.stack((left, np.zeros(length-1))))

    right_end = spectrum.values[-1, 0] + spacing*length
    right = np.linspace(spectrum.values[-1, 0] + spacing, right_end, endpoint = True, num = length - 1)
    right_arr = np.transpose(np.stack((right, np.zeros(length-1))))

    arr_with_zeros = np.concatenate([left_arr, spectrum.values, right_arr])

    return pd.DataFrame(arr_with_zeros)



def replace_with_zeros(spectrum, start, end):
    '''
    Replace numbers on Y-axis of "spectrum" from "start" to "end" with zeroes. "start" and "end" are in X-axis' units.
    '''

    import numpy as np
    import pandas as pd

    data = spectrum.values.copy()
    if start == "min":
        s = 0
    else:
        s = np.searchsorted(spectrum.values[:, 0], start)

    if end == "max":
        e = spectrum.shape[0]
    else:
        e = np.searchsorted(spectrum.values[:, 0], end)

    data[s: e, 1] = np.zeros(e-s)

    return pd.DataFrame(data)


def smart_shift(spectrum, shift = "center"):
    '''
    Shift spectrum by rolling Y-axis.
     
    ARGUMENTS:

    spectrum - standard pd.DataFrame spectrum.

    shift - either value of shift in X axis units or \"center\". In the latter case the spectrum is shifted so, that 1/2-order quantile for Y axis is reached for x = 0.
    
    RETURNS:

    shifted_spectrum - standard pd.DataFrame spectrum.    
    '''

    import numpy as np
    import pandas as pd
    from math import floor
    import spectral_analysis as sa

    data = spectrum.values.copy()
    shift2 = shift
    
    if isinstance(shift, int):
        pass
    if shift == "center":
        shift2 = -sa.quantile(spectrum, 1/2)

    spacing = np.mean(np.diff(data[:, 0]))
    index_shift = floor(np.real(shift2)/np.real(spacing))

    data[:, 1] = np.roll(data[:, 1], index_shift)
        
    return pd.DataFrame(data)


def find_shift(spectrum1, spectrum2, spectrum_in = "wl"):
    '''
    Returns translation between two spectra in THz or nm. Least squares is loss function to be minimized. Shift is found by brute force: checking number of shifts equal to number of points on X axis.

    ARGUMENTS:

    spectrum1 - first spectrum in standard pd.DataFrame format.
    
    spectrum1 - second spectrum in standard pd.DataFrame format.

    spectrum_in - specify X-axis units. Either "wl" or "THz".

    RETURNS:

    shift - scalar value of shift, which unit is determined by \"spectrum_in\".

    '''
    import pandas as pd
    import numpy as np
    import spectral_analysis as sa

    if spectrum_in not in ["wl", "freq"]:
        raise Exception("spectrum_in must be either \"wl\" or \"freq\".")

    if spectrum_in == "wl":

        spectrum1 = sa.wl_to_freq(spectrum1)
        spectrum2 = sa.wl_to_freq(spectrum2)
        spectrum1 = sa.constant_spacing(spectrum1)
        spectrum2 = sa.constant_spacing(spectrum2)

    spacing = np.mean(np.diff(spectrum1.values[:, 0]))

    values1 = spectrum1.values[:, 1]
    values2 = spectrum2.values[:, 1]

    def error(v_1, v_2):
        return np.sum(np.abs(v_1-v_2)**2)
    
    min = np.sum(np.abs(values1)**2)
    idx = 0
    for i in range(len(values1)):
         if min > error(values1, np.roll(a = values2, shift = i)):
             min = error(values1, np.roll(a = values2, shift = i))
             idx = i

    if idx > len(values1)/2:
        idx = len(values1) - idx
    
    return spacing*idx



def recover_pulse(phase_df, intensity_df):
    '''
    Reconstructs the pulse (or any type of spectrum), given DataFrame of spectral phase and DataFrame of spectral intensity.

    ARGUMENTS:
    
    phase_df - pd.DataFrame with values of spectral phase in second column.
    
    intensity_df - pd.DataFrame with values of spectral intensity in second column.
    
    RETURNS:

    pulse_df - pd.DataFrame with reconstructed pulse spectrum. 
    '''

    import numpy as np
    import pandas as pd
    import spectral_analysis as sa

    if len(phase_df.values[:, 0]) != len(intensity_df.values[:, 0]):
        raise Exception("Frequency axes of phase and intensity are not equal.")
    
    X = phase_df.values[:, 0]
    complex_Y = intensity_df.values[:, 1].copy()
    complex_Y = complex_Y.astype(complex) # surprisingly that line is necessary
    for i in range(len(complex_Y)):
        complex_Y[i] *= np.exp(1j*phase_df.values[i, 1])

    spectrum = pd.DataFrame(np.transpose(np.stack((X, complex_Y))))
    return sa.fourier(spectrum)


#                   ----------------
#                   SPIDER ALGORITHM
#                   ----------------


def spider(phase_spectrum, 
           temporal_spectrum, 
           shear, 
           spectrum_in = "wl",
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

    spectrum_in - either "wl" or "freq". Specify units of the first column of loaded spectra.

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

    if isinstance(phase_spectrum, pd.DataFrame):
        spectrum = phase_spectrum

    elif isinstance(phase_spectrum, str):    
        spectrum = pd.read_csv(phase_spectrum, skiprows = 2)

    spectrum = sa.zero_padding(spectrum, 1)

    # load data - temporal phase

    if isinstance(temporal_spectrum, pd.DataFrame):
        t_spectrum = temporal_spectrum

    elif isinstance(temporal_spectrum, str):    
        t_spectrum = pd.read_csv(temporal_spectrum, skiprows = 2)

    t_spectrum = sa.zero_padding(t_spectrum, 1)

    # plot OSA

    min = sa.quantile(spectrum, 0.1)
    max = sa.quantile(spectrum, 0.9)
    delta = (max-min)
    min_wl = min-delta
    max_wl = max+delta 

    if plot_steps and spectrum_in == "wl":
        sa.plot(spectrum, "orange", title = "Data from OSA", start = min_wl, end = max_wl, x_type = "wl")
    if plot_steps and spectrum_in == "freq":
        sa.plot(spectrum, "orange", title = "Data from OSA", start = min_wl, end = max_wl, x_type = "freq")

    # transform X-axis to frequency

    # spider

    if spectrum_in == "wl":
        s_freq = sa.wl_to_freq(spectrum)
        s_freq = sa.constant_spacing(s_freq)
        min = sa.quantile(s_freq, 0.1)
        max = sa.quantile(s_freq, 0.9) 
        delta = (max-min)
        min_freq = min-delta
        max_freq = max+delta
        if plot_steps: 
            sa.plot(s_freq,"orange", title = "Wavelength to frequency", x_type = "freq", start = min_freq, end = max_freq)

    elif spectrum_in == "freq":
        s_freq = spectrum
        min = sa.quantile(s_freq, 0.1)
        max = sa.quantile(s_freq, 0.9) 
        delta = (max-min)
        min_freq = min-delta
        max_freq = max+delta

    s_freq_for_later = s_freq.copy()

    # temporal

    if spectrum_in == "wl":
        s_freq_t = sa.wl_to_freq(t_spectrum)
        s_freq_t = sa.constant_spacing(s_freq_t)

    elif spectrum_in == "freq":
        s_freq_t = t_spectrum

    # fourier transform

    s_ft = sa.fourier(s_freq, absolute = False)         # spider
    s_ft_t = sa.fourier(s_freq_t, absolute = False)     # temporal

    s_shear = s_ft.copy()       # SPOILER: we will use it later to find the shear
    s_shear_t = s_ft_t.copy()

    min = sa.quantile(s_ft, 0.1)
    max = sa.quantile(s_ft, 0.9)
    delta = (max-min)/3
    min_time = min-delta
    max_time = max+delta
   
    if plot_steps: 
        sa.plot(s_ft, title = "Fourier transformed", x_type = "time", start = min_time, end = max_time) 

    # estimate time delay
    
    if vis_param == None:
        height = sa.find_visibility(s_freq_t)
    else:
        height = vis_param
    period = sa.find_period(s_freq_t, height)[0]
    delay = 1/period

    # find exact value of time delay

    s_ft2 = s_ft.copy()
    s_ft2 = sa.replace_with_zeros(s_ft2, start = "min", end = delay*0.5)
    s_ft2 = sa.replace_with_zeros(s_ft2, start = delay*1.5, end = "max")
    
    idx = s_ft2.values[:, 1].argmax()
    if isinstance(idx, np.ndarray): 
        idx = idx[0]
    delay2 = s_ft.values[idx, 0]
    
    # and filter the spectrum to keep only one of site peaks
    
    s_ft = sa.replace_with_zeros(s_ft, start = "min", end = delay2*0.5)         # spider
    s_ft = sa.replace_with_zeros(s_ft, start = delay2*1.5, end = "max")

    s_ft_t = sa.replace_with_zeros(s_ft_t, start = "min", end = delay2*0.5)     # temporal
    s_ft_t = sa.replace_with_zeros(s_ft_t, start = delay2*1.5, end = "max")

    if plot_steps: 
        sa.plot(s_ft, title = "Filtered", x_type = "time", start = -2*delay2, end = 2*delay2)

    # let's find the shear
    
    if shear == None:

        s_shear = sa.replace_with_zeros(s_shear, start = "min", end = -delay2*0.5)          # spider
        s_shear = sa.replace_with_zeros(s_shear, start = delay2*0.5, end = "max")

        s_shear_t = sa.replace_with_zeros(s_shear_t, start = "min", end = -delay2*0.5)      # temporal
        s_shear_t = sa.replace_with_zeros(s_shear_t, start = delay2*0.5, end = "max")
        
        s_shear = sa.inv_fourier(s_shear)
        s_shear_t = sa.inv_fourier(s_shear_t)

        X_shear = np.real(s_shear.values[:, 0])
        Y_shear = np.abs(s_shear.values[:, 1])
        Y_shear_t = np.abs(s_shear_t.values[:, 1])

        Y_shear_t /= 2
        Y_shear -= Y_shear_t

        Y_shear = pd.DataFrame(np.transpose(np.stack((X_shear, Y_shear))))
        Y_shear_t = pd.DataFrame(np.transpose(np.stack((X_shear, Y_shear_t))))

        Y_shear = sa.replace_with_zeros(Y_shear, start = "min", end = -0.5)
        Y_shear = sa.replace_with_zeros(Y_shear, start = 0.5, end = "max")

        Y_shear_t = sa.replace_with_zeros(Y_shear_t, start = "min", end = -0.5)
        Y_shear_t = sa.replace_with_zeros(Y_shear_t, start = 0.5, end = "max")
        
        shear = sa.find_shift(Y_shear, Y_shear_t, spectrum_in = "freq")
        if True:
            sa.compare_plots([Y_shear, Y_shear_t], 
                             start = -0.6, 
                             end = 0.6, 
                             abs = True, 
                             title = "Shear of {} THz".format(round(shear,5)),
                             legend = ["Sheared", "Not sheared"])

    integrate_interval = flr(shear/(np.mean(np.diff(s_freq_for_later.values[:, 0]))))
    mean = np.mean(s_freq_for_later.values[:, 0])
        
    # inverse fourier

    s_ift = sa.inv_fourier(s_ft)        # spider
    s_ift_t = sa.inv_fourier(s_ft_t)    # temporal
    if plot_steps:
        s_ift2 = s_ift.copy()
        s_ift2.values[:, 0] += mean 
        sa.plot(s_ift2, title = "Inverse Fourier transformed", x_type = "freq", start = min_freq, end = max_freq, what_to_plot = "abs")

    # cut spectrum to area of significant phase

    if phase_borders == None:
        min_phase = sa.quantile(s_ift, 0.05)
        max_phase = sa.quantile(s_ift, 0.95)

    else:
        min_phase = phase_borders[0] - mean
        max_phase = phase_borders[1] - mean

    s_cut = sa.cut(s_ift, start = min_phase, end = max_phase)
    s_cut_t = sa.cut(s_ift_t, start = min_phase, end = max_phase)

    # extract phase differences

    phase_values = s_cut.values[:, 1].copy()
    temporal_phase = s_cut_t.values[:, 1].copy()
    X = s_cut.values[:, 0].copy()

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

    # and plot phase

    X = X[::integrate_interval]
    values = values[::integrate_interval]

    Y = np.cumsum(values)

    phase_frame_old = pd.DataFrame(np.transpose(np.stack((X, Y))))
    interpolated_phase_old = sa.interpolate(phase_frame_old, X2)

    Y2 = interpolated_phase_old.values[:, 1]
    Y -= Y2[flr(len(Y2)/2)]

    phase_frame = pd.DataFrame(np.transpose(np.stack((X, Y))))
    interpolated_phase = sa.interpolate(phase_frame, X2)

    if plot_phase:

        plt.scatter(X, Y, color = "orange", s = 20)
        plt.plot(interpolated_phase.values[:,0], interpolated_phase.values[:,1], color = "green")
        plt.title("Spectral phase of original pulse")
        plt.legend(["Phase reconstructed from SPIDER", "Interpolated phase"], bbox_to_anchor = [1, 1])
        plt.xlabel("Frequency [THz]")
        plt.ylabel("Spectral phase [rad]")
        plt.grid()
        plt.show()

    # recover intensity spectrum

    if intensity_spectrum == None:
        intensity = s_cut.copy()
        s_cut.values[:,:] = np.abs(s_cut.values[:,:])
    
    else:
        if isinstance(intensity_spectrum, pd.DataFrame):
            intensity = intensity_spectrum
        elif isinstance(intensity_spectrum, str):
            intensity = pd.read_csv(intensity_spectrum, skiprows = 2)
        if spectrum_in == "wl":
            intensity = sa.wl_to_freq(intensity)
            intensity = sa.constant_spacing(intensity)
        start = np.searchsorted(intensity.values[:, 0], interpolated_phase.values[0, 0])
        num = len(interpolated_phase.values[:, 0])
        end = start + num
        intensity = sa.cut(intensity, start = start, end = end, how = "index")

    # extract the pulse

    interpolated_phase2 = sa.zero_padding(interpolated_phase, 2)  
    intensity = sa.zero_padding(intensity, 2) 

    pulse = sa.recover_pulse(interpolated_phase2, intensity)

    if plot_pulse:
        min_pulse = sa.quantile(pulse, 0.25)
        max_pulse = sa.quantile(pulse, 0.75)
        delta = (max_pulse-min_pulse)*3
        sa.plot(pulse, color = "red", title = "Recovered pulse", x_type = "time", start = min_pulse-delta, end = max_pulse+delta, what_to_plot = "abs")

    # return what's needed

    if what_to_return == "pulse":
        return pulse
    elif what_to_return == "phase":
        return phase_frame, interpolated_phase