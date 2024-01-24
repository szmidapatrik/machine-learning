import numpy as np

class EstimatePDF:

    # Constructor
    def __init__(self):
        pass

    # PDF Function shape estimate
    # x : data
    # windows_size : window size
    # window_number : window number
    # interval_min : start point of interval
    # interval_max : end point of interval
    def pdf(self, x, windows_size, window_number, interval_min=None, interval_max=None):
        
        x_min = self.set_interval_start(x, interval_min)
        x_max = self.set_interval_end(x, interval_max)

        x_axis = np.linspace(x_min, x_max, window_number)

        fx = []
        for x_value in x_axis:
            fx.append(x[x > x_value - windows_size/2][x[x > x_value - windows_size/2] < x_value + windows_size/2].shape[0] / (x.shape[0] * windows_size))
        return x_axis, fx


    # Smoothed PDF Function shape estimate
    def smooth_pdf(self, x, window_number, interval_min=None, interval_max=None):
        
        x_min = self.set_interval_start(x, interval_min)
        x_max = self.set_interval_end(x, interval_max)

        x_interval = np.max(x) - x_min
        x_axis = np.linspace(x_min, x_max, window_number)
        windows_size_list = [1000, 950, 900, 850, 800, 750, 700, 650, 600, 550, 500, 250, 100, 75, 50, 25, 10]



        for idx, windows_size in enumerate(windows_size_list):
            _, fx_temp = self.pdf(x, x_interval / windows_size, window_number, interval_min, interval_max)
            fx_temp = np.array(fx_temp)
            if idx == 0:
                fx = fx_temp
            else:
                fx = np.mean([fx, fx_temp], axis=0)
        
        return x_axis, fx
    


    # Set the interval start
    def set_interval_start(self, x, value):
        return np.min(x) if value is None else value
    
    # Set the interval end
    def set_interval_end(self, x, value):
        return np.max(x) if value is None else value