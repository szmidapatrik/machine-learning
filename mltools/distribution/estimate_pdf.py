import numpy as np

class EstimatePDF:

    # Constructor
    def __init__(self):
        pass

    # PDF Function shape estimate
    def pdf(self, x, w_s, w_n):
        x_min = np.min(x)
        x_max = np.max(x)

        x_axis = np.linspace(x_min, x_max, w_n)

        fx = []
        for x_value in x_axis:
            fx.append(x[x > x_value - w_s/2][x[x > x_value - w_s/2] < x_value + w_s/2].shape[0] / (x.shape[0] * w_s))
        return x_axis, fx


    # Smoothed PDF Function shape estimate
    def smooth_pdf(self, x, w_n):
        x_min = np.min(x)
        x_max = np.max(x)

        interval = x_max - x_min
        x_axis = np.linspace(x_min, x_max, w_n)
        w_s_list = [1000, 950, 900, 850, 800, 750, 700, 650, 600, 550, 500, 250, 100, 75, 50, 25, 10]



        for idx, w_s in enumerate(w_s_list):
            _, fx_temp = self.pdf(x, interval / w_s, w_n)
            fx_temp = np.array(fx_temp)
            if idx == 0:
                fx = fx_temp
            else:
                fx = np.mean([fx, fx_temp], axis=0)
        
        return x_axis, fx