import numpy as np
from mltools.generative.estimate_pdf import EstimatePDF
from mltools.distribution import NormalDistribution

class RejectionSampling:

    # Constructor
    def __init__(self):
        pass

    # Sample from the distribution of x data
    def sample(self, data, sample_number, smoothing_rate=1000, q='normal'):

        data_pdf_x, data_pdf_y = EstimatePDF().smooth_pdf(data, smoothing_rate)

        if q == 'normal':

            # Normal distribution pdf
            estimation_dist = NormalDistribution(np.mean(data), np.std(data))
            q_pdf = estimation_dist.pdf(data_pdf_x)

            
            # Index where the difference between data_pdf_y and q_pdf is the biggest
            max_difference = np.max(data_pdf_y - q_pdf)
            max_difference_idx = np.where(data_pdf_y - q_pdf == max_difference)[0][0]

            # Multiplier; raising the PDF of the normal distribution over the data PDF
            M = (data_pdf_y[max_difference_idx] / q_pdf[max_difference_idx]) + 0.01

        elif q == 'uniform':

            # Uniform distribution pdf
            estimation_dist = np.linspace(np.min(data), np.max(data), smoothing_rate)
            q_pdf = np.ones(smoothing_rate)

            # Multiplier; raising the PDF of the uniform distribution over the data PDF
            M = (data_pdf_y[0] / q_pdf[0]) + 0.01


        # Acceptance-Rejection sampling
        accepted = []
        while len(accepted) < sample_number:

            if q == 'normal':

                # Sample from the normal distribution
                samples = estimation_dist.sample(500)
            elif q == 'uniform':

                # Sample from the uniform distribution
                samples = np.random.uniform(np.min(data), np.max(data), 500)


            for sample in samples:

                # Find the closest data point to the sample
                idx = self.find_closest(data_pdf_x, sample)

                # Accept the sample with probability p(x) / Mq(x)
                accept_proba = data_pdf_y[idx] / (M * q_pdf[idx])

                print('------------------')
                print('sample: ', sample)
                print('clusest_x: ', data_pdf_x[idx])
                print('idx: ', idx)
                print('M: ', M)
                print('data_pdf_y[idx]: ', data_pdf_y[idx])
                print('q_pdf[idx]: ', q_pdf[idx] * M)
                print('accept_proba: ', accept_proba)

                if np.random.rand() < accept_proba and self.check_sample_range(sample, data):
                    accepted.append(sample)

        return np.array(accepted)


    # Find closest data point in the data array
    def find_closest(self, array, number):
        array = np.asarray(array)
        idx = (np.abs(array - number)).argmin()
        return idx
    


    # Check whether a generated sample is in the right range
    def check_sample_range(self, sample, data):
        if sample > np.min(data) and sample < np.max(data):
            return True
        else:
            return False