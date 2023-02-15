import numpy as np


class OnlineMeanVariance:
    
    def __init__(self):
        self.count = 0
        self.mean = 0.
        self.M2 = 0.
    
    def update(self, x):
        x = np.array(x)
        self.count += 1
        delta = x - self.mean
        self.mean = self.mean + delta / float(self.count)
        delta2 = x - self.mean
        self.M2 = self.M2 + delta * delta2
    
    def calculate_variance(self):
        return self.M2 / (self.count - 1.)
    
    def calculate_standard_error(self):
        return np.sqrt(self.calculate_variance() / float(self.count))


def histogram(*series, n_bins=100):
    min_x = min(np.min(data) for data in series)
    max_x = max(np.max(data) for data in series)
    bins = np.linspace(min_x, max_x, n_bins)
    series = [np.histogram(data, bins=bins)[0] for data in series]
    pdfs = [data / np.sum(data) for data in series]
    result = np.column_stack(pdfs)
    return result, bins

