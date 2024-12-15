class MersenneTwister:
    # Constants for MT19937
    w, n, m, r = 32, 624, 397, 31
    a = 0x9908B0DF
    u, d = 11, 0xFFFFFFFF
    s, b = 7, 0x9D2C5680
    t, c = 15, 0xEFC60000
    l = 18
    f = 1812433253

    def __init__(self, seed):
        # Initialize the generator with a seed
        self.index = self.n
        self.mt = [0] * self.n
        self.mt[0] = seed
        for i in range(1, self.n):
            self.mt[i] = self.int_32(
                self.f * (self.mt[i - 1] ^ (self.mt[i - 1] >> (self.w - 2))) + i
            )

    def int_32(self, number):
        # Get the 32 least significant bits
        return number & 0xFFFFFFFF

    def twist(self):
        for i in range(self.n):
            x = (self.mt[i] & 0x80000000) + (self.mt[(i + 1) % self.n] & 0x7fffffff)
            xA = x >> 1
            if x % 2 != 0:  # lowest bit of x is 1
                xA = xA ^ self.a
            self.mt[i] = self.mt[(i + self.m) % self.n] ^ xA
        self.index = 0

    def random(self):
        if self.index >= self.n:
            self.twist()

        y = self.mt[self.index]
        y ^= (y >> self.u) & self.d
        y ^= (y << self.s) & self.b
        y ^= (y << self.t) & self.c
        y ^= y >> self.l

        self.index += 1
        return self.int_32(y) / float(2**self.w)

    def random_interval(self, min_value, max_value):
        # Scale the random number to fit within the interval [min_value, max_value]
        return min_value + (max_value - min_value) * self.random()


# Example usage:
seed = 12345
mt = MersenneTwister(seed)



# Generate a list of random numbers within an interval, for example [-10, 10]







import numpy as np
import matplotlib.pyplot as plt

# Custom functions for moments
def calculate_mean(data):
    return sum(data) / len(data)

def calculate_variance(data, mean):
    return sum((x - mean) ** 2 for x in data) / len(data)

def calculate_skewness(data, mean, variance):
    n = len(data)
    skewness = sum((x - mean) ** 3 for x in data) / (n * (variance ** 1.5))
    return skewness

def calculate_kurtosis(data, mean, variance):
    n = len(data)
    kurt = sum((x - mean) ** 4 for x in data) / (n * (variance ** 2)) - 3
    return kurt

# Function to generate statistics with both predefined and custom calculations
def generate_statistics(xmin, xmax, num_samples=100000):
    # Generate uniform random numbers from the above defined m_twister
    data = random_numbers =  [mt.random_interval(xmin, xmax) for _ in range(num_samples)]
    # Calculate moments using custom functions
    custom_mean = calculate_mean(data)
    custom_variance = calculate_variance(data, custom_mean)
    custom_skewness = calculate_skewness(data, custom_mean, custom_variance)
    custom_kurtosis = calculate_kurtosis(data, custom_mean, custom_variance)
    
    # Calculate moments using predefined NumPy functions
    np_mean = np.mean(data)
    np_variance = np.var(data)
    np_skewness = (np.sum((data - np_mean)**3) / num_samples) / (np_variance ** 1.5)
    np_kurtosis = (np.sum((data - np_mean)**4) / num_samples) / (np_variance ** 2) - 3
    
    # Print statistics for custom and predefined calculations
    print(f"Statistics for Uniform({xmin}, {xmax}):")
    print("Custom Calculations:")
    print(f"Mean: {custom_mean}")
    print(f"Variance: {custom_variance}")
    print(f"Skewness: {custom_skewness}")
    print(f"Kurtosis: {custom_kurtosis}")
    
    print("\nNumPy Predefined Calculations:")
    print(f"Mean: {np_mean}")
    print(f"Variance: {np_variance}")
    print(f"Skewness: {np_skewness}")
    print(f"Kurtosis: {np_kurtosis}")
    print("\n" + "-"*50 + "\n")
    
    # Plot histogram
    plt.hist(data, bins=50, alpha=0.7, color='blue', edgecolor='black')
    plt.title(f"Histogram of Uniform({xmin}, {xmax})")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.show()

# Question 1: Generate 100,000 RN uniformly between 0 and 1
print("Question 1:")
generate_statistics(0, 1)

# Question 2: Generate 100,000 RN uniformly between given ranges
ranges = [
    (0, 1),
    (-1, 1),
    (0, 10),
    (-10, 10)
]

print("Question 2:")
for i, (xmin, xmax) in enumerate(ranges, start=1):
    print(f"Statistics for Histo_{i}:")
    generate_statistics(xmin, xmax)
