import numpy as np
import matplotlib.pyplot as plt

def generate_soft_max_distribution(z: list[float], N: int) -> tuple[list[float]]:
    """
    Generate a soft max distribution 
    """

    denominator = 0
    for j in range(len(z)):
        denominator += np.exp(z[j])
    
    f_x = []

    for z_element in z:
        f_x.append(np.exp(z_element)/denominator)
    
    random_sample = np.random.choice(a=z, p=f_x, size=N)

    relative_frequencies = np.bincount(random_sample) / N

    return f_x, relative_frequencies[z[0]:]


z_test = [4, 5, 6, 7, 8, 9, 10]
N_test = 10000

soft_max_distribution, random_sample = generate_soft_max_distribution(z_test, N_test)

plt.stem(z_test, soft_max_distribution, 'r--', label="ACTUAL DISTRIBUTION")

plt.stem(z_test, random_sample, label="RELATIVE FREQUENCIES")
plt.legend()
plt.show()
