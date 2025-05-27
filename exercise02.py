import numpy as np
import matplotlib.pyplot as plt


# Heads -> X(omega) = 1
# Tails -> X(omega) = 0

# Confidence interval:
# C_n = (p^_n - epislon_n, p^_n + epislon_n)
# Where:
# epislon_n = sqrt(ln(2/alpha)/(2*n))


def epislon(n_: int, alpha_: float) -> float:
    """
    This function computes the epislon from
     the parameters n and alpha.
    The epislon is used to calculate the confidence interval

    Parameters
    ----------
    n_: int
        The number of coin tosses
    alpha_: float
        Used to define the lenght of the confidence interval
        C_n is a (1-alpha)-confidence interval for parameter p

    Returns
    -------
    epislon: float
        The
    """
    return np.sqrt(np.log(2 / alpha_) / (2 * n_))


def p_hat(X_: np.ndarray) -> float:
    """
    Computes the estimator p_hat of the parameter p,
    based on the previous results of the coin tosses.
    This is done using a simple mean.

    Parameters
    ----------
    X_: np.ndarray
        The previous results of all the coin tosses

    Returns
    -------
    p_hat: float
        Estimator for the parameter p
    """
    sum = 0
    n_ = X_.size
    for i in range(n_):
        sum += X_[i]

    return sum / n_


def confidence_interval(X_: np.ndarray, alpha_: float) -> tuple[float, float]:
    """
    Computes the confidence interval in which we are confident that our estimator p_hat is.

    Parameters
    ----------
    X_: np.ndarray
        The previous results of all the coin tosses
    alpha_: float
        Used to define the lenght of the confidence interval
        C_n is a (1-alpha)-confidence interval for parameter p

    Returns
    -------
    lower_bound: float
        The lower bound of the confidence interval
    upper_bound: float
        The upper bound of the confidence interval
    """

    n_ = X_.size
    epislon_ = epislon(n_, alpha_)
    p_hat_ = p_hat(X_)

    return p_hat_ - epislon_, p_hat_ + epislon_


"""
1.1. Conduct a simulation study to see how often the confidence interval contains p (called
    the coverage). Do this for various values of n between 1 and 1000. Plot the coverage
    versus n
"""


def coverage(X_: np.ndarray, alpha_: float, p_: float) -> bool:
    """
    Given the results of the coin tosses and the parameter p,
    is our estimator inside the confidence interval?

    Parameters
    ----------
    X_: np.ndarray
        The previous results of all the coin tosses
    alpha_: float
        Used to define the lenght of the confidence interval
        C_n is a (1-alpha)-confidence interval for parameter p
    p_: float
        The true probability

    Returns
    -------
    True, if the estimator is inside the confidence interval
    False, if the estimator is not inside the confidence interval
    """

    lower_bound, upper_bound = confidence_interval(X_, alpha_)

    if p_ > lower_bound and p_ < upper_bound:
        return True

    return False


def main():
    n_max = 1000
    alpha = 0.05
    p = 0.4  # True probability of heads
    num_simulations = 100  # Number of repetitions for Monte Carlo

    n_values = np.arange(1, n_max + 1)
    coverage_values = []
    interval_lengths = []

    for n in n_values:
        covered = 0

        for _ in range(num_simulations):
            # Simulate n coin tosses with probability p of heads (1)
            X = np.random.binomial(1, p, n)

            if coverage(X, alpha, p):
                covered += 1

        # Compute coverage probability
        coverage_prob = covered / num_simulations
        coverage_values.append(coverage_prob)

        # Compute interval length (same for all simulations with fixed n and alpha)
        eps = epislon(n, alpha)
        interval_lengths.append(2 * eps)

    # Plot coverage vs n
    plt.figure(figsize=(10, 5))
    plt.scatter(n_values, coverage_values, label="Coverage probability")
    plt.axhline(
        1 - alpha, color="red", linestyle="--", label="Expected coverage (0.95)"
    )
    plt.xlabel("n")
    plt.ylabel("Coverage Probability")
    plt.title("Coverage vs n")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot interval length vs n
    plt.figure(figsize=(10, 5))
    plt.plot(n_values, interval_lengths, label="Interval Length")
    plt.axhline(0.05, color="green", linestyle="--", label="Target length (0.05)")
    plt.xlabel("n")
    plt.ylabel("Interval Length")
    plt.title("Length of Confidence Interval vs n")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Find minimal n such that interval length ≤ 0.05
    for n, length in zip(n_values, interval_lengths):
        if length <= 0.05:
            print(f"The smallest n such that interval length ≤ 0.05 is: {n}")
            break


if __name__ == "__main__":
    main()
