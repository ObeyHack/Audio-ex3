import numpy as np


def p2z(p, blank):
    z = [blank]
    for char in p:
        z.append(char)
        z.append(blank)
    return z


def ctc(output_path, label_str, alphabet_str):
    """
    z = [eps, p_{1}, eps, p_{2}, eps, p_{3}, eps, ..., eps, p_{|p|}, eps]

    dynamic programming:
    a_{s},{t} = { (a_{s-1},{t-1} + a_{s},{t-1}) * y_{t}{z_{s}}                  if z_{s} = eps or z_{s} = z_{s-2}
                   (a_{s-1},{t-1} + a_{s},{t-1} + a_{s-2},{t-1}) * y_{t}{z_{s}}  otherwise
    a_{1},{1} = y_{1}{eps}
    a_{2},{1} = y_{1}{z_{1}}
    a_{s},{1} = 0 for s > 2

    :param output_path: A path to a 2D numpy matrix of network outputs (y). This should be loaded using numpy.load.
                        y = is a matrix with the shape of T × K where T is the number of time steps, and K is the
                        amount of phonemes. Each row i of y is a distribution over K phonemes at time i.
    :param label_str: A string of the labeling you wish to calculate the probability for (e.g., “aaabb” means we
                      want the probability of aaabb).
    :param alphabet_str: A string specifying the possible output tokens (e.g., for an alphabet of [a,b,c] the string
                         should be “abc”).
    :return: P(p|y) - The probability of the labeling given the network outputs.
    """
    y = np.load(output_path)
    T, K = y.shape
    p = list(label_str)
    eps = ''
    alphabet = [eps] + list(alphabet_str)

    # Extend p to include blanks
    z = p2z(p, eps)
    L = len(z)

    # Initialize alpha
    alpha = np.zeros((L, T))
    # a_{1},{1} = y_{1}{eps}
    alpha[0, 0] = y[0, alphabet.index(eps)]
    # a_{2},{1} = y_{1}{z_{1}}
    alpha[1, 0] = y[0, alphabet.index(z[1])]

    # Dynamic programming

    for t in range(1, T):
        for s in range(L):
            # a_{s},{1} = 0 for s > 2
            if s > 1 and t == 0:
                alpha[s, t] = 0

            # (a_{s-1},{t-1} + a_{s},{t-1}) * y_{t}{z_{s}}   if z_{s} = eps or z_{s} = z_{s-2}
            elif z[s] == eps or z[s] == z[s - 2]:
                alpha[s, t] = (alpha[s, t - 1] + alpha[s - 1, t - 1]) * y[t, alphabet.index(z[s])]

            # (a_{s-1},{t-1} + a_{s},{t-1} + a_{s-2},{t-1}) * y_{t}{z_{s}}  otherwise
            else:
                alpha[s, t] = (alpha[s, t - 1] + alpha[s - 1, t - 1] + alpha[s - 2, t - 1]) * y[t, alphabet.index(z[s])]

    # Calculate probability P(p|y)
    p_y_given_p = alpha[-1, -1] + alpha[-2, -1]
    return p_y_given_p


def print_p(p: float):
    print("%.3f" % p)


if __name__ == '__main__':
    path = "mat.npy"
    labels = "aaabb"
    possible_labels = "abc"
    print_p(ctc(path, labels, possible_labels))