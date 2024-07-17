import torch
import torch.nn as nn
import numpy as np


T = 50      # Input sequence length
C = 20      # Number of classes (including blank)
N = 16      # Batch size
S = 30      # Target sequence length of longest target in batch
S_min = 10  # Minimum target length, for demonstration purposes

# Initialize random batch of input vectors, for *size = (T,N,C)
input = torch.randn(T, N, C).log_softmax(2).detach().requires_grad_()

# Initialize random batch of targets (0 = blank, 1:C = classes)
target = torch.randint(low=1, high=C, size=(N, S), dtype=torch.long)

input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long)
target_lengths = torch.randint(low=S_min, high=S, size=(N,), dtype=torch.long)
ctc_loss = nn.CTCLoss()
loss = ctc_loss(input, target, input_lengths, target_lengths)
loss.backward()

print(loss)
import numpy as np


def ctc_loss(y_path, label_str, alphabet_str):
    y = np.load(y_path)
    T, K = y.shape
    p = list(label_str)
    alphabet = list(alphabet_str)

    # Define blank token
    blank = ''

    # Extend p to include blanks
    z = [blank]
    for char in p:
        z.append(char)
        z.append(blank)

    L = len(z)

    # Initialize alpha
    alpha = np.zeros((L, T))

    alpha[1, 0] = y[0, alphabet.index(z[1])]
    alpha[0, 0] = y[0, alphabet.index(z[0])]

    # Dynamic programming
    for t in range(1, T):
        for s in range(L):
            if s == 0:
                alpha[s, t] = alpha[s, t - 1] * y[t, alphabet.index(z[s])]
            elif s == 1:
                alpha[s, t] = (alpha[s, t - 1] + alpha[s - 1, t - 1]) * y[t, alphabet.index(z[s])]
            else:
                alpha[s, t] = (alpha[s, t - 1] + alpha[s - 1, t - 1] + alpha[s - 2, t - 1] * (z[s] != z[s - 2])) * y[
                    t, alphabet.index(z[s])]

    # Calculate probability P(p|y)
    p_y_given_p = alpha[-1, -1] + alpha[-2, -1]

    return p_y_given_p


def print_p(p: float):
    print("%.3f" % p)


# Example usage
if __name__ == "__main__":
    import sys

    y_path = sys.argv[1]
    label_str = sys.argv[2]
    alphabet_str = sys.argv[3]

    p = ctc_loss(y_path, label_str, alphabet_str)
    print_p(p)
