# Mascis Larson and Michael Marsico

from typing import Tuple
from neural import *
from sklearn.model_selection import train_test_split

def parse_line(line: str) -> Tuple[List[float], List[float]]:
    """Splits line of CSV into inputs and output (transormfing output as appropriate)

    Args:
        line - one line of the CSV as a string

    Returns:
        tuple of input list and output list
    """
    tokens = line.split(",")
    out = [int(tokens[len(tokens) - 1])]

    inpt = [float(x) for x in tokens[:len(tokens) - 1]]
    return (inpt, out)


def normalize(data: List[Tuple[List[float], List[float]]]):
    """Makes the data range for each input feature from 0 to 1

    Args:
        data - list of (input, output) tuples

    Returns:
        normalized data where input features are mapped to 0-1 range (output already
        mapped in parse_line)
    """
    leasts = len(data[0][0]) * [100.0]
    mosts = len(data[0][0]) * [0.0]

    for i in range(len(data)):
        for j in range(len(data[i][0])):
            if data[i][0][j] < leasts[j]:
                leasts[j] = data[i][0][j]
            if data[i][0][j] > mosts[j]:
                mosts[j] = data[i][0][j]

    for i in range(len(data)):
        for j in range(len(data[i][0])):
            data[i][0][j] = (data[i][0][j] - leasts[j]) / (mosts[j] - leasts[j])
    return data


with open("data/spambase.data", "r") as f:
    training_data = [parse_line(line) for line in f.readlines() if len(line) > 4]

# print(training_data)
td = normalize(training_data)
# print(td)

train, test = train_test_split(td)

nn = NeuralNet(57, 26, 1)
nn.train(train, iters=100, print_interval=1, learning_rate=0.5)

for i in nn.test_with_expected(test):
    actual = i[1][0]
    predicted = i[2][0]
    confidence = abs((0.5 - round(abs(predicted - actual), 3)) / 0.5) * 100

    actual = "Spam" if actual == 1 else "Not Spam"
    predicted = "Spam" if predicted == 1 else "Not Spam"

    print(f"-------------------------------------------")
    print(f"Predicted Classification: {predicted} with {confidence}% Confidence")
    print(f"Actual Classification: {actual}")
