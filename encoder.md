# Encoder

We're encoding the numbers from 1 to 8, inputted as one-hot vectors. For this reason the network has 8 input neurons and 8 output neurons. The hidden layer of the network consists in 3 neurons that will create and internal representation of the the input space.

### One-hot encoding

In a one-hot encoding the number 2 can be represented as:

|2|
|---------------------------|
|-1|**+1**|-1|-1|-1|-1|-1|-1|

### Interpretation of the hidden layer output

In order to understand the internal representation created by the network, we consider the sign of the output of the hidden layer. Given that there are only three neurons we can represent it as a binary string.

Running the training algorithm several times we get:

| In | Hidden  | Out |
|----|---------|-----|
| 1  | 0  1  1 | 1   |
| 2  | 0  1  0 | 2   |
| 3  | 0  0  1 | 3   |
| 4  | 0  1  1 | 4   |
| 5  | 1  1  0 | 5   |
| 6  | 1  1  1 | 6   |
| 7  | 1  1  1 | 7   |
| 8  | 0  1  0 | 8   |

| In | Hidden  | Out |
|----|---------|-----|
| 1  | 0  1  1 | 1   |
| 2  | 0  1  0 | 2   |
| 3  | 0  0  1 | 3   |
| 4  | 1  1  0 | 4   |
| 5  | 1  0  0 | 5   |
| 6  | 1  1  1 | 6   |
| 7  | 1  0  1 | 7   |
| 8  | 0  1  0 | 8   |

| In | Hidden  | Out |
|----|---------|-----|
| 1  | 0  1  0 | 1   |
| 2  | 0  1  0 | 2   |
| 3  | 1  0  0 | 3   |
| 4  | 1  1  0 | 4   |
| 5  | 0  0  1 | 5   |
| 6  | 1  1  1 | 6   |
| 7  | 1  0  1 | 7   |
| 8  | 0  1  1 | 8   |

We can immediately note that the encoding that the network has learned is not the usual binary encoding that's used in informatics. Also, different runs can yield different encodings. We can interpret this by observing that there is no particular advantage of one encoding over another, so every time the network learns it picks randomly one encoding that works.

Finally, notice that sometimes the binary strings encoding one number are the same (number 6 and 7 in the first table). Remember however that we are only printing the sign of the hidden output and that differences in the actual values are what allows the network to discriminate two numbers.
