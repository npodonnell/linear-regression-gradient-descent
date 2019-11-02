#!/usr/bin/env python3

ITERS = 10000
LEARNING_RATE = 0.01


def predict(x, m, b):
    """
    Make a prediction for y given x using line formula
    """
    return (m * x) + b


def error(points, m, b):
    """
    Error function. Average of squared difference
    between actual and expected output. This is
    the function we wish to minimise.
    """
    sum_of_squares = 0.0
    for x, y in points:
        sum_of_squares += (y - predict(x, m, b)) ** 2
    return sum_of_squares / len(points)


def error_gradient(points, m, b):
    """
    Evaluates the gradient of the error function. The
    gradient is the vector of the partial derivatives
    with respect to m and b. 
    """
    m_grad = 0.0
    b_grad = 0.0
    for x, y in points:
        diff = y - predict(x, m, b)
        m_grad -= x * diff
        b_grad -= diff
    return (
        (2 * m_grad) / len(points), 
        (2 * b_grad) / len(points)
    )


def main():
    points = [(4.0, 7.0), (7.0, 6.0), (3.0, 2.0)]
    m = 0.0
    b = 0.0
    for i in range(ITERS):
        m_change, b_change = error_gradient(points, m, b)
        m -= m_change * LEARNING_RATE
        b -= b_change * LEARNING_RATE
        print("Iteration:", i, "Error:", error(points, m, b), "m:", m, "b:", b)


if __name__ == "__main__":
    main()
