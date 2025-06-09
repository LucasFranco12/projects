import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

def linearRegression(x, y, learning_rate=0.01, iters=1000):
    # Initialize parameters
    m = np.random.uniform(0, 1) * -1
    c = np.random.uniform(0, 1) * -1
    y_guess = []
    # Initialize loss list
    loss = []

    for i in range(iters):
        # Forward propagation (m is slope and c is y-intercept)
        y_pred = m * x + c

        # Cost function (Mean Squared Error)
        cost = np.mean((y - y_pred) ** 2)
        loss.append(cost)

        # Backward propagation
        df = y_pred - y
        dm = 2 * np.mean(x * df)
        dc = 2 * np.mean(df)

        # Update parameters
        m -= learning_rate * dm
        c -= learning_rate * dc

        # Print loss every 100 iterations
        if (i + 1) % 100 == 0:
            print(f"Iteration {i + 1}, Loss: {cost}")
        y_guess.append(y_pred)

    # Calculate R-squared (how well our line fits)
    ss_total = np.sum((y - np.mean(y)) ** 2)
    ss_residual = np.sum((y - y_pred) ** 2)
    r_squared = 1 - (ss_residual / ss_total)

    return m, c, y_pred, cost, r_squared, y_guess

def main():
    #np.random.seed(0)
    x = 4 * np.random.rand(500, 1)
    y = 4 + 3 * x + np.random.randn(500, 1) * 1
    
    # Get results
    slope, bias, y_pred, mse, r_squared, y_guess = linearRegression(x, y)
    
    # Print results
    print(f"Estimated slope: {slope}")
    print(f"Estimated y-intercept: {bias}")
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared Score: {r_squared}")
    
    # Plot results
    fig, ax = plt.subplots()
    ax.scatter(x, y, color='blue', label='Data points')
    line, = ax.plot(x, y_guess[0], color='red', label='Linear regression')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Linear Regression Results')
    plt.legend()

    def update(frame):
        line.set_ydata(y_guess[frame])
        return line,

    ani = FuncAnimation(fig, update, frames=len(y_guess), interval=5, blit=True)
    plt.show()

if __name__ == '__main__':
    main()