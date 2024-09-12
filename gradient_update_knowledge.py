import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Generate random data
np.random.seed(42)
x = 2 * np.random.rand(100, 1)
y = 4 + 3 * x + np.random.randn(100, 1)

# Initialize parameters
theta0 = 0
theta1 = 0
learning_rate = 0.01
iterations = 1000

# To store history
cost_history = []
theta0_history = []
theta1_history = []

# Gradient Descent with tracking
def gradient_descent_with_tracking(x, y, theta0, theta1, learning_rate, iterations):
    m = len(y)
    
    for i in range(iterations):
        predictions = theta0 + theta1 * x
        d_theta0 = (1 / m) * np.sum(predictions - y)
        d_theta1 = (1 / m) * np.sum((predictions - y) * x)
        
        theta0 -= learning_rate * d_theta0
        theta1 -= learning_rate * d_theta1
        
        cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
        cost_history.append(cost)
        theta0_history.append(theta0)
        theta1_history.append(theta1)
    
    return theta0, theta1, cost_history

theta0, theta1, cost_history = gradient_descent_with_tracking(x, y, theta0, theta1, learning_rate, iterations)

# Create figure and axis
fig, ax = plt.subplots()
ax.set_xlim(0, 2)
ax.set_ylim(0, 15)
line, = ax.plot([], [], color='red', linewidth=2)

# Scatter plot of the data
ax.scatter(x, y, color='blue', label='Data Points')

# Initialize the line
def init():
    line.set_data([], [])
    return line,

# Update the line
def update(i):
    y_pred = theta0_history[i] + theta1_history[i] * x
    line.set_data(x, y_pred)
    return line,

# Create the animation with a 0.5-second interval (500 ms)
ani = FuncAnimation(fig, update, frames=len(theta0_history), init_func=init, blit=True, repeat=False, interval=30)

# Additional elements for clarity
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Gradient Descent Best-Fit Line Animation")
ax.legend()

# Show the animation
plt.show()
