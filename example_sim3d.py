import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import threading
import time
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import env_wrapper

env_wrapper.register_env()
env = gym.make("RocketSim-v0")
observation, _ = env.reset()

# lists to store historical data
x = []
y = []
z = []

# function to update the plot
def create_plot():
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('Time')
    ax.set_ylabel('Altitude')
    ax.set_zlabel('Displacement')
    plt.title("Rocket Simulation")

    # Plot the ground as a flat plane at z=0
    X = np.arange(-0.2, 0.2, 0.25)
    Y = np.arange(-0.2, 0.2, 0.25)
    X, Y = np.meshgrid(X, Y)
    Z = np.zeros(X.shape)
    ax.plot_surface(X, Y, Z, alpha=0.5)



create_plot() 
fig = plt.gcf()

def animate(i):
    plt.cla()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title("SATURN Rocket Simulation")
    
    ax.plot(x, y, z)

    # ground as a flat plane z=0
    X = np.arange(-0.2, 0.2, 0.25)
    Y = np.arange(-0.2, 0.2, 0.25)
    X, Y = np.meshgrid(X, Y)
    Z = np.zeros(X.shape)
    ax.plot_surface(X, Y, Z, alpha=0.5)



ani = animation.FuncAnimation(fig, animate, interval=1000)


def data_collection():
    while True:
        action = env.action_space.sample()
        observation, reward, terminated, truncated, _ = env.step(action)
        
        x.append(observation["displacement"][0])
        y.append(observation["displacement"][1])
        z.append(observation["displacement"][2])
        
        # ADD YOUR RL CODE HERE

        time.sleep(1)
        if terminated or truncated:
            print()
            print("----------------- Env Reset -----------------")
            observation, _ = env.reset()
            break

# data collection in a separate thread to plot
data_thread = threading.Thread(target=data_collection)
data_thread.daemon = True
data_thread.start()

plt.show()
