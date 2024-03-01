import torch
import torch.nn as nn
import random
from tkinter import *


# Define the neural network using PyTorch integrated functions.
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(6, 4)
        self.fc2 = nn.Linear(4, 2)
        self.fc3 = nn.Linear(2, 1)

    # Defines the forward pass of the neural network model. It takes an input tensor 'x' and passes it through three
    # fully connected layers sequentially, applying ReLU activation after the first two layers and sigmoid activation
    # after the third layer. This function computes the output of the neural network for a given input.
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


# Initialize the neural network
net = Net()

# Define the loss function and optimizer
# Part of what narrows down the color choices
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

# Initialize data for training
data = [
    {"input": torch.tensor([0, 0, 0, 0, 0, 0], dtype=torch.float32), "output": torch.tensor([0], dtype=torch.float32)},
    {"input": torch.tensor([1, 1, 1, 1, 1, 1], dtype=torch.float32), "output": torch.tensor([0], dtype=torch.float32)}
]


# Function to get a random color value
def get_random_color_value():
    return random.random()


# Function to update the colors and prediction
def update_colors_and_prediction():
    global colors
    colors = [get_random_color_value() for _ in range(6)]
    input_tensor = torch.tensor(colors, dtype=torch.float32)
    prediction = net(input_tensor).item()
    update_canvas(colors[:3], colors[3:])
    update_prediction_label(prediction)


# Function to update the canvas with the new colors
def update_canvas(circle_color, rect_color):
    canvas.delete("all")
    canvas.create_rectangle(10, 10, 190, 190, fill=rgb_to_hex(rect_color), outline="")
    canvas.create_oval(60, 60, 140, 140, fill=rgb_to_hex(circle_color), outline="")


# Function to update the prediction label
def update_prediction_label(prediction):
    prediction_label.config(text=f"Chance You May Like it: {prediction * 100:.1f}%")


# Function to train the network with the new data
def train_network(value):
    global data
    data.append(
        {"input": torch.tensor(colors, dtype=torch.float32), "output": torch.tensor([value], dtype=torch.float32)})
    for epoch in range(100):  # number of epochs can be adjusted
        for sample in data:
            optimizer.zero_grad()
            output = net(sample["input"])
            loss = criterion(output, sample["output"])
            loss.backward()
            optimizer.step()
    update_colors_and_prediction()


# Function to convert RGB values to hexadecimal
def rgb_to_hex(rgb):
    return f'#{int(rgb[0] * 255):02x}{int(rgb[1] * 255):02x}{int(rgb[2] * 255):02x}'


# Create the tkinter window
root = Tk()
root.title("Neural Network Recoloring Predictor")

# Create a canvas to display the colors. I don't lock the canvas, but it is intended for 200x200
canvas = Canvas(root, width=200, height=200)
canvas.pack()

# Create buttons for user feedback. The weights of each user choice is
# displayed in the train_network argument (1, 0.75, 0.5, 0).
Button(root, text="Love it", command=lambda: train_network(1)).pack(side=LEFT)
Button(root, text="Like it", command=lambda: train_network(0.75)).pack(side=LEFT)
Button(root, text="Meh", command=lambda: train_network(0.5)).pack(side=LEFT)
Button(root, text="Hate it", command=lambda: train_network(0)).pack(side=LEFT)

# Create a label to display the prediction
prediction_label = Label(root, text="Chance You May Like it: 0%")
prediction_label.pack()

# Initialize the colors and prediction
colors = [0.74901960784, 0.03921568627, 0.18823529412, 1, 0.8431372549, 0]
update_canvas(colors[:3], colors[3:])
update_prediction_label(net(torch.tensor(colors, dtype=torch.float32)).item())  # Updates the prediction

# Start the tkinter main loop (run the actual code)
root.mainloop()
