import torch
import torch

# define two classes, the Layer class and the Model class, where the model class has an array of layers. the Layer class should implement a forward function that looks like Ax + B with torch functions
# randomly initialize A and B of Layer


class Layer(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(Layer, self).__init__()
        self.A = torch.nn.Parameter(torch.randn(input_size, output_size))
        self.B = torch.nn.Parameter(torch.randn(output_size))

    def forward(self, x):
        return torch.nn.LeakyReLU(0.2)(torch.matmul(x, self.A) + self.B)


class Model(torch.nn.Module):
    def __init__(self, input_size, output_size, layer_sizes):
        super(Model, self).__init__()
        self.layers = torch.nn.ModuleList([Layer(input_size, layer_sizes[0])])
        for i in range(1, len(layer_sizes)):
            self.layers.append(Layer(layer_sizes[i - 1], layer_sizes[i]))
        self.layers.append(Layer(layer_sizes[-1], output_size))

    def forward(self, x, debug=False):
        for layer in self.layers:
            x = layer.forward(x)
            if debug:
                print(x)
        return x

    def learn(self, epochs=1000, lr=0.1):
        loss_fn = torch.nn.MSELoss()
        # use adam optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        for epoch in range(epochs):
            x = torch.randn(100000, 2)
            y = torch.mul(x[:, 0], x[:, 1]).unsqueeze(1)
            y_pred = self.forward(x)
            loss = loss_fn(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: loss = {loss.item()}")


model = Model(2, 1, [10, 100, 10])
# generate training data, network will learn x1, x2 -> x1 * x2


model.learn()

x = torch.randn(2)
y = torch.mul(x[0], x[1]).unsqueeze(0)
print(x, y, model.forward(x))
# print(model.forward(x[1]))
