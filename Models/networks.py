import time
import os

import numpy as np
from torch import nn
import torch


class MLP(nn.Module):
    def __init__(self, in_dims: int, out_dims: int, layers: list[int], final_activation_function=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # List to add all layers to.
        model = list()

        in_dims = [in_dims] + layers
        out_dims = layers + [out_dims]

        for idx, (in_dim, out_dim) in enumerate(zip(in_dims, out_dims)):
            # Possibly use LazyLinear but not an issue.
            model.append(nn.Linear(in_dim, out_dim))

            # Check if it's not the last layer, if it isn't add the activation
            # Maybe add a param to set this function, but not worth the to-do
            if idx < len(layers):
                model.append(nn.ReLU())  # TODO: Test ['nn.LeakyReLU()']

        if type(final_activation_function) == list:
            model += final_activation_function

        elif final_activation_function is not None:
            model.append(final_activation_function)

        # Combine into a model
        self.main = nn.Sequential(*model)
        self.is_compiled = False

        self.try_compile()

    def try_compile(self):
        # Check if running on a system that can compile, I can't test if this will actually work or what it may break.
        try:
            t1 = time.time()
            self.main = torch.compile(self.main)
            print(f"Network compiled in {time.time() - t1} seconds")
            self.is_compiled = True
        except RuntimeError:
            print("Compiling is not supported on this platform")

    def forward(self, values):
        if isinstance(values, np.ndarray):
            values = torch.tensor(values, dtype=torch.float32)
        return self.main(values)

    def get_saved_location(self, path, name):
        location = os.path.join(path, name)
        return location + "_compiled" if self.is_compiled else "" + ".pth"

    def save(self, path, name):
        torch.save(self.state_dict(), self.get_saved_location(path, name))

    def load(self, path, name):
        self.load_state_dict(torch.load(self.get_saved_location(path, name)))


if __name__ == "__main__":
    x = torch.Tensor(list(range(10)))
    network = MLP(10, 2, [64, 128, 32], nn.Softmax(dim=0))
    network.save("SavedModels/TestModels/", "TestModel")
    print(network)
    result_saved = network(x).detach()

    del network
    network = MLP(10, 2, [64, 128, 32], nn.Softmax(dim=0))
    network.load("SavedModels/TestModels/", "TestModel")
    result_loaded = network(x).detach()

    if torch.equal(result_saved, result_loaded):
        print("Networks produce same result")
    else:
        print(f"Networks produce differing results: {result_saved} vs {result_loaded}")
