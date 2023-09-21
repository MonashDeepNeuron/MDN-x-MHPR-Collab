import wandb


class Logger:
    def __init__(self, print_order: list | None = None, wandb_details: dict | None = None, print_every: int = 100):
        self.log_dict = {}
        self.print_every = print_every
        self.print_order = print_order

        self.step_count = 0
        self.using_wandb = False
        self.wandb_details = wandb_details

        if wandb_details is not None:
            self.using_wandb = True
            self.run = wandb.init(project=wandb_details["project"], config=wandb_details["config"])

    def step(self):
        self.step_count += 1

        end = "\n" if self.step_count % self.print_every == 0 else ""

        line = []

        if self.print_order is not None:
            for key in self.print_order:
                line.append(f"{key}: {self.log_dict[key]} \t")
        else:
            for key, value in self.log_dict.items():
                line.append(f"{key}: {value} \t")

        print("\r" + "".join(line), end=end)

        if self.using_wandb:
            for detail in self.wandb_details["log_keys"]:
                wandb.log({detail: self.log_dict[detail]}, step=self.step_count)

    def initialize(self, values):
        for key, value in values.items():
            if key not in self.log_dict:
                self.log_dict[key] = value

    def __setitem__(self, key, value):
        self.log_dict[key] = value

    def __getitem__(self, key):
        return self.log_dict[key]


if __name__ == "__main__":
    logger = Logger()
    logger["a"] = 1
    logger["a"] += 2
    logger["b"] = [1, 2]
    logger["b"] += [3, 4]
    print(logger.log_dict)
