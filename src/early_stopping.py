class EarlyStopper:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.num_bad = 0

    def step(self, current):
        improved = current < (self.best - self.min_delta)
        if improved:
            self.best = current
            self.num_bad = 0
        else:
            self.num_bad += 1
        return self.num_bad >= self.patience, improved

