import math


class EpsilonScheduler:

    def __init__(self, epsilon_start, epsilon_final, epsilon_decay):
        self.start = epsilon_start
        self.final = epsilon_final
        self.decay = epsilon_decay

    def __call__(self, episode):
        return self.final + (self.start - self.final) * math.exp(-1.0 * episode / self.decay)


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    length = 10_000
    scheduler = EpsilonScheduler(
        epsilon_start=1.0,
        epsilon_final=0.05,
        epsilon_decay=3_000,
    )

    plt.plot([scheduler(i) for i in range(length)])
    plt.show()
