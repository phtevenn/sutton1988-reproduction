import numpy as np
import matplotlib.pyplot as plt
import argparse

Z_TRUE = [1 / 6, 1 / 3, 1 / 2, 2 / 3, 5 / 6]
N_SEQS_PER_SET = 10
N_TRAINING_SETS = 100
np.random.seed(42)


class RandomWalk:
    def __init__(self, n_steps, start):
        self.n_steps = n_steps
        self.start = start
        start_state = np.zeros(n_steps, dtype=int)
        start_state[start] = 1
        self.states = [start_state]
        self.state = start  # mid point
        self.end = False

    def get_state_matrix(self):
        return np.vstack(self.states)

    def check_end(self):
        if self.state == 0 or self.state == self.n_steps - 1:
            self.end = True
        else:
            return False

    def step(self):
        if not self.end:
            self.state += np.random.choice([-1, 1])
            state_vec = np.zeros(self.n_steps, dtype=int)
            state_vec[self.state] = 1
            self.states.append(state_vec)
            # self.states.append(self.state)
            self.check_end()
        return self.state

    def gen_walk(self):
        while not self.end:
            self.step()
        return self.get_state_matrix()

    def reset(self):
        start_state = np.zeros(self.n_steps)
        start_state[self.start] = 1
        self.states = [start_state]
        self.state = self.start  # mid point
        self.end = False

    def build_seqs(self, n_runs):
        runs = []
        for i in range(n_runs):
            runs.append(self.gen_walk())
            self.reset()
        return runs


def RMSE(z, z_true):
    return np.sqrt(np.mean((z - z_true) ** 2))


class TD_Agent:
    def __init__(self, alpha, lmbda):
        self.alpha = alpha
        self.lmbda = lmbda

    # training set struct
    # training_set[set index][run index]
    # each run is one hot encoded matrix where each column is one step
    # 1 for the current state, 0 for other states

    def update(self, full_seq, p):
        seq = full_seq[:-1, 1:-1]  # truncate ending
        n_steps, n_states = seq.shape

        z = int(full_seq[-1, -1])
        lambdas = np.ones(1)
        delta_w = np.zeros(n_states)
        for step in range(n_steps):
            curr_seq = seq[: step + 1]
            # print(curr_seq)
            if step < n_steps - 1:
                delta_p = np.dot(p, seq[step + 1, :]) - np.dot(p, seq[step, :])
            else:  # terminal state
                delta_p = z - np.dot(p, seq[-1, :])
            delta_w += (
                self.alpha * delta_p * np.sum(curr_seq * lambdas[:, None], axis=0)
            )

            # update lambdas
            lambdas *= self.lmbda
            lambdas = np.concatenate((lambdas, np.ones(1)))  # discount/add lambda
        return delta_w

    def repeated_repr_training(self, training_set, iterations=100, tol=1e-5):
        weights = np.zeros(5) + 0.5
        deltas = np.zeros(5)
        no_change_count = 0
        for i in range(iterations):
            weights_prev = weights.copy()
            for train in training_set:
                for seq in train:
                    deltas += self.update(seq, weights)
                weights += deltas
                # print(weights)
                deltas = np.zeros(5)
            if np.sum(weights - weights_prev) < tol:
                # consider "converged" when 3 iterations with no change
                no_change_count += 1
                if no_change_count > 3:
                    print(f"Converged in {i} iterations")
                    break
                else:
                    no_change_count = 0
        return weights

    def train(self, training_set):
        all_weights = []
        for train in training_set:
            weights = np.zeros(5) + 0.5
            for seq in train:
                weights += self.update(seq, weights)
            all_weights.append(weights)
        return all_weights


def generate_figure(fig, generate_all=False):
    rw_agent = RandomWalk(7, 3)
    training_set = []
    for i in range(N_TRAINING_SETS):
        training_set.append(rw_agent.build_seqs(N_SEQS_PER_SET))
        rw_agent.reset()
    if fig == 3 or generate_all:
        lambdas = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
        RMSEs = []
        agent = TD_Agent(0.001, None)
        for lb in lambdas:
            agent.lmbda = lb
            agent.lmbda = lb
            result = agent.repeated_repr_training(training_set, 20)
            RMSEs.append(RMSE(result, Z_TRUE))
        plt.plot(lambdas, RMSEs, marker="s", markerfacecolor="white")
        plt.xlabel("Lambda")
        plt.ylabel("RMSE")
        plt.title("Replication of Sutton 1988 Figure 3")
        plt.savefig("fig_3.png")
    elif fig == 4 or generate_all:
        fig4_lambdas = [0, 0.3, 0.8, 1]
        # paper samples different amount of alphas for each lambda, but easier if we do same amount
        fig4_alphas = np.arange(0.001, 0.5, 0.03)

        agent = TD_Agent(None, None)
        fig4_RMSEs = []
        for lb in fig4_lambdas:
            agent.lmbda = lb
            lb_RMSEs = []
            for alpha in fig4_alphas:
                agent.alpha = alpha
                preds = agent.train(training_set)
                RMSEs = []
                for p in preds:
                    RMSEs.append(RMSE(p, Z_TRUE))
                lb_RMSEs.append(np.mean(RMSEs))
            fig4_RMSEs.append(lb_RMSEs)

        for i, fig4_RMSE in enumerate(fig4_RMSEs):
            plt.plot(
                fig4_alphas,
                fig4_RMSE,
                marker="s",
                markerfacecolor="white",
                label=fig4_lambdas[i],
            )
        plt.legend(title="Lambdas")
        plt.xlabel("Alpha")
        plt.ylabel("RMSE")
        plt.title("Replication of Sutton 1988 Figure 4")
        plt.savefig("fig_4.png")
    elif fig == 5 or generate_all:
        fig5_lambdas = np.arange(0, 1.1, 0.1)
        fig5_alphas = np.logspace(-5, 0, 20)
        fig5_RMSEs = []
        best_alphas = []

        for lb in fig5_lambdas:
            RMSEs = []
            agent = TD_Agent(None, lb)
            for alpha in fig5_alphas:
                set_RMSEs = []
                agent.alpha = alpha
                preds = agent.train(training_set)
                for p in preds:
                    set_RMSEs.append(RMSE(p, Z_TRUE))
                RMSEs.append(np.mean(set_RMSEs))
            best_index = np.argmin(RMSEs)
            fig5_RMSEs.append(RMSEs[best_index])
            best_alphas.append(fig5_alphas[best_index])
        plt.plot(fig5_lambdas, fig5_RMSEs, marker="s", markerfacecolor="white")
        plt.xlabel("Lambdas")
        plt.ylabel("RMSE with best alpha")
        plt.title("Replication of Sutton 1988 Figure 5")
        plt.savefig("fig_5.png")
    if not generate_all:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)

    parser.add_argument("-f", action="store", dest="fig", default=False)
    fig = parser.parse_args().fig
    if fig == "all":
        generate_figure(None, generate_all=True)
    else:
        generate_figure(int(fig))
