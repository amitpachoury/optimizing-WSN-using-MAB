import math
import random
import matplotlib.pyplot as plt


# -----------------------------
# 1) Environment (Non-stationary)
# -----------------------------
class NonStationaryWSN:
        """
        3-channel WSN where the success probabilities flip at 'switch_t'.
        Example:
        before: [0.80, 0.40, 0.20]
        after : [0.10, 0.40, 0.80]
        """
        def __init__(self, probs_before, probs_after, switch_t):
            assert len(probs_before) == len(probs_after)
            self.K = len(probs_before)
            self.p0 = probs_before
            self.p1 = probs_after
            self.switch_t = switch_t
            self.t = 0


        def step(self, arm: int) -> int:
            """
            One transmission on 'arm' (channel).
            Returns: 1 (ACK success) or 0 (fail).
            """
            self.t += 1
            p = self.p0[arm] if self.t <= self.switch_t else self.p1[arm]
            return 1 if random.random() < p else 0




# -----------------------------
# 2) Algorithms
# -----------------------------
class UCB1:
        """Classic UCB1 (Auer et al., 2002) — best for stationary problems."""
        def __init__(self, K: int):
            self.K = K
            self.n = [0] * K        # pulls per arm
            self.s = [0.0] * K      # total reward per arm
            self.t = 0              # global steps


        def select(self) -> int:
            self.t += 1
            # Pull each arm once
            for a in range(self.K):
                if self.n[a] == 0:
                    return a
            # UCB index
            idx = []
            for a in range(self.K):
                mean = self.s[a] / self.n[a]
                bonus = math.sqrt(2.0 * math.log(self.t) / self.n[a])
                idx.append(mean + bonus)
            return max(range(self.K), key=lambda a: idx[a])


        def update(self, a: int, r: float) -> None:
            self.n[a] += 1
            self.s[a] += r




class DiscountedUCB:
        """Discounted-UCB (good for non-stationary): exponential forgetting via gamma."""
        def __init__(self, K: int, gamma: float = 0.90):
            self.K = K
            self.gamma = gamma
            self.n = [0.0] * K      # discounted pulls
            self.s = [0.0] * K      # discounted rewards
            self.t = 0


        def select(self) -> int:
            self.t += 1
            # Ensure initial exploration
            for a in range(self.K):
                if self.n[a] < 1.0:
                    return a
            total = sum(self.n)
            idx = []
            for a in range(self.K):
                mean = self.s[a] / self.n[a]
                bonus = math.sqrt(2.0 * math.log(total) / self.n[a])
                idx.append(mean + bonus)
            return max(range(self.K), key=lambda a: idx[a])


        def update(self, a: int, r: float) -> None:
            g = self.gamma
            for i in range(self.K):
                self.n[i] *= g
                self.s[i] *= g
            self.n[a] += 1.0
            self.s[a] += r




class ThompsonBernoulli:
        """Thompson Sampling for Bernoulli rewards with Beta posterior."""
        def __init__(self, K: int):
            self.K = K
            self.alpha = [1.0] * K
            self.beta =  [1.0] * K


        def select(self) -> int:
            samples = [random.betavariate(self.alpha[a], self.beta[a]) for a in range(self.K)]
            return max(range(self.K), key=lambda a: samples[a])


        def update(self, a: int, r: float) -> None:
            if r >= 0.5:
                self.alpha[a] += 1.0
            else:
                self.beta[a] += 1.0




# -----------------------------
# 3) Runner
# -----------------------------
def run(env: NonStationaryWSN, agent, T: int):
        """Returns cumulative reward list of length T for the (env, agent) pair."""
        cum = []
        total = 0.0
        for _ in range(T):
            a = agent.select()
            r = env.step(a)
            agent.update(a, r)
            total += r
            cum.append(total)
        return cum




# -----------------------------
# 4) Main — plot and save
# -----------------------------
if __name__ == "__main__":
        # Reproducibility
        random.seed(99)


        # Problem setup (same as the report)
        K = 3
        T = 500
        switch_t = 250
        probs_before = [0.80, 0.40, 0.20]
        probs_after  = [0.10, 0.40, 0.80]


        # We run each algorithm on an independent copy of the environment
        alg_builders = [
            ("UCB1",            lambda: UCB1(K)),
            ("Discounted-UCB",   lambda: DiscountedUCB(K, gamma=0.90)),
            ("Thompson-Beta",   lambda: ThompsonBernoulli(K)),
        ]


        curves = {}
        for name, ctor in alg_builders:
            env = NonStationaryWSN(probs_before, probs_after, switch_t)
            agent = ctor()
            curves[name] = run(env, agent, T)


        # --- Plot ---
        plt.figure(figsize=(7.5, 4.8))
        for name, data in curves.items():
            plt.plot(range(1, T + 1), data, label=name, linewidth=2)


        # Interference flip marker
        plt.axvline(x=switch_t, color="red", linestyle="--", label="Interference Flip")


        plt.xlabel("Transmission Count")
        plt.ylabel("Cumulative Reward (ACK successes)")
        plt.title("WSN Bandits under Non-Stationarity")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()


        # Save + show
        plt.savefig("img_plot.png", dpi=160)
        print("Saved figure -> img_plot.png")
        plt.show()
