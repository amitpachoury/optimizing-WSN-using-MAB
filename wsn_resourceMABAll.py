
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# -----------------------------
# Utilities: Bandit algorithms
# -----------------------------
@dataclass
class BanditConfig:
	n_arms: int
	T: int
	c_ucb: float = 1.5
	gamma: float = 0.97  # discount for Discounted-UCB / Discounted updates
	lambda_var: float = 1.0  # variance weight for DVS

@dataclass
class RunResult:
	rewards: np.ndarray
	actions: np.ndarray
	energy: np.ndarray
	regrets: np.ndarray
	pdr_series: np.ndarray  # observed successes over time (0/1)


def run_ucb(probs: np.ndarray, energy_costs: np.ndarray, cfg: BanditConfig, reward_weight: float = 1.0, energy_weight: float = 1.0) -> RunResult:
	"""UCB1 with stationary assumption. Reward = reward_weight*success - energy_weight*energy_cost."""
	T, K = probs.shape
	mu = np.zeros(K)
	n = np.zeros(K)
	rewards = np.zeros(T)
	energy = np.zeros(T)
	actions = np.zeros(T, dtype=int)
	regrets = np.zeros(T)
	pdr_series = np.zeros(T)

	# warm-start: play each arm once
	for i in range(K):
    	t = i
    	p = probs[t, i]
    	x = np.random.rand() < p
    	r = reward_weight * x - energy_weight * energy_costs[i]
    	mu[i] = r
    	n[i] = 1
    	rewards[t] = r
    	energy[t] = energy_costs[i]
    	actions[t] = i
    	pdr_series[t] = x
    	optimal = np.max(reward_weight * probs[t, :] - energy_weight * energy_costs)
    	regrets[t] = optimal - r
	# continue
	for t in range(K, T):
    	ucb = mu + cfg.c_ucb * np.sqrt(np.log(max(t,1)) / np.maximum(n, 1e-9))
    	a = int(np.argmax(ucb))
    	x = np.random.rand() < probs[t, a]
    	r = reward_weight * x - energy_weight * energy_costs[a]
    	# update
    	n[a] += 1
    	mu[a] += (r - mu[a]) / n[a]
    	# log
    	rewards[t] = r
    	energy[t] = energy_costs[a]
    	actions[t] = a
    	pdr_series[t] = x
    	optimal = np.max(reward_weight * probs[t, :] - energy_weight * energy_costs)
    	regrets[t] = optimal - r
	return RunResult(rewards, actions, energy, regrets, pdr_series)


def run_discounted_ucb(probs: np.ndarray, energy_costs: np.ndarray, cfg: BanditConfig, reward_weight: float = 1.0, energy_weight: float = 1.0) -> RunResult:
	T, K = probs.shape
	sum_r = np.zeros(K)
	sum_w = np.zeros(K) + 1e-9
	rewards = np.zeros(T)
	energy = np.zeros(T)
	actions = np.zeros(T, dtype=int)
	regrets = np.zeros(T)
	pdr_series = np.zeros(T)

	# warm-start
	for i in range(K):
    	t = i
    	p = probs[t, i]
    	x = np.random.rand() < p
    	r = reward_weight * x - energy_weight * energy_costs[i]
    	sum_r[i] += r
    	sum_w[i] += 1
    	rewards[t] = r
    	energy[t] = energy_costs[i]
    	actions[t] = i
    	pdr_series[t] = x
    	optimal = np.max(reward_weight * probs[t, :] - energy_weight * energy_costs)
    	regrets[t] = optimal - r

	for t in range(K, T):
    	# discount
    	sum_r *= cfg.gamma
    	sum_w *= cfg.gamma
    	mu = sum_r / sum_w
    	n_eff = sum_w
    	ucb = mu + cfg.c_ucb * np.sqrt(np.log(max(t,1)) / np.maximum(n_eff, 1e-9))
    	a = int(np.argmax(ucb))
    	x = np.random.rand() < probs[t, a]
    	r = reward_weight * x - energy_weight * energy_costs[a]
    	sum_r[a] += r
    	sum_w[a] += 1
    	rewards[t] = r
    	energy[t] = energy_costs[a]
    	actions[t] = a
    	pdr_series[t] = x
    	optimal = np.max(reward_weight * probs[t, :] - energy_weight * energy_costs)
    	regrets[t] = optimal - r

	return RunResult(rewards, actions, energy, regrets, pdr_series)


def run_thompson(probs: np.ndarray, energy_costs: np.ndarray, cfg: BanditConfig, reward_weight: float = 1.0, energy_weight: float = 1.0, discount: Optional[float]=None) -> RunResult:
	T, K = probs.shape
	alpha = np.ones(K)
	beta = np.ones(K)
	rewards = np.zeros(T)
	energy = np.zeros(T)
	actions = np.zeros(T, dtype=int)
	regrets = np.zeros(T)
	pdr_series = np.zeros(T)

	for t in range(T):
    	# optional discounting for non-stationarity
    	if discount is not None and t > 0:
        	alpha = 1 + discount * (alpha - 1)
        	beta = 1 + discount * (beta - 1)
    	theta = np.random.beta(alpha, beta)
    	# subtract energy cost by translating to effective score
    	score = theta - energy_weight * energy_costs / max(reward_weight, 1e-9)
    	a = int(np.argmax(score))
    	x = np.random.rand() < probs[t, a]
    	r = reward_weight * x - energy_weight * energy_costs[a]
    	alpha[a] += x
    	beta[a] += (1 - x)
    	rewards[t] = r
    	energy[t] = energy_costs[a]
    	actions[t] = a
    	pdr_series[t] = x
    	optimal = np.max(reward_weight * probs[t, :] - energy_weight * energy_costs)
    	regrets[t] = optimal - r

	return RunResult(rewards, actions, energy, regrets, pdr_series)


def run_dvs(probs: np.ndarray, energy_costs: np.ndarray, cfg: BanditConfig, reward_weight: float = 1.0, energy_weight: float = 1.0, discount: Optional[float]=0.98) -> RunResult:
	"""Dynamic Variance Sampling: TS + variance bonus scaled by lambda_var.
	We maintain Beta posteriors with optional exponential forgetting.
	"""
	T, K = probs.shape
	alpha = np.ones(K)
	beta = np.ones(K)
	rewards = np.zeros(T)
	energy = np.zeros(T)
	actions = np.zeros(T, dtype=int)
	regrets = np.zeros(T)
	pdr_series = np.zeros(T)

	for t in range(T):
    	if discount is not None and t > 0:
        	alpha = 1 + discount * (alpha - 1)
        	beta = 1 + discount * (beta - 1)
    	mu = alpha / (alpha + beta)
    	var = (alpha * beta) / (((alpha + beta)**2) * (alpha + beta + 1))
    	theta = np.random.beta(alpha, beta)
    	score = theta + cfg.lambda_var * np.sqrt(var) - energy_weight * energy_costs / max(reward_weight, 1e-9)
    	a = int(np.argmax(score))
    	x = np.random.rand() < probs[t, a]
    	r = reward_weight * x - energy_weight * energy_costs[a]
    	alpha[a] += x
    	beta[a] += (1 - x)
    	rewards[t] = r
    	energy[t] = energy_costs[a]
    	actions[t] = a
    	pdr_series[t] = x
    	optimal = np.max(reward_weight * probs[t, :] - energy_weight * energy_costs)
    	regrets[t] = optimal - r

	return RunResult(rewards, actions, energy, regrets, pdr_series)

# -----------------------------
# Trace generators / loaders
# -----------------------------

def load_or_generate_channel_trace(T=1200, K=8, seed=7) -> np.ndarray:
	"""Generate a non-stationary per-channel success-probability trace.
	Calibrated to industrial 802.15.4/802.15.4g PDR ranges (approx 0.66–0.87) with occasional deep fades.
	"""
	rng = np.random.default_rng(seed)
	base = rng.uniform(0.66, 0.87, size=K)  # as reported ranges for industrial deployments
	probs = np.zeros((T, K))
    probs[0, :] = base
    # Random walks + occasional interference spikes on random channels
    for t in range(1, T):
        drift = rng.normal(0, 0.002, size=K)
        probs[t, :] = np.clip(probs[t-1, :] + drift, 0.05, 0.98)
        if t % 150 == 0:
            j = rng.integers(0, K)
            # temporary interference: drop PDR for ~30 steps
            dur = 30
            drop = rng.uniform(0.25, 0.45)
            for s in range(t, min(T, t+dur)):
                probs[s, j] = np.clip(probs[s, j] - drop, 0.05, 0.98)
    return probs


def load_or_generate_routing_trace(T=1200, K=4, seed=21) -> np.ndarray:
    """Per-neighbor PRR traces, inspired by FlockLab multi-node link measurements.
    Create distinct link qualities with sporadic fades and recoveries.
    """
    rng = np.random.default_rng(seed)
    base = np.array([0.9, 0.8, 0.75, 0.65])
    probs = np.zeros((T, K))
    probs[0, :] = base
    for t in range(1, T):
        drift = rng.normal(0, 0.002, size=K)
        probs[t, :] = np.clip(probs[t-1, :] + drift, 0.05, 0.99)
        # independent fades
        if rng.random() < 0.01:
            j = rng.integers(0, K)
            dur = rng.integers(20, 60)
            depth = rng.uniform(0.3, 0.5)
            for s in range(t, min(T, t+dur)):
                probs[s, j] = np.clip(probs[s, j] - depth, 0.05, 0.99)
    return probs


def load_or_generate_power_trace(T=1200, power_levels: List[int]=[-25,-15,-10,-5,0], seed=5) -> Tuple[np.ndarray, np.ndarray]:
    """Map underlying SNR trace to success probabilities per power level using a sigmoid.
    Returns (probs[T,K], energy_costs[K]). Energy scales with linear power (mW) proxy.
    """
    rng = np.random.default_rng(seed)
    # underlying SNR-like process with occasional dips
    snr = np.zeros(T)
    snr[0] = rng.uniform(-5, 5)
    for t in range(1, T):
        snr[t] = snr[t-1] + rng.normal(0, 0.4)
        snr[t] = np.clip(snr[t], -10, 15)
        if t % 200 == 0:
            snr[t:t+40] = snr[t:t+40] - rng.uniform(5, 10)
    # per power level: effective SNR = snr + power_offset
    K = len(power_levels)
	probs = np.zeros((T, K))
	# Map dBm to linear mW for energy; assume cost proportional to mW
	mW = 10 ** (np.array(power_levels)/10)
	energy_costs = (mW - mW.min()) / (mW.max() - mW.min() + 1e-9) * 1.0 + 0.05  # normalize to ~[0.05, 1.05]
	for i, p_dbm in enumerate(power_levels):
    	eff = snr + (p_dbm - (-25)) * 0.4  # stronger power shifts SNR
    	probs[:, i] = 1 / (1 + np.exp(-(eff - 2.0)))  # threshold ~2 dB
    	probs[:, i] = np.clip(probs[:, i], 0.05, 0.995)
	return probs, energy_costs


def load_or_generate_dutycycle_trace(T=1200, duty_cycles: List[float]=[0.05,0.1,0.2,0.4,0.6], seed=9) -> Tuple[np.ndarray, np.ndarray]:
	"""Effective success probability is constrained by both link PRR and radio availability (duty cycle).
	Energy scales roughly linearly with duty cycle.
	"""
	rng = np.random.default_rng(seed)
	# link PRR baseline dynamics
	link_prr = load_or_generate_channel_trace(T=T, K=1, seed=seed)[:,0]
	traffic_load = np.clip(rng.normal(0.3, 0.1, size=T), 0.05, 0.9)  # fraction of slots that carry data
	K = len(duty_cycles)
	probs = np.zeros((T, K))
	energy_costs = np.array(duty_cycles) * 1.0 + 0.02  # linear energy + small overhead
	for i, dc in enumerate(duty_cycles):
    	# availability constraint: if dc < load, backlog leads to drops; approximate effective factor
    	availability = np.minimum(1.0, dc / (traffic_load + 1e-9))
    	probs[:, i] = np.clip(link_prr * 0.9 * availability, 0.02, 0.98)
	return probs, energy_costs

# -----------------------------
# Plot helpers
# -----------------------------
def plot_results(title: str, results: Dict[str, RunResult], outfile: str):
	plt.figure(figsize=(12,8))
	T = len(next(iter(results.values())).rewards)
	x = np.arange(T)

	# (1) Cumulative PDR (success ratio)
	plt.subplot(2,2,1)
	for name, res in results.items():
    	cum_success = np.cumsum(res.pdr_series)
    	pdr = cum_success / (x+1)
    	plt.plot(x, pdr, label=name)
	plt.title('累積 PDR（成功率） / Cumulative PDR')
	plt.xlabel('時間ステップ / Time step')
	plt.ylabel('PDR')
	plt.legend()

	# (2) Cumulative energy
	plt.subplot(2,2,2)
	for name, res in results.items():
		plt.plot(x, np.cumsum(res.energy), label=name)
	plt.title('累積エネルギー / Cumulative energy')
	plt.xlabel('時間ステップ / Time step')
	plt.ylabel('Energy (normalized units)')
	plt.legend()

	# (3) Cumulative reward
	plt.subplot(2,2,3)
	for name, res in results.items():
		plt.plot(x, np.cumsum(res.rewards), label=name)
	plt.title('累積報酬 / Cumulative reward')
	plt.xlabel('時間ステップ / Time step')
	plt.ylabel('Reward')
	plt.legend()

	# (4) Cumulative regret
	plt.subplot(2,2,4)
	for name, res in results.items():
		plt.plot(x, np.cumsum(res.regrets), label=name)
	plt.title('累積後悔（レグレット） / Cumulative regret')
	plt.xlabel('時間ステップ / Time step')
	plt.ylabel('Regret')
	plt.legend()

	plt.suptitle(title)
	plt.tight_layout(rect=[0,0.03,1,0.95])
	plt.savefig(outfile, dpi=160)
	plt.close()

# -----------------------------
# Per-layer simulations
# -----------------------------
def simulate_channel_selection():
	T = 1200
	K = 8
	probs = load_or_generate_channel_trace(T=T, K=K, seed=42)
	# equal energy per TX per channel (normalized); small difference to reflect hardware diversity
	energy_costs = np.linspace(0.08, 0.12, K)
	cfg = BanditConfig(n_arms=K, T=T, c_ucb=1.2, gamma=0.97, lambda_var=0.8)

	res = {}
	res['UCB'] = run_ucb(probs, energy_costs, cfg)
	res['D-UCB'] = run_discounted_ucb(probs, energy_costs, cfg)
	res['TS'] = run_thompson(probs, energy_costs, cfg)
	res['DVS'] = run_dvs(probs, energy_costs, cfg)

	plot_results('Channel Selection (802.15.4/15.4g) — Bandit Comparison', res, 'mab_channel.png')
	return res


def simulate_routing_selection():
	T = 1200
	K = 4
	probs = load_or_generate_routing_trace(T=T, K=K, seed=21)
	# Energy per neighbor approximates ETX (1/PRR) times tx cost; we normalize using initial PRR
	base_prr = probs[0]
	etx = 1.0 / np.clip(base_prr, 0.05, 0.99)
	energy_costs = 0.05 + 0.15 * (etx - etx.min()) / (etx.max() - etx.min() + 1e-9)
	cfg = BanditConfig(n_arms=K, T=T, c_ucb=1.2, gamma=0.97, lambda_var=1.0)

	res = {}
	res['UCB'] = run_ucb(probs, energy_costs, cfg)
	res['D-UCB'] = run_discounted_ucb(probs, energy_costs, cfg)
	res['TS'] = run_thompson(probs, energy_costs, cfg)
	res['DVS'] = run_dvs(probs, energy_costs, cfg)

	plot_results('Routing (Next-hop) Selection — Bandit Comparison', res, 'mab_routing.png')
	return res


def simulate_power_control():
	T = 1200
	power_levels = [-25,-15,-10,-5,0]
	probs, energy_costs = load_or_generate_power_trace(T=T, power_levels=power_levels, seed=5)
	cfg = BanditConfig(n_arms=len(power_levels), T=T, c_ucb=1.2, gamma=0.97, lambda_var=0.8)

	res = {}
	res['UCB'] = run_ucb(probs, energy_costs, cfg)
	res['D-UCB'] = run_discounted_ucb(probs, energy_costs, cfg)
	# TS with small discount to adapt to SNR dynamics
	res['TS'] = run_thompson(probs, energy_costs, cfg, discount=0.99)
	res['DVS'] = run_dvs(probs, energy_costs, cfg, discount=0.99)

	plot_results('Transmit Power Control (dBm levels) — Bandit Comparison', res, 'mab_power.png')
	return res


def simulate_duty_cycle():
	T = 1200
	duty_cycles = [0.05, 0.1, 0.2, 0.4, 0.6]
	probs, energy_costs = load_or_generate_dutycycle_trace(T=T, duty_cycles=duty_cycles, seed=9)
	cfg = BanditConfig(n_arms=len(duty_cycles), T=T, c_ucb=1.2, gamma=0.97, lambda_var=1.2)

	res = {}
	res['UCB'] = run_ucb(probs, energy_costs, cfg)
	res['D-UCB'] = run_discounted_ucb(probs, energy_costs, cfg)
	res['TS'] = run_thompson(probs, energy_costs, cfg, discount=0.99)
	res['DVS'] = run_dvs(probs, energy_costs, cfg, discount=0.99)

	plot_results('Duty-Cycle Optimization — Bandit Comparison', res, 'mab_dutycycle.png')
	return res

# -----------------------------
# Execute all
# -----------------------------
if __name__ == '__main__':
	ch = simulate_channel_selection()
	rt = simulate_routing_selection()
	pw = simulate_power_control()
	dc = simulate_duty_cycle()
	# Save quick CSV summaries
	def summarize(res: Dict[str, RunResult]) -> pd.DataFrame:
    	rows = []
    	for name, r in res.items():
        	rows.append({
            	'Algorithm': name,
            	'Final PDR': np.sum(r.pdr_series)/len(r.pdr_series),
            	'Cumulative Energy': np.sum(r.energy),
            	'Cumulative Reward': np.sum(r.rewards),
            	'Cumulative Regret': np.sum(r.regrets)
        	})
    	return pd.DataFrame(rows).sort_values('Cumulative Reward', ascending=False)

	summarize(ch).to_csv('summary_channel.csv', index=False)
	summarize(rt).to_csv('summary_routing.csv', index=False)
	summarize(pw).to_csv('summary_power.csv', index=False)
	summarize(dc).to_csv('summary_dutycycle.csv', index=False)
	print('Generated plots: mab_channel.png, mab_routing.png, mab_power.png, mab_dutycycle.png')
	print('Summaries: summary_channel.csv, summary_routing.csv, summary_power.csv, summary_dutycycle.csv')
