# ga.py — Genetic Algorithm optimizer

import numpy as np
from backtest import decode, decode_t, backtest, backtest_with_signal

# Gene bounds — without and with transformer weight
BOUNDS = [
    (10,  40),   # RSI buy threshold
    (60,  90),   # RSI sell threshold
    (10,  50),   # MA short period
    (50, 200),   # MA long period
    (0.01, 0.10), # Stop loss
    (0.05, 0.30)  # Take profit
]

BOUNDS_T = BOUNDS + [(0.0, 1.0)]  # + transformer weight gene


def _fitness(chrom, train_df, use_transformer):
    """Evaluate one chromosome — returns Sharpe Ratio"""
    params = decode_t(chrom) if use_transformer else decode(chrom)
    sma_s  = f'SMA_{params[2]}'
    sma_l  = f'SMA_{params[3]}'
    if sma_s not in train_df.columns or sma_l not in train_df.columns:
        return -999
    fn = backtest_with_signal if use_transformer else backtest
    return fn(train_df, *params)['sharpe']


def run_ga(train_df, pop_size=50, generations=40,
           use_transformer=True):
    """
    Run Genetic Algorithm
    Returns: (best_chromosome, history_of_best_sharpe)
    """
    bounds = BOUNDS_T if use_transformer else BOUNDS

    def random_chrom():
        return [np.random.uniform(lo, hi) for lo, hi in bounds]

    def tournament_select(pop, scores, k=5):
        """Tournament selection — pick k random, return best"""
        new_pop = []
        for _ in range(len(pop)):
            idx  = np.random.choice(len(pop), k, replace=False)
            best = idx[np.argmax([scores[i] for i in idx])]
            new_pop.append(pop[best][:])
        return new_pop

    def crossover(p1, p2, rate=0.8):
        """Single-point crossover"""
        if np.random.rand() < rate:
            pt = np.random.randint(1, len(p1))
            return p1[:pt] + p2[pt:], p2[:pt] + p1[pt:]
        return p1[:], p2[:]

    def mutate(ch, rate=0.2):
        """Gaussian mutation"""
        ch = ch[:]
        for i in range(len(ch)):
            if np.random.rand() < rate:
                lo, hi = bounds[i]
                ch[i]  = np.clip(
                    ch[i] + np.random.normal(0, (hi - lo) * 0.1),
                    lo, hi)
        return ch

    # Initialize population
    population = [random_chrom() for _ in range(pop_size)]
    best_score = -np.inf
    best_chrom = None
    history    = []

    for gen in range(generations):
        # Evaluate all chromosomes
        scores = [_fitness(c, train_df, use_transformer)
                  for c in population]
        idx    = np.argmax(scores)

        # Track best
        if scores[idx] > best_score:
            best_score = scores[idx]
            best_chrom = population[idx][:]

        history.append(round(best_score, 4))
        print(f"GA Gen {gen+1}/{generations} | Best Sharpe: {best_score:.4f}")

        # Create next generation
        selected = tournament_select(population, scores)
        next_gen = []
        for i in range(0, pop_size, 2):
            c1, c2 = crossover(selected[i], selected[i + 1])
            next_gen += [mutate(c1), mutate(c2)]
        population = next_gen[:pop_size]

    return best_chrom, history