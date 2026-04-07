# pso.py — Particle Swarm Optimization

import numpy as np
from backtest import decode, decode_t, backtest, backtest_with_signal

BOUNDS = [
    (10,  40),
    (60,  90),
    (10,  50),
    (50, 200),
    (0.01, 0.10),
    (0.05, 0.30)
]

BOUNDS_T = BOUNDS + [(0.0, 1.0)]


def _fitness(chrom, train_df, use_transformer):
    """Evaluate one particle position — returns Sharpe Ratio"""
    params = decode_t(chrom) if use_transformer else decode(chrom)
    sma_s  = f'SMA_{params[2]}'
    sma_l  = f'SMA_{params[3]}'
    if sma_s not in train_df.columns or sma_l not in train_df.columns:
        return -999
    fn = backtest_with_signal if use_transformer else backtest
    return fn(train_df, *params)['sharpe']


def run_pso(train_df, n_particles=50, n_iters=40,
            use_transformer=True):
    """
    Run Particle Swarm Optimization
    Returns: (best_position, history_of_best_sharpe)
    """
    bounds  = BOUNDS_T if use_transformer else BOUNDS
    W       = 0.7    # inertia weight
    C1      = 1.5    # cognitive (personal best)
    C2      = 1.5    # social (global best)
    W_DECAY = 0.99   # slowly reduce inertia

    # Initialize particles
    positions  = [[np.random.uniform(lo, hi) for lo, hi in bounds]
                  for _ in range(n_particles)]
    velocities = [[np.random.uniform(-(hi - lo) * 0.2, (hi - lo) * 0.2)
                   for lo, hi in bounds]
                  for _ in range(n_particles)]

    # Personal bests
    pbest_pos    = [p[:] for p in positions]
    pbest_scores = [_fitness(p, train_df, use_transformer)
                    for p in positions]

    # Global best
    best_idx    = np.argmax(pbest_scores)
    gbest_pos   = pbest_pos[best_idx][:]
    gbest_score = pbest_scores[best_idx]
    history     = []
    w           = W

    for it in range(n_iters):
        for i in range(n_particles):
            r1 = np.random.rand(len(bounds))
            r2 = np.random.rand(len(bounds))

            # Update velocity
            for d in range(len(bounds)):
                velocities[i][d] = (
                    w  * velocities[i][d]
                    + C1 * r1[d] * (pbest_pos[i][d] - positions[i][d])
                    + C2 * r2[d] * (gbest_pos[d]    - positions[i][d])
                )
                # Clip velocity to ±30% of range
                limit            = (bounds[d][1] - bounds[d][0]) * 0.3
                velocities[i][d] = np.clip(
                    velocities[i][d], -limit, limit)

            # Update position
            for d in range(len(bounds)):
                positions[i][d] = np.clip(
                    positions[i][d] + velocities[i][d],
                    bounds[d][0], bounds[d][1])

            # Evaluate new position
            score = _fitness(positions[i], train_df, use_transformer)

            if score > pbest_scores[i]:
                pbest_scores[i] = score
                pbest_pos[i]    = positions[i][:]

            if score > gbest_score:
                gbest_score = score
                gbest_pos   = positions[i][:]

        # Decay inertia
        w *= W_DECAY
        history.append(round(gbest_score, 4))
        print(f"PSO Iter {it+1}/{n_iters} | Best Sharpe: {gbest_score:.4f}")

    return gbest_pos, history