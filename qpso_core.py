"""
qpso_core.py  (FIXED)
=======================================================================
Two critical fixes vs your previous version:

FIX 1 — RAW PPF vs FITNESS were being conflated.
  fitness_fn(patterns) (from fitness_utils.make_fitness_fn) returns a
  PENALIZED score: raw_mean + W_UNC*sigma + floor_penalty. This is the
  right thing to select ON (it keeps the optimizer honest about
  uncertainty/OOD), but it is NOT a PPF value. Your previous code did
  `out['best_fitness']` and treated it as PPF directly -- that's why W5
  showed ppf=1.000+/-0.000 (a pure floor-penalty artifact, not a real
  design) and W1 showed ppf~1.10 (fitness units, already penalized).
  FIX: every optimizer now ALSO threads through a `diag_fn` (from
  fitness_utils.make_fitness_fn's get_last_diagnostics) so it can report
  the REAL raw oracle mean/sigma at whatever pattern it selects as best,
  separately from the fitness score used for selection. Both are
  returned in `out`.

FIX 2 — QPSO-DM: differential mutation added to QuantumPSO.
  Per your mentor's QPSO-DM reference (Quantum PSO + Differential
  Mutation, used on WWER-1000/Bushehr loading-pattern optimization),
  standard QPSO's only exploration mechanism is the delta-potential-well
  random walk, which stagnates in a discrete/multimodal space like this
  one. QPSO-DM periodically injects a DE-style mutant vector at the
  probability-distribution level:
      q_mutant = q_r1 + F * (q_r2 - q_r3)     (three distinct random particles)
  clipped to a valid simplex, then greedily replaces the target particle
  only if the mutant's collapsed pattern improves personal best. This is
  a genuinely different exploration mechanism than the potential well
  (structured recombination of THREE particles' distributions, not just
  drift toward two), applied every generation to a fraction of the swarm
  (DM_RATE), plus more aggressively during stagnation bursts.

fitness_fn must be: fitness_fn(patterns: (N, n_pos) int array) -> (N,) array,
LOWER is better (this module minimizes).
"""

import numpy as np


class QuantumParticle:
    def __init__(self, n_pos, n_types, rng, init_bias=None):
        self.n_pos = n_pos
        self.n_types = n_types
        if init_bias is not None:
            self.q = init_bias.copy()
        else:
            self.q = np.ones((n_pos, n_types), dtype=np.float64) / n_types
        self.pbest_q = self.q.copy()
        self.pbest_fitness = np.inf
        self.pbest_raw = np.inf
        self.pbest_sigma = np.nan
        self.fitness = np.inf
        self.pattern = None

    def collapse(self, temperature, rng):
        probs = self.q ** (1.0 / max(temperature, 1e-3))
        probs = probs / probs.sum(axis=1, keepdims=True)
        pattern = np.array([
            rng.choice(self.n_types, p=probs[p]) + 1 for p in range(self.n_pos)
        ], dtype=np.int32)
        self.pattern = pattern
        return pattern

    def entropy(self):
        p = np.clip(self.q, 1e-12, 1.0)
        return float(-np.mean(np.sum(p * np.log(p), axis=1)))


def _to_prob(pattern, n_pos, n_types, sharpness=0.84):
    q = np.ones((n_pos, n_types), dtype=np.float64) * ((1 - sharpness) / (n_types - 1))
    for p in range(n_pos):
        q[p, int(pattern[p]) - 1] = sharpness
    return q / q.sum(axis=1, keepdims=True)


def _project_simplex(q, eps=1e-4):
    """Clip a (n_pos, n_types) array back to a valid probability simplex
    per row -- needed after DE-style arithmetic (q_r1 + F*(q_r2-q_r3))
    can produce negative or >1 entries."""
    q = np.clip(q, eps, None)
    return q / q.sum(axis=1, keepdims=True)


class QuantumPSO:
    """
    Quantum-behaved PSO (Sun-Feng-Xu delta-potential-well update) over a
    discrete, categorical, probability-vector-encoded search space, now
    with QPSO-DM differential mutation for stronger, more legitimate
    exploration in this discrete/multimodal setting.
    """

    def __init__(self, n_pos, n_types=9, n_particles=80, n_gens=250,
                 seed=42, free_mask=None, beta_start=1.0, beta_end=0.4,
                 stagnation_limit=20, dm_rate=0.15, dm_F=0.6, dm_CR=0.5):
        self.n_pos = n_pos
        self.n_types = n_types
        self.n_particles = n_particles
        self.n_gens = n_gens
        self.rng = np.random.default_rng(seed)
        self.free_mask = free_mask if free_mask is not None else np.ones(n_pos, dtype=bool)
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.stagnation_limit = stagnation_limit
        # QPSO-DM hyperparameters
        self.dm_rate = dm_rate   # fraction of swarm subjected to DE mutation per gen
        self.dm_F = dm_F         # DE differential weight
        self.dm_CR = dm_CR       # DE crossover rate (per-position, in q-space rows)

    def run(self, fitness_fn, warm_start_patterns=None, verbose=True, tag="",
            diag_fn=None):
        """
        diag_fn: optional callable (no args) -> dict with 'last_mean' and
        'last_sigma' (N,) arrays ALIGNED to the most recent fitness_fn(...)
        call's batch order. Pass fitness_utils.make_fitness_fn's
        get_last_diagnostics here. If None, best_raw/best_sigma fall back
        to NaN (fitness is still reported correctly either way).
        """
        rng = self.rng
        particles = []
        for i in range(self.n_particles):
            bias = _to_prob(warm_start_patterns[i], self.n_pos, self.n_types) \
                if warm_start_patterns is not None and i < len(warm_start_patterns) else None
            particles.append(QuantumParticle(self.n_pos, self.n_types, rng, bias))

        gbest_pattern, gbest_fitness = None, np.inf
        gbest_raw, gbest_sigma = np.inf, np.nan
        gbest_q = None
        history = {'gen': [], 'best_fitness': [], 'best_raw_ppf': [], 'mean_fitness': [],
                   'entropy': [], 'stagnation': []}
        stagnation = 0

        def _eval(patterns):
            fits = np.asarray(fitness_fn(patterns), dtype=np.float64)
            if diag_fn is not None:
                diag = diag_fn()
                raw = np.asarray(diag.get('last_mean'), dtype=np.float64) \
                    if diag.get('last_mean') is not None else np.full(len(patterns), np.nan)
                sig = np.asarray(diag.get('last_sigma'), dtype=np.float64) \
                    if diag.get('last_sigma') is not None else np.full(len(patterns), np.nan)
            else:
                raw = np.full(len(patterns), np.nan)
                sig = np.full(len(patterns), np.nan)
            return fits, raw, sig

        for gen in range(self.n_gens):
            beta = self.beta_start - (self.beta_start - self.beta_end) * (gen / max(self.n_gens - 1, 1))
            temperature = 1.0 + 1.5 * (1.0 - gen / max(self.n_gens - 1, 1))

            patterns = np.stack([p.collapse(temperature, rng) for p in particles])
            fits, raws, sigs = _eval(patterns)

            improved_this_gen = False
            for p, f, raw, sig in zip(particles, fits, raws, sigs):
                p.fitness = f
                if f < p.pbest_fitness:
                    p.pbest_fitness = f
                    p.pbest_q = p.q.copy()
                    p.pbest_raw = raw
                    p.pbest_sigma = sig
                if f < gbest_fitness:
                    gbest_fitness = f
                    gbest_pattern = p.pattern.copy()
                    gbest_q = p.q.copy()
                    gbest_raw = raw
                    gbest_sigma = sig
                    improved_this_gen = True
            stagnation = 0 if improved_this_gen else stagnation + 1

            # ---- QPSO delta-potential-well update ----
            mbest = np.mean([p.pbest_q for p in particles], axis=0)
            for p in particles:
                phi = rng.uniform(0, 1, size=(self.n_pos, self.n_types))
                attractor = phi * p.pbest_q + (1 - phi) * gbest_q
                u = rng.uniform(1e-6, 1.0, size=(self.n_pos, self.n_types))
                direction = rng.choice([-1.0, 1.0], size=(self.n_pos, self.n_types))
                new_q = attractor + direction * beta * np.abs(mbest - p.q) * np.log(1.0 / u)
                new_q = _project_simplex(new_q)
                new_q[~self.free_mask] = gbest_q[~self.free_mask]
                p.q = new_q

            # ---- QPSO-DM: differential mutation on a subset of the swarm ----
            n_dm = max(1, int(round(self.dm_rate * self.n_particles)))
            dm_targets = rng.choice(self.n_particles, size=n_dm, replace=False)
            dm_patterns = []
            dm_target_idx = []
            for ti in dm_targets:
                r1, r2, r3 = rng.choice(
                    [i for i in range(self.n_particles) if i != ti], size=3, replace=False)
                mutant_q = particles[r1].pbest_q + self.dm_F * (
                    particles[r2].pbest_q - particles[r3].pbest_q)
                mutant_q = _project_simplex(mutant_q)
                # binomial crossover with the target's current distribution
                cross_mask = rng.random((self.n_pos, self.n_types)) < self.dm_CR
                trial_q = np.where(cross_mask, mutant_q, particles[ti].q)
                trial_q = _project_simplex(trial_q)
                trial_q[~self.free_mask] = particles[ti].q[~self.free_mask]
                trial_pattern = particles[ti].collapse(temperature, rng) \
                    if False else None  # placeholder, replaced below
                # collapse the TRIAL distribution (not current q) to evaluate it
                probs = trial_q ** (1.0 / max(temperature, 1e-3))
                probs = probs / probs.sum(axis=1, keepdims=True)
                trial_pattern = np.array([
                    rng.choice(self.n_types, p=probs[p]) + 1 for p in range(self.n_pos)
                ], dtype=np.int32)
                dm_patterns.append(trial_pattern)
                dm_target_idx.append((ti, trial_q))

            if dm_patterns:
                dm_fits, dm_raws, dm_sigs = _eval(np.stack(dm_patterns))
                for (ti, trial_q), f, raw, sig, pat in zip(dm_target_idx, dm_fits, dm_raws, dm_sigs, dm_patterns):
                    # greedy DE selection: only accept if it improves this
                    # particle's own personal best (never overwrites pbest
                    # blindly -- this is what keeps DM from wrecking a
                    # converged, uncertainty-safe particle)
                    if f < particles[ti].pbest_fitness:
                        particles[ti].q = trial_q
                        particles[ti].pattern = pat
                        particles[ti].fitness = f
                        particles[ti].pbest_fitness = f
                        particles[ti].pbest_q = trial_q.copy()
                        particles[ti].pbest_raw = raw
                        particles[ti].pbest_sigma = sig
                        if f < gbest_fitness:
                            gbest_fitness = f
                            gbest_pattern = pat.copy()
                            gbest_q = trial_q.copy()
                            gbest_raw = raw
                            gbest_sigma = sig
                            stagnation = 0

            # ---- stagnation-triggered stronger DM burst (replaces old
            # naive "reset to uniform" injection with a more targeted DE
            # burst around gbest, still followed by a diversity floor) ----
            if stagnation >= self.stagnation_limit:
                n_inject = max(1, self.n_particles // 4)
                inject_idx = rng.choice(self.n_particles, size=n_inject, replace=False)
                for idx in inject_idx:
                    others = [i for i in range(self.n_particles) if i != idx]
                    r1, r2, r3 = rng.choice(others, size=3, replace=False)
                    mutant_q = particles[r1].pbest_q + (self.dm_F * 1.5) * (
                        particles[r2].pbest_q - particles[r3].pbest_q)
                    mutant_q = _project_simplex(mutant_q)
                    mutant_q[~self.free_mask] = gbest_q[~self.free_mask]
                    particles[idx].q = mutant_q
                stagnation = 0

            mean_ent = float(np.mean([p.entropy() for p in particles]))
            history['gen'].append(gen)
            history['best_fitness'].append(gbest_fitness)
            history['best_raw_ppf'].append(gbest_raw)
            history['mean_fitness'].append(float(np.mean(fits)))
            history['entropy'].append(mean_ent)
            history['stagnation'].append(stagnation)

            if verbose and (gen % 25 == 0 or gen == self.n_gens - 1):
                raw_str = f" raw_ppf={gbest_raw:.4f}" if np.isfinite(gbest_raw) else ""
                print(f"  [{tag}] gen {gen:4d}/{self.n_gens} | fitness={gbest_fitness:.4f}"
                      f"{raw_str} mean_fit={np.mean(fits):.4f} H={mean_ent:.3f} stag={stagnation}")

        return {'best_pattern': gbest_pattern, 'best_fitness': gbest_fitness,
                'best_raw_ppf': gbest_raw, 'best_sigma': gbest_sigma,
                'history': history}


class ClassicalGA:
    """Plain elitist GA -- no quantum encoding, the floor control. Same
    raw-PPF-tracking fix as QuantumPSO."""

    def __init__(self, n_pos, n_types=9, pop_size=80, n_gens=250, seed=42,
                 mutation_rate=0.03, tournament_size=3, free_mask=None):
        self.n_pos = n_pos
        self.n_types = n_types
        self.pop_size = pop_size
        self.n_gens = n_gens
        self.rng = np.random.default_rng(seed)
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.free_mask = free_mask if free_mask is not None else np.ones(n_pos, dtype=bool)

    def run(self, fitness_fn, warm_start_patterns=None, verbose=True, tag="",
            diag_fn=None):
        rng = self.rng
        if warm_start_patterns is not None and len(warm_start_patterns) > 0:
            base = warm_start_patterns[rng.integers(0, len(warm_start_patterns),
                                                      size=self.pop_size) %
                                        len(warm_start_patterns)]
            pop = base.copy()
        else:
            pop = rng.integers(1, self.n_types + 1, size=(self.pop_size, self.n_pos))

        best_pattern, best_fitness = None, np.inf
        best_raw, best_sigma = np.inf, np.nan
        history = {'gen': [], 'best_fitness': [], 'best_raw_ppf': [], 'mean_fitness': []}

        def _eval(p):
            fits = np.asarray(fitness_fn(p), dtype=np.float64)
            if diag_fn is not None:
                diag = diag_fn()
                raw = np.asarray(diag.get('last_mean'), dtype=np.float64) \
                    if diag.get('last_mean') is not None else np.full(len(p), np.nan)
                sig = np.asarray(diag.get('last_sigma'), dtype=np.float64) \
                    if diag.get('last_sigma') is not None else np.full(len(p), np.nan)
            else:
                raw = np.full(len(p), np.nan)
                sig = np.full(len(p), np.nan)
            return fits, raw, sig

        for gen in range(self.n_gens):
            fits, raws, sigs = _eval(pop)
            gi = np.argmin(fits)
            if fits[gi] < best_fitness:
                best_fitness = fits[gi]
                best_pattern = pop[gi].copy()
                best_raw = raws[gi]
                best_sigma = sigs[gi]

            history['gen'].append(gen)
            history['best_fitness'].append(best_fitness)
            history['best_raw_ppf'].append(best_raw)
            history['mean_fitness'].append(float(np.mean(fits)))

            new_pop = np.empty_like(pop)
            new_pop[0] = best_pattern  # elitism
            for i in range(1, self.pop_size):
                cand = rng.choice(self.pop_size, size=self.tournament_size, replace=False)
                parent1 = pop[cand[np.argmin(fits[cand])]]
                cand2 = rng.choice(self.pop_size, size=self.tournament_size, replace=False)
                parent2 = pop[cand2[np.argmin(fits[cand2])]]
                mask = rng.random(self.n_pos) < 0.5
                child = np.where(mask, parent1, parent2)
                mut_mask = (rng.random(self.n_pos) < self.mutation_rate) & self.free_mask
                child = child.copy()
                child[mut_mask] = rng.integers(1, self.n_types + 1, size=mut_mask.sum())
                new_pop[i] = child
            pop = new_pop

            if verbose and (gen % 25 == 0 or gen == self.n_gens - 1):
                raw_str = f" raw_ppf={best_raw:.4f}" if np.isfinite(best_raw) else ""
                print(f"  [{tag}] gen {gen:4d}/{self.n_gens} | fitness={best_fitness:.4f}"
                      f"{raw_str} mean_fit={np.mean(fits):.4f}")

        return {'best_pattern': best_pattern, 'best_fitness': best_fitness,
                'best_raw_ppf': best_raw, 'best_sigma': best_sigma,
                'history': history}
