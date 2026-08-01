"""
qpso_core.py
=======================================================================
Quantum-behaved Particle Swarm Optimization (QPSO) for the discrete
9-type / N-position loading-pattern space, PLUS a classical (non-quantum)
GA control used by W5 to isolate whether "quantum" is earning its keep,
mirroring the plain-GA-vs-QICA comparison you already ran for the NN
pipeline.

Design choice, explained: standard QPSO (Sun, Feng & Xu 2004) is defined
for CONTINUOUS position vectors, using an attractor point between each
particle's personal best and the swarm's global best (the "delta
potential well" model), then a random walk around that attractor via
   x(t+1) = p +/- (L/2) * ln(1/u),   u ~ Uniform(0,1)
where p is the attractor and L depends on the "mean best" (mbest) of the
swarm. To use this on a DISCRETE, 9-type categorical space, this
implementation keeps QICA's encoding trick (each position is a
probability vector over the 9 types, matching your existing
QuantumCountry design) but replaces ICA's empire/assimilation update
with the QPSO delta-potential-well update applied directly to the
probability vectors, followed by a softmax-style renormalization. This
is a genuinely different optimizer mechanism from QICA (swarm-based,
no empires/colonies/revolution), while staying in the same "quantum
probability over discrete types" representation family so results are
comparable to your existing QICA runs.

fitness_fn must be: fitness_fn(patterns: (N, n_pos) int array) -> (N,) array
LOWER is better throughout (this module minimizes PPF-style objectives,
matching your existing QICA convention).
"""

import numpy as np


class QuantumParticle:
    """One particle = a probability matrix (n_pos, n_types) over assembly
    types, exactly analogous to QICA's QuantumCountry.q."""

    def __init__(self, n_pos, n_types, rng, init_bias=None):
        self.n_pos = n_pos
        self.n_types = n_types
        if init_bias is not None:
            self.q = init_bias.copy()
        else:
            self.q = np.ones((n_pos, n_types), dtype=np.float64) / n_types
        self.pbest_q = self.q.copy()
        self.pbest_fitness = np.inf
        self.fitness = np.inf
        self.pattern = None  # last collapsed concrete pattern

    def collapse(self, temperature, rng):
        """Sample a concrete integer pattern from the probability matrix.
        temperature > 1 flattens the distribution (more exploration),
        < 1 sharpens it (more exploitation) -- same convention as your
        QICA collapse()."""
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
    """Turn a concrete pattern into a peaked probability matrix -- used
    to build the QPSO attractor point from pbest/gbest concrete
    patterns."""
    q = np.ones((n_pos, n_types), dtype=np.float64) * ((1 - sharpness) / (n_types - 1))
    for p in range(n_pos):
        q[p, int(pattern[p]) - 1] = sharpness
    return q / q.sum(axis=1, keepdims=True)


class QuantumPSO:
    """
    Quantum-behaved PSO over the discrete assembly-type search space.

    Parameters mirror your QICA signature where possible so results are
    directly comparable in the final verdict table (n_particles ~ pop,
    n_gens ~ gens, mc_samples for uncertainty-aware fitness_fn variants).
    """

    def __init__(self, n_pos, n_types=9, n_particles=80, n_gens=250,
                 seed=42, free_mask=None, beta_start=1.0, beta_end=0.4,
                 stagnation_limit=20):
        self.n_pos = n_pos
        self.n_types = n_types
        self.n_particles = n_particles
        self.n_gens = n_gens
        self.rng = np.random.default_rng(seed)
        # free_mask: boolean (n_pos,) -- positions NOT in the mask are
        # frozen to the most-likely training type (sensitivity or
        # entropy trust region, set by the calling workflow).
        self.free_mask = free_mask if free_mask is not None else np.ones(n_pos, dtype=bool)
        self.beta_start = beta_start   # contraction-expansion coefficient
        self.beta_end = beta_end
        self.stagnation_limit = stagnation_limit

    def run(self, fitness_fn, warm_start_patterns=None, verbose=True, tag=""):
        rng = self.rng
        particles = []
        for i in range(self.n_particles):
            if warm_start_patterns is not None and i < len(warm_start_patterns):
                bias = _to_prob(warm_start_patterns[i], self.n_pos, self.n_types)
            else:
                bias = None
            particles.append(QuantumParticle(self.n_pos, self.n_types, rng, bias))

        gbest_pattern, gbest_fitness = None, np.inf
        gbest_q = None
        history = {'gen': [], 'best_fitness': [], 'mean_fitness': [],
                   'entropy': [], 'stagnation': []}
        stagnation = 0

        for gen in range(self.n_gens):
            beta = self.beta_start - (self.beta_start - self.beta_end) * (gen / max(self.n_gens - 1, 1))
            temperature = 1.0 + 1.5 * (1.0 - gen / max(self.n_gens - 1, 1))

            patterns = np.stack([p.collapse(temperature, rng) for p in particles])
            fits = np.asarray(fitness_fn(patterns), dtype=np.float64)

            improved_this_gen = False
            for p, f in zip(particles, fits):
                p.fitness = f
                if f < p.pbest_fitness:
                    p.pbest_fitness = f
                    p.pbest_q = p.q.copy()
                if f < gbest_fitness:
                    gbest_fitness = f
                    gbest_pattern = p.pattern.copy()
                    gbest_q = p.q.copy()
                    improved_this_gen = True

            stagnation = 0 if improved_this_gen else stagnation + 1

            # ---- QPSO delta-potential-well update on probability space ----
            mbest = np.mean([p.pbest_q for p in particles], axis=0)  # mean-best attractor
            for p in particles:
                phi = rng.uniform(0, 1, size=(self.n_pos, self.n_types))
                attractor = phi * p.pbest_q + (1 - phi) * gbest_q
                u = rng.uniform(1e-6, 1.0, size=(self.n_pos, self.n_types))
                direction = rng.choice([-1.0, 1.0], size=(self.n_pos, self.n_types))
                new_q = attractor + direction * beta * np.abs(mbest - p.q) * np.log(1.0 / u)
                new_q = np.clip(new_q, 1e-4, None)
                new_q = new_q / new_q.sum(axis=1, keepdims=True)
                # respect trust region: frozen positions snap back to
                # gbest's (near-deterministic) distribution there
                new_q[~self.free_mask] = gbest_q[~self.free_mask]
                p.q = new_q

            # stagnation-triggered diversity injection (same fix you
            # already validated works for QICA's own stagnation problem)
            if stagnation >= self.stagnation_limit:
                n_inject = max(1, self.n_particles // 4)
                inject_idx = rng.choice(self.n_particles, size=n_inject, replace=False)
                for idx in inject_idx:
                    n_mut = rng.integers(4, max(5, self.n_pos // 3))
                    mut_pos = rng.choice(self.n_pos, size=n_mut, replace=False)
                    q = _to_prob(gbest_pattern, self.n_pos, self.n_types)
                    q[mut_pos] = 1.0 / self.n_types
                    particles[idx].q = q
                stagnation = 0

            mean_ent = float(np.mean([p.entropy() for p in particles]))
            history['gen'].append(gen)
            history['best_fitness'].append(gbest_fitness)
            history['mean_fitness'].append(float(np.mean(fits)))
            history['entropy'].append(mean_ent)
            history['stagnation'].append(stagnation)

            if verbose and (gen % 25 == 0 or gen == self.n_gens - 1):
                print(f"  [{tag}] gen {gen:4d}/{self.n_gens} | best={gbest_fitness:.4f} "
                      f"mean={np.mean(fits):.4f} H={mean_ent:.3f} stag={stagnation}")

        return {'best_pattern': gbest_pattern, 'best_fitness': gbest_fitness,
                'history': history}


class ClassicalGA:
    """
    Plain elitist GA, NO quantum probability encoding at all -- integer
    tournament selection + uniform crossover + random-resample mutation.
    This is W5: the floor control that tells you whether QPSO's quantum
    representation is doing anything QICA's quantum representation
    wasn't already shown to do (your earlier plain-GA-vs-QICA result).
    """

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

    def run(self, fitness_fn, warm_start_patterns=None, verbose=True, tag=""):
        rng = self.rng
        if warm_start_patterns is not None and len(warm_start_patterns) > 0:
            base = warm_start_patterns[rng.integers(0, len(warm_start_patterns),
                                                      size=self.pop_size) %
                                        len(warm_start_patterns)]
            pop = base.copy()
        else:
            pop = rng.integers(1, self.n_types + 1, size=(self.pop_size, self.n_pos))

        best_pattern, best_fitness = None, np.inf
        history = {'gen': [], 'best_fitness': [], 'mean_fitness': []}

        for gen in range(self.n_gens):
            fits = np.asarray(fitness_fn(pop), dtype=np.float64)
            gen_best_idx = np.argmin(fits)
            if fits[gen_best_idx] < best_fitness:
                best_fitness = fits[gen_best_idx]
                best_pattern = pop[gen_best_idx].copy()

            history['gen'].append(gen)
            history['best_fitness'].append(best_fitness)
            history['mean_fitness'].append(float(np.mean(fits)))

            # tournament selection
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
                print(f"  [{tag}] gen {gen:4d}/{self.n_gens} | best={best_fitness:.4f} "
                      f"mean={np.mean(fits):.4f}")

        return {'best_pattern': best_pattern, 'best_fitness': best_fitness,
                'history': history}
