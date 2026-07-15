# Complexity analysis

- One closed-PDA group evaluation is O(I_pda * (A*D + E_a*I_cw)), where A is active emitter count, D is mean path depth, E_a is active pipes, I_pda is pressure-flow iterations, and I_cw is Colebrook iterations.
- A complete plan with K groups is O(K * I_pda * (A*D + E_a*I_cw)).
- NSGA-II non-dominated sorting is O(M*N^2) per generation for M objectives and population N; the hydraulic evaluator is usually the dominant measured cost.
- L1 pipe caching removes repeated Colebrook work, L2 group caching removes repeated converged PDA solves for identical group/head states, and L3 plan caching removes repeated complete-plan evaluations in the same immutable context.
- Caches reduce expected constants and repeated work; they do not improve worst-case asymptotic complexity.
