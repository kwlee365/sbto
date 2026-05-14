import time
import copy
import numpy as np
import numpy.typing as npt
from tqdm import trange
from typing import Tuple, Optional, Any

from sbto.run.stats import OptimizationStats
from sbto.sim.sim_base import SimRolloutBase
from sbto.tasks.task_base import OCPBase
from sbto.solvers.solver_base import SamplingBasedSolver, SolverState

Array = npt.NDArray[np.float64]

def compute_cost(
    u_knots,
    sim: SimRolloutBase,
    task: OCPBase,
    **kwargs
    ):
    return task.cost(*sim.rollout(u_knots)[1:])


def _has_device_path(sim, task) -> bool:
    return hasattr(sim, "rollout_t_steps_device") and hasattr(task, "cost_jax")


def compute_cost_t_end(
    u_knots,
    sim: SimRolloutBase,
    task: OCPBase,
    t_end: int,
    **kwargs
    ):
    # Rescale cost based on the number of timesteps
    scale = sim.T / t_end
    if _has_device_path(sim, task):
        t_actual, x_d, u_d, obs_d = sim.rollout_t_steps_device(u_knots, t_end)
        costs_d = task.cost_jax(x_d, u_d, obs_d, t_end=t_actual)
        return scale * np.asarray(costs_d)
    return scale * task.cost(*sim.rollout_t_steps(u_knots, t_end)[1:])

def compute_cost_multiple_shooting(
    u_knots,
    sim: SimRolloutBase,
    task: OCPBase,
    **kwargs
    ):
    x_shooting = task.ref.x[sim.t_knots]
    return task.cost(*sim.rollout_multiple_shooting(u_knots, x_shooting)[1:])

def _optimize(
    sim: SimRolloutBase,
    task: OCPBase,
    solver: SamplingBasedSolver,
    compute_cost_fn: Any,
    init_state_solver: Optional[SolverState] = None,
    opt_stats: Optional[OptimizationStats] = None,
    ) -> Tuple[SolverState, Array, Array]:
    
    all_costs = []
    all_samples = []
    best_samples_it = []

    pbar = trange(solver.cfg.N_it, desc="Optimizing", leave=True)
    pbar_postfix = {}

    if not init_state_solver is None:
        solver.state = copy.deepcopy(init_state_solver)

    if opt_stats is None:
        opt_stats = OptimizationStats()

    start = time.time()
    for _ in pbar:
        opt_stats.add_iteration(sim.Nknots, sim.T)
        samples = solver.get_samples()
        all_samples.append(samples.copy())

        costs = compute_cost_fn(samples, sim, task)
        all_costs.append(costs)

        solver.update(samples, costs)
        opt_stats.end_iteration()

        best_samples_it.append(solver.state.best.copy())

        pbar_postfix["min_cost"] = solver.state.min_cost_all
        pbar_postfix["cost"] = solver.state.min_cost

        pbar.set_postfix(pbar_postfix)

    end = time.time()
    duration = end - start
    print(f"Solving time: {duration:.2f}s")

    all_samples_arr = np.asarray(all_samples)
    all_costs_arr = np.asarray(all_costs)
    last_solver_state = copy.deepcopy(solver.state)

    return last_solver_state, all_samples_arr, best_samples_it, all_costs_arr, opt_stats

def optimize_single_shooting(
    sim: SimRolloutBase,
    task: OCPBase,
    solver: SamplingBasedSolver,
    init_state_solver: Optional[SolverState] = None,
    opt_stats: Optional[OptimizationStats] = None,
    ) -> Tuple[SolverState, Array, Array]:
    return _optimize(
        sim,
        task,
        solver,
        compute_cost,
        init_state_solver,
        opt_stats,
    )

def optimize_mutiple_shooting(
    sim: SimRolloutBase,
    task: OCPBase,
    solver: SamplingBasedSolver,
    init_state_solver: Optional[SolverState] = None,
    opt_stats: Optional[OptimizationStats] = None,
    ) -> Tuple[SolverState, Array, Array]:
    return _optimize(
        sim,
        task,
        solver,
        compute_cost_multiple_shooting,
        init_state_solver,
        opt_stats,
    )

def optimize_incremental_opt(
    sim: SimRolloutBase,
    task: OCPBase,
    solver: SamplingBasedSolver,
    init_state_solver: Optional[SolverState] = None,
    opt_stats: Optional[OptimizationStats] = None,
    N_max_it_per_knots: int = 50,
    min_std_next: float = 1.e-2,
    min_std_final: float = 1.e-3,
    ) -> Tuple[SolverState, Array, Array]:

    all_costs = []
    all_samples = []
    best_samples_it = []

    if not init_state_solver is None:
        solver.state = copy.deepcopy(init_state_solver)

    if opt_stats is None:
        opt_stats = OptimizationStats()

    start = time.time()

    reset_best_knots_all = True
    pbar_knots = trange(2, sim.Nknots+1, desc="Optimizing", leave=True)
    pbar_postfix = {}
    pbar_it_postfix = {}
    nit_total = 0

    for N_knots_to_opt in pbar_knots:
        nit = 0
        increment_value = np.inf
        
        t_end = sim.t_knots[N_knots_to_opt-1]
        n_dim = N_knots_to_opt * sim.Nu
        solver.opt_first_dim(n_dim)

        # For last iterations
        all_knots_optimized = N_knots_to_opt == sim.Nknots
        if all_knots_optimized:
            min_std_next = min_std_final
            N_max_it_per_knots *= 2

        pbar_knots.set_description_str(f"Opt. first {N_knots_to_opt} knots")
        pbar_it = trange(N_max_it_per_knots, leave=False)

        while min_std_next < increment_value and nit < N_max_it_per_knots:

            opt_stats.add_iteration(N_knots_to_opt, 0)

            # Reset best samples when all knots are optimized
            # That ensures that the best samples are full dimensions
            if reset_best_knots_all and all_knots_optimized:
                solver.state.min_cost_all = np.inf
                reset_best_knots_all = False
            
            # Sample and compute cost
            samples = solver.get_samples()
            costs = compute_cost_t_end(samples, sim, task, t_end=t_end)
            
            # Save samples and cost
            all_costs.append(costs)
            best_samples_it.append(solver.state.best.copy())
            if all_knots_optimized: all_samples.append(samples.copy())

            # Update solver distribution
            solver.update(samples, costs)
            increment_value = solver.increment_value()

            pbar_postfix["min_cost"] = solver.state.min_cost_all
            pbar_postfix["it"] = nit_total
            pbar_it_postfix["std_max"] = increment_value
            pbar_it_postfix["cost"] = solver.state.min_cost
            pbar_knots.set_postfix(pbar_postfix)
            pbar_it.set_postfix(pbar_it_postfix)
            pbar_it.update(1)
            nit += 1
            nit_total += 1

            opt_stats.iterations[opt_stats.i].n_sim_steps_rollout = sim.nstep_allocated
            opt_stats.end_iteration()
        
        del pbar_it

    end = time.time()
    duration = end - start
    print(f"Solving time: {duration:.2f}s")

    all_costs_arr = np.asarray(all_costs)
    all_samples_arr = np.asarray(all_samples)
    last_solver_state = copy.deepcopy(solver.state)

    return last_solver_state, all_samples_arr, best_samples_it, all_costs_arr, opt_stats
