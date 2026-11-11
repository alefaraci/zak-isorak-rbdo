"""Text summaries of the state of the search, shown in the live display at every generation.

There is one function per analysis: each one formats the entries of the status dictionary of the
searcher that are relevant to that analysis.
"""

from evotorch.algorithms.searchalgorithm import LazyStatusDict


def log_ddo(status: LazyStatusDict) -> str:
    """Format the status of a deterministic design optimization (DDO) generation.

    Args:
        status: Status dictionary of the searcher.

    Returns:
        One line per entry: median and mean fitness, best design and its fitness, standard
        deviation of the fitness of the population, and cost.
    """
    max_key_length = len("solution fitness")
    median = f"{'median_eval'.rjust(max_key_length)} : {status['median_eval']}\n"
    mean = f"{'mean_eval'.rjust(max_key_length)} : {status['mean_eval']}\n"
    solution = f"{'solution'.rjust(max_key_length)} : {[float(k) for k in status['pop_best'].values.cpu()]}\n"
    solution_opt_eval = (
        f"{'solution fitness'.rjust(max_key_length)} : {float(status['pop_best'].evals.cpu())}\n"
    )
    stdev_pop = f"{'pop_evals_stdev'.rjust(max_key_length)} : {float(status['pop_evals_stdev'])}\n"
    cost = f"{'cost'.rjust(max_key_length)} : {status['cost']}\n"
    return median + mean + solution + solution_opt_eval + stdev_pop + cost


def log_rbdo(status: LazyStatusDict) -> str:
    """Format the status of an RBDO generation.

    Args:
        status: Status dictionary of the searcher.

    Returns:
        The lines of `log_ddo`, plus the feasibility of the best design.
    """
    max_key_length = len("solution fitness")
    median = f"{'median_eval'.rjust(max_key_length)} : {status['median_eval']}\n"
    mean = f"{'mean_eval'.rjust(max_key_length)} : {status['mean_eval']}\n"
    solution = f"{'solution'.rjust(max_key_length)} : {[float(k) for k in status['pop_best'].values.cpu()]}\n"
    solution_opt_eval = (
        f"{'solution fitness'.rjust(max_key_length)} : {float(status['pop_best'].evals.cpu())}\n"
    )
    feasibility = f"{'feasibility'.rjust(max_key_length)} : {bool(status['feasibility'])}\n"
    stdev_pop = f"{'pop_evals_stdev'.rjust(max_key_length)} : {float(status['pop_evals_stdev'])}\n"
    cost = f"{'cost'.rjust(max_key_length)} : {status['cost']}\n"
    return median + mean + solution + solution_opt_eval + feasibility + stdev_pop + cost


def log_zakrbdo(status: LazyStatusDict, doe_log: list[str]) -> str:
    """Format the status of a ZAK-RBDO generation.

    Args:
        status: Status dictionary of the searcher.
        doe_log: One string per design of experiments, with its number of limit-state calls.

    Returns:
        The lines of `log_rbdo`, plus the maximum relative error "eps_max" of the estimated
        failure probability and the number of limit-state calls.
    """
    max_key_length = len("solution fitness")
    median = f"{'median_eval'.rjust(max_key_length)} : {status['median_eval']}\n"
    mean = f"{'mean_eval'.rjust(max_key_length)} : {status['mean_eval']}\n"
    solution = f"{'solution'.rjust(max_key_length)} : {[float(k) for k in status['pop_best'].values.cpu()]}\n"
    solution_opt_eval = (
        f"{'solution fitness'.rjust(max_key_length)} : {float(status['pop_best'].evals.cpu())}\n"
    )
    feasibility = f"{'feasibility'.rjust(max_key_length)} : {bool(status['feasibility'])}\n"
    stdev_pop = f"{'pop_evals_stdev'.rjust(max_key_length)} : {float(status['pop_evals_stdev'])}\n"
    cost = f"{'cost'.rjust(max_key_length)} : {status['cost']}\n"
    eps_max = f"{'eps_max'.rjust(max_key_length)} : {status['eps_max']}\n"
    DoE = f"{'enrichments'.rjust(max_key_length)} : {''.join(doe_log)}\n"
    return (
        median
        + mean
        + solution
        + solution_opt_eval
        + feasibility
        + stdev_pop
        + cost
        + eps_max
        + DoE
    )


def log_isorakrbdo(status: LazyStatusDict, doe_log: list[str]) -> str:
    """Format the status of an ISORAK-RBDO generation.

    Args:
        status: Status dictionary of the searcher.
        doe_log: One string per design of experiments, with its number of limit-state calls.

    Returns:
        The lines of `log_zakrbdo`, plus the level alpha of the isocost region and the two flags
        "rho_alpha" (the region is well explored) and "admissibility" (the region holds a point
        that satisfies all the limit states).
    """
    max_key_length = len("solution fitness")
    median = f"{'median_eval'.rjust(max_key_length)} : {status['median_eval']}\n"
    mean = f"{'mean_eval'.rjust(max_key_length)} : {status['mean_eval']}\n"
    solution = f"{'solution'.rjust(max_key_length)} : {[float(k) for k in status['pop_best'].values.cpu()]}\n"
    solution_opt_eval = (
        f"{'solution fitness'.rjust(max_key_length)} : {float(status['pop_best'].evals.cpu())}\n"
    )
    feasibility = f"{'feasibility'.rjust(max_key_length)} : {bool(status['feasibility'])}\n"
    stdev_pop = f"{'pop_evals_stdev'.rjust(max_key_length)} : {float(status['pop_evals_stdev'])}\n"
    cost = f"{'cost'.rjust(max_key_length)} : {status['cost']}\n"
    eps_max = f"{'eps_max'.rjust(max_key_length)} : {status['eps_max']}\n"
    alpha = f"{'alpha'.rjust(max_key_length)} : {status['alpha']}\n"
    rho_alpha = f"{'rho_alpha'.rjust(max_key_length)} : {status['rho_alpha']}\n"
    admissibility = f"{'admissibility'.rjust(max_key_length)} : {status['admissibility']}\n"

    DoE = f"{'enrichments'.rjust(max_key_length)} : {''.join(doe_log)}\n"
    return (
        median
        + mean
        + solution
        + solution_opt_eval
        + feasibility
        + stdev_pop
        + cost
        + eps_max
        + alpha
        + rho_alpha
        + admissibility
        + DoE
    )


def log_soft_constr(benchmark, design_opt) -> str:
    """Format the check of the soft constraints of a nominal-tolerance design.

    The constraints are expected in pairs, one pair per variable: first the lower-bound
    constraint, then the upper-bound one.

    Args:
        benchmark: The benchmark, which provides the soft constraints.
        design_opt: Design to check: nominal values followed by tolerances.

    Returns:
        Rich markup with one line per constraint: the inequality with its numerical values and
        whether it is satisfied.
    """
    half_dim = int(benchmark.dim / 2)
    nominal = design_opt[:half_dim]
    tolerance = design_opt[half_dim:]
    if not hasattr(benchmark, "soft") or not benchmark.soft:
        return "[bold red]No soft constraints found in benchmark\n[/]"
    # A soft constraint is satisfied when f <= 0.
    results = [bool(f.fn(design_opt) <= 0) for f in benchmark.soft]
    res = {}
    for i in range(len(benchmark.soft)):
        if i >= len(results):
            continue
        try:
            constraint = benchmark.soft[i]
            # Constraints 2j and 2j + 1 are the lower and upper bounds of variable j.
            var_idx = i // 2
            if var_idx >= len(nominal) or var_idx >= len(tolerance):
                continue
            is_lower = i % 2 == 0
            bound_idx = 0 if is_lower else 1
            if hasattr(constraint, "bounds_xvar") and constraint.bounds_xvar is not None:  # noqa: SIM102
                if (
                    bound_idx < constraint.bounds_xvar.shape[0]
                    and var_idx < constraint.bounds_xvar.t().shape[1]
                ):
                    bound_value = constraint.bounds_xvar.t()[bound_idx][var_idx].item()
                    if is_lower:
                        constraint_str = f"T >= lb + t/2 : {nominal[var_idx].item():6.3f} >= {bound_value:6.3f} + {tolerance[var_idx].item():6.3f} / 2 = {(bound_value + tolerance[var_idx] / 2).item():6.3f}"
                    else:
                        constraint_str = f"T <= ub - t/2 : {nominal[var_idx].item():6.3f} <= {bound_value:6.3f} - {tolerance[var_idx].item():6.3f} / 2 = {(bound_value - tolerance[var_idx] / 2).item():6.3f}"
                    res[constraint_str] = results[i]
        except (IndexError, AttributeError):
            res[f"Error processing constraint {i}"] = False
    log_str = "\n\n[bold dark_orange]Soft constraints check for nominals T and tolerance t:[/]\n"
    for n, (k, v) in enumerate(res.items()):
        bool_col = "[green]" if v is True else "[red]"
        log_str += f"[white]({n})[cyan] {k} [bold light_coral]→ {bool_col}{v}\n"
    return log_str
