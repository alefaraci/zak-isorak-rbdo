"""Per-generation frame renderers of the evolution animations.

Each public function renders the frame(s) of one generation of a finished run and saves them as PDF
files in ``<run directory>/evolution/<animation name>/``. They are module-level functions with
picklable arguments, so that the post-processing can dispatch one call per generation to a process
pool; every call reloads the pickled problem and ``results.json`` from the run directory.

Frame names, common to all the renderers:

* ``frame_0000``: the initial population, rendered together with generation 0;
* ``frame_<gen + 1>``: the population of generation ``gen`` and its best design;
* ``frame_<gen + 1>_bis`` (ISORAK-RBDO only): the population re-initialised for a new iso-cost
  level, rendered when the ``iter`` counter of the results restarts at 1.
"""

# Each renderer imports what it needs when it is called, in the worker process that runs it.
# ruff: noqa: PLC0415


def animation_evolution_g_space_frame(path, gen, show, aug_w) -> None:
    """Render the design-space frame(s) of one generation.

    The population and its best design are drawn over the 2D picture of the benchmark (cost
    contours, limit states, soft constraints). For the adaptive-Kriging analyses the limit states
    are the surrogates of that generation, shown with the initial design of experiments, the
    enrichment points added so far and, in a lighter colour, those added at this generation.

    With ``aug_w`` the frame shows the augmented space: the Monte Carlo cloud of realisations of
    the random variables is added around every individual that is feasible at its own design
    point, and in a darker colour around the best design, to show which part of the augmented
    space the population is probing.

    Frames are saved in ``<path>/evolution/evo_g_space``, or ``<path>/evolution/evo_g_aug_space``
    with ``aug_w``.

    Args:
        path: Run directory, holding the pickled problem and ``results.json``.
        gen: Zero-based index of the generation to render.
        show: Also display each frame briefly on screen.
        aug_w: Render the augmented-space variant.
    """
    import torch
    import polars as pl

    from zak import utils, plotter
    from zak.plotter.benchmark_2d import BaseModel2D

    from ..gaussianprocess.gp import ListGPRBoTorch

    def _create_visualization(
        problem,
        benchmark_2d,
        individuals,
        filename,
        show,
        best_pop=None,
        surrogate_2d=None,
        train_x=None,
        points_added=None,
        points_to_add=None,
        metamodels=None,
    ) -> None:
        """Create and save a single visualization frame."""
        augmented = "rbdo" in problem.analysis
        if surrogate_2d is not None and train_x is not None:
            surrogate_2d.visualize(augmented_space=augmented)
            plotter.plt.scatter(train_x[:, 0], train_x[:, 1], **plotter.scatter_style["train"])
            if points_added is not None:
                if points_to_add is not None:
                    points_added_to_plot = [
                        utils.tensor_difference(points_added[k], points_to_add[k])
                        for k in range(len(points_added))
                    ]
                    for i, pi in enumerate(points_added_to_plot):
                        for pj in pi:
                            plotter.plt.scatter(
                                pj[0],
                                pj[1],
                                **plotter.scatter_style[f"enrich_base_circle_{i + 1}"],
                            )
                            plotter.plt.scatter(pj[0], pj[1], **plotter.scatter_style["train"])
                    for i, pi in enumerate(points_to_add):
                        for pj in pi:
                            plotter.plt.scatter(
                                pj[0],
                                pj[1],
                                **plotter.scatter_style[f"enrich_base_circle_{i + 1}"],
                            )
                            plotter.plt.scatter(pj[0], pj[1], **plotter.scatter_style["train_new"])
                else:
                    for i, pi in enumerate(points_added):
                        for pj in pi:
                            plotter.plt.scatter(
                                pj[0],
                                pj[1],
                                **plotter.scatter_style[f"enrich_base_circle_{i + 1}"],
                            )
                            plotter.plt.scatter(pj[0], pj[1], **plotter.scatter_style["train"])
        else:
            benchmark_2d.visualize(augmented_space=augmented)
        plotter.plt.scatter(
            individuals[:, 0],
            individuals[:, 1],
            **plotter.scatter_style["ga_individual"],
        )
        if (
            "rbdo" in problem.analysis
            and problem.benchmark.id not in ("column_under_compression", "truss_bridge")
            and aug_w
        ):
            augmented_size = min(problem.config["rbdo"]["size_mc"], 1e4)
            w = benchmark_2d.benchmark.get_W_MCS(
                N_SAMPLES=int(augmented_size),
                design=individuals,
            )
            f = benchmark_2d.benchmark.evaluate_soft(individuals)  # (n_individuals, n_soft)
            # A single sample is the design itself: feasibility is checked without scatter, and
            # only the feasible individuals get their cloud drawn.
            w_individuals = benchmark_2d.benchmark.get_W(N_SAMPLES=1, design=individuals)
            if surrogate_2d is not None:
                mean, _ = metamodels.evaluate_surrogate(w_individuals)  # (n_individuals, n_hard)
            else:
                mean, _ = benchmark_2d.benchmark.evaluate_hard(w_individuals)
            g_ok = (mean > 0).all(dim=1)
            if f is not None:
                f_ok = (f <= 0).all(dim=1)
                feas = g_ok & f_ok
            else:
                feas = g_ok
            plotter.plt.scatter(w[:, feas, 0], w[:, feas, 1], **plotter.scatter_style["nebula"])
            if best_pop is not None:
                w_best = benchmark_2d.benchmark.get_W(N_SAMPLES=1, design=best_pop)
                f = benchmark_2d.benchmark.evaluate_soft(best_pop)
                if surrogate_2d is not None:
                    mean, _ = metamodels.evaluate_surrogate(w_best)
                else:
                    mean, _ = benchmark_2d.benchmark.evaluate_hard(w_best)
                g_ok = (mean > 0).all()
                if f is not None:
                    f_ok = (f <= 0).all()
                    feas = g_ok & f_ok
                else:
                    feas = g_ok
                if feas:
                    w = benchmark_2d.benchmark.get_W_MCS(
                        N_SAMPLES=int(augmented_size),
                        design=best_pop,
                    )
                    plotter.plt.scatter(
                        w[..., 0],
                        w[..., 1],
                        **plotter.scatter_style["nebula_best"],
                    )
        if best_pop is not None:
            plotter.plt.scatter(best_pop[0], best_pop[1], **plotter.scatter_style["best"])
        plotter.set_square_ratio(plotter.plt.gca())
        name_anim = "evo_g_space" if aug_w is False else "evo_g_aug_space"
        plotter.utils.save(
            plotter.plt,
            path=path / "evolution" / name_anim,
            filename=filename,
            pdf_dir=False,
        )
        plotter.plt.pause(0.1) if show else None
        plotter.plt.close()

    problem = utils.load(path=path, filename="opt_problem")["opt_problem"]
    benchmark_2d = BaseModel2D(benchmark=problem.benchmark)
    results = pl.read_ndjson(path / "results.json")
    if "iso" in problem.analysis:
        benchmark_2d.benchmark.cost._alpha = results["alpha"][gen]

    if "ak" in problem.analysis:
        train_x = [torch.tensor(doe_k, dtype=torch.float64) for doe_k in results["doe"][0]]
        doe = [torch.tensor(doe_k, dtype=torch.float64) for doe_k in results["doe"][gen]]
        # The benchmark YAML sits next to the `results` directory, four levels above the run.
        params = utils.load_yaml(
            path=path.parent.parent.parent.parent,
            filename=f"{problem.benchmark.id}.yaml",
        )[problem.analysis.upper()]

        # The surrogates are not stored with the results: they are rebuilt from the design of
        # experiments recorded at this generation.
        metamodels = ListGPRBoTorch(
            covar_module=params["gp"]["covar_module"],
            meta_bounds=problem.benchmark.aug_bounds,
            ng=len(problem.benchmark.hard),
        )
        with torch.no_grad():
            train_y = [gk.fn(doe[k]) for k, gk in enumerate(problem.benchmark.hard)]
        metamodels.train(DoE=doe, train_y=train_y)
        surrogate_2d = BaseModel2D(benchmark=problem.benchmark, hard=metamodels.hard)
        if gen == 0:
            _create_visualization(
                problem,
                benchmark_2d,
                problem.init_pop
                if "iso" not in problem.analysis
                else torch.tensor(results["pop_init_alpha"][gen]),
                f"frame_{gen:04d}",
                show,
                best_pop=None,
                surrogate_2d=surrogate_2d,
                train_x=train_x[0],
                metamodels=metamodels,
            )
        if "iso" in problem.analysis and gen > 0 and results["iter"][gen] == 1:
            doe_ = [torch.tensor(doe_k, dtype=torch.float64) for doe_k in results["doe"][gen - 1]]
            points_added = [utils.tensor_difference(doe[k], train_x[k]) for k in range(len(doe))]
            points_to_add = [utils.tensor_difference(doe[k], doe_[k]) for k in range(len(doe))]
            _create_visualization(
                problem,
                benchmark_2d,
                torch.tensor(results["pop_init_alpha"][gen]),
                f"frame_{gen + 1:04d}_bis",
                show,
                best_pop=None,
                surrogate_2d=surrogate_2d,
                train_x=train_x[0],
                points_added=points_added,
                points_to_add=points_to_add,
                metamodels=metamodels,
            )
        individuals = torch.tensor(results["individuals"][gen])
        best_pop = torch.tensor(results["pop_best"][gen])
        doe_ = [torch.tensor(doe_k, dtype=torch.float64) for doe_k in results["doe"][gen - 1]]
        points_added = [utils.tensor_difference(doe[k], train_x[k]) for k in range(len(doe))]
        points_to_add = [utils.tensor_difference(doe[k], doe_[k]) for k in range(len(doe))]
        _create_visualization(
            problem,
            benchmark_2d,
            individuals,
            f"frame_{gen + 1:04d}",
            show,
            best_pop=best_pop,
            surrogate_2d=surrogate_2d,
            train_x=train_x[0],
            points_added=points_added,
            points_to_add=points_to_add,
            metamodels=metamodels,
        )
    else:
        # ddo and rbdo: no surrogate, the true limit states are drawn.
        if gen == 0:
            _create_visualization(problem, benchmark_2d, problem.init_pop, f"frame_{gen:04d}", show)
        individuals = torch.tensor(results["individuals"][gen])
        best_pop = torch.tensor(results["pop_best"][gen])
        _create_visualization(
            problem,
            benchmark_2d,
            individuals,
            f"frame_{gen + 1:04d}",
            show,
            best_pop,
        )


def animation_evolution_cost_space_frame(path, gen, show) -> None:
    """Render the cost-space frame(s) of one generation.

    The population and its best design are drawn over the filled cost contours in the design
    space, with the soft constraint of the column benchmark and, for ISORAK-RBDO, the mask of the
    designs beyond the current iso-cost level. Frames are saved in
    ``<path>/evolution/evo_cost_space``.

    Args:
        path: Run directory, holding the pickled problem and ``results.json``.
        gen: Zero-based index of the generation to render.
        show: Also display each frame briefly on screen.
    """
    import torch
    import polars as pl

    from zak import utils, plotter

    def _create_visualization(benchmark, individuals, filename, show, best_pop=None) -> None:
        """Draw and save one frame."""
        benchmark.cost.visualize(design_bounds=benchmark.bounds, bench_id=benchmark.id)
        if "column" in benchmark.id:
            benchmark.soft[0].visualize(design_bounds=benchmark.bounds)
        if "iso" in problem.analysis:
            benchmark.cost.visualize_mask_iso(design_bounds=benchmark.bounds)
        plotter.plt.scatter(individuals[:, 0], individuals[:, 1], **plotter.scatter_style["trace"])
        if best_pop is not None:
            plotter.plt.scatter(best_pop[0], best_pop[1], **plotter.scatter_style["best"])
        plotter.plt.pause(0.1) if show else None
        plotter.utils.save(
            plotter.plt,
            path=path / "evolution" / "evo_cost_space",
            filename=filename,
            pdf_dir=False,
        )
        plotter.plt.pause(0.1) if show else None
        plotter.plt.close()

    problem = utils.load(path=path, filename="opt_problem")["opt_problem"]
    benchmark = problem.benchmark
    results = pl.read_ndjson(path / "results.json")

    if "iso" in problem.analysis:
        benchmark.cost._alpha = results["alpha"][gen]

    if gen == 0:
        _create_visualization(
            benchmark,
            problem.init_pop
            if "iso" not in problem.analysis
            else torch.tensor(results["pop_init_alpha"][gen]),
            f"frame_{gen:04d}",
            show,
        )
    if "iso" in problem.analysis and gen > 0 and results["iter"][gen] == 1:
        _create_visualization(
            benchmark,
            torch.tensor(results["pop_init_alpha"][gen]),
            f"frame_{gen + 1:04d}_bis",
            show,
        )
    individuals = torch.tensor(results["individuals"][gen])
    best_pop = torch.tensor(results["pop_best"][gen])
    _create_visualization(benchmark, individuals, f"frame_{gen + 1:04d}", show, best_pop)


def animation_evolution_tol_g_space_frame(path, gen, show) -> None:
    """Render the tolerance-interval frame(s) of one generation of a tolerance benchmark.

    Every individual is drawn as a rectangle centred on its nominal values, whose sides are its
    tolerances, over the limit states (true or surrogate) in the plane of the two toleranced
    dimensions. In the linear stack-up benchmarks the nominal point is fixed and only the
    rectangles change. Frames are saved in ``<path>/evolution/evo_tol_g_space``.

    Args:
        path: Run directory, holding the pickled problem and ``results.json``.
        gen: Zero-based index of the generation to render.
        show: Also display each frame briefly on screen.
    """
    from matplotlib.patches import Rectangle
    from matplotlib.collections import PatchCollection
    import torch
    import polars as pl

    from zak import utils, plotter
    from zak.plotter.benchmark_2d import BaseModel2D

    from ..gaussianprocess.gp import ListGPRBoTorch

    def _create_visualization(
        problem,
        benchmark_2d,
        individuals,
        filename,
        show,
        best_pop=None,
        surrogate_2d=None,
        train_x=None,
        points_added=None,
        points_to_add=None,
    ) -> None:
        """Draw and save one frame."""
        nominal_tolerance = benchmark_2d.benchmark.nominal_tolerance_design
        aug_space = "rbdo" in problem.analysis
        if surrogate_2d is not None and train_x is not None:
            surrogate_2d.visualize(augmented_space=aug_space, nominal_tolerance=nominal_tolerance)
            plotter.plt.scatter(train_x[:, 0], train_x[:, 1], **plotter.scatter_style["train"])
            if points_added is not None:
                if points_to_add is not None:
                    points_added_to_plot = [
                        utils.tensor_difference(points_added[k], points_to_add[k])
                        for k in range(len(points_added))
                    ]
                    for i, pi in enumerate(points_added_to_plot):
                        for pj in pi:
                            plotter.plt.scatter(
                                pj[0],
                                pj[1],
                                **plotter.scatter_style[f"enrich_base_circle_{i + 1}"],
                            )
                            plotter.plt.scatter(pj[0], pj[1], **plotter.scatter_style["train"])
                    for i, pi in enumerate(points_to_add):
                        for pj in pi:
                            plotter.plt.scatter(
                                pj[0],
                                pj[1],
                                **plotter.scatter_style[f"enrich_base_circle_{i + 1}"],
                            )
                            plotter.plt.scatter(pj[0], pj[1], **plotter.scatter_style["train_new"])
                else:
                    for i, pi in enumerate(points_added):
                        for pj in pi:
                            plotter.plt.scatter(
                                pj[0],
                                pj[1],
                                **plotter.scatter_style[f"enrich_base_circle_{i + 1}"],
                            )
                            plotter.plt.scatter(pj[0], pj[1], **plotter.scatter_style["train"])
        else:
            benchmark_2d.visualize(augmented_space=aug_space, nominal_tolerance=nominal_tolerance)
        if "linear_stackup" in benchmark_2d.benchmark.id:
            nominal = benchmark_2d.benchmark.nominal
            tolerance = individuals
            plotter.plt.scatter(
                nominal[-1, 0],
                nominal[-1, 1],
                **plotter.scatter_style["best"],
            )
            rectangles = [
                Rectangle(
                    (
                        float(nominal[-1, 0] - individuals[j, 0] / 2),
                        float(nominal[-1, 1] - individuals[j, 1] / 2),
                    ),
                    float(individuals[j, 0]),
                    float(individuals[j, 1]),
                )
                for j in range(tolerance.shape[0])
            ]
        else:
            nominal = individuals[:, : int(benchmark_2d.benchmark.dim / 2)]
            tolerance = individuals[:, int(benchmark_2d.benchmark.dim / 2) :]
            plotter.plt.scatter(
                individuals[:, 0],
                individuals[:, 1],
                **plotter.scatter_style["ga_individual"],
            )
            rectangles = [
                Rectangle(
                    (
                        float(individuals[j, 0] - individuals[j, 2] / 2),
                        float(individuals[j, 1] - individuals[j, 3] / 2),
                    ),
                    float(individuals[j, 2]),
                    float(individuals[j, 3]),
                )
                for j in range(individuals.shape[0])
            ]
            if best_pop is not None:
                plotter.plt.scatter(
                    best_pop[: int(benchmark_2d.benchmark.dim / 2)][0],
                    best_pop[: int(benchmark_2d.benchmark.dim / 2)][1],
                    **plotter.scatter_style["best"],
                )
        collection = PatchCollection(
            rectangles,
            alpha=0.1,
            color="#00BFC0",
            edgecolor="#22242B",
            zorder=95,
        )
        plotter.plt.gca().add_collection(collection)
        plotter.plt.tight_layout()
        plotter.plt.pause(0.1) if show else None
        plotter.utils.save(
            plotter.plt,
            path=path / "evolution" / "evo_tol_g_space",
            filename=filename,
            pdf_dir=False,
        )
        plotter.plt.pause(0.1) if show else None
        plotter.plt.close()

    problem = utils.load(path=path, filename="opt_problem")["opt_problem"]
    benchmark_2d = BaseModel2D(benchmark=problem.benchmark)
    results = pl.read_ndjson(path / "results.json")

    if "ak" in problem.analysis:
        train_x = [torch.tensor(doe_k, dtype=torch.float64) for doe_k in results["doe"][0]]
        doe = [torch.tensor(doe_k, dtype=torch.float64) for doe_k in results["doe"][gen]]
        params = utils.load_yaml(
            path=path.parent.parent.parent.parent,
            filename=f"{problem.benchmark.id}.yaml",
        )[problem.analysis.upper()]
        # Surrogates rebuilt from the design of experiments recorded at this generation.
        metamodels = ListGPRBoTorch(
            covar_module=params["gp"]["covar_module"],
            meta_bounds=problem.benchmark.aug_bounds,
            ng=len(problem.benchmark.hard),
        )
        with torch.no_grad():
            train_y = [gk.fn(doe[k]) for k, gk in enumerate(problem.benchmark.hard)]
        metamodels.train(DoE=doe, train_y=train_y)
        surrogate_2d = BaseModel2D(benchmark=problem.benchmark, hard=metamodels.hard)
        if gen == 0:
            _create_visualization(
                problem,
                benchmark_2d,
                problem.init_pop
                if "iso" not in problem.analysis
                else torch.tensor(results["pop_init_alpha"][gen]),
                f"frame_{gen:04d}",
                show,
                best_pop=None,
                surrogate_2d=surrogate_2d,
                train_x=train_x[0],
            )
        if "iso" in problem.analysis and gen > 0 and results["iter"][gen] == 1:
            _create_visualization(
                problem,
                benchmark_2d,
                torch.tensor(results["pop_init_alpha"][gen]),
                f"frame_{gen + 1:04d}_bis",
                show,
                best_pop=None,
                surrogate_2d=surrogate_2d,
                train_x=train_x[0],
            )
        individuals = torch.tensor(results["individuals"][gen])
        best_pop = torch.tensor(results["pop_best"][gen])
        doe_ = [torch.tensor(doe_k, dtype=torch.float64) for doe_k in results["doe"][gen - 1]]
        points_added = [utils.tensor_difference(doe[k], train_x[k]) for k in range(len(doe))]
        points_to_add = [utils.tensor_difference(doe[k], doe_[k]) for k in range(len(doe))]
        _create_visualization(
            problem,
            benchmark_2d,
            individuals,
            f"frame_{gen + 1:04d}",
            show,
            best_pop=best_pop,
            surrogate_2d=surrogate_2d,
            train_x=train_x[0],
            points_added=points_added,
            points_to_add=points_to_add,
        )
    else:
        # ddo and rbdo: no surrogate, the true limit states are drawn.
        if gen == 0:
            _create_visualization(problem, benchmark_2d, problem.init_pop, f"frame_{gen:04d}", show)
        individuals = torch.tensor(results["individuals"][gen])
        best_pop = torch.tensor(results["pop_best"][gen])
        _create_visualization(
            problem,
            benchmark_2d,
            individuals,
            f"frame_{gen + 1:04d}",
            show,
            best_pop,
        )


def animation_evolution_tol_space_frame(path, gen, show) -> None:
    """Render the tolerance-space frame(s) of one generation of a tolerance benchmark.

    The tolerances of the population and of the best design are drawn over the filled cost
    contours in the plane of the two tolerances, with the iso-cost mask for ISORAK-RBDO. Frames
    are saved in ``<path>/evolution/evo_tol_space``.

    Args:
        path: Run directory, holding the pickled problem and ``results.json``.
        gen: Zero-based index of the generation to render.
        show: Also display each frame briefly on screen.
    """
    import torch
    import polars as pl

    from zak import utils, plotter

    def _create_visualization(benchmark, individuals, filename, show, best_pop=None) -> None:
        """Draw and save one frame."""
        # The cost depends on the tolerances only: the whole design vector for the linear
        # stack-up benchmarks, its second half for the [nominal values, tolerances] designs.
        if "linear_stack" in benchmark.id:
            tolerance = individuals
            bounds = benchmark.bounds
        else:
            tolerance = individuals[:, int(benchmark.dim / 2) :]
            bounds = benchmark.bounds[int(benchmark.dim / 2) :]
        benchmark.cost.visualize(design_bounds=bounds, bench_id=benchmark.id)
        if "iso" in problem.analysis:
            benchmark.cost.visualize_mask_iso(design_bounds=bounds)
        plotter.plt.scatter(tolerance[:, 0], tolerance[:, 1], **plotter.scatter_style["trace"])
        if best_pop is not None:
            plotter.plt.scatter(best_pop[0], best_pop[1], **plotter.scatter_style["best"])
        plotter.plt.tight_layout()
        plotter.plt.pause(0.1) if show else None
        plotter.utils.save(
            plotter.plt,
            path=path / "evolution" / "evo_tol_space",
            filename=filename,
            pdf_dir=False,
        )
        plotter.plt.pause(0.1) if show else None
        plotter.plt.close()

    problem = utils.load(path=path, filename="opt_problem")["opt_problem"]
    benchmark = problem.benchmark
    results = pl.read_ndjson(path / "results.json")

    if "iso" in problem.analysis:
        benchmark.cost._alpha = results["alpha"][gen]

    if gen == 0:
        _create_visualization(
            benchmark,
            problem.init_pop
            if "iso" not in problem.analysis
            else torch.tensor(results["pop_init_alpha"][gen]),
            f"frame_{gen:04d}",
            show,
        )
    if "iso" in problem.analysis and gen > 0 and results["iter"][gen] == 1:
        _create_visualization(
            benchmark,
            torch.tensor(results["pop_init_alpha"][gen]),
            f"frame_{gen + 1:04d}_bis",
            show,
        )
    individuals = torch.tensor(results["individuals"][gen])
    if "linear_stack" in benchmark.id:
        best_pop = torch.tensor(results["pop_best"][gen])
    else:
        best_pop = torch.tensor(results["pop_best"][gen])[int(benchmark.dim / 2) :]
    _create_visualization(benchmark, individuals, f"frame_{gen + 1:04d}", show, best_pop)


def animation_evolution_nom_tol_adim_space_frame(path, gen, show) -> None:
    """Render the dimensionless nominal/tolerance frame(s) of one generation.

    For the cantilever beam benchmarks, every individual contributes one point per toleranced
    dimension, at its nominal value and tolerance both scaled to [0, 1] by their bounds, over the
    background of ``plot_tolerance_2d.plot_nom_tol_adim_space``. The points of the best design are
    highlighted with one colour per dimension. Frames are saved in
    ``<path>/evolution/evo_nom_tol_adim_space``.

    Args:
        path: Run directory, holding the pickled problem and ``results.json``.
        gen: Zero-based index of the generation to render.
        show: Also display each frame briefly on screen.
    """
    import torch
    import polars as pl

    from zak import utils, plotter

    def _create_visualization(benchmark, evo_design, filename, show, best_pop=None) -> None:
        """Draw and save one frame."""
        if "cantilever_4d" in benchmark.id:
            labels = ["h", "L", "b", "f"]
            points_lab = ["nom_tol_1", "nom_tol_2", "nom_tol_3", "nom_tol_4"]
        elif "cantilever_2d" in benchmark.id:
            labels = ["h", "L"]
            points_lab = ["nom_tol_1", "nom_tol_2"]
        else:
            return
        mean_shift = "mean_shift" in benchmark.id
        plotter.plot_tolerance_2d.plot_nom_tol_adim_space(mean_shift=mean_shift)
        nominal = evo_design[:, : int(benchmark.dim / 2)]
        tolerance = evo_design[:, int(benchmark.dim / 2) :]
        bounds_nom = benchmark.bounds[: len(labels), :]
        bounds_tol = benchmark.bounds[len(labels) :, :]
        nominal_norm = (nominal - bounds_nom.min(dim=1).values) / (
            bounds_nom.max(dim=1).values - bounds_nom.min(dim=1).values
        )
        tolerance_norm = (tolerance - bounds_tol.min(dim=1).values) / (
            bounds_tol.max(dim=1).values - bounds_tol.min(dim=1).values
        )
        for nom, tol in zip(nominal_norm.t(), tolerance_norm.t(), strict=False):
            plotter.plt.scatter(nom, tol, **plotter.scatter_style["ga_individual"])
        if best_pop is not None:
            best_pop_nominal = best_pop[: int(benchmark.dim / 2)]
            best_pop_tolerance = best_pop[int(benchmark.dim / 2) :]
            best_pop_norm_nominal = (best_pop_nominal - bounds_nom.min(dim=1).values) / (
                bounds_nom.max(dim=1).values - bounds_nom.min(dim=1).values
            )
            best_pop_norm_tolerance = (best_pop_tolerance - bounds_tol.min(dim=1).values) / (
                bounds_tol.max(dim=1).values - bounds_tol.min(dim=1).values
            )
            leg_list = [
                plotter.plt.scatter(
                    best_pop_norm_nominal[k],
                    best_pop_norm_tolerance[k],
                    **plotter.scatter_style[points_lab[k]],
                )
                for k in range(len(points_lab))
            ]
            plotter.plt.legend(leg_list, [rf"${label}$" for label in labels], loc="upper right")
        plotter.set_square_ratio(plotter.plt.gca())
        plotter.plt.tight_layout()
        plotter.plt.pause(0.1) if show else None
        plotter.utils.save(
            plotter.plt,
            path=path / "evolution" / "evo_nom_tol_adim_space",
            filename=filename,
            pdf_dir=False,
        )
        plotter.plt.pause(0.1) if show else None
        plotter.plt.close()

    problem = utils.load(path=path, filename="opt_problem")["opt_problem"]
    benchmark = problem.benchmark
    results = pl.read_ndjson(path / "results.json")

    if gen == 0:
        _create_visualization(
            benchmark,
            problem.init_pop
            if "iso" not in problem.analysis
            else torch.tensor(results["pop_init_alpha"][gen]),
            f"frame_{gen:04d}",
            show,
        )
    if "iso" in problem.analysis and gen > 0 and results["iter"][gen] == 1:
        _create_visualization(
            benchmark,
            torch.tensor(results["pop_init_alpha"][gen]),
            f"frame_{gen + 1:04d}_bis",
            show,
        )
    individuals = torch.tensor(results["individuals"][gen])
    best_pop = torch.tensor(results["pop_best"][gen])
    _create_visualization(benchmark, individuals, f"frame_{gen + 1:04d}", show, best_pop)


def animation_evolution_tol_mean_shift_adim_space_frame(path, gen, show) -> None:
    """Render the dimensionless mean-shift frame(s) of one generation.

    Only the best design is drawn: one point per toleranced dimension at ``(t* / 2, t*)``, where
    ``t*`` is its tolerance scaled to [0, 1] by its bounds, over the background of
    ``plot_tolerance_2d.plot_tol_mean_shift_adim_space``. The population itself is not shown, so
    the frames without a best design contain the background only. Frames are saved in
    ``<path>/evolution/evo_tol_mean_shift_adim_space``.

    Args:
        path: Run directory, holding the pickled problem and ``results.json``.
        gen: Zero-based index of the generation to render.
        show: Also display each frame briefly on screen.
    """
    import torch
    import polars as pl

    from zak import utils, plotter

    def _create_visualization(benchmark, _, filename, show, best_pop=None) -> None:
        """Draw and save one frame (the population argument is ignored)."""
        if "cantilever_4d" in benchmark.id:
            points_lab = ["nom_tol_1", "nom_tol_2", "nom_tol_3", "nom_tol_4"]
        elif "cantilever_2d" in benchmark.id or "linear_stackup" in benchmark.id:
            points_lab = ["nom_tol_1", "nom_tol_2"]
        else:
            return
        labels = benchmark.label_X
        plotter.plot_tolerance_2d.plot_tol_mean_shift_adim_space()
        bounds_tol = benchmark.bounds[len(labels) :, :]
        if best_pop is not None:
            best_pop_tolerance = best_pop[int(benchmark.dim / 2) :]
            best_pop_norm_tolerance = (best_pop_tolerance - bounds_tol.min(dim=1).values) / (
                bounds_tol.max(dim=1).values - bounds_tol.min(dim=1).values
            )
            best_pop_norm_delta = best_pop_norm_tolerance / 2
            leg_list = [
                plotter.plt.scatter(
                    best_pop_norm_delta[k],
                    best_pop_norm_tolerance[k],
                    **plotter.scatter_style[points_lab[k]],
                )
                for k in range(len(points_lab))
            ]
            plotter.plt.legend(leg_list, [rf"${label}$" for label in labels], loc="upper right")
        plotter.set_square_ratio(plotter.plt.gca())
        plotter.plt.tight_layout()
        plotter.plt.pause(0.1) if show else None
        plotter.utils.save(
            plotter.plt,
            path=path / "evolution" / "evo_tol_mean_shift_adim_space",
            filename=filename,
            pdf_dir=False,
        )
        plotter.plt.pause(0.1) if show else None
        plotter.plt.close()

    problem = utils.load(path=path, filename="opt_problem")["opt_problem"]
    benchmark = problem.benchmark
    results = pl.read_ndjson(path / "results.json")

    if gen == 0:
        _create_visualization(
            benchmark,
            problem.init_pop
            if "iso" not in problem.analysis
            else torch.tensor(results["pop_init_alpha"][gen]),
            f"frame_{gen:04d}",
            show,
        )
    if "iso" in problem.analysis and gen > 0 and results["iter"][gen] == 1:
        _create_visualization(
            benchmark,
            torch.tensor(results["pop_init_alpha"][gen]),
            f"frame_{gen + 1:04d}_bis",
            show,
        )
    individuals = torch.tensor(results["individuals"][gen])
    best_pop = torch.tensor(results["pop_best"][gen])
    _create_visualization(benchmark, individuals, f"frame_{gen + 1:04d}", show, best_pop)
