# ============================================================
# StringParticleResearch
# Phase I — Baseline String Experiment
# ============================================================

using Printf

function run_baseline()

    # --------------------------------------------------------
    # 1. Physical parameters
    # --------------------------------------------------------

    L = 1.0
    c = 1.0

    # --------------------------------------------------------
    # 2. Numerical parameters
    # --------------------------------------------------------

    N = 201
    T = 10.0

    dx = L / (N - 1)

    # CFL condition
    CFL = 0.5
    dt = CFL * dx / c

    num_steps = Int(floor(T / dt))

    x = range(0.0, L, length=N)

    # --------------------------------------------------------
    # 3. Initial condition
    # --------------------------------------------------------

    X = sin.(π .* x ./ L)

    # Initial velocity
    V = zeros(N)

    # Fixed endpoints
    X[1] = 0.0
    X[end] = 0.0

    # --------------------------------------------------------
    # 4. Analytical frequency
    # --------------------------------------------------------

    ω_exact = π * c / L

    println()
    println("==============================================")
    println(" BASELINE STRING EXPERIMENT")
    println("==============================================")
    println()

    @printf("String length L:       %.4f\n", L)
    @printf("Wave speed c:          %.4f\n", c)
    @printf("Grid points N:         %d\n", N)
    @printf("Spatial step dx:       %.6e\n", dx)
    @printf("Time step dt:          %.6e\n", dt)
    @printf("Simulation time T:     %.4f\n", T)

    println()

    @printf(
        "Analytical frequency:  %.10f\n",
        ω_exact
    )

    # --------------------------------------------------------
    # 5. First time step
    # --------------------------------------------------------

    X_old = copy(X)

    acceleration = zeros(N)

    for i in 2:N-1
        acceleration[i] =
            c^2 * (X[i+1] - 2X[i] + X[i-1]) / dx^2
    end

    X_new =
        X .+
        dt .* V .+
        0.5 .* dt^2 .* acceleration

    # Fixed boundaries
    X_new[1] = 0.0
    X_new[end] = 0.0

    # --------------------------------------------------------
    # 6. Time evolution
    # --------------------------------------------------------

    for step in 2:num_steps

        for i in 2:N-1
            acceleration[i] =
                c^2 *
                (X_new[i+1] - 2X_new[i] + X_new[i-1]) /
                dx^2
        end

        X_next =
            2.0 .* X_new .-
            X_old .+
            dt^2 .* acceleration

        # Fixed boundaries
        X_next[1] = 0.0
        X_next[end] = 0.0

        # Advance solution
        X_old = X_new
        X_new = X_next

    end

    # --------------------------------------------------------
    # 7. Analytical comparison
    # --------------------------------------------------------

    t_final = num_steps * dt

    X_exact =
        sin.(π .* x ./ L) .* cos.(ω_exact * t_final)

    absolute_error =
        maximum(abs.(X_new .- X_exact))

    relative_error =
        absolute_error /
        maximum(abs.(X_exact))

    println()
    println("----------------------------------------------")
    println(" NUMERICAL VALIDATION")
    println("----------------------------------------------")

    @printf("Final time:             %.10f\n", t_final)

    @printf(
        "Maximum absolute error: %.6e\n",
        absolute_error
    )

    @printf(
        "Relative error:         %.6e\n",
        relative_error
    )

    println()

    if relative_error < 1e-2
        println(
            "PASS: Numerical solution agrees with analytical solution."
        )
    else
        println(
            "WARNING: Error is larger than expected."
        )
    end

    println()
    println("Simulation completed.")
    println("==============================================")
    println()

end

# Run the experiment
run_baseline()