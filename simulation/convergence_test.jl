# ============================================================
# StringParticleResearch
# Phase I — Numerical Convergence Test
# ============================================================

using Printf


# ------------------------------------------------------------
# Simulate the baseline string
# ------------------------------------------------------------

function simulate_string(N)

    L = 1.0
    c = 1.0
    T = 10.0

    dx = L / (N - 1)

    CFL = 0.5
    dt = CFL * dx / c

    num_steps = Int(floor(T / dt))

    x = collect(range(0.0, L, length=N))

    # Initial displacement: fundamental mode
    X = sin.(π .* x ./ L)

    # Initial velocity
    V = zeros(N)

    # Fixed endpoints
    X[1] = 0.0
    X[end] = 0.0

    # Exact fundamental frequency
    ω_exact = π * c / L

    # --------------------------------------------------------
    # First time step
    # --------------------------------------------------------

    acceleration = zeros(N)

    for i in 2:N-1

        acceleration[i] =
            c^2 *
            (X[i+1] -
             2.0 * X[i] +
             X[i-1]) /
            dx^2

    end

    X_old = copy(X)

    X_new =
        X .+
        dt .* V .+
        0.5 .* dt^2 .* acceleration

    X_new[1] = 0.0
    X_new[end] = 0.0

    # --------------------------------------------------------
    # Time evolution
    # --------------------------------------------------------

    for step in 2:num_steps

        for i in 2:N-1

            acceleration[i] =
                c^2 *
                (X_new[i+1] -
                 2.0 * X_new[i] +
                 X_new[i-1]) /
                dx^2

        end

        X_next =
            2.0 .* X_new .-
            X_old .+
            dt^2 .* acceleration

        X_next[1] = 0.0
        X_next[end] = 0.0

        X_old = X_new
        X_new = X_next

    end

    # --------------------------------------------------------
    # Exact solution at final time
    # --------------------------------------------------------

    t_final = num_steps * dt

    X_exact =
        sin.(π .* x ./ L) .*
        cos.(ω_exact * t_final)

    # --------------------------------------------------------
    # Error
    # --------------------------------------------------------

    absolute_error =
        maximum(abs.(X_new .- X_exact))

    return dx, dt, absolute_error

end


# ============================================================
# Convergence study
# ============================================================

function convergence_test()

    grid_sizes = [101, 201, 401, 801]

    println()
    println("==============================================")
    println(" CONVERGENCE TEST")
    println("==============================================")
    println()

    @printf(
        "%8s %15s %15s %18s %15s\n",
        "N",
        "dx",
        "dt",
        "Max Error",
        "Order"
    )

    println("-" ^ 82)

    previous_error = nothing

    for N in grid_sizes

        dx, dt, error =
            simulate_string(N)

        if previous_error === nothing

            @printf(
                "%8d %15.6e %15.6e %18.6e %15s\n",
                N,
                dx,
                dt,
                error,
                "---"
            )

        else

            observed_order =
                log(previous_error / error) /
                log(2.0)

            @printf(
                "%8d %15.6e %15.6e %18.6e %15.6f\n",
                N,
                dx,
                dt,
                error,
                observed_order
            )

        end

        previous_error = error

    end

    println()
    println("Convergence test completed.")
    println("==============================================")
    println()

end


# Run convergence study
convergence_test()