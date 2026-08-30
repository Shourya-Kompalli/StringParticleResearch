# ============================================================
# StringParticleResearch
# Phase II — Task 1A
#
# Corrected Nonlinear String Solver
#
# Equation:
#
#     X_tt = c^2 X_xx - λ X^3
#
# This version tracks position AND velocity consistently.
# ============================================================

using Printf


# ------------------------------------------------------------
# Acceleration
# ------------------------------------------------------------

function calculate_acceleration(
    X,
    dx,
    c,
    λ
)

    N = length(X)

    acceleration = zeros(N)

    for i in 2:N-1

        spatial =
            (
                X[i+1]
                - 2.0 * X[i]
                + X[i-1]
            ) / dx^2

        nonlinear =
            λ * X[i]^3

        acceleration[i] =
            c^2 * spatial - nonlinear

    end

    # Fixed endpoints
    acceleration[1] = 0.0
    acceleration[end] = 0.0

    return acceleration

end


# ------------------------------------------------------------
# Energy
# ------------------------------------------------------------

function calculate_energy(
    X,
    V,
    dx,
    c,
    λ
)

    N = length(X)

    total = 0.0

    for i in 1:N-1

        # Midpoint velocity
        V_mid =
            0.5 * (V[i] + V[i+1])

        # Spatial derivative
        X_gradient =
            (X[i+1] - X[i]) / dx

        density =
            0.5 * V_mid^2 +
            0.5 * c^2 * X_gradient^2 +
            (λ / 4.0) *
            (
                0.5 *
                (X[i]^4 + X[i+1]^4)
            )

        total += density * dx

    end

    return total

end


# ------------------------------------------------------------
# Simulation
# ------------------------------------------------------------

function simulate(
    N,
    T,
    λ
)

    L = 1.0
    c = 1.0

    dx =
        L / (N - 1)

    CFL = 0.4

    dt =
        CFL * dx / c

    num_steps =
        Int(floor(T / dt))

    x =
        collect(
            range(
                0.0,
                L,
                length=N
            )
        )

    # --------------------------------------------------------
    # Initial displacement
    # --------------------------------------------------------

    X =
        sin.(π .* x ./ L) .+
        0.5 .* sin.(2π .* x ./ L) .+
        0.25 .* sin.(3π .* x ./ L)

    # Initial velocity
    V =
        zeros(N)

    # Fixed boundaries
    X[1] = 0.0
    X[end] = 0.0

    V[1] = 0.0
    V[end] = 0.0

    # --------------------------------------------------------
    # Energy history
    # --------------------------------------------------------

    energy =
        zeros(num_steps + 1)

    energy[1] =
        calculate_energy(
            X,
            V,
            dx,
            c,
            λ
        )

    # --------------------------------------------------------
    # Initial acceleration
    # --------------------------------------------------------

    acceleration =
        calculate_acceleration(
            X,
            dx,
            c,
            λ
        )

    # --------------------------------------------------------
    # Velocity Verlet
    # --------------------------------------------------------

    for step in 1:num_steps

        # Position update

        X =
            X .+
            dt .* V .+
            0.5 .* dt^2 .* acceleration

        X[1] = 0.0
        X[end] = 0.0

        # New acceleration

        new_acceleration =
            calculate_acceleration(
                X,
                dx,
                c,
                λ
            )

        # Velocity update

        V =
            V .+
            0.5 .* dt .*
            (
                acceleration
                +
                new_acceleration
            )

        V[1] = 0.0
        V[end] = 0.0

        # Store new acceleration

        acceleration =
            new_acceleration

        # Energy

        energy[step + 1] =
            calculate_energy(
                X,
                V,
                dx,
                c,
                λ
            )

    end

    return energy, dt

end


# ============================================================
# Main
# ============================================================

function main()

    println()
    println("==============================================")
    println(" CORRECTED NONLINEAR STRING SOLVER")
    println("==============================================")
    println()

    N = 401
    T = 20.0
    λ = 0.1

    @printf(
        "Nonlinear coupling λ: %.6f\n",
        λ
    )

    @printf(
        "Grid points N:        %d\n",
        N
    )

    @printf(
        "Simulation time T:    %.4f\n",
        T
    )

    println()

    energy, dt =
        simulate(
            N,
            T,
            λ
        )

    E0 =
        energy[1]

    Ef =
        energy[end]

    Emin =
        minimum(energy)

    Emax =
        maximum(energy)

    final_drift =
        abs(Ef - E0) / E0

    total_range =
        (Emax - Emin) / E0

    println("----------------------------------------------")
    println(" ENERGY VALIDATION")
    println("----------------------------------------------")

    @printf(
        "Initial energy:        %.12e\n",
        E0
    )

    @printf(
        "Final energy:          %.12e\n",
        Ef
    )

    @printf(
        "Relative final drift:  %.12e\n",
        final_drift
    )

    @printf(
        "Relative energy range: %.12e\n",
        total_range
    )

    println()

    if final_drift < 1e-4

        println(
            "PASS: Energy drift is below 0.01%."
        )

    elseif final_drift < 1e-3

        println(
            "PASS: Energy drift is below 0.1%."
        )

    else

        println(
            "FAIL: Energy drift is too large."
        )

    end

    println()
    println("==============================================")
    println(" Corrected nonlinear simulation completed.")
    println("==============================================")
    println()

end


main()