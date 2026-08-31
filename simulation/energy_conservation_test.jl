using Printf
using Statistics

# ============================================================
# ENERGY CONSERVATION TEST
#
# Model:
#     X_tt = X_ss - λ X^3
#
# Boundary conditions:
#     X(0,t) = X(1,t) = 0
#
# Purpose:
# Test whether the numerical evolution approximately
# conserves the Hamiltonian energy.
# ============================================================

const N = 401
const Δσ = 1.0 / (N - 1)

const DT_VALUES = [0.005, 0.0025, 0.00125]
const T_VALUES  = [40.0, 80.0, 160.0, 320.0]

const LAMBDA = 1.0
const AMPLITUDE = 1.5
const MODE = 3

const OUTPUT_FILE = "data/energy_conservation_test.csv"


# ============================================================
# INITIAL CONDITION
# ============================================================

function initial_displacement(N, A, mode)

    X = zeros(Float64, N)

    for j in 1:N
        σ = (j - 1) / (N - 1)
        X[j] = A * sin(mode * π * σ)
    end

    return X
end


# ============================================================
# SPATIAL LAPLACIAN
# ============================================================

function spatial_laplacian(X, Δσ)

    N = length(X)

    Xss = zeros(Float64, N)

    invΔσ2 = 1.0 / (Δσ^2)

    @inbounds for j in 2:N-1

        Xss[j] =
            (X[j+1] - 2.0 * X[j] + X[j-1]) *
            invΔσ2

    end

    Xss[1] = 0.0
    Xss[end] = 0.0

    return Xss
end


# ============================================================
# ACCELERATION
#
# X_tt = X_ss - λX^3
# ============================================================

function acceleration(X, Δσ, λ)

    Xss = spatial_laplacian(X, Δσ)

    A = zeros(Float64, length(X))

    @inbounds for j in eachindex(X)

        A[j] = Xss[j] - λ * X[j]^3

    end

    A[1] = 0.0
    A[end] = 0.0

    return A
end


# ============================================================
# ENERGY
#
# E = ∫ [ 1/2 X_t²
#       + 1/2 X_σ²
#       + λ/4 X⁴ ] dσ
#
# ============================================================

function calculate_energy(X, V, Δσ, λ)

    N = length(X)

    kinetic = 0.0
    gradient = 0.0
    potential = 0.0

    @inbounds for j in 1:N

        kinetic += 0.5 * V[j]^2

        potential +=
            (λ / 4.0) * X[j]^4

    end

    @inbounds for j in 1:N-1

        gradient +=
            0.5 *
            ((X[j+1] - X[j]) / Δσ)^2

    end

    kinetic *= Δσ
    potential *= Δσ
    gradient *= Δσ

    return kinetic + gradient + potential
end


# ============================================================
# VELOCITY-VERLET SIMULATION
# ============================================================

function simulate_energy(
    N,
    Δσ,
    dt,
    T,
    λ,
    A0,
    mode
)

    X = initial_displacement(
        N,
        A0,
        mode
    )

    V = zeros(Float64, N)

    Acc = acceleration(
        X,
        Δσ,
        λ
    )

    E0 = calculate_energy(
        X,
        V,
        Δσ,
        λ
    )

    min_energy = E0
    max_energy = E0

    steps = Int(round(T / dt))

    for step in 1:steps

        # ----------------------------------------------------
        # Position update
        # ----------------------------------------------------

        @inbounds for j in 2:N-1

            X[j] +=
                V[j] * dt +
                0.5 * Acc[j] * dt^2

        end

        X[1] = 0.0
        X[end] = 0.0


        # ----------------------------------------------------
        # New acceleration
        # ----------------------------------------------------

        Acc_new = acceleration(
            X,
            Δσ,
            λ
        )


        # ----------------------------------------------------
        # Velocity update
        # ----------------------------------------------------

        @inbounds for j in 2:N-1

            V[j] +=
                0.5 *
                (Acc[j] + Acc_new[j]) *
                dt

        end

        V[1] = 0.0
        V[end] = 0.0

        Acc = Acc_new


        # ----------------------------------------------------
        # Energy monitoring
        # ----------------------------------------------------

        if step % max(1, steps ÷ 100) == 0

            E = calculate_energy(
                X,
                V,
                Δσ,
                λ
            )

            min_energy = min(
                min_energy,
                E
            )

            max_energy = max(
                max_energy,
                E
            )

        end

    end


    Ef = calculate_energy(
        X,
        V,
        Δσ,
        λ
    )

    final_relative_drift =
        (Ef - E0) / E0

    maximum_relative_deviation =
        max(
            abs(min_energy - E0),
            abs(max_energy - E0)
        ) / E0

    return (
        E0,
        Ef,
        final_relative_drift,
        maximum_relative_deviation
    )

end


# ============================================================
# MAIN EXPERIMENT
# ============================================================

function main()

    mkpath("data")

    println()
    println("==============================================")
    println(" ENERGY CONSERVATION TEST")
    println("==============================================")
    println()

    println("Model: X_tt = X_ss - λ X^3")
    println("Grid points N:        ", N)
    println(
        "Spatial step Δσ:      ",
        @sprintf("%.7f", Δσ)
    )

    println("λ:                    ", LAMBDA)
    println("Initial amplitude A:  ", AMPLITUDE)
    println("Initial mode:         n = ", MODE)

    println()

    println("----------------------------------------------")
    println("ENERGY FUNCTION")
    println("----------------------------------------------")
    println()

    println(
        "E = ∫ [ 1/2 X_t² + 1/2 X_σ² + λ/4 X⁴ ] dσ"
    )

    println()

    println("----------------------------------------------")
    println("TIME-STEP CONVERGENCE")
    println("----------------------------------------------")
    println()


    # --------------------------------------------------------
    # STORAGE
    # --------------------------------------------------------

    results = NamedTuple[]

    total_runs =
        length(DT_VALUES) *
        length(T_VALUES)

    counter = 0


    # --------------------------------------------------------
    # RUN
    # --------------------------------------------------------

    for dt in DT_VALUES

        println()
        println(
            "dt = ",
            @sprintf("%.6f", dt)
        )

        println("----------------------------------------------")

        for T in T_VALUES

            counter += 1

            print(
                "[",
                lpad(counter, 2),
                "/",
                lpad(total_runs, 2),
                "] T = ",
                @sprintf("%.2f", T),
                " ..."
            )

            E0, Ef, drift, maxdev =
                simulate_energy(
                    N,
                    Δσ,
                    dt,
                    T,
                    LAMBDA,
                    AMPLITUDE,
                    MODE
                )

            push!(
                results,
                (
                    dt = dt,
                    T = T,
                    initial_energy = E0,
                    final_energy = Ef,
                    relative_drift = drift,
                    maximum_relative_deviation = maxdev
                )
            )

            println()

            println(
                "      E_initial = ",
                @sprintf("%.12e", E0)
            )

            println(
                "      E_final   = ",
                @sprintf("%.12e", Ef)
            )

            println(
                "      drift     = ",
                @sprintf("%+.6e", drift)
            )

            println(
                "      max dev   = ",
                @sprintf("%.6e", maxdev)
            )

        end
    end


    # ========================================================
    # WRITE CSV
    # ========================================================

    open(OUTPUT_FILE, "w") do io

        println(
            io,
            "dt,T,initial_energy,final_energy,relative_drift,maximum_relative_deviation"
        )

        for r in results

            println(
                io,
                @sprintf(
                    "%.8f,%.4f,%.12e,%.12e,%+.12e,%.12e",
                    r.dt,
                    r.T,
                    r.initial_energy,
                    r.final_energy,
                    r.relative_drift,
                    r.maximum_relative_deviation
                )
            )

        end

    end


    # ========================================================
    # SUMMARY
    # ========================================================

    println()
    println("----------------------------------------------")
    println("ENERGY CONSERVATION SUMMARY")
    println("----------------------------------------------")
    println()

    println(
        "dt          T          relative drift       maximum deviation"
    )

    println(
        "----------------------------------------------------------------"
    )

    for r in results

        println(
            @sprintf(
                "%.6f    %6.1f      %+.6e          %.6e",
                r.dt,
                r.T,
                r.relative_drift,
                r.maximum_relative_deviation
            )
        )

    end


    # ========================================================
    # BEST RESULT
    # ========================================================

    best_index = argmin(
        [
            abs(r.relative_drift)
            for r in results
        ]
    )

    best = results[best_index]

    println()
    println("----------------------------------------------")
    println("BEST ENERGY CONSERVATION")
    println("----------------------------------------------")
    println()

    println(
        "dt = ",
        @sprintf("%.6f", best.dt)
    )

    println(
        "T  = ",
        @sprintf("%.1f", best.T)
    )

    println(
        "Final relative drift = ",
        @sprintf(
            "%+.8e",
            best.relative_drift
        )
    )

    println(
        "Maximum relative deviation = ",
        @sprintf(
            "%.8e",
            best.maximum_relative_deviation
        )
    )


    # ========================================================
    # GLOBAL DIAGNOSTIC
    # ========================================================

    maximum_drift = maximum(
        [
            abs(r.relative_drift)
            for r in results
        ]
    )

    println()
    println("----------------------------------------------")
    println("NUMERICAL DIAGNOSTIC")
    println("----------------------------------------------")
    println()

    println(
        "Maximum absolute final drift: ",
        @sprintf("%.6e", maximum_drift)
    )


    if maximum_drift < 1e-3

        println()
        println("GOOD ENERGY CONSERVATION:")
        println(
            "The numerical evolution shows relative"
        )
        println(
            "energy drift below 0.1% for all tested"
        )
        println(
            "configurations."
        )

    elseif maximum_drift < 1e-2

        println()
        println("MODERATE ENERGY DRIFT:")
        println(
            "The numerical solution remains usable,"
        )
        println(
            "but additional timestep convergence"
        )
        println(
            "testing is recommended."
        )

    else

        println()
        println("SIGNIFICANT ENERGY DRIFT:")
        println(
            "The numerical integration requires"
        )
        println(
            "further investigation before physical"
        )
        println(
            "interpretation."
        )

    end


    # ========================================================
    # SCIENTIFIC NOTE
    # ========================================================

    println()
    println("IMPORTANT:")
    println(
        "Energy conservation does NOT prove"
    )
    println(
        "the existence of a new particle."
    )

    println()
    println(
        "It tests whether the numerical solver"
    )
    println(
        "faithfully evolves the nonlinear system"
    )
    println(
        "without introducing large artificial"
    )
    println(
        "energy errors."
    )

    println()
    println("----------------------------------------------")
    println("Results saved to:")
    println(OUTPUT_FILE)
    println("----------------------------------------------")

    println()
    println("==============================================")
    println(" Energy conservation test completed.")
    println("==============================================")

end


# ============================================================
# RUN
# ============================================================

main()