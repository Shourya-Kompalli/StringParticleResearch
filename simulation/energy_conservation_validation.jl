using Printf

# ============================================================
# ENERGY CONSERVATION VALIDATION
# ============================================================
#
# Model:
#       X_tt = X_ss - λ X^3
#
# Energy:
#       E = ∫ [ 1/2 X_t^2 + 1/2 X_s^2 + λ/4 X^4 ] ds
#
# Purpose:
#   Validate the numerical integrator by checking energy
#   conservation as the time step is reduced.
#
# ============================================================

println()
println("==============================================")
println(" ENERGY CONSERVATION VALIDATION")
println("==============================================")
println()

# ------------------------------------------------------------
# PARAMETERS
# ------------------------------------------------------------

const N = 401
const σmin = 0.0
const σmax = 1.0

const λ = 1.0
const A = 1.5
const MODE = 3

const dt_values = [0.005, 0.0025, 0.00125, 0.000625]

const T_values = [40.0, 80.0, 160.0, 320.0]

const Δσ = (σmax - σmin) / (N - 1)

σ = collect(range(σmin, σmax, length=N))

println("Model: X_tt = X_ss - λ X^3")
println("Grid points N:        ", N)
println("Spatial step Δσ:      ", @sprintf("%.7f", Δσ))
println("λ:                    ", λ)
println("Initial amplitude A:  ", A)
println("Initial mode:         n = ", MODE)
println()

# ------------------------------------------------------------
# INITIAL CONDITIONS
# ------------------------------------------------------------

function initial_displacement(σ, A, mode)
    return A .* sin.(mode .* π .* σ)
end

function initial_velocity(σ)
    return zeros(length(σ))
end

# ------------------------------------------------------------
# SECOND DERIVATIVE
# ------------------------------------------------------------

function second_derivative(X, Δσ)

    N = length(X)

    Xss = zeros(Float64, N)

    @inbounds for i in 2:N-1
        Xss[i] =
            (X[i+1] - 2.0 * X[i] + X[i-1]) / Δσ^2
    end

    # Dirichlet boundaries
    Xss[1] = 0.0
    Xss[N] = 0.0

    return Xss
end

# ------------------------------------------------------------
# ACCELERATION
# ------------------------------------------------------------

function acceleration(X, λ, Δσ)

    Xss = second_derivative(X, Δσ)

    return Xss .- λ .* X.^3
end

# ------------------------------------------------------------
# ENERGY
# ------------------------------------------------------------

function total_energy(X, V, λ, Δσ)

    Xs = zeros(Float64, length(X))

    @inbounds for i in 2:length(X)-1
        Xs[i] =
            (X[i+1] - X[i-1]) / (2.0 * Δσ)
    end

    # One-sided boundary derivatives
    Xs[1] =
        (X[2] - X[1]) / Δσ

    Xs[end] =
        (X[end] - X[end-1]) / Δσ

    density =
        0.5 .* V.^2 .+
        0.5 .* Xs.^2 .+
        (λ / 4.0) .* X.^4

    # Trapezoidal integration
    E = Δσ * (
        0.5 * density[1] +
        sum(density[2:end-1]) +
        0.5 * density[end]
    )

    return E
end

# ------------------------------------------------------------
# RK4 INTEGRATOR
# ------------------------------------------------------------

function rk4_step(X, V, dt, λ, Δσ)

    # k1
    k1_X = V
    k1_V = acceleration(X, λ, Δσ)

    # k2
    X2 = X .+ 0.5 .* dt .* k1_X
    V2 = V .+ 0.5 .* dt .* k1_V

    k2_X = V2
    k2_V = acceleration(X2, λ, Δσ)

    # k3
    X3 = X .+ 0.5 .* dt .* k2_X
    V3 = V .+ 0.5 .* dt .* k2_V

    k3_X = V3
    k3_V = acceleration(X3, λ, Δσ)

    # k4
    X4 = X .+ dt .* k3_X
    V4 = V .+ dt .* k3_V

    k4_X = V4
    k4_V = acceleration(X4, λ, Δσ)

    X_new =
        X .+
        (dt / 6.0) .* (
            k1_X .+
            2.0 .* k2_X .+
            2.0 .* k3_X .+
            k4_X
        )

    V_new =
        V .+
        (dt / 6.0) .* (
            k1_V .+
            2.0 .* k2_V .+
            2.0 .* k3_V .+
            k4_V
        )

    # Enforce boundary conditions
    X_new[1] = 0.0
    X_new[end] = 0.0

    V_new[1] = 0.0
    V_new[end] = 0.0

    return X_new, V_new
end

# ------------------------------------------------------------
# SINGLE SIMULATION
# ------------------------------------------------------------

function run_simulation(T, dt)

    X = initial_displacement(σ, A, MODE)
    V = initial_velocity(σ)

    E_initial = total_energy(X, V, λ, Δσ)

    steps = Int(round(T / dt))

    max_relative_deviation = 0.0

    E_final = E_initial

    stable = true

    for step in 1:steps

        X, V = rk4_step(X, V, dt, λ, Δσ)

        # Check numerical validity
        if any(!isfinite, X) || any(!isfinite, V)
            stable = false
            E_final = NaN
            max_relative_deviation = NaN
            break
        end

        # Check energy every 100 steps
        if step % 100 == 0 || step == steps

            E = total_energy(X, V, λ, Δσ)

            if !isfinite(E)
                stable = false
                E_final = NaN
                max_relative_deviation = NaN
                break
            end

            relative_error =
                abs(E - E_initial) / abs(E_initial)

            max_relative_deviation =
                max(max_relative_deviation, relative_error)

            E_final = E
        end
    end

    if stable
        relative_drift =
            (E_final - E_initial) / E_initial
    else
        relative_drift = NaN
    end

    return (
        stable = stable,
        E_initial = E_initial,
        E_final = E_final,
        relative_drift = relative_drift,
        max_relative_deviation = max_relative_deviation
    )
end

# ------------------------------------------------------------
# CREATE DATA DIRECTORY
# ------------------------------------------------------------

mkpath("data")

output_file =
    "data/energy_conservation_validation.csv"

open(output_file, "w") do io

    println(
        io,
        "dt,T,stable,E_initial,E_final,relative_drift,max_relative_deviation"
    )

    total_runs =
        length(dt_values) * length(T_values)

    counter = 0

    println("----------------------------------------------")
    println("RUNNING VALIDATION")
    println("----------------------------------------------")
    println()

    println("Total simulations: ", total_runs)
    println()

    for dt in dt_values

        println(
            "dt = ",
            @sprintf("%.6f", dt)
        )

        println("----------------------------------------------")

        for T in T_values

            counter += 1

            print(
                "[ ",
                lpad(counter, 2),
                "/",
                lpad(total_runs, 2),
                " ] T = ",
                @sprintf("%.2f", T),
                " ..."
            )

            result =
                run_simulation(T, dt)

            if result.stable

                println()

                println(
                    "      E_initial = ",
                    @sprintf("%.10e", result.E_initial)
                )

                println(
                    "      E_final   = ",
                    @sprintf("%.10e", result.E_final)
                )

                println(
                    "      drift     = ",
                    @sprintf("%+.6e", result.relative_drift)
                )

                println(
                    "      max dev   = ",
                    @sprintf("%.6e",
                            result.max_relative_deviation)
                )

            else

                println(
                    " UNSTABLE / NaN"
                )

            end

            println(
                io,
                @sprintf(
                    "%.7f,%.1f,%s,%.12e,%.12e,%.12e,%.12e",
                    dt,
                    T,
                    result.stable ? "true" : "false",
                    result.E_initial,
                    result.E_final,
                    result.relative_drift,
                    result.max_relative_deviation
                )
            )
        end

        println()
    end
end

# ------------------------------------------------------------
# READ RESULTS FOR ANALYSIS
# ------------------------------------------------------------

results = NamedTuple[]

open(output_file, "r") do io

    # Skip header
    readline(io)

    for line in eachline(io)

        parts = split(line, ",")

        dt = parse(Float64, parts[1])
        T = parse(Float64, parts[2])
        stable = parts[3] == "true"

        E_initial = parse(Float64, parts[4])
        E_final = parse(Float64, parts[5])
        drift = parse(Float64, parts[6])
        maxdev = parse(Float64, parts[7])

        push!(
            results,
            (
                dt = dt,
                T = T,
                stable = stable,
                E_initial = E_initial,
                E_final = E_final,
                drift = drift,
                maxdev = maxdev
            )
        )
    end
end

# ------------------------------------------------------------
# SUMMARY
# ------------------------------------------------------------

println()
println("----------------------------------------------")
println("ENERGY CONSERVATION SUMMARY")
println("----------------------------------------------")
println()

println(
    "dt          T          stable       relative drift       maximum deviation"
)

println(
    "------------------------------------------------------------------------"
)

for r in results

    if r.stable

        println(
            @sprintf(
                "%.6f   %7.1f      YES       %+.6e          %.6e",
                r.dt,
                r.T,
                r.drift,
                r.maxdev
            )
        )

    else

        println(
            @sprintf(
                "%.6f   %7.1f      NO        unstable             unstable",
                r.dt,
                r.T
            )
        )

    end
end

# ------------------------------------------------------------
# CONVERGENCE ANALYSIS
# ------------------------------------------------------------

println()
println("----------------------------------------------")
println("TIME-STEP CONVERGENCE")
println("----------------------------------------------")
println()

println(
    "For a second-order accurate method, reducing dt by"
)
println(
    "a factor of 2 should reduce the numerical error"
)
println(
    "approximately by a factor of 4."
)
println()

for T in T_values

    println(
        "T = ",
        @sprintf("%.1f", T)
    )

    previous_error = nothing
    previous_dt = nothing

    for dt in dt_values

        matching = [
            r for r in results
            if r.dt == dt &&
               r.T == T &&
               r.stable
        ]

        if !isempty(matching)

            current_error = matching[1].maxdev

            if previous_error !== nothing

                ratio =
                    previous_error / current_error

                println(
                    @sprintf(
                        "  dt %.6f → %.6f : error ratio = %.4f",
                        previous_dt,
                        dt,
                        ratio
                    )
                )
            end

            previous_error = current_error
            previous_dt = dt
        end
    end

    println()
end

# ------------------------------------------------------------
# FIND BEST STABLE RESULT
# ------------------------------------------------------------

stable_results =
    [r for r in results if r.stable]

println()
println("----------------------------------------------")
println("BEST STABLE RESULT")
println("----------------------------------------------")
println()

if isempty(stable_results)

    println(
        "No stable simulation was obtained."
    )

else

    best =
        stable_results[argmin(
            [r.maxdev for r in stable_results]
        )]

    println(
        "dt = ",
        @sprintf("%.6f", best.dt)
    )

    println(
        "T  = ",
        @sprintf("%.1f", best.T)
    )

    println(
        "Maximum relative energy deviation = ",
        @sprintf("%.6e", best.maxdev)
    )

    println(
        "Final relative drift = ",
        @sprintf("%+.6e", best.drift)
    )
end

# ------------------------------------------------------------
# STABILITY CHECK
# ------------------------------------------------------------

println()
println("----------------------------------------------")
println("NUMERICAL STABILITY DIAGNOSTIC")
println("----------------------------------------------")
println()

unstable_count =
    count(r -> !r.stable, results)

println(
    "Unstable simulations: ",
    unstable_count,
    " / ",
    length(results)
)

if unstable_count > 0

    println()
    println(
        "At least one timestep is numerically unstable."
    )

    println(
        "Those runs are excluded from the physical"
    )

    println(
        "interpretation and from the best-result selection."
    )

else

    println()
    println(
        "All tested timesteps remained numerically stable."
    )
end

# ------------------------------------------------------------
# SCIENTIFIC INTERPRETATION
# ------------------------------------------------------------

println()
println("----------------------------------------------")
println("SCIENTIFIC INTERPRETATION")
println("----------------------------------------------")
println()

println(
    "The energy test evaluates whether the numerical"
)
println(
    "integration faithfully preserves the conserved"
)
println(
    "energy of the nonlinear string model."
)
println()

println(
    "A reduction in energy error as dt decreases"
)
println(
    "provides evidence of numerical convergence."
)
println()

println(
    "The result does not establish a new particle."
)
println(
    "It establishes whether the numerical dynamics"
)
println(
    "are sufficiently reliable for subsequent spectral"
)
println(
    "and mode-coupling analysis."
)

println()
println("----------------------------------------------")
println("Results saved to:")
println(output_file)
println("----------------------------------------------")
println()

println("==============================================")
println(" Energy conservation validation completed.")
println("==============================================")