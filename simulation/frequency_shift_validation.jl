using FFTW
using Statistics
using Printf

# ============================================================
# FREQUENCY SHIFT VALIDATION
# ============================================================
#
# Model:
#
#     X_tt = X_ss - λ X^3
#
# Boundary conditions:
#
#     X(0,t) = X(1,t) = 0
#
# Initial condition:
#
#     X(σ,0)  = A sin(nπσ)
#     X_t(σ,0) = 0
#
# Target spatial mode:
#
#     n = 3
#
# This program compares:
#
#     numerical nonlinear frequency
#
# against a perturbative analytical prediction.
#
# ============================================================


# ------------------------------------------------------------
# GLOBAL PARAMETERS
# ------------------------------------------------------------

const N = 401
const T = 160.0
const DT = 0.0025

const TARGET_MODE = 3

const LAMBDA_VALUES = [
    0.0,
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
    1.00
]

const AMPLITUDE_VALUES = [
    0.50,
    0.75,
    1.00,
    1.25,
    1.50
]

const OUTPUT_FILE =
    joinpath("data", "frequency_shift_validation.csv")


# ------------------------------------------------------------
# REFERENCE FREQUENCY
# ------------------------------------------------------------

function linear_frequency(n)

    return n * π

end


# ------------------------------------------------------------
# ANALYTICAL PERTURBATIVE FREQUENCY
# ------------------------------------------------------------
#
# For a dominant single spatial mode:
#
#     X ≈ A sin(kσ) cos(ωt)
#
# the cubic nonlinearity generates a self-interaction
# correction.
#
# The projection of sin^3(kσ) onto sin(kσ) gives:
#
#     sin^3(kσ)
#       = (3/4)sin(kσ) - (1/4)sin(3kσ)
#
# Therefore the leading single-mode equation is
#
#     q'' + ω0² q + (3/4)λ q³ = 0
#
# which gives the Duffing hardening correction:
#
#     ω ≈ ω0 [1 + (3 λ A²)/(32 ω0²)]
#
# ------------------------------------------------------------

function theoretical_frequency(λ, A, ω0)

    correction =
        1.0 +
        (3.0 * λ * A^2) /
        (32.0 * ω0^2)

    return ω0 * correction

end


# ------------------------------------------------------------
# SPATIAL GRID
# ------------------------------------------------------------

function create_grid()

    x = collect(
        range(
            0.0,
            1.0,
            length = N
        )
    )

    dx = x[2] - x[1]

    return x, dx

end


# ------------------------------------------------------------
# INITIAL CONDITION
# ------------------------------------------------------------

function initial_displacement(x, A)

    return A .* sin.(TARGET_MODE * π .* x)

end


# ------------------------------------------------------------
# LAPLACIAN
# ------------------------------------------------------------

function laplacian(u, dx)

    n = length(u)

    result = zeros(Float64, n)

    @inbounds for i in 2:(n - 1)

        result[i] =
            (
                u[i + 1]
                -
                2.0 * u[i]
                +
                u[i - 1]
            ) / dx^2

    end

    result[1] = 0.0
    result[end] = 0.0

    return result

end


# ------------------------------------------------------------
# ACCELERATION
# ------------------------------------------------------------

function acceleration(u, λ, dx)

    return laplacian(u, dx) .- λ .* u.^3

end


# ------------------------------------------------------------
# NONLINEAR SIMULATION
# ------------------------------------------------------------

function simulate(λ, A)

    x, dx = create_grid()

    u =
        initial_displacement(
            x,
            A
        )

    v =
        zeros(Float64, N)

    # Fixed boundaries

    u[1] = 0.0
    u[end] = 0.0

    v[1] = 0.0
    v[end] = 0.0

    nsteps =
        Int(
            round(
                T / DT
            )
        )

    # Sample every 4 integration steps.

    sample_every = 4

    nsamples =
        div(
            nsteps,
            sample_every
        ) + 1

    signal =
        zeros(Float64, nsamples)

    times =
        zeros(Float64, nsamples)

    # Target mode shape

    mode_shape =
        sin.(
            TARGET_MODE * π .* x
        )

    # Initial modal coordinate

    signal[1] =
        (
            2.0 / (N - 1)
        ) *
        sum(
            u .* mode_shape
        )

    times[1] = 0.0

    a =
        acceleration(
            u,
            λ,
            dx
        )

    sample_index = 1

    for step in 1:nsteps

        # Velocity half-step

        v .+=
            0.5 *
            DT *
            a

        # Position update

        u .+=
            DT .* v

        # Boundary conditions

        u[1] = 0.0
        u[end] = 0.0

        # New acceleration

        a =
            acceleration(
                u,
                λ,
                dx
            )

        # Velocity second half-step

        v .+=
            0.5 *
            DT *
            a

        v[1] = 0.0
        v[end] = 0.0

        # Store modal coordinate

        if step % sample_every == 0

            sample_index += 1

            signal[sample_index] =
                (
                    2.0 / (N - 1)
                ) *
                sum(
                    u .* mode_shape
                )

            times[sample_index] =
                step * DT

        end

    end

    return times, signal

end


# ------------------------------------------------------------
# SPECTRUM
# ------------------------------------------------------------

function calculate_spectrum(signal, dt)

    n =
        length(signal)

    # Remove DC component

    centered =
        signal .-
        mean(signal)

    # Hann window

    window =
        0.5 .-
        0.5 .* cos.(
            2π .* (0:n-1) ./ (n-1)
        )

    weighted =
        centered .* window

    spectrum =
        abs.(
            rfft(weighted)
        )

    frequencies =
        2π .* rfftfreq(
            n,
            1.0 / dt
        )

    return frequencies, spectrum

end


# ------------------------------------------------------------
# NUMERICAL FREQUENCY NEAR TARGET
# ------------------------------------------------------------

function numerical_frequency(
    signal,
    dt,
    target
)

    frequencies, spectrum =
        calculate_spectrum(
            signal,
            dt
        )

    # Search in a broad window around
    # the expected nonlinear mode.

    lower =
        target - 1.0

    upper =
        target + 1.0

    indices =
        findall(
            (frequencies .>= lower)
            .&
            (frequencies .<= upper)
        )

    if isempty(indices)

        return NaN, NaN

    end

    local_spectrum =
        spectrum[indices]

    local_index =
        argmax(
            local_spectrum
        )

    index =
        indices[local_index]

    return (
        frequencies[index],
        spectrum[index]
    )

end


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

function main()

    println()
    println("==============================================")
    println(" FREQUENCY SHIFT VALIDATION")
    println("==============================================")
    println()

    println(
        "Model: X_tt = X_ss - λ X^3"
    )

    println(
        "Target spatial mode: n = ",
        TARGET_MODE
    )

    println(
        "Grid points N:        ",
        N
    )

    println(
        "Simulation time T:    ",
        @sprintf("%.1f", T)
    )

    println(
        "Time step dt:         ",
        @sprintf("%.4f", DT)
    )

    println()

    ω0 =
        linear_frequency(
            TARGET_MODE
        )

    println("----------------------------------------------")
    println("REFERENCE")
    println("----------------------------------------------")

    println(
        @sprintf(
            "ω₃ = %.12f",
            ω0
        )
    )

    println()

    println("----------------------------------------------")
    println("PERTURBATIVE MODEL")
    println("----------------------------------------------")

    println(
        "ω_theory ≈ ω₃ [1 + 3λA²/(32ω₃²)]"
    )

    println()

    total =
        length(LAMBDA_VALUES) *
        length(AMPLITUDE_VALUES)

    println(
        "Total simulations: ",
        total
    )

    println()

    # --------------------------------------------------------
    # CSV RESULTS
    # --------------------------------------------------------

    mkpath("data")

    open(
        OUTPUT_FILE,
        "w"
    ) do io

        println(
            io,
            "lambda,amplitude,omega_linear,omega_numerical,omega_theory,numerical_shift,theoretical_shift,absolute_error,relative_error,relative_numerical_shift"
        )

        counter = 0

        # IMPORTANT:
        # counter is local to main(), so there is
        # no Julia soft-scope problem.

        for λ in LAMBDA_VALUES

            for A in AMPLITUDE_VALUES

                counter += 1

                println(
                    @sprintf(
                        "[%3d/%3d] λ = %.2f, A = %.2f ...",
                        counter,
                        total,
                        λ,
                        A
                    )
                )

                # Numerical simulation

                times, signal =
                    simulate(
                        λ,
                        A
                    )

                sample_dt =
                    times[2] -
                    times[1]

                ω_num, amplitude =
                    numerical_frequency(
                        signal,
                        sample_dt,
                        ω0
                    )

                # Analytical prediction

                ω_theory =
                    theoretical_frequency(
                        λ,
                        A,
                        ω0
                    )

                # Frequency shifts

                numerical_shift =
                    ω_num - ω0

                theoretical_shift =
                    ω_theory - ω0

                # Difference between simulation
                # and theory

                absolute_error =
                    abs(
                        ω_num -
                        ω_theory
                    )

                relative_error =
                    absolute_error /
                    abs(ω_theory)

                relative_numerical_shift =
                    numerical_shift /
                    ω0

                println(
                    @sprintf(
                        "      numerical ω = %.10f",
                        ω_num
                    )
                )

                println(
                    @sprintf(
                        "      theory ω    = %.10f",
                        ω_theory
                    )
                )

                println(
                    @sprintf(
                        "      numerical Δω = %+.6e",
                        numerical_shift
                    )
                )

                println(
                    @sprintf(
                        "      theory Δω    = %+.6e",
                        theoretical_shift
                    )
                )

                println(
                    @sprintf(
                        "      relative error = %.6e",
                        relative_error
                    )
                )

                println()

                # Write CSV row

                println(
                    io,
                    join(
                        [
                            @sprintf("%.12e", λ),
                            @sprintf("%.12e", A),
                            @sprintf("%.12e", ω0),
                            @sprintf("%.12e", ω_num),
                            @sprintf("%.12e", ω_theory),
                            @sprintf("%.12e", numerical_shift),
                            @sprintf("%.12e", theoretical_shift),
                            @sprintf("%.12e", absolute_error),
                            @sprintf("%.12e", relative_error),
                            @sprintf("%.12e", relative_numerical_shift)
                        ],
                        ","
                    )
                )

            end

        end

    end

    # --------------------------------------------------------
    # READ RESULTS BACK FOR SUMMARY
    # --------------------------------------------------------

    numerical_values =
        Float64[]

    theoretical_values =
        Float64[]

    numerical_shifts =
        Float64[]

    theoretical_shifts =
        Float64[]

    errors =
        Float64[]

    open(
        OUTPUT_FILE,
        "r"
    ) do io

        # Skip header

        readline(io)

        for line in eachline(io)

            fields =
                split(
                    line,
                    ","
                )

            ω_num =
                parse(
                    Float64,
                    fields[4]
                )

            ω_theory =
                parse(
                    Float64,
                    fields[5]
                )

            num_shift =
                parse(
                    Float64,
                    fields[6]
                )

            theory_shift =
                parse(
                    Float64,
                    fields[7]
                )

            error =
                parse(
                    Float64,
                    fields[8]
                )

            if !isnan(ω_num)

                push!(
                    numerical_values,
                    ω_num
                )

                push!(
                    theoretical_values,
                    ω_theory
                )

                push!(
                    numerical_shifts,
                    num_shift
                )

                push!(
                    theoretical_shifts,
                    theory_shift
                )

                push!(
                    errors,
                    error
                )

            end

        end

    end

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    println("----------------------------------------------")
    println("VALIDATION SUMMARY")
    println("----------------------------------------------")

    if !isempty(numerical_values)

        mean_num =
            mean(
                numerical_values
            )

        mean_theory =
            mean(
                theoretical_values
            )

        mean_num_shift =
            mean(
                numerical_shifts
            )

        mean_theory_shift =
            mean(
                theoretical_shifts
            )

        mean_error =
            mean(
                errors
            )

        maximum_error =
            maximum(
                errors
            )

        println(
            @sprintf(
                "Mean numerical ω:       %.10f",
                mean_num
            )
        )

        println(
            @sprintf(
                "Mean theoretical ω:     %.10f",
                mean_theory
            )
        )

        println()

        println(
            @sprintf(
                "Mean numerical Δω:      %+.10e",
                mean_num_shift
            )
        )

        println(
            @sprintf(
                "Mean theoretical Δω:    %+.10e",
                mean_theory_shift
            )
        )

        println()

        println(
            @sprintf(
                "Mean absolute error:     %.10e",
                mean_error
            )
        )

        println(
            @sprintf(
                "Maximum absolute error:  %.10e",
                maximum_error
            )
        )

    end

    println()

    # --------------------------------------------------------
    # λ DEPENDENCE
    # --------------------------------------------------------

    println("----------------------------------------------")
    println("λ DEPENDENCE")
    println("----------------------------------------------")

    println(
        "λ          A       ω_num        ω_theory      Δω_num"
    )

    println(
        "----------------------------------------------------------------"
    )

    # Show A = 1.0 as a clean λ scan

    for λ in LAMBDA_VALUES

        A = 1.0

        times, signal =
            simulate(
                λ,
                A
            )

        sample_dt =
            times[2] -
            times[1]

        ω_num, _ =
            numerical_frequency(
                signal,
                sample_dt,
                ω0
            )

        ω_theory =
            theoretical_frequency(
                λ,
                A,
                ω0
            )

        println(
            @sprintf(
                "%.2f       %.2f    %.8f    %.8f    %+.6e",
                λ,
                A,
                ω_num,
                ω_theory,
                ω_num - ω0
            )
        )

    end

    println()

    # --------------------------------------------------------
    # AMPLITUDE DEPENDENCE
    # --------------------------------------------------------

    println("----------------------------------------------")
    println("AMPLITUDE DEPENDENCE")
    println("----------------------------------------------")

    println(
        "λ = 1.00"
    )

    println()

    println(
        "A          ω_num        ω_theory      Δω_num"
    )

    println(
        "----------------------------------------------------------"
    )

    λ = 1.0

    for A in AMPLITUDE_VALUES

        times, signal =
            simulate(
                λ,
                A
            )

        sample_dt =
            times[2] -
            times[1]

        ω_num, _ =
            numerical_frequency(
                signal,
                sample_dt,
                ω0
            )

        ω_theory =
            theoretical_frequency(
                λ,
                A,
                ω0
            )

        println(
            @sprintf(
                "%.2f       %.8f    %.8f    %+.6e",
                A,
                ω_num,
                ω_theory,
                ω_num - ω0
            )
        )

    end

    println()

    # --------------------------------------------------------
    # SCIENTIFIC INTERPRETATION
    # --------------------------------------------------------

    println("----------------------------------------------")
    println("SCIENTIFIC INTERPRETATION")
    println("----------------------------------------------")

    println()

    println(
        "The numerical experiment tests whether the"
    )

    println(
        "mode-3 frequency changes systematically under"
    )

    println(
        "the cubic nonlinear interaction."
    )

    println()

    println(
        "The analytical comparison is a leading-order"
    )

    println(
        "single-mode perturbative prediction."
    )

    println()

    println(
        "Agreement between numerical and theoretical"
    )

    println(
        "frequency shifts supports the interpretation"
    )

    println(
        "of the observed shift as a nonlinear effect."
    )

    println()

    println(
        "Disagreement does not automatically invalidate"
    )

    println(
        "the numerical result; it may indicate higher"
    )

    println(
        "mode coupling or higher-order nonlinear"
    )

    println(
        "corrections."
    )

    println()

    println(
        "IMPORTANT:"
    )

    println(
        "A nonlinear frequency shift is not evidence"
    )

    println(
        "of a new fundamental particle by itself."
    )

    println()

    println("----------------------------------------------")
    println("Results saved to:")
    println(OUTPUT_FILE)
    println("----------------------------------------------")

    println()

    println("==============================================")
    println(" Frequency validation completed.")
    println("==============================================")

end


# ------------------------------------------------------------
# RUN
# ------------------------------------------------------------

main()