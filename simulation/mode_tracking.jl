# ============================================================
# StringParticleResearch
# Phase II — Experiment 8
#
# NONLINEAR MODE TRACKING
#
# Purpose:
# Identify which spatial normal modes contain the nonlinear
# spectral features observed in the high-resolution sweep.
#
# ============================================================

using FFTW
using Printf
using Statistics
using DelimitedFiles


# ============================================================
# PARAMETERS
# ============================================================

const L = 1.0
const C = 1.0

const N = 401
const T = 160.0

const CFL = 0.4

const MAX_MODE = 8


# ============================================================
# ACCELERATION
# ============================================================

function acceleration(X, dx, c, λ)

    Nlocal = length(X)

    A = zeros(Float64, Nlocal)

    for i in 2:Nlocal-1

        spatial =
            (
                X[i+1]
                - 2.0 * X[i]
                + X[i-1]
            ) / dx^2

        A[i] =
            c^2 * spatial -
            λ * X[i]^3

    end

    A[1] = 0.0
    A[end] = 0.0

    return A

end


# ============================================================
# SIMULATION
# ============================================================

function simulate(λ)

    dx =
        L / (N - 1)

    dt =
        CFL * dx / C

    steps =
        Int(floor(T / dt))

    x =
        collect(
            range(
                0.0,
                L,
                length=N
            )
        )

    # Initial state
    X =
        sin.(π .* x ./ L)

    V =
        zeros(Float64, N)

    X[1] = 0.0
    X[end] = 0.0

    V[1] = 0.0
    V[end] = 0.0

    # Number of recorded samples
    #
    # We don't need every time step for the modal spectrum.
    # Recording every 4th step reduces memory substantially.

    record_every = 4

    recorded_steps =
        Int(floor(steps / record_every)) + 1

    mode_signal =
        zeros(
            Float64,
            MAX_MODE,
            recorded_steps
        )

    # Precompute sine basis
    basis =
        zeros(
            Float64,
            MAX_MODE,
            N
        )

    for n in 1:MAX_MODE

        for i in 1:N

            basis[n, i] =
                sin(
                    n * π * x[i] / L
                )

        end

    end

    # --------------------------------------------------------
    # Initial acceleration
    # --------------------------------------------------------

    A =
        acceleration(
            X,
            dx,
            C,
            λ
        )

    record_index = 1

    # Record initial modal amplitudes

    for n in 1:MAX_MODE

        mode_signal[n, record_index] =
            2.0 * dx *
            sum(
                X .* basis[n, :]
            )

    end


    # ========================================================
    # TIME INTEGRATION
    # ========================================================

    for step in 1:steps

        X =
            X .+
            dt .* V .+
            0.5 .* dt^2 .* A

        X[1] = 0.0
        X[end] = 0.0

        A_new =
            acceleration(
                X,
                dx,
                C,
                λ
            )

        V =
            V .+
            0.5 .* dt .* (
                A +
                A_new
            )

        V[1] = 0.0
        V[end] = 0.0

        A =
            A_new


        # ----------------------------------------------------
        # Record modal amplitudes
        # ----------------------------------------------------

        if step % record_every == 0

            record_index += 1

            for n in 1:MAX_MODE

                mode_signal[n, record_index] =
                    2.0 * dx *
                    sum(
                        X .* basis[n, :]
                    )

            end

        end

    end


    return mode_signal, dt * record_every

end


# ============================================================
# MODAL FOURIER SPECTRUM
# ============================================================

function calculate_spectrum(
    signal,
    dt
)

    Nsignal =
        length(signal)

    centered =
        signal .-
        mean(signal)

    window =
        0.5 .-
        0.5 .* cos.(
            2π .* (0:Nsignal-1) /
            (Nsignal - 1)
        )

    weighted =
        centered .* window

    F =
        abs.(
            fft(weighted)
        )

    half =
        Int(floor(Nsignal / 2))

    frequencies =
        (0:half-1) /
        (Nsignal * dt)

    omega =
        2π .* frequencies

    amplitude =
        F[1:half]

    amplitude[1] = 0.0

    max_amp =
        maximum(amplitude)

    if max_amp > 0

        amplitude =
            amplitude ./ max_amp

    end

    return omega, amplitude

end


# ============================================================
# PEAK DETECTION
# ============================================================

function find_peaks(
    omega,
    amplitude;
    threshold=0.001
)

    peaks =
        Tuple{Float64,Float64}[]

    for i in 2:length(amplitude)-1

        if amplitude[i] >
           amplitude[i-1] &&
           amplitude[i] >
           amplitude[i+1] &&
           amplitude[i] >
           threshold

            push!(
                peaks,
                (
                    omega[i],
                    amplitude[i]
                )
            )

        end

    end

    sort!(
        peaks,
        by = p -> p[2],
        rev = true
    )

    return peaks

end


# ============================================================
# MAIN ANALYSIS
# ============================================================

function analyze(λ)

    println()
    println("----------------------------------------------")
    println(
        "λ = ",
        @sprintf("%.2f", λ)
    )
    println("----------------------------------------------")

    modes, dt =
        simulate(λ)

    println()

    println(
        "Mode       Dominant ω       Relative amplitude"
    )

    println(
        "--------------------------------------------------"
    )


    for n in 1:MAX_MODE

        signal =
            modes[n, :]

        omega, amplitude =
            calculate_spectrum(
                signal,
                dt
            )

        peaks =
            find_peaks(
                omega,
                amplitude
            )


        if isempty(peaks)

            @printf(
                "%4d       %12s       %12s\n",
                n,
                "none",
                "none"
            )

        else

            peak =
                peaks[1]

            @printf(
                "%4d       %12.7f       %12.8f\n",
                n,
                peak[1],
                peak[2]
            )

        end

    end

    return modes, dt

end


# ============================================================
# MAIN
# ============================================================

function main()

    println()
    println("==============================================")
    println(" NONLINEAR MODE TRACKING")
    println("==============================================")
    println()

    println(
        "Grid points N:        ",
        N
    )

    println(
        "Simulation time T:    ",
        T
    )

    println(
        "Tracked modes:        ",
        MAX_MODE
    )

    println()

    println(
        "Reference frequencies:"
    )

    for n in 1:MAX_MODE

        @printf(
            "n = %2d       ω = %.8f\n",
            n,
            n * π
        )

    end


    # ========================================================
    # TEST THREE COUPLING VALUES
    # ========================================================

    lambdas =
        [
            0.0,
            0.5,
            1.0
        ]


    # ========================================================
    # OUTPUT FILE
    # ========================================================

    output_file =
        "data/mode_tracking.csv"

    open(
        output_file,
        "w"
    ) do io

        println(
            io,
            "lambda,mode,dominant_omega,relative_amplitude"
        )

    end


    # ========================================================
    # ANALYZE
    # ========================================================

    for λ in lambdas

        modes, dt =
            analyze(λ)


        for n in 1:MAX_MODE

            signal =
                modes[n, :]

            omega, amplitude =
                calculate_spectrum(
                    signal,
                    dt
                )

            peaks =
                find_peaks(
                    omega,
                    amplitude
                )


            if isempty(peaks)

                open(
                    output_file,
                    "a"
                ) do io

                    @printf(
                        io,
                        "%.4f,%d,NaN,NaN\n",
                        λ,
                        n
                    )

                end

            else

                peak =
                    peaks[1]

                open(
                    output_file,
                    "a"
                ) do io

                    @printf(
                        io,
                        "%.4f,%d,%.10f,%.10e\n",
                        λ,
                        n,
                        peak[1],
                        peak[2]
                    )

                end

            end

        end

    end


    println()
    println("----------------------------------------------")

    println(
        "Results saved to:"
    )

    println(
        output_file
    )

    println("----------------------------------------------")

    println()
    println("==============================================")
    println(" Mode tracking completed.")
    println("==============================================")
    println()

end


# ============================================================
# RUN
# ============================================================

main()