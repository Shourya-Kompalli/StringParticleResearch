using FFTW
using Printf
using Statistics
using DelimitedFiles

# ============================================================
# AUTOMATED CANDIDATE SPECTRAL SEARCH
# ============================================================

const L = 1.0
const C = 1.0

const N = 401
const T = 160.0
const CFL = 0.4

const MAX_MODE = 8

# Couplings to investigate
const LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0]

# Number of peaks retained per mode
const MAX_PEAKS = 12

# Minimum relative amplitude for a peak
const PEAK_THRESHOLD = 1.0e-4


# ============================================================
# NONLINEAR EQUATION
#
# X_tt = c² X_ss - λ X³
#
# Fixed endpoints:
# X(0,t) = X(L,t) = 0
# ============================================================

function acceleration(X, dx, c, λ)

    n = length(X)

    A = zeros(Float64, n)

    for i in 2:n-1

        laplacian =
            (X[i+1] - 2.0*X[i] + X[i-1]) / dx^2

        A[i] =
            c^2 * laplacian -
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

    dx = L / (N - 1)
    dt = CFL * dx / C

    steps = Int(floor(T / dt))

    x = collect(range(0.0, L, length=N))

    X = sin.(π .* x ./ L)

    V = zeros(Float64, N)

    X[1] = 0.0
    X[end] = 0.0

    V[1] = 0.0
    V[end] = 0.0

    A = acceleration(X, dx, C, λ)

    # Reduce storage by recording every few integration steps
    record_every = 4

    recorded_steps =
        Int(floor(steps / record_every)) + 1

    modal_signal =
        zeros(
            Float64,
            MAX_MODE,
            recorded_steps
        )

    # Sine basis
    basis =
        zeros(
            Float64,
            MAX_MODE,
            N
        )

    for n in 1:MAX_MODE

        basis[n, :] =
            sin.(n .* π .* x ./ L)

    end

    record_index = 1

    # --------------------------------------------------------
    # Initial modal amplitudes
    # --------------------------------------------------------

    for n in 1:MAX_MODE

        modal_signal[n, record_index] =
            2.0 * dx *
            sum(X .* basis[n, :])

    end

    # --------------------------------------------------------
    # Time integration
    # --------------------------------------------------------

    for step in 1:steps

        X =
            X +
            dt .* V +
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
            V +
            0.5 .* dt .* (A + A_new)

        V[1] = 0.0
        V[end] = 0.0

        A = A_new

        if step % record_every == 0

            record_index += 1

            for n in 1:MAX_MODE

                modal_signal[n, record_index] =
                    2.0 * dx *
                    sum(X .* basis[n, :])

            end

        end

    end

    return modal_signal, dt * record_every
end


# ============================================================
# SPECTRUM
# ============================================================

function spectrum(signal, dt)

    n = length(signal)

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

    F =
        abs.(fft(weighted))

    half =
        Int(floor(n / 2))

    omega =
        2π .* (
            (0:half-1) ./ (n * dt)
        )

    amplitude =
        F[1:half]

    amplitude[1] = 0.0

    # Preserve absolute amplitude
    absolute_max =
        maximum(amplitude)

    relative =
        if absolute_max > 0
            amplitude ./ absolute_max
        else
            zeros(length(amplitude))
        end

    return omega, amplitude, relative
end


# ============================================================
# PEAK DETECTION
# ============================================================

function detect_peaks(omega, relative)

    candidates =
        Tuple{Float64,Float64}[]

    for i in 2:length(relative)-1

        if relative[i] >
           relative[i-1] &&
           relative[i] >
           relative[i+1] &&
           relative[i] >= PEAK_THRESHOLD

            push!(
                candidates,
                (
                    omega[i],
                    relative[i]
                )
            )

        end

    end

    sort!(
        candidates,
        by = x -> x[2],
        rev = true
    )

    return candidates[1:min(MAX_PEAKS, length(candidates))]
end


# ============================================================
# DISTANCE FROM ORDINARY STRING SPECTRUM
# ============================================================

function nearest_linear_mode(ω)

    n =
        round(Int, ω / π)

    expected =
        n * π

    difference =
        abs(ω - expected)

    return n, expected, difference
end


# ============================================================
# MAIN
# ============================================================

function main()

    println()
    println("==============================================")
    println(" AUTOMATED CANDIDATE SPECTRAL SEARCH")
    println("==============================================")
    println()

    println("Grid points N:        ", N)
    println("Simulation time T:    ", T)
    println("Tracked modes:        ", MAX_MODE)
    println()

    println(
        "Frequency resolution: ",
        @sprintf(
            "%.10f",
            2π / T
        )
    )

    println()

    mkpath("data")

    output =
        "data/candidate_spectrum.csv"

    open(output, "w") do io

        println(
            io,
            "lambda,spatial_mode,peak_rank,omega,"
        *
            "relative_amplitude,nearest_linear_n,"
        *
            "nearest_linear_omega,difference_from_linear"
        )
    end


    # ========================================================
    # COUPLING SCAN
    # ========================================================

    for λ in LAMBDAS

        println()
        println("----------------------------------------------")

        println(
            "λ = ",
            @sprintf("%.2f", λ)
        )

        println("----------------------------------------------")

        modal_signal, dt =
            simulate(λ)

        for mode in 1:MAX_MODE

            signal =
                modal_signal[mode, :]

            omega, absolute, relative =
                spectrum(
                    signal,
                    dt
                )

            peaks =
                detect_peaks(
                    omega,
                    relative
                )

            println()
            println(
                "Mode ",
                mode,
                " — strongest spectral peaks:"
            )

            println(
                "Rank       ω             Relative"
            )

            println(
                "------------------------------------------"
            )

            for (rank, peak) in enumerate(peaks)

                ω =
                    peak[1]

                amp =
                    peak[2]

                n_linear,
                ω_linear,
                difference =
                    nearest_linear_mode(ω)

                @printf(
                    "%4d   %12.7f    %12.6e\n",
                    rank,
                    ω,
                    amp
                )

                open(output, "a") do io

                    @printf(
                        io,
                        "%.5f,%d,%d,%.10f,%.10e,%d,%.10f,%.10f\n",
                        λ,
                        mode,
                        rank,
                        ω,
                        amp,
                        n_linear,
                        ω_linear,
                        difference
                    )

                end

            end
        end
    end


    # ========================================================
    # COMPLETION
    # ========================================================

    println()
    println("==============================================")
    println(" Candidate search completed.")
    println("==============================================")
    println()

    println(
        "Results saved to:"
    )

    println(output)

    println()
    println(
        "IMPORTANT:"
    )

    println(
        "A spectral peak is NOT automatically a new particle."
    )

    println(
        "Candidate states require further stability and"
    )

    println(
        "robustness tests."
    )

    println()
end


main()