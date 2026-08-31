# ============================================================
# StringParticleResearch
# Phase II — Task 3
#
# SINGLE-MODE NONLINEAR SPECTRUM
#
# Initial condition:
#
#     X(σ,0) = sin(πσ)
#     X_t(σ,0) = 0
#
# The purpose is to determine which modes are generated
# dynamically by the nonlinear interaction.
#
# ============================================================

using FFTW
using Printf
using Statistics


# ============================================================
# ACCELERATION
# ============================================================

function acceleration(X, dx, c, λ)

    N = length(X)

    A = zeros(N)

    for i in 2:N-1

        Xxx =
            (
                X[i+1]
                - 2.0 * X[i]
                + X[i-1]
            ) / dx^2

        A[i] =
            c^2 * Xxx -
            λ * X[i]^3

    end

    return A

end


# ============================================================
# SIMULATION
# ============================================================

function simulate(
    N,
    T,
    λ
)

    L = 1.0
    c = 1.0

    dx = L / (N - 1)

    CFL = 0.4

    dt = CFL * dx / c

    steps = Int(floor(T / dt))

    x =
        collect(
            range(
                0.0,
                L,
                length=N
            )
        )

    # --------------------------------------------------------
    # SINGLE INITIAL MODE
    # --------------------------------------------------------

    X =
        sin.(π .* x ./ L)

    V =
        zeros(N)

    X[1] = 0.0
    X[end] = 0.0

    V[1] = 0.0
    V[end] = 0.0

    # --------------------------------------------------------
    # Record displacement at midpoint
    # --------------------------------------------------------

    midpoint =
        Int((N + 1) ÷ 2)

    signal =
        zeros(steps + 1)

    signal[1] =
        X[midpoint]

    # --------------------------------------------------------
    # Initial acceleration
    # --------------------------------------------------------

    A =
        acceleration(
            X,
            dx,
            c,
            λ
        )

    # --------------------------------------------------------
    # Velocity Verlet
    # --------------------------------------------------------

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
                c,
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

        signal[step + 1] =
            X[midpoint]

    end

    return signal, dt

end


# ============================================================
# FOURIER SPECTRUM
# ============================================================

function calculate_spectrum(
    signal,
    dt
)

    N =
        length(signal)

    # Remove mean
    signal =
        signal .-
        mean(signal)

    # Hann window
    window =
        0.5 .-
        0.5 .* cos.(
            2π .* (0:N-1) ./ (N-1)
        )

    weighted =
        signal .* window

    F =
        abs.(
            fft(weighted)
        )

    half =
        Int(floor(N / 2))

    frequency =
        (0:half-1) ./ (N * dt)

    omega =
        2π .* frequency

    amplitude =
        F[1:half]

    amplitude[1] = 0.0

    # Normalize
    maximum_amplitude =
        maximum(amplitude)

    if maximum_amplitude > 0

        amplitude =
            amplitude ./ maximum_amplitude

    end

    return omega, amplitude

end


# ============================================================
# FIND STRONG PEAKS
# ============================================================

function find_peaks(
    omega,
    amplitude
)

    peaks = []

    threshold = 0.005

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
# ANALYZE CASE
# ============================================================

function analyze(
    λ
)

    println()
    println("----------------------------------------------")

    @printf(
        "λ = %.3f\n",
        λ
    )

    println("----------------------------------------------")

    signal, dt =
        simulate(
            401,
            40.0,
            λ
        )

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

    println()

    println(
        "Strongest spectral peaks:"
    )

    println()

    println(
        "Rank          ω              Relative amplitude"
    )

    println(
        "------------------------------------------------"
    )

    number =
        min(
            15,
            length(peaks)
        )

    for i in 1:number

        @printf(
            "%3d      %12.7f       %.8f\n",
            i,
            peaks[i][1],
            peaks[i][2]
        )

    end

end


# ============================================================
# MAIN
# ============================================================

function main()

    println()
    println("==============================================")
    println(" SINGLE-MODE NONLINEAR SPECTRUM")
    println("==============================================")

    println()
    println(
        "Initial condition: X = sin(πσ)"
    )

    println(
        "Initial velocity:   X_t = 0"
    )

    println()
    println(
        "Reference frequencies:"
    )

    for n in 1:10

        @printf(
            "n = %2d     ω = %.8f\n",
            n,
            n * π
        )

    end

    analyze(0.0)

    analyze(0.1)

    analyze(0.5)

    analyze(1.0)

    println()
    println("==============================================")
    println(" Single-mode spectral search completed.")
    println("==============================================")
    println()

end


main()