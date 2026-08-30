# ============================================================
# StringParticleResearch
# Phase II — Task 2
#
# Nonlinear Excitation Spectrum
#
# Compares the linear string (λ = 0) with the nonlinear
# string (λ > 0) and searches for spectral components.
#
# ============================================================

using Printf
using FFTW
using Statistics


# ============================================================
# ACCELERATION
# ============================================================

function acceleration(X, dx, c, λ)

    N = length(X)

    A = zeros(N)

    for i in 2:N-1

        spatial =
            (
                X[i+1]
                - 2.0 * X[i]
                + X[i-1]
            ) / dx^2

        nonlinear =
            λ * X[i]^3

        A[i] =
            c^2 * spatial -
            nonlinear

    end

    A[1] = 0.0
    A[end] = 0.0

    return A

end


# ============================================================
# MODAL PROJECTION
# ============================================================

function project_modes(
    X,
    x,
    dx,
    L,
    num_modes
)

    amplitudes =
        zeros(num_modes)

    for n in 1:num_modes

        mode =
            sin.(n * π .* x ./ L)

        integrand =
            X .* mode

        integral =
            sum(
                (
                    integrand[1:end-1]
                    +
                    integrand[2:end]
                ) .* 0.5 .* dx
            )

        amplitudes[n] =
            (2.0 / L) * integral

    end

    return amplitudes

end


# ============================================================
# SIMULATION
# ============================================================

function simulate(
    N,
    T,
    λ,
    num_modes
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
    # Initial condition
    # --------------------------------------------------------

    X =
        sin.(π .* x ./ L) .+
        0.5 .* sin.(2π .* x ./ L) .+
        0.25 .* sin.(3π .* x ./ L)

    V =
        zeros(N)

    X[1] = 0.0
    X[end] = 0.0

    # --------------------------------------------------------
    # Store modal amplitudes
    # --------------------------------------------------------

    modes =
        zeros(
            num_modes,
            num_steps + 1
        )

    modes[:, 1] =
        project_modes(
            X,
            x,
            dx,
            L,
            num_modes
        )

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

    for step in 1:num_steps

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

        modes[:, step + 1] =
            project_modes(
                X,
                x,
                dx,
                L,
                num_modes
            )

    end

    return modes, dt

end


# ============================================================
# SPECTRUM
# ============================================================

function spectrum(
    signal,
    dt
)

    N =
        length(signal)

    centered =
        signal .-
        mean(signal)

    # Hann window
    window =
        0.5 .-
        0.5 .* cos.(
            2π .* (0:N-1) ./ (N-1)
        )

    windowed =
        centered .* window

    F =
        abs.(
            fft(windowed)
        )

    half =
        Int(floor(N / 2))

    frequencies =
        (0:half-1) ./ (N * dt)

    ω =
        2π .* frequencies

    F_positive =
        F[1:half]

    F_positive[1] = 0.0

    return ω, F_positive

end


# ============================================================
# FIND PEAKS
# ============================================================

function find_peaks(
    ω,
    S,
    threshold
)

    peaks = []

    for i in 2:length(S)-1

        if S[i] > S[i-1] &&
           S[i] > S[i+1] &&
           S[i] > threshold

            push!(
                peaks,
                (
                    ω[i],
                    S[i]
                )
            )

        end

    end

    sort!(
        peaks,
        by = x -> x[2],
        rev = true
    )

    return peaks

end


# ============================================================
# ANALYZE ONE CASE
# ============================================================

function analyze_case(
    λ,
    label
)

    println()
    println("----------------------------------------------")
    println(label)
    println("----------------------------------------------")

    N = 401
    T = 40.0

    num_modes = 5

    modes, dt =
        simulate(
            N,
            T,
            λ,
            num_modes
        )

    for n in 1:num_modes

        signal =
            modes[n, :]

        ω, S =
            spectrum(
                signal,
                dt
            )

        threshold =
            0.02 *
            maximum(S)

        peaks =
            find_peaks(
                ω,
                S,
                threshold
            )

        println()
        @printf(
            "Mode %d:\n",
            n
        )

        if isempty(peaks)

            println(
                "  No peaks detected."
            )

        else

            number =
                min(
                    length(peaks),
                    8
                )

            for j in 1:number

                ω_peak =
                    peaks[j][1]

                amplitude =
                    peaks[j][2]

                @printf(
                    "  ω = %12.8f    amplitude = %.6e\n",
                    ω_peak,
                    amplitude
                )

            end

        end

    end

end


# ============================================================
# MAIN
# ============================================================

function main()

    println()
    println("==============================================")
    println(" NONLINEAR EXCITATION SPECTRUM")
    println("==============================================")

    println()
    println(
        "Comparing λ = 0 with λ = 0.1"
    )

    # --------------------------------------------------------
    # Linear reference
    # --------------------------------------------------------

    analyze_case(
        0.0,
        "LINEAR REFERENCE (λ = 0)"
    )

    # --------------------------------------------------------
    # Nonlinear case
    # --------------------------------------------------------

    analyze_case(
        0.1,
        "NONLINEAR CASE (λ = 0.1)"
    )

    println()
    println("==============================================")
    println(" Spectral analysis completed.")
    println("==============================================")
    println()

end


main()