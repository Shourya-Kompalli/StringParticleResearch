# ============================================================
# StringParticleResearch
# Phase II — Task 4
#
# NONLINEAR COUPLING SPECTRAL SWEEP
#
# We vary λ and track the dominant spectral peaks.
#
# Initial condition:
#
#     X(σ,0) = sin(πσ)
#     X_t(σ,0) = 0
#
# ============================================================

using FFTW
using Printf
using Statistics
using DelimitedFiles


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

        A[i] =
            c^2 * spatial -
            λ * X[i]^3

    end

    return A

end


# ============================================================
# SIMULATION
# ============================================================

function simulate(N, T, λ)

    L = 1.0
    c = 1.0

    dx =
        L / (N - 1)

    CFL = 0.4

    dt =
        CFL * dx / c

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

    # Single-mode initial condition

    X =
        sin.(π .* x ./ L)

    V =
        zeros(N)

    X[1] = 0.0
    X[end] = 0.0

    V[1] = 0.0
    V[end] = 0.0

    midpoint =
        Int((N + 1) ÷ 2)

    signal =
        zeros(steps + 1)

    signal[1] =
        X[midpoint]

    A =
        acceleration(
            X,
            dx,
            c,
            λ
        )

    # Velocity Verlet

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

function calculate_spectrum(signal, dt)

    N =
        length(signal)

    centered =
        signal .-
        mean(signal)

    window =
        0.5 .-
        0.5 .* cos.(
            2π .* (0:N-1) ./ (N-1)
        )

    weighted =
        centered .* window

    F =
        abs.(
            fft(weighted)
        )

    half =
        Int(floor(N / 2))

    frequencies =
        (0:half-1) ./ (N * dt)

    omega =
        2π .* frequencies

    amplitude =
        F[1:half]

    amplitude[1] = 0.0

    maximum_amplitude =
        maximum(amplitude)

    if maximum_amplitude > 0

        amplitude =
            amplitude ./ maximum_amplitude

    end

    return omega, amplitude

end


# ============================================================
# PEAK DETECTION
# ============================================================

function find_peaks(
    omega,
    amplitude
)

    peaks = []

    threshold = 0.002

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
# MAIN
# ============================================================

function main()

    println()
    println("==============================================")
    println(" NONLINEAR COUPLING SPECTRAL SWEEP")
    println("==============================================")
    println()

    println(
        "Scanning λ from 0.0 to 1.0"
    )

    println()

    # --------------------------------------------------------
    # Output file
    # --------------------------------------------------------

    output_file =
        "data/nonlinear_coupling_sweep.csv"

    open(
        output_file,
        "w"
    ) do io

        println(
            io,
            "lambda,peak1_omega,peak1_amp,peak2_omega,peak2_amp,peak3_omega,peak3_amp"
        )

    end


    # --------------------------------------------------------
    # Coupling values
    # --------------------------------------------------------

    lambdas =
        collect(
            0.0:0.05:1.0
        )


    println(
        " λ          Peak 1 ω        Peak 2 ω        Peak 3 ω"
    )

    println(
        "----------------------------------------------------------"
    )


    # --------------------------------------------------------
    # Scan
    # --------------------------------------------------------

    for λ in lambdas

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


        # ----------------------------------------------------
        # Extract three strongest peaks
        # ----------------------------------------------------

        p1 =
            length(peaks) >= 1 ?
            peaks[1] :
            (NaN, NaN)

        p2 =
            length(peaks) >= 2 ?
            peaks[2] :
            (NaN, NaN)

        p3 =
            length(peaks) >= 3 ?
            peaks[3] :
            (NaN, NaN)


        @printf(
            "%5.2f      %12.7f      %12.7f      %12.7f\n",
            λ,
            p1[1],
            p2[1],
            p3[1]
        )


        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        open(
            output_file,
            "a"
        ) do io

            @printf(
                io,
                "%.4f,%.10f,%.10e,%.10f,%.10e,%.10f,%.10e\n",
                λ,
                p1[1],
                p1[2],
                p2[1],
                p2[2],
                p3[1],
                p3[2]
            )

        end

    end


    println()
    println("----------------------------------------------")

    println(
        "Data saved to:"
    )

    println(
        output_file
    )

    println("----------------------------------------------")

    println()
    println("==============================================")
    println(" Coupling sweep completed.")
    println("==============================================")
    println()

end


main()