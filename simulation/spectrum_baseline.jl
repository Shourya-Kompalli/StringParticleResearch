# ============================================================
# StringParticleResearch
# Phase I — Task 2
#
# Multi-Mode Baseline Spectrum
#
# Purpose:
# Numerically measure the excitation frequencies of a
# fixed-end string and compare them with the analytical
# spectrum.
# ============================================================

using Printf
using FFTW
using Statistics


# ------------------------------------------------------------
# String simulation
# ------------------------------------------------------------

function simulate_string(N, T)

    # --------------------------------------------------------
    # Physical parameters
    # --------------------------------------------------------

    L = 1.0
    c = 1.0

    # --------------------------------------------------------
    # Numerical parameters
    # --------------------------------------------------------

    dx = L / (N - 1)

    CFL = 0.5
    dt = CFL * dx / c

    num_steps = Int(floor(T / dt))

    x = collect(range(0.0, L, length=N))

    # --------------------------------------------------------
    # Multi-mode initial condition
    #
    # X(x,0) =
    #
    # sin(pi*x/L)
    # + 0.5 sin(2*pi*x/L)
    # + 0.25 sin(3*pi*x/L)
    # --------------------------------------------------------

    X =
        sin.(π .* x ./ L) .+
        0.5 .* sin.(2π .* x ./ L) .+
        0.25 .* sin.(3π .* x ./ L)

    # Initial velocity
    V = zeros(N)

    # Fixed endpoints
    X[1] = 0.0
    X[end] = 0.0

    # --------------------------------------------------------
    # Store displacement at the midpoint
    #
    # This gives us a time series for Fourier analysis.
    # --------------------------------------------------------

    midpoint = Int(ceil(N / 2))

    signal = zeros(num_steps + 1)

    signal[1] = X[midpoint]

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

    signal[2] = X_new[midpoint]

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

        signal[step + 1] = X_new[midpoint]

    end

    return signal, dt, x

end


# ------------------------------------------------------------
# Discrete Fourier transform
# ------------------------------------------------------------

function calculate_spectrum(signal, dt)

    N = length(signal)

    # Remove DC component
    signal_centered =
        signal .- mean(signal)

    # Fourier transform
    spectrum =
        abs.(fft(signal_centered))

    # Frequencies in cycles/time
    frequencies =
        (0:N-1) ./ (N * dt)

    # Convert to angular frequency
    angular_frequencies =
        2π .* frequencies

    return angular_frequencies, spectrum

end


# ============================================================
# Main experiment
# ============================================================

function main()

    println()
    println("==============================================")
    println(" MULTI-MODE BASELINE SPECTRUM")
    println("==============================================")
    println()

    N = 401
    T = 20.0

    signal, dt, x =
        simulate_string(N, T)

    ω, spectrum =
        calculate_spectrum(signal, dt)

    # --------------------------------------------------------
    # Only analyze positive frequencies
    # --------------------------------------------------------

    half =
        Int(floor(length(ω) / 2))

    ω_positive =
        ω[1:half]

    spectrum_positive =
        spectrum[1:half]

    # --------------------------------------------------------
    # Find strongest spectral peaks
    # --------------------------------------------------------

    peak_indices =
        sortperm(
            spectrum_positive,
            rev=true
        )

    println("Strongest spectral components:")
    println()

    println(
        "Rank        ω_numeric"
    )

    println("-" ^ 30)

    count = 0

    for index in peak_indices

        # Ignore the zero-frequency component
        if ω_positive[index] > 0.1

            @printf(
                "%4d        %.8f\n",
                count + 1,
                ω_positive[index]
            )

            count += 1

            if count == 10
                break
            end
        end
    end

    # --------------------------------------------------------
    # Analytical frequencies
    # --------------------------------------------------------

    L = 1.0
    c = 1.0

    println()
    println("Analytical frequencies:")
    println()

    for n in 1:5

        ω_exact =
            n * π * c / L

        @printf(
            "n = %d       ω = %.8f\n",
            n,
            ω_exact
        )

    end

    println()
    println("==============================================")
    println(" Spectrum calculation completed.")
    println("==============================================")
    println()

end


main()