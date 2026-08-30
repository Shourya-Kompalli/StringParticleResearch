# ============================================================
# StringParticleResearch
# Phase I — Task 2B
#
# Modal Spectrum Analysis
#
# Purpose:
# Decompose the numerical string solution into its normal
# modes and determine the frequency associated with each mode.
# ============================================================

using Printf
using FFTW
using Statistics


# ------------------------------------------------------------
# Simulate the string and record its spatial configuration
# ------------------------------------------------------------

function simulate_string(N, T)

    # Physical parameters
    L = 1.0
    c = 1.0

    # Numerical parameters
    dx = L / (N - 1)

    CFL = 0.5
    dt = CFL * dx / c

    num_steps = Int(floor(T / dt))

    x = collect(range(0.0, L, length=N))

    # --------------------------------------------------------
    # Multi-mode initial condition
    # --------------------------------------------------------

    X =
        sin.(π .* x ./ L) .+
        0.5 .* sin.(2π .* x ./ L) .+
        0.25 .* sin.(3π .* x ./ L)

    V = zeros(N)

    X[1] = 0.0
    X[end] = 0.0

    # --------------------------------------------------------
    # Mode functions
    # --------------------------------------------------------

    num_modes = 5

    modes = zeros(num_modes, N)

    for n in 1:num_modes

        modes[n, :] =
            sin.(n * π .* x ./ L)

    end

    # --------------------------------------------------------
    # Store modal amplitudes
    # --------------------------------------------------------

    modal_amplitudes =
        zeros(num_modes, num_steps + 1)

    # --------------------------------------------------------
    # Projection function
    #
    # a_n(t) ≈ (2/L) ∫ X(x,t) sin(nπx/L) dx
    #
    # We use the trapezoidal rule numerically.
    # --------------------------------------------------------

    function calculate_modal_amplitudes!(column, X_current)

        for n in 1:num_modes

            integrand =
                X_current .* modes[n, :]

            integral =
                sum(
                    (integrand[1:end-1] +
                     integrand[2:end]) .* 0.5 .* dx
                )

            column[n] =
                (2.0 / L) * integral

        end

    end

    # Initial modal amplitudes
    column = zeros(num_modes)

    calculate_modal_amplitudes!(
        column,
        X
    )

    modal_amplitudes[:, 1] = column

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

    calculate_modal_amplitudes!(
        column,
        X_new
    )

    modal_amplitudes[:, 2] = column

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

        calculate_modal_amplitudes!(
            column,
            X_new
        )

        modal_amplitudes[:, step + 1] =
            column

    end

    return modal_amplitudes, dt

end


# ------------------------------------------------------------
# Determine dominant frequency of a signal
# ------------------------------------------------------------

function dominant_frequency(signal, dt)

    N = length(signal)

    # Remove mean
    centered =
        signal .- mean(signal)

    # Apply Hann window
    window =
        0.5 .-
        0.5 .* cos.(
            2π .* (0:N-1) ./ (N-1)
        )

    windowed =
        centered .* window

    # FFT
    spectrum =
        abs.(fft(windowed))

    # Positive frequencies only
    half =
        Int(floor(N / 2))

    frequencies =
        (0:half-1) ./ (N * dt)

    angular_frequencies =
        2π .* frequencies

    spectrum_positive =
        spectrum[1:half]

    # Ignore DC
    spectrum_positive[1] = 0.0

    peak =
        argmax(spectrum_positive)

    return (
        angular_frequencies[peak],
        spectrum_positive[peak]
    )

end


# ============================================================
# Main experiment
# ============================================================

function main()

    println()
    println("==============================================")
    println(" MODAL SPECTRUM ANALYSIS")
    println("==============================================")
    println()

    N = 401
    T = 40.0

    modal_amplitudes, dt =
        simulate_string(N, T)

    num_modes =
        size(modal_amplitudes, 1)

    println("Mode       Numerical ω       Analytical ω")
    println("-" ^ 50)

    L = 1.0
    c = 1.0

    for n in 1:num_modes

        signal =
            modal_amplitudes[n, :]

        ω_numeric, amplitude =
            dominant_frequency(
                signal,
                dt
            )

        ω_exact =
            n * π * c / L

        @printf(
            "%4d       %14.8f       %14.8f\n",
            n,
            ω_numeric,
            ω_exact
        )

    end

    println()
    println("==============================================")
    println(" Modal spectrum analysis completed.")
    println("==============================================")
    println()

end


main()