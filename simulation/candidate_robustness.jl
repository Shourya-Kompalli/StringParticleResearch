using FFTW
using Statistics
using Printf
using DelimitedFiles

# ============================================================
# CANDIDATE ROBUSTNESS TEST
# ============================================================
#
# Model:
#     X_tt = X_ss - λ X^3
#
# Boundary conditions:
#     X(0,t) = X(1,t) = 0
#
# Target:
#     n = 3 excitation
#
# Purpose:
#     Test whether the candidate spectral state near 3π
#     remains identifiable when numerical and physical
#     parameters are varied.
#
# ============================================================

const L = 1.0
const C = 1.0
const TARGET_MODE = 3
const TARGET_FREQUENCY = TARGET_MODE * π

const LAMBDA_VALUES = [0.25, 0.5, 0.75, 1.0]
const AMPLITUDE_VALUES = [0.5, 1.0, 1.5]
const GRID_VALUES = [201, 401, 801]
const TIME_VALUES = [80.0, 160.0, 320.0]

const DT_FACTOR = 0.5
const PEAK_SEARCH_WIDTH = 1.5
const MIN_RELATIVE_PEAK = 0.005


# ============================================================
# INITIAL CONDITION
# ============================================================

function initial_displacement(x, amplitude)
    return amplitude .* sin.(TARGET_MODE .* π .* x)
end


# ============================================================
# LAPLACIAN
# ============================================================

function laplacian(X, dx)
    result = similar(X)

    result[1] = 0.0
    result[end] = 0.0

    @inbounds for i in 2:length(X)-1
        result[i] =
            (X[i+1] - 2.0*X[i] + X[i-1]) / dx^2
    end

    return result
end


# ============================================================
# ACCELERATION
# ============================================================

function acceleration(X, λ, dx)
    lap = laplacian(X, dx)

    result = similar(X)

    @inbounds for i in 2:length(X)-1
        result[i] = lap[i] - λ * X[i]^3
    end

    result[1] = 0.0
    result[end] = 0.0

    return result
end


# ============================================================
# ENERGY
# ============================================================

function calculate_energy(X, V, λ, dx)
    kinetic = 0.0
    gradient = 0.0
    nonlinear = 0.0

    @inbounds for i in 2:length(X)-1
        kinetic += 0.5 * V[i]^2

        gradient +=
            0.5 * ((X[i+1] - X[i]) / dx)^2

        nonlinear +=
            0.25 * λ * X[i]^4
    end

    return dx * (kinetic + gradient + nonlinear)
end


# ============================================================
# STABLE VELOCITY-VERLET SOLVER
# ============================================================

function simulate(λ, amplitude, N, T)

    dx = L / (N - 1)

    # Stability condition for explicit wave solver.
    dt = DT_FACTOR * dx / C

    steps = Int(floor(T / dt))

    x = collect(range(0.0, L, length=N))

    X = initial_displacement(x, amplitude)

    V = zeros(Float64, N)

    # Enforce boundaries
    X[1] = 0.0
    X[end] = 0.0

    # Acceleration at t = 0
    A = acceleration(X, λ, dx)

    # We do not need to store every spatial point.
    # Store only the target modal coordinate.

    modal_signal = zeros(Float64, steps + 1)

    # Initial modal projection
    basis = sin.(TARGET_MODE .* π .* x)

    norm_basis = dx * sum(basis.^2)

    modal_signal[1] =
        dx * sum(X .* basis) / norm_basis

    # Time integration
    for step in 1:steps

        # Position update
        @inbounds for i in 2:N-1
            X[i] =
                X[i] +
                dt * V[i] +
                0.5 * dt^2 * A[i]
        end

        X[1] = 0.0
        X[end] = 0.0

        # New acceleration
        A_new = acceleration(X, λ, dx)

        # Velocity update
        @inbounds for i in 2:N-1
            V[i] =
                V[i] +
                0.5 * dt * (A[i] + A_new[i])
        end

        V[1] = 0.0
        V[end] = 0.0

        A = A_new

        modal_signal[step + 1] =
            dx * sum(X .* basis) / norm_basis
    end

    return modal_signal, dt
end


# ============================================================
# SPECTRUM
# ============================================================

function calculate_spectrum(signal, dt)

    N = length(signal)

    # Remove DC component
    signal_centered = signal .- mean(signal)

    # Hann window
    window = 0.5 .- 0.5 .* cos.(2π .* (0:N-1) ./ (N-1))

    weighted = signal_centered .* window

    spectrum = abs.(rfft(weighted))

    frequencies =
        2π .* collect(0:length(spectrum)-1) ./ (N * dt)

    return frequencies, spectrum
end


# ============================================================
# FIND CANDIDATE PEAK NEAR TARGET
# ============================================================

function find_target_peak(frequencies, spectrum)

    # Search around the expected nonlinear candidate region.
    lower = TARGET_FREQUENCY - PEAK_SEARCH_WIDTH
    upper = TARGET_FREQUENCY + PEAK_SEARCH_WIDTH

    indices = findall(
        f -> lower <= f <= upper,
        frequencies
    )

    if isempty(indices)
        return NaN, NaN
    end

    local_spectrum = spectrum[indices]

    max_index = argmax(local_spectrum)

    idx = indices[max_index]

    frequency = frequencies[idx]

    amplitude = spectrum[idx]

    return frequency, amplitude
end


# ============================================================
# RELATIVE AMPLITUDE
# ============================================================

function relative_amplitude(spectrum, peak_amplitude)

    maximum_amplitude = maximum(spectrum)

    if maximum_amplitude == 0.0
        return 0.0
    end

    return peak_amplitude / maximum_amplitude
end


# ============================================================
# SINGLE ROBUSTNESS TEST
# ============================================================

function run_test(λ, amplitude, N, T)

    signal, dt =
        simulate(
            λ,
            amplitude,
            N,
            T
        )

    frequencies, spectrum =
        calculate_spectrum(
            signal,
            dt
        )

    peak_frequency, peak_amplitude =
        find_target_peak(
            frequencies,
            spectrum
        )

    relative =
        relative_amplitude(
            spectrum,
            peak_amplitude
        )

    frequency_shift =
        peak_frequency - TARGET_FREQUENCY

    relative_shift =
        frequency_shift / TARGET_FREQUENCY

    robust =
        isfinite(peak_frequency) &&
        relative >= MIN_RELATIVE_PEAK

    return (
        λ = λ,
        amplitude = amplitude,
        N = N,
        T = T,
        dt = dt,
        peak_frequency = peak_frequency,
        frequency_shift = frequency_shift,
        relative_shift = relative_shift,
        relative_amplitude = relative,
        robust = robust
    )
end


# ============================================================
# MAIN EXPERIMENT
# ============================================================

function main()

    println()
    println("==============================================")
    println(" CANDIDATE ROBUSTNESS TEST")
    println("==============================================")
    println()

    println("Model: X_tt = X_ss - λ X^3")
    println("Boundary conditions: X(0,t) = X(1,t) = 0")
    println()

    println("Target mode: n = ", TARGET_MODE)

    @printf(
        "Reference frequency: %.10f\n",
        TARGET_FREQUENCY
    )

    println()

    println("Testing:")
    println("  λ values:        ", LAMBDA_VALUES)
    println("  amplitudes:      ", AMPLITUDE_VALUES)
    println("  grid sizes:      ", GRID_VALUES)
    println("  simulation times: ", TIME_VALUES)

    total =
        length(LAMBDA_VALUES) *
        length(AMPLITUDE_VALUES) *
        length(GRID_VALUES) *
        length(TIME_VALUES)

    println()
    println("Total simulations: ", total)
    println()

    println("----------------------------------------------")

    results = NamedTuple[]

    counter = 0

    for λ in LAMBDA_VALUES

        println()
        @printf("λ = %.2f\n", λ)
        println()

        for amplitude in AMPLITUDE_VALUES

            for N in GRID_VALUES

                for T in TIME_VALUES

                    counter += 1

                    result =
                        run_test(
                            λ,
                            amplitude,
                            N,
                            T
                        )

                    push!(
                        results,
                        result
                    )

                    @printf(
                        "[%3d/%3d] λ=%4.2f  A=%3.1f  N=%4d  T=%5.1f  ω=%11.7f  rel=%9.5f  %s\n",
                        counter,
                        total,
                        λ,
                        amplitude,
                        N,
                        T,
                        result.peak_frequency,
                        result.relative_amplitude,
                        result.robust ? "PASS" : "FAIL"
                    )

                end
            end
        end
    end

    # ========================================================
    # SUMMARY
    # ========================================================

    println()
    println("----------------------------------------------")
    println(" ROBUSTNESS SUMMARY")
    println("----------------------------------------------")

    robust_count =
        count(r -> r.robust, results)

    total_count =
        length(results)

    robustness_fraction =
        robust_count / total_count

    @printf(
        "Robust cases:       %d / %d\n",
        robust_count,
        total_count
    )

    @printf(
        "Robustness fraction: %.4f\n",
        robustness_fraction
    )

    frequencies =
        [
            r.peak_frequency
            for r in results
            if isfinite(r.peak_frequency)
        ]

    if !isempty(frequencies)

        mean_frequency =
            mean(frequencies)

        std_frequency =
            std(frequencies)

        minimum_frequency =
            minimum(frequencies)

        maximum_frequency =
            maximum(frequencies)

        println()

        @printf(
            "Mean candidate ω:    %.10f\n",
            mean_frequency
        )

        @printf(
            "Std. deviation:      %.10f\n",
            std_frequency
        )

        @printf(
            "Minimum candidate ω: %.10f\n",
            minimum_frequency
        )

        @printf(
            "Maximum candidate ω: %.10f\n",
            maximum_frequency
        )

        @printf(
            "Reference ω:         %.10f\n",
            TARGET_FREQUENCY
        )

    end

    # ========================================================
    # SAVE CSV
    # ========================================================

    mkpath("data")

    output_file =
        "data/candidate_robustness.csv"

    open(output_file, "w") do io

        println(
            io,
            "lambda,amplitude,N,T,dt,peak_frequency,frequency_shift,relative_shift,relative_amplitude,robust"
        )

        for r in results

            @printf(
                io,
                "%.8f,%.8f,%d,%.8f,%.12e,%.12f,%.12f,%.12e,%.12e,%s\n",
                r.λ,
                r.amplitude,
                r.N,
                r.T,
                r.dt,
                r.peak_frequency,
                r.frequency_shift,
                r.relative_shift,
                r.relative_amplitude,
                r.robust
            )

        end
    end

    println()
    println("----------------------------------------------")
    println("Results saved to:")
    println(output_file)
    println("----------------------------------------------")

    println()
    println("==============================================")
    println(" Candidate robustness test completed.")
    println("==============================================")
    println()

end


# ============================================================
# RUN
# ============================================================

main()