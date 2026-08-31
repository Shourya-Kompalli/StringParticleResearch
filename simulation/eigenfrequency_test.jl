using FFTW
using Statistics
using DelimitedFiles
using Printf

# ============================================================
# NONLINEAR EIGENFREQUENCY TEST
# Equation:
#     X_tt = X_ss - λ X^3
#
# Boundary conditions:
#     X(0,t) = X(1,t) = 0
#
# Initial condition:
#     X(σ,0) = A sin(3πσ)
#     X_t(σ,0) = 0
#
# Purpose:
# Determine whether the nonlinear candidate near 3ω₁
# behaves like a reproducible nonlinear eigenfrequency.
# ============================================================

const OUTPUT_FILE = joinpath("data", "eigenfrequency_results.csv")

# ------------------------------------------------------------
# Parameters
# ------------------------------------------------------------

const N = 401
const T = 160.0
const DT = 0.0025

const LAMBDA_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0]
const AMPLITUDE_VALUES = [0.5, 1.0, 1.5]

const TARGET_MODE = 3

# ------------------------------------------------------------
# Reference frequencies
# ------------------------------------------------------------

function reference_frequency(n)
    return n * π
end

# ------------------------------------------------------------
# Initial condition
# ------------------------------------------------------------

function initial_displacement(x, amplitude)
    return amplitude .* sin.(3π .* x)
end

# ------------------------------------------------------------
# Spatial second derivative
# ------------------------------------------------------------

function laplacian(u, dx)
    Nloc = length(u)
    acc = zeros(Float64, Nloc)

    @inbounds for i in 2:(Nloc-1)
        acc[i] = (u[i+1] - 2.0*u[i] + u[i-1]) / dx^2
    end

    # Fixed boundaries
    acc[1] = 0.0
    acc[end] = 0.0

    return acc
end

# ------------------------------------------------------------
# Nonlinear acceleration
# ------------------------------------------------------------

function acceleration(u, λ, dx)
    return laplacian(u, dx) .- λ .* u.^3
end

# ------------------------------------------------------------
# Symplectic velocity-Verlet integrator
# ------------------------------------------------------------

function simulate(λ, amplitude)

    x = collect(range(0.0, 1.0, length=N))
    dx = x[2] - x[1]

    u = initial_displacement(x, amplitude)
    v = zeros(Float64, N)

    u[1] = 0.0
    u[end] = 0.0
    v[1] = 0.0
    v[end] = 0.0

    nsteps = Int(round(T / DT))

    # Sample every few steps to keep memory reasonable
    sample_every = 4
    nsamples = div(nsteps, sample_every) + 1

    signal = zeros(Float64, nsamples)
    times = zeros(Float64, nsamples)

    sample_index = 1

    # Project onto mode 3 at each sampled time
    mode_shape = sin.(3π .* x)

    signal[sample_index] =
        (2.0 / (N-1)) * sum(u .* mode_shape)

    times[sample_index] = 0.0

    a = acceleration(u, λ, dx)

    for step in 1:nsteps

        # Half velocity update
        v .+= 0.5 .* DT .* a

        # Position update
        u .+= DT .* v

        # Boundary conditions
        u[1] = 0.0
        u[end] = 0.0

        # New acceleration
        a = acceleration(u, λ, dx)

        # Second half velocity update
        v .+= 0.5 .* DT .* a

        # Boundary velocity
        v[1] = 0.0
        v[end] = 0.0

        if step % sample_every == 0
            sample_index += 1

            signal[sample_index] =
                (2.0 / (N-1)) * sum(u .* mode_shape)

            times[sample_index] = step * DT
        end
    end

    return times, signal
end

# ------------------------------------------------------------
# Frequency spectrum
# ------------------------------------------------------------

function calculate_spectrum(signal, dt)

    n = length(signal)

    # Remove mean to suppress DC component
    centered = signal .- mean(signal)

    # Hann window
    window = 0.5 .- 0.5 .* cos.(2π .* (0:n-1) ./ (n-1))

    weighted = centered .* window

    spectrum = abs.(rfft(weighted))

    frequencies = 2π .* rfftfreq(n, 1.0 / dt)

    return frequencies, spectrum
end

# ------------------------------------------------------------
# Find dominant frequency
# ------------------------------------------------------------

function dominant_frequency(signal, dt)

    frequencies, spectrum = calculate_spectrum(signal, dt)

    # Ignore DC
    start_index = 2

    if length(spectrum) < start_index
        return NaN, NaN
    end

    local_spectrum = spectrum[start_index:end]

    index = argmax(local_spectrum)

    actual_index = index + start_index - 1

    ω = frequencies[actual_index]

    amplitude = spectrum[actual_index]

    return ω, amplitude
end

# ------------------------------------------------------------
# Find frequency close to target
# ------------------------------------------------------------

function frequency_near_target(signal, dt, target)

    frequencies, spectrum = calculate_spectrum(signal, dt)

    # Search within ±0.5 rad/s
    lower = target - 0.5
    upper = target + 0.5

    valid = findall(
        (frequencies .>= lower) .&
        (frequencies .<= upper)
    )

    if isempty(valid)
        return NaN, NaN
    end

    local_index = argmax(spectrum[valid])
    index = valid[local_index]

    return frequencies[index], spectrum[index]
end

# ------------------------------------------------------------
# Main experiment
# ------------------------------------------------------------

function main()

    println()
    println("==============================================")
    println(" NONLINEAR EIGENFREQUENCY TEST")
    println("==============================================")
    println()

    println("Grid points N:        ", N)
    println("Simulation time T:    ", @sprintf("%.1f", T))
    println("Time step dt:         ", @sprintf("%.4f", DT))

    dx = 1.0 / (N - 1)

    println("Spatial step Δσ:      ", @sprintf("%.4f", dx))
    println()

    println("Testing λ values:     ", LAMBDA_VALUES)
    println("Testing amplitudes:   ", AMPLITUDE_VALUES)
    println()

    ω1 = reference_frequency(1)
    ω3 = reference_frequency(3)
    three_ω1 = 3.0 * ω1

    println("----------------------------------------------")
    println("REFERENCE FREQUENCIES")
    println("----------------------------------------------")

    println(@sprintf("ω₁ = %.10f", ω1))
    println(@sprintf("ω₃ = %.10f", ω3))
    println(@sprintf("3ω₁ = %.10f", three_ω1))

    println()

    total = length(LAMBDA_VALUES) * length(AMPLITUDE_VALUES)

    println("----------------------------------------------")
    println("RUNNING EIGENFREQUENCY EXPERIMENT")
    println("----------------------------------------------")
    println()

    println("Total simulations: ", total)
    println()

    # Results matrix
    results = Vector{Vector{Any}}()

    push!(
        results,
        Any[
            "lambda",
            "amplitude",
            "dominant_frequency",
            "near_target_frequency",
            "frequency_shift",
            "relative_shift",
            "spectral_amplitude"
        ]
    )

    counter = 0

    for λ in LAMBDA_VALUES

        for amplitude in AMPLITUDE_VALUES

            counter += 1

            println(
                @sprintf(
                    "[%3d/%3d] λ = %.2f, A = %.2f ...",
                    counter,
                    total,
                    λ,
                    amplitude
                )
            )

            times, signal = simulate(λ, amplitude)

            sample_dt = times[2] - times[1]

            dominant_ω, dominant_amp =
                dominant_frequency(signal, sample_dt)

            near_ω, near_amp =
                frequency_near_target(
                    signal,
                    sample_dt,
                    ω3
                )

            # The frequency relevant to the candidate is
            # the peak near ω₃ = 3π.
            candidate_ω = near_ω

            shift = candidate_ω - ω3

            relative_shift = shift / ω3

            println(
                @sprintf(
                    "      candidate ω = %.10f",
                    candidate_ω
                )
            )

            println(
                @sprintf(
                    "      shift       = %+.10e",
                    shift
                )
            )

            println(
                @sprintf(
                    "      amplitude   = %.6e",
                    near_amp
                )
            )

            println()

            push!(
                results,
                Any[
                    λ,
                    amplitude,
                    dominant_ω,
                    candidate_ω,
                    shift,
                    relative_shift,
                    near_amp
                ]
            )
        end
    end

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    mkpath("data")

    open(OUTPUT_FILE, "w") do io

        for row in results

            println(
                io,
                join(
                    [
                        x isa AbstractString ?
                        x :
                        @sprintf("%.12e", Float64(x))
                        for x in row
                    ],
                    ","
                )
            )

        end

    end

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    numeric_results = results[2:end]

    candidate_frequencies =
        Float64[
            Float64(row[4])
            for row in numeric_results
            if !isnan(Float64(row[4]))
        ]

    shifts =
        Float64[
            Float64(row[5])
            for row in numeric_results
            if !isnan(Float64(row[5]))
        ]

    println("----------------------------------------------")
    println("SUMMARY")
    println("----------------------------------------------")

    if !isempty(candidate_frequencies)

        println(
            @sprintf(
                "Mean candidate frequency: %.10f",
                mean(candidate_frequencies)
            )
        )

        println(
            @sprintf(
                "Frequency standard deviation: %.10f",
                std(candidate_frequencies)
            )
        )

        println(
            @sprintf(
                "Minimum candidate frequency: %.10f",
                minimum(candidate_frequencies)
            )
        )

        println(
            @sprintf(
                "Maximum candidate frequency: %.10f",
                maximum(candidate_frequencies)
            )
        )

        println(
            @sprintf(
                "Mean frequency shift: %+.10e",
                mean(shifts)
            )
        )

        println(
            @sprintf(
                "Maximum |frequency shift|: %.10e",
                maximum(abs.(shifts))
            )
        )

    end

    println()

    # --------------------------------------------------------
    # Per-lambda summary
    # --------------------------------------------------------

    println("----------------------------------------------")
    println("DEPENDENCE ON λ")
    println("----------------------------------------------")

    println(
        "λ          Mean ω          Std ω        Shift"
    )

    println(
        "----------------------------------------------------------"
    )

    for λ in LAMBDA_VALUES

        values = Float64[]

        for row in numeric_results

            if isapprox(Float64(row[1]), λ; atol=1e-12)

                ω = Float64(row[4])

                if !isnan(ω)
                    push!(values, ω)
                end
            end
        end

        if !isempty(values)

            m = mean(values)
            s = length(values) > 1 ? std(values) : 0.0

            println(
                @sprintf(
                    "%.2f      %.10f      %.10f    %+.10e",
                    λ,
                    m,
                    s,
                    m - ω3
                )
            )
        end
    end

    println()

    # --------------------------------------------------------
    # Per-amplitude summary
    # --------------------------------------------------------

    println("----------------------------------------------")
    println("DEPENDENCE ON INITIAL AMPLITUDE")
    println("----------------------------------------------")

    println(
        "A          Mean ω          Std ω        Shift"
    )

    println(
        "----------------------------------------------------------"
    )

    for amplitude in AMPLITUDE_VALUES

        values = Float64[]

        for row in numeric_results

            if isapprox(Float64(row[2]), amplitude; atol=1e-12)

                ω = Float64(row[4])

                if !isnan(ω)
                    push!(values, ω)
                end
            end
        end

        if !isempty(values)

            m = mean(values)
            s = length(values) > 1 ? std(values) : 0.0

            println(
                @sprintf(
                    "%.2f      %.10f      %.10f    %+.10e",
                    amplitude,
                    m,
                    s,
                    m - ω3
                )
            )
        end
    end

    println()

    # --------------------------------------------------------
    # Final interpretation
    # --------------------------------------------------------

    println("----------------------------------------------")
    println("INTERPRETATION")
    println("----------------------------------------------")

    println(
        "The experiment measures the nonlinear frequency"
    )

    println(
        "associated with the mode-3 spectral candidate."
    )

    println()

    println(
        "A systematic frequency shift with λ or amplitude"
    )

    println(
        "would indicate nonlinear renormalization of the"
    )

    println(
        "mode frequency."
    )

    println()

    println(
        "This test does NOT establish a new particle."
    )

    println(
        "It tests whether the candidate behaves like a"
    )

    println(
        "well-defined nonlinear eigenfrequency."
    )

    println()

    println("----------------------------------------------")
    println("Results saved to:")
    println(OUTPUT_FILE)
    println("----------------------------------------------")

    println()

    println("==============================================")
    println(" Eigenfrequency test completed.")
    println("==============================================")
end

# ------------------------------------------------------------
# Run
# ------------------------------------------------------------

main()