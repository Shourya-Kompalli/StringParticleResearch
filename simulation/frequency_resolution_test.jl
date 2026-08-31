using FFTW
using Statistics
using Printf

println()
println("==============================================")
println(" FREQUENCY RESOLUTION CONVERGENCE TEST")
println("==============================================")
println()

# ============================================================
# MODEL PARAMETERS
# ============================================================

const N = 401
const dt = 0.0025

const λ = 1.0
const A = 1.5

const target_mode = 3

const T_values = [40.0, 80.0, 160.0, 320.0, 640.0]

const L = 1.0
const dx = L / (N - 1)

const ω3 = target_mode * π

println("Model: X_tt = X_ss - λ X^3")
println("Target mode: n = ", target_mode)
println()
println("N = ", N)
println("dt = ", dt)
println("λ = ", λ)
println("A = ", A)
println()

println("----------------------------------------------")
println("REFERENCE")
println("----------------------------------------------")

@printf("ω₃ = %.12f\n", ω3)

println()
println("----------------------------------------------")
println("EXPECTED FFT RESOLUTION")
println("----------------------------------------------")

for T in T_values
    Δω = 2π / T
    @printf("T = %6.1f   Δω ≈ %.10f\n", T, Δω)
end

# ============================================================
# INITIAL CONDITION
# ============================================================

function initial_displacement(x)
    return A * sin(target_mode * π * x)
end

function initial_velocity(x)
    return 0.0
end

# ============================================================
# FINITE DIFFERENCE EVOLUTION
# ============================================================

function run_simulation(T)

    steps = Int(round(T / dt))

    x = collect(range(0.0, L, length=N))

    u = zeros(Float64, N)
    v = zeros(Float64, N)

    # Initial condition
    for i in 1:N
        u[i] = initial_displacement(x[i])
        v[i] = initial_velocity(x[i])
    end

    # Enforce boundaries
    u[1] = 0.0
    u[end] = 0.0

    # --------------------------------------------------------
    # We measure the target modal coordinate
    # q₃(t) = integral X(x,t) sin(3πx) dx
    # --------------------------------------------------------

    signal = zeros(Float64, steps + 1)

    function modal_coordinate(u)
        value = 0.0

        for i in 2:N-1
            value += u[i] * sin(target_mode * π * x[i])
        end

        return 2.0 * dx * value
    end

    signal[1] = modal_coordinate(u)

    # --------------------------------------------------------
    # Time integration
    #
    # Leapfrog / central difference scheme
    # --------------------------------------------------------

    acceleration = zeros(Float64, N)

    for i in 2:N-1
        laplacian =
            (u[i+1] - 2.0*u[i] + u[i-1]) / dx^2

        acceleration[i] =
            laplacian - λ * u[i]^3
    end

    u_previous = u - dt .* v + 0.5 .* dt^2 .* acceleration

    for step in 1:steps

        # Calculate acceleration at current time
        for i in 2:N-1

            laplacian =
                (u[i+1] - 2.0*u[i] + u[i-1]) / dx^2

            acceleration[i] =
                laplacian - λ * u[i]^3
        end

        # Central difference update
        u_next = zeros(Float64, N)

        for i in 2:N-1
            u_next[i] =
                2.0*u[i] -
                u_previous[i] +
                dt^2 * acceleration[i]
        end

        # Boundary conditions
        u_next[1] = 0.0
        u_next[end] = 0.0

        u_previous = u
        u = u_next

        signal[step + 1] = modal_coordinate(u)
    end

    return signal
end

# ============================================================
# FREQUENCY ANALYSIS
# ============================================================

function find_frequency(signal, T)

    n = length(signal)

    # Remove DC component
    signal_centered = signal .- mean(signal)

    # Hann window
    window = 0.5 .- 0.5 .* cos.(2π .* (0:n-1) ./ (n-1))

    weighted_signal = signal_centered .* window

    spectrum = abs.(rfft(weighted_signal))

    frequencies =
        2π .* (0:length(spectrum)-1) ./ T

    # Ignore zero frequency
    spectrum[1] = 0.0

    # Search near the expected mode-3 frequency
    search_min = 0.5 * ω3
    search_max = 1.5 * ω3

    candidates = findall(
        f -> search_min <= f <= search_max,
        frequencies
    )

    if isempty(candidates)
        return NaN
    end

    local_spectrum = spectrum[candidates]

    index =
        candidates[argmax(local_spectrum)]

    # --------------------------------------------------------
    # Parabolic interpolation around spectral maximum
    # --------------------------------------------------------

    if index > 1 && index < length(spectrum)

        y1 = log(max(spectrum[index-1], eps()))
        y2 = log(max(spectrum[index], eps()))
        y3 = log(max(spectrum[index+1], eps()))

        denominator =
            y1 - 2.0*y2 + y3

        if abs(denominator) > 1e-14

            δ =
                0.5 *
                (y1 - y3) /
                denominator

            frequency_resolution =
                2π / T

            return frequencies[index] +
                   δ * frequency_resolution
        end
    end

    return frequencies[index]
end

# ============================================================
# RUN EXPERIMENT
# ============================================================

results = []

println()
println("----------------------------------------------")
println("RUNNING CONVERGENCE EXPERIMENT")
println("----------------------------------------------")
println()

for (counter, T) in enumerate(T_values)

    @printf(
        "[ %d/%d ] T = %.1f ...\n",
        counter,
        length(T_values),
        T
    )

    signal = run_simulation(T)

    ω = find_frequency(signal, T)

    shift = ω - ω3

    Δω_resolution = 2π / T

    relative_shift = shift / ω3

    push!(
        results,
        (
            T,
            Δω_resolution,
            ω,
            shift,
            relative_shift
        )
    )

    @printf(
        "      frequency = %.10f\n",
        ω
    )

    @printf(
        "      shift     = %+.10e\n",
        shift
    )

    @printf(
        "      resolution = %.10e\n",
        Δω_resolution
    )

    println()
end

# ============================================================
# SUMMARY
# ============================================================

println("----------------------------------------------")
println("CONVERGENCE SUMMARY")
println("----------------------------------------------")

println()
println(
    "T          Δω_resolution      ω_measured        Δω"
)

println(
    "--------------------------------------------------------------"
)

for result in results

    T,
    resolution,
    ω,
    shift,
    relative_shift = result

    @printf(
        "%6.1f     %.10f       %.10f     %+.10e\n",
        T,
        resolution,
        ω,
        shift
    )
end

# ============================================================
# CONVERGENCE DIAGNOSTIC
# ============================================================

println()
println("----------------------------------------------")
println("CONVERGENCE DIAGNOSTIC")
println("----------------------------------------------")

frequencies = [r[3] for r in results]

if length(frequencies) >= 3

    differences = abs.(diff(frequencies))

    println()
    println("Successive frequency differences:")

    for i in eachindex(differences)

        @printf(
            "T = %.1f → %.1f : %.10e\n",
            T_values[i],
            T_values[i+1],
            differences[i]
        )
    end

    final_difference =
        differences[end]

    final_frequency =
        frequencies[end]

    println()

    @printf(
        "Final measured frequency: %.12f\n",
        final_frequency
    )

    @printf(
        "Reference frequency:      %.12f\n",
        ω3
    )

    @printf(
        "Final shift:              %+.12e\n",
        final_frequency - ω3
    )

    println()

    if final_difference < 1e-3

        println(
            "CONVERGENCE INDICATION:"
        )

        println(
            "The measured frequency is becoming"
        )

        println(
            "stable as simulation time increases."
        )

    else

        println(
            "CONVERGENCE NOT YET ESTABLISHED:"
        )

        println(
            "The measured frequency is still"
        )

        println(
            "changing significantly with T."
        )

    end
end

# ============================================================
# SAVE RESULTS
# ============================================================

mkpath("data")

output_file =
    "data/frequency_resolution_test.csv"

open(output_file, "w") do io

    println(
        io,
        "T,resolution,frequency,shift,relative_shift"
    )

    for result in results

        T,
        resolution,
        ω,
        shift,
        relative_shift = result

        println(
            io,
            "$(T),$(resolution),$(ω),$(shift),$(relative_shift)"
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
println(" Frequency resolution test completed.")
println("==============================================")
println()