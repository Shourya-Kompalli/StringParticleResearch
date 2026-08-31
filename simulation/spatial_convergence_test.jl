using Printf
using FFTW
using Statistics

# ============================================================
# SPATIAL CONVERGENCE TEST
# Model: X_tt = X_ss - λ X^3
# ============================================================

const λ = 1.0
const A = 1.5
const MODE = 3

const dt = 0.00125
const T = 80.0

const N_VALUES = [101, 201, 401, 801]

println()
println("==============================================")
println(" SPATIAL CONVERGENCE TEST")
println("==============================================")
println()
println("Model: X_tt = X_ss - λ X^3")
println("Target mode: n = $(MODE)")
println()
println("λ = $(λ)")
println("A = $(A)")
println("dt = $(dt)")
println("T = $(T)")
println()
println("----------------------------------------------")
println("REFERENCE")
println("----------------------------------------------")

ω_ref = MODE * π

@printf("ω_%d = %.12f\n", MODE, ω_ref)

# ------------------------------------------------------------
# Initial conditions
# ------------------------------------------------------------

function initial_displacement(x)
    return A .* sin.(MODE .* π .* x)
end

function initial_velocity(x)
    return zeros(length(x))
end

# ------------------------------------------------------------
# Energy
# ------------------------------------------------------------

function calculate_energy(u, v, dx, λ)
    ux = zeros(length(u))

    ux[1] = (u[2] - u[1]) / dx
    ux[end] = (u[end] - u[end-1]) / dx

    for i in 2:length(u)-1
        ux[i] = (u[i+1] - u[i-1]) / (2dx)
    end

    density = 0.5 .* v.^2 .+
              0.5 .* ux.^2 .+
              (λ / 4.0) .* u.^4

    return sum(density) * dx
end

# ------------------------------------------------------------
# Second spatial derivative
# ------------------------------------------------------------

function laplacian(u, dx)
    result = zeros(length(u))

    for i in 2:length(u)-1
        result[i] = (u[i+1] - 2u[i] + u[i-1]) / dx^2
    end

    return result
end

# ------------------------------------------------------------
# RHS of equation
# ------------------------------------------------------------

function acceleration(u, dx, λ)
    a = laplacian(u, dx)

    for i in 2:length(u)-1
        a[i] -= λ * u[i]^3
    end

    a[1] = 0.0
    a[end] = 0.0

    return a
end

# ------------------------------------------------------------
# Modal projection
# ------------------------------------------------------------

function modal_amplitude(u, x, n)
    basis = sin.(n .* π .* x)

    return 2.0 * sum(u .* basis) / (length(x) - 1)
end

# ------------------------------------------------------------
# Frequency extraction
# ------------------------------------------------------------

function dominant_frequency(signal, dt)
    n = length(signal)

    signal = signal .- mean(signal)

    window = 0.5 .- 0.5 .* cos.(2π .* (0:n-1) ./ (n-1))

    weighted = signal .* window

    spectrum = abs.(rfft(weighted))

    frequencies = (0:length(spectrum)-1) ./ (n * dt)
    angular = 2π .* frequencies

    # Search around expected mode frequency.
    lower = 0.5 * ω_ref
    upper = 1.5 * ω_ref

    valid = findall(i -> lower <= angular[i] <= upper, eachindex(angular))

    if isempty(valid)
        return NaN
    end

    local_index = valid[argmax(spectrum[valid])]

    # Parabolic interpolation around spectral maximum.
    if local_index > 1 && local_index < length(spectrum)

        y1 = spectrum[local_index-1]
        y2 = spectrum[local_index]
        y3 = spectrum[local_index+1]

        denominator = y1 - 2y2 + y3

        if abs(denominator) > 1e-14
            offset = 0.5 * (y1 - y3) / denominator
        else
            offset = 0.0
        end

    else
        offset = 0.0
    end

    bin_width = 2π / (n * dt)

    return angular[local_index] + offset * bin_width
end

# ------------------------------------------------------------
# Single simulation
# ------------------------------------------------------------

function run_simulation(N)

    dx = 1.0 / (N - 1)

    x = collect(range(0.0, 1.0, length=N))

    u = initial_displacement(x)
    v = initial_velocity(x)

    steps = Int(round(T / dt))

    # Sample the modal amplitude.
    sample_interval = max(1, Int(round(0.05 / dt)))

    samples = Int(floor(steps / sample_interval)) + 1

    mode_signal = zeros(samples)

    energy_initial = calculate_energy(u, v, dx, λ)

    energy_max_deviation = 0.0

    sample_counter = 1
    mode_signal[sample_counter] = modal_amplitude(u, x, MODE)

    for step in 1:steps

        # Velocity Verlet / leapfrog-style update.

        a = acceleration(u, dx, λ)

        v_half = v .+ 0.5 .* dt .* a

        u_new = u .+ dt .* v_half

        # Enforce fixed boundaries.
        u_new[1] = 0.0
        u_new[end] = 0.0

        a_new = acceleration(u_new, dx, λ)

        v_new = v_half .+ 0.5 .* dt .* a_new

        v_new[1] = 0.0
        v_new[end] = 0.0

        u = u_new
        v = v_new

        if step % sample_interval == 0

            sample_counter += 1

            if sample_counter <= length(mode_signal)
                mode_signal[sample_counter] =
                    modal_amplitude(u, x, MODE)
            end

            E = calculate_energy(u, v, dx, λ)

            relative_error =
                abs(E - energy_initial) / abs(energy_initial)

            energy_max_deviation =
                max(energy_max_deviation, relative_error)
        end

        if any(!isfinite, u) || any(!isfinite, v)
            return (
                stable = false,
                frequency = NaN,
                shift = NaN,
                energy_drift = NaN,
                max_energy_deviation = NaN
            )
        end
    end

    E_final = calculate_energy(u, v, dx, λ)

    relative_drift =
        (E_final - energy_initial) / energy_initial

    frequency =
        dominant_frequency(mode_signal, dt * sample_interval)

    shift = frequency - ω_ref

    return (
        stable = true,
        frequency = frequency,
        shift = shift,
        energy_drift = relative_drift,
        max_energy_deviation = energy_max_deviation
    )
end

# ------------------------------------------------------------
# Output CSV
# ------------------------------------------------------------

mkpath("data")

csv_file = "data/spatial_convergence_test.csv"

open(csv_file, "w") do io

    println(
        io,
        "N,dx,stable,frequency,frequency_shift," *
        "relative_energy_drift,max_energy_deviation"
    )

    println()
    println("----------------------------------------------")
    println("RUNNING CONVERGENCE EXPERIMENT")
    println("----------------------------------------------")
    println()

    for (index, N) in enumerate(N_VALUES)

        dx = 1.0 / (N - 1)

        @printf(
            "[ %d/%d ] N = %d, dx = %.6f ...\n",
            index,
            length(N_VALUES),
            N,
            dx
        )

        result = run_simulation(N)

        if result.stable

            @printf(
                "      frequency = %.12f\n",
                result.frequency
            )

            @printf(
                "      shift     = %+ .12e\n",
                result.shift
            )

            @printf(
                "      drift     = %+ .12e\n",
                result.energy_drift
            )

            @printf(
                "      max dev   = %.12e\n",
                result.max_energy_deviation
            )

            println()

            println(
                io,
                "$(N),$(dx),YES," *
                "$(result.frequency)," *
                "$(result.shift)," *
                "$(result.energy_drift)," *
                "$(result.max_energy_deviation)"
            )

        else

            println("      UNSTABLE / NaN")
            println()

            println(
                io,
                "$(N),$(dx),NO,NaN,NaN,NaN,NaN"
            )
        end
    end
end

# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

println()
println("----------------------------------------------")
println("SPATIAL CONVERGENCE SUMMARY")
println("----------------------------------------------")
println()

println(
    "N          dx             ω_measured        Δω"
)

println(
    "--------------------------------------------------------------"
)

# Read results back from the CSV.
lines = readlines(csv_file)

for line in lines[2:end]

    fields = split(line, ",")

    N = parse(Int, fields[1])
    dx = parse(Float64, fields[2])
    stable = fields[3]

    if stable == "YES"

        frequency = parse(Float64, fields[4])
        shift = parse(Float64, fields[5])

        @printf(
            "%-10d %-14.8f %-17.12f %+ .8e\n",
            N,
            dx,
            frequency,
            shift
        )

    else

        @printf(
            "%-10d %-14.8f UNSTABLE          UNSTABLE\n",
            N,
            dx
        )
    end
end

println()
println("----------------------------------------------")
println("CONVERGENCE DIAGNOSTIC")
println("----------------------------------------------")
println()

frequencies = Float64[]
Ns_stable = Int[]

for line in lines[2:end]

    fields = split(line, ",")

    if fields[3] == "YES"

        push!(Ns_stable, parse(Int, fields[1]))
        push!(frequencies, parse(Float64, fields[4]))

    end
end

if length(frequencies) >= 2

    println("Successive frequency differences:")

    for i in 2:length(frequencies)

        difference =
            abs(frequencies[i] - frequencies[i-1])

        @printf(
            "N = %d → %d : %.12e\n",
            Ns_stable[i-1],
            Ns_stable[i],
            difference
        )
    end
end

println()
println("----------------------------------------------")
println("REFERENCE COMPARISON")
println("----------------------------------------------")
println()

if !isempty(frequencies)

    final_frequency = frequencies[end]
    final_shift = final_frequency - ω_ref

    @printf(
        "Final measured frequency: %.12f\n",
        final_frequency
    )

    @printf(
        "Reference frequency:      %.12f\n",
        ω_ref
    )

    @printf(
        "Final frequency shift:    %+ .12e\n",
        final_shift
    )
end

println()
println("----------------------------------------------")
println("SCIENTIFIC INTERPRETATION")
println("----------------------------------------------")
println()

println(
    "This test determines whether the measured nonlinear"
)
println(
    "frequency is sensitive to the spatial grid resolution."
)
println()
println(
    "A converged frequency should approach a stable value"
)
println(
    "as N increases and dx decreases."
)
println()
println(
    "If the frequency and energy diagnostics stabilize,"
)
println(
    "the numerical resolution is adequate for subsequent"
)
println(
    "mode-coupling analysis."
)
println()
println(
    "This test does not establish the existence of a"
)
println(
    "new particle."
)

println()
println("Results saved to:")
println(csv_file)

println()
println("==============================================")
println(" Spatial convergence test completed.")
println("==============================================")