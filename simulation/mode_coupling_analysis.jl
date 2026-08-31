using DelimitedFiles
using FFTW
using Printf
using Statistics

# ============================================================
# NONLINEAR MODE COUPLING ANALYSIS
# ============================================================
#
# Model:
#       X_tt = X_ss - λ X^3
#
# Boundary conditions:
#       X(0,t) = X(1,t) = 0
#
# Purpose:
# Determine whether the nonlinear shift of mode n=3
# is associated with energy transfer between spatial modes.
#
# ============================================================

println()
println("==============================================")
println(" NONLINEAR MODE COUPLING ANALYSIS")
println("==============================================")
println()

# ------------------------------------------------------------
# PARAMETERS
# ------------------------------------------------------------

const N = 401
const T = 160.0
const dt = 0.0025

const λ_values = [0.0, 0.25, 0.5, 0.75, 1.0]
const A_values = [0.5, 1.0, 1.5]

const MAX_MODE = 8

const σ = range(0.0, 1.0, length=N)
const dx = 1.0 / (N - 1)

const ω_ref = [n * π for n in 1:MAX_MODE]

println("Grid points N:        ", N)
println("Simulation time T:    ", T)
println("Time step dt:         ", dt)
println("Spatial step Δσ:      ", dx)
println("Tracked modes:        ", MAX_MODE)
println()

println("Reference frequencies:")
for n in 1:MAX_MODE
    @printf("n = %d       ω = %.10f\n", n, ω_ref[n])
end

# ------------------------------------------------------------
# INITIAL CONDITIONS
# ------------------------------------------------------------

function initial_displacement(A)
    return A .* sin.(π .* σ)
end

function initial_velocity()
    return zeros(Float64, N)
end

# ------------------------------------------------------------
# SECOND SPATIAL DERIVATIVE
# ------------------------------------------------------------

function second_derivative(X)
    Xss = zeros(Float64, N)

    @inbounds for i in 2:N-1
        Xss[i] =
            (X[i+1] - 2.0*X[i] + X[i-1]) / dx^2
    end

    Xss[1] = 0.0
    Xss[N] = 0.0

    return Xss
end

# ------------------------------------------------------------
# ACCELERATION
# ------------------------------------------------------------

function acceleration(X, λ)
    return second_derivative(X) .- λ .* X.^3
end

# ------------------------------------------------------------
# TIME INTEGRATION
# Velocity-Verlet / leapfrog style integration
# ------------------------------------------------------------

function simulate(A, λ)

    Nt = Int(round(T / dt)) + 1

    X = initial_displacement(A)
    V = initial_velocity()

    # Number of stored samples.
    # We store every step because the FFT needs a regular time grid.
    X_history = Matrix{Float64}(undef, N, Nt)

    X_history[:, 1] .= X

    acc = acceleration(X, λ)

    for k in 2:Nt

        # Position update
        X_new = X .+ V .* dt .+ 0.5 .* acc .* dt^2

        # Enforce boundary conditions
        X_new[1] = 0.0
        X_new[N] = 0.0

        # New acceleration
        acc_new = acceleration(X_new, λ)

        # Velocity update
        V_new = V .+ 0.5 .* (acc .+ acc_new) .* dt

        # Update
        X = X_new
        V = V_new
        acc = acc_new

        X_history[:, k] .= X
    end

    return X_history
end

# ------------------------------------------------------------
# PROJECT SPATIAL FIELD ONTO SIN(nπσ)
# ------------------------------------------------------------

function modal_coordinates(X_history)

    Nt = size(X_history, 2)

    q = Matrix{Float64}(undef, MAX_MODE, Nt)

    for n in 1:MAX_MODE

        basis = sin.(n .* π .* σ)

        # Discrete approximation of:
        #
        # q_n(t) = 2 ∫ X(σ,t) sin(nπσ)dσ
        #
        for k in 1:Nt
            q[n, k] =
                2.0 * sum(X_history[:, k] .* basis) * dx
        end
    end

    return q
end

# ------------------------------------------------------------
# FFT FREQUENCY ANALYSIS
# ------------------------------------------------------------

function spectrum(signal)

    Nt = length(signal)

    # Remove DC component
    x = signal .- mean(signal)

    # Hann window
    window = 0.5 .- 0.5 .* cos.(2π .* (0:Nt-1) ./ (Nt-1))

    xw = x .* window

    F = fft(xw)

    freqs = 2π .* (0:Nt-1) ./ (Nt * dt)

    half = 1:div(Nt, 2)

    amplitudes = abs.(F[half])

    frequencies = freqs[half]

    return frequencies, amplitudes
end

# ------------------------------------------------------------
# FIND STRONGEST PEAK NEAR EXPECTED MODE FREQUENCY
# ------------------------------------------------------------

function find_peak_near(freqs, amps, target)

    # Search within ±1.5 frequency units.
    width = 1.5

    indices = findall(
        f -> abs(f - target) <= width,
        freqs
    )

    if isempty(indices)
        return NaN, NaN
    end

    local_amps = amps[indices]

    j = argmax(local_amps)

    idx = indices[j]

    return freqs[idx], amps[idx]
end

# ------------------------------------------------------------
# MODAL ENERGY
# ------------------------------------------------------------

function modal_energy(qn, ω)

    # Approximate oscillator energy:
    #
    # E = 1/2 (qdot^2 + ω^2 q^2)

    Nt = length(qn)

    qdot = zeros(Float64, Nt)

    for k in 2:Nt-1
        qdot[k] =
            (qn[k+1] - qn[k-1]) / (2dt)
    end

    qdot[1] =
        (qn[2] - qn[1]) / dt

    qdot[Nt] =
        (qn[Nt] - qn[Nt-1]) / dt

    E = 0.5 .* (qdot.^2 .+ ω^2 .* qn.^2)

    return E
end

# ------------------------------------------------------------
# OUTPUT DIRECTORY
# ------------------------------------------------------------

mkpath("data")

output_file = "data/mode_coupling_analysis.csv"

open(output_file, "w") do io

    println(
        io,
        "lambda,amplitude,mode,reference_frequency," *
        "dominant_frequency,frequency_shift," *
        "relative_peak_amplitude,mean_energy," *
        "max_energy,energy_variation"
    )

    total = length(λ_values) * length(A_values)

    counter = 0

    for λ in λ_values

        println()
        println("----------------------------------------------")
        @printf("λ = %.2f\n", λ)
        println("----------------------------------------------")

        for A in A_values

            counter += 1

            @printf(
                "\n[%2d/%2d] Running λ = %.2f, A = %.2f ...\n",
                counter,
                total,
                λ,
                A
            )

            # ------------------------------------------------
            # SIMULATE
            # ------------------------------------------------

            X_history = simulate(A, λ)

            # ------------------------------------------------
            # MODAL DECOMPOSITION
            # ------------------------------------------------

            q = modal_coordinates(X_history)

            # ------------------------------------------------
            # ANALYZE EACH MODE
            # ------------------------------------------------

            peak_frequencies = zeros(Float64, MAX_MODE)
            peak_amplitudes = zeros(Float64, MAX_MODE)
            mean_energies = zeros(Float64, MAX_MODE)
            max_energies = zeros(Float64, MAX_MODE)
            energy_variations = zeros(Float64, MAX_MODE)

            for n in 1:MAX_MODE

                freqs, amps = spectrum(q[n, :])

                peakω, peakA =
                    find_peak_near(
                        freqs,
                        amps,
                        ω_ref[n]
                    )

                peak_frequencies[n] = peakω
                peak_amplitudes[n] = peakA

                E =
                    modal_energy(
                        q[n, :],
                        ω_ref[n]
                    )

                meanE = mean(E)
                maxE = maximum(E)

                # Relative temporal variation
                variation =
                    meanE == 0.0 ?
                    0.0 :
                    (maxE - minimum(E)) / meanE

                mean_energies[n] = meanE
                max_energies[n] = maxE
                energy_variations[n] = variation
            end

            # ------------------------------------------------
            # NORMALIZE PEAK AMPLITUDES
            # ------------------------------------------------

            maximum_peak =
                maximum(
                    peak_amplitudes[
                        isfinite.(peak_amplitudes)
                    ]
                )

            relative_amplitudes =
                peak_amplitudes ./ maximum_peak

            # ------------------------------------------------
            # PRINT RESULTS
            # ------------------------------------------------

            println()
            println(
                "Mode       ω_peak        Shift        " *
                "Relative peak"
            )
            println(
                "--------------------------------------------------"
            )

            for n in 1:MAX_MODE

                shift =
                    peak_frequencies[n] - ω_ref[n]

                @printf(
                    "%3d     %11.7f   %+10.6f     %10.6f\n",
                    n,
                    peak_frequencies[n],
                    shift,
                    relative_amplitudes[n]
                )
            end

            # ------------------------------------------------
            # SAVE
            # ------------------------------------------------

            for n in 1:MAX_MODE

                shift =
                    peak_frequencies[n] - ω_ref[n]

                println(
                    io,
                    @sprintf(
                        "%.4f,%.4f,%d,%.10f,%.10f,%.10f,%.10e,%.10e,%.10e,%.10e",
                        λ,
                        A,
                        n,
                        ω_ref[n],
                        peak_frequencies[n],
                        shift,
                        relative_amplitudes[n],
                        mean_energies[n],
                        max_energies[n],
                        energy_variations[n]
                    )
                )
            end

            # ------------------------------------------------
            # ENERGY FRACTIONS
            # ------------------------------------------------

            total_energy =
                sum(mean_energies)

            println()
            println("Mean modal energy fractions:")

            for n in 1:MAX_MODE

                fraction =
                    total_energy == 0.0 ?
                    0.0 :
                    mean_energies[n] / total_energy

                @printf(
                    "Mode %d: %.6f\n",
                    n,
                    fraction
                )
            end

        end
    end
end

# ------------------------------------------------------------
# SUMMARY
# ------------------------------------------------------------

println()
println("----------------------------------------------")
println(" ANALYSIS COMPLETE")
println("----------------------------------------------")
println()

println("Results saved to:")
println(output_file)

println()
println("The CSV contains:")
println("  • dominant frequency of each mode")
println("  • nonlinear frequency shift")
println("  • relative spectral amplitude")
println("  • mean modal energy")
println("  • maximum modal energy")
println("  • modal energy variation")

println()
println("IMPORTANT:")
println("A frequency shift alone does not establish")
println("a new particle.")
println()
println("The purpose of this experiment is to determine")
println("whether the candidate mode remains isolated or")
println("exchanges energy with other spatial modes.")
println()

println("==============================================")
println(" Mode coupling analysis completed.")
println("==============================================")
println()