# ============================================================
# StringParticleResearch
# Phase II — High Resolution Spectral Sweep
#
# Purpose:
# Determine whether the apparent nonlinear spectral feature
# near omega ~ 9.58 is a physical frequency shift or merely
# an FFT-resolution artifact.
#
# Observation time:
#       T = 160
#
# Frequency resolution:
#       Δω = 2π/T ≈ 0.03927
#
# Coupling values:
#       λ = 0.0, 0.1, ..., 1.0
#
# ============================================================

using FFTW
using Printf
using Statistics
using DelimitedFiles


# ============================================================
# PHYSICAL / NUMERICAL PARAMETERS
# ============================================================

const L = 1.0
const C = 1.0

const N = 401
const T = 160.0

const CFL = 0.4


# ============================================================
# ACCELERATION
# ============================================================

function acceleration(X, dx, c, λ)

    Nlocal = length(X)

    A = zeros(Float64, Nlocal)

    for i in 2:Nlocal-1

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

    # Fixed endpoints
    A[1] = 0.0
    A[end] = 0.0

    return A

end


# ============================================================
# NONLINEAR STRING SIMULATION
# ============================================================

function simulate(N, T, λ)

    dx =
        L / (N - 1)

    dt =
        CFL * dx / C

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

    # --------------------------------------------------------
    # Initial condition
    #
    # X(σ,0) = sin(πσ)
    # X_t(σ,0) = 0
    # --------------------------------------------------------

    X =
        sin.(π .* x ./ L)

    V =
        zeros(Float64, N)

    X[1] = 0.0
    X[end] = 0.0

    V[1] = 0.0
    V[end] = 0.0

    midpoint =
        Int((N + 1) ÷ 2)

    signal =
        zeros(Float64, steps + 1)

    signal[1] =
        X[midpoint]

    A =
        acceleration(
            X,
            dx,
            C,
            λ
        )

    # --------------------------------------------------------
    # Velocity-Verlet integration
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
                C,
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

    Nsignal =
        length(signal)

    # Remove DC component
    centered =
        signal .-
        mean(signal)

    # Hann window
    window =
        0.5 .-
        0.5 .* cos.(
            2π .* (0:Nsignal-1) ./ (Nsignal-1)
        )

    weighted =
        centered .* window

    F =
        abs.(
            fft(weighted)
        )

    half =
        Int(floor(Nsignal / 2))

    frequencies =
        (0:half-1) ./ (Nsignal * dt)

    omega =
        2π .* frequencies

    amplitude =
        F[1:half]

    # Remove DC
    amplitude[1] = 0.0

    maximum_amplitude =
        maximum(amplitude)

    if maximum_amplitude > 0.0

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

    peaks =
        Tuple{Float64,Float64}[]

    # Ignore extremely small numerical features
    threshold = 0.001

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

    # Strongest peaks first
    sort!(
        peaks,
        by = p -> p[2],
        rev = true
    )

    return peaks

end


# ============================================================
# FIND PEAK NEAREST TO A TARGET FREQUENCY
# ============================================================

function nearest_peak(
    peaks,
    target
)

    if isempty(peaks)

        return (
            NaN,
            NaN
        )

    end

    distances =
        [
            abs(p[1] - target)
            for p in peaks
        ]

    index =
        argmin(distances)

    return peaks[index]

end


# ============================================================
# MAIN
# ============================================================

function main()

    println()
    println("==============================================")
    println(" HIGH-RESOLUTION NONLINEAR SPECTRAL SWEEP")
    println("==============================================")
    println()

    println(
        "Grid points N:        ",
        N
    )

    println(
        "Simulation time T:    ",
        T
    )

    println(
        "Expected Δω = 2π/T:   ",
        @sprintf(
            "%.10f",
            2π / T
        )
    )

    println()

    println(
        "Scanning λ = 0.0 to 1.0"
    )

    println()


    # ========================================================
    # OUTPUT FILE
    # ========================================================

    output_file =
        "data/high_resolution_sweep.csv"

    open(
        output_file,
        "w"
    ) do io

        println(
            io,
            "lambda,peak1_omega,peak1_amp,peak2_omega,peak2_amp,peak3_omega,peak3_amp,near_3pi_omega,near_3pi_amp"
        )

    end


    # ========================================================
    # COUPLING VALUES
    # ========================================================

    lambdas =
        collect(
            0.0:0.1:1.0
        )


    println(
        " λ          Peak 1 ω        Peak 2 ω        Peak 3 ω        Near 3π"
    )

    println(
        "--------------------------------------------------------------------------"
    )


    # ========================================================
    # PARAMETER SWEEP
    # ========================================================

    for λ in lambdas

        println(
            "Running λ = ",
            @sprintf(
                "%.2f",
                λ
            ),
            " ..."
        )


        # ----------------------------------------------------
        # Run simulation
        # ----------------------------------------------------

        signal, dt =
            simulate(
                N,
                T,
                λ
            )


        # ----------------------------------------------------
        # Calculate spectrum
        # ----------------------------------------------------

        omega, amplitude =
            calculate_spectrum(
                signal,
                dt
            )


        # ----------------------------------------------------
        # Detect peaks
        # ----------------------------------------------------

        peaks =
            find_peaks(
                omega,
                amplitude
            )


        # ----------------------------------------------------
        # Three strongest peaks
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


        # ----------------------------------------------------
        # Track the region around 3π
        #
        # 3π ≈ 9.42478
        #
        # This lets us distinguish a genuine frequency shift
        # from simply selecting the neighboring FFT bin.
        # ----------------------------------------------------

        near_3pi =
            nearest_peak(
                peaks,
                3π
            )


        @printf(
            "%5.2f      %12.7f      %12.7f      %12.7f      %12.7f\n",
            λ,
            p1[1],
            p2[1],
            p3[1],
            near_3pi[1]
        )


        # ----------------------------------------------------
        # Save raw results
        # ----------------------------------------------------

        open(
            output_file,
            "a"
        ) do io

            @printf(
                io,
                "%.4f,%.10f,%.10e,%.10f,%.10e,%.10f,%.10e,%.10f,%.10e\n",
                λ,
                p1[1],
                p1[2],
                p2[1],
                p2[2],
                p3[1],
                p3[2],
                near_3pi[1],
                near_3pi[2]
            )

        end

    end


    # ========================================================
    # COMPLETION
    # ========================================================

    println()
    println("----------------------------------------------")

    println(
        "Frequency resolution:"
    )

    println(
        @sprintf(
            "Δω = %.10f",
            2π / T
        )
    )

    println()

    println(
        "Data saved to:"
    )

    println(
        output_file
    )

    println("----------------------------------------------")

    println()
    println("==============================================")
    println(" High-resolution sweep completed.")
    println("==============================================")
    println()

end


# ============================================================
# RUN
# ============================================================

main()