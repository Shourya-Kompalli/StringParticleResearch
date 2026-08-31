using Statistics
using Printf
using DelimitedFiles

# ============================================================
# ROBUSTNESS ANALYSIS
# ============================================================
#
# Reads:
#     data/candidate_robustness.csv
#
# Analyzes:
#     1. Overall candidate detection
#     2. Dependence on λ
#     3. Dependence on amplitude
#     4. Dependence on grid resolution
#     5. Dependence on simulation time
#     6. Frequency convergence
#
# ============================================================


# ============================================================
# CSV READER
# ============================================================

function load_results(filename)

    lines = readlines(filename)

    if length(lines) < 2
        error("CSV file contains no simulation results.")
    end

    results = NamedTuple[]

    for line in lines[2:end]

        isempty(strip(line)) && continue

        fields = split(line, ",")

        λ = parse(Float64, fields[1])
        amplitude = parse(Float64, fields[2])
        N = parse(Int, fields[3])
        T = parse(Float64, fields[4])
        dt = parse(Float64, fields[5])
        peak_frequency = parse(Float64, fields[6])
        frequency_shift = parse(Float64, fields[7])
        relative_shift = parse(Float64, fields[8])
        relative_amplitude = parse(Float64, fields[9])
        robust = lowercase(fields[10]) == "true"

        push!(
            results,
            (
                λ = λ,
                amplitude = amplitude,
                N = N,
                T = T,
                dt = dt,
                peak_frequency = peak_frequency,
                frequency_shift = frequency_shift,
                relative_shift = relative_shift,
                relative_amplitude = relative_amplitude,
                robust = robust
            )
        )
    end

    return results
end


# ============================================================
# GROUP FILTER
# ============================================================

function filter_results(results; λ=nothing,
                        amplitude=nothing,
                        N=nothing,
                        T=nothing)

    filtered = NamedTuple[]

    for r in results

        if λ !== nothing && r.λ != λ
            continue
        end

        if amplitude !== nothing && r.amplitude != amplitude
            continue
        end

        if N !== nothing && r.N != N
            continue
        end

        if T !== nothing && r.T != T
            continue
        end

        push!(filtered, r)
    end

    return filtered
end


# ============================================================
# STATISTICS
# ============================================================

function safe_mean(values)

    values = filter(isfinite, values)

    isempty(values) && return NaN

    return mean(values)
end


function safe_std(values)

    values = filter(isfinite, values)

    length(values) < 2 && return NaN

    return std(values)
end


function robustness_fraction(group)

    isempty(group) && return NaN

    return count(r -> r.robust, group) / length(group)
end


# ============================================================
# MAIN
# ============================================================

function main()

    println()
    println("==============================================")
    println(" ROBUSTNESS ANALYSIS")
    println("==============================================")
    println()

    filename = "data/candidate_robustness.csv"

    if !isfile(filename)
        error(
            "Could not find $filename. " *
            "Run candidate_robustness.jl first."
        )
    end

    results = load_results(filename)

    println("Input file:")
    println(filename)

    println()
    println("Total records: ", length(results))

    println()
    println("----------------------------------------------")
    println(" OVERALL RESULTS")
    println("----------------------------------------------")

    robust_count =
        count(r -> r.robust, results)

    total_count =
        length(results)

    fraction =
        robust_count / total_count

    @printf(
        "Robust cases:        %d / %d\n",
        robust_count,
        total_count
    )

    @printf(
        "Robustness fraction:  %.6f\n",
        fraction
    )

    frequencies =
        [
            r.peak_frequency
            for r in results
            if isfinite(r.peak_frequency)
        ]

    shifts =
        [
            r.frequency_shift
            for r in results
            if isfinite(r.frequency_shift)
        ]

    amplitudes =
        [
            r.relative_amplitude
            for r in results
            if isfinite(r.relative_amplitude)
        ]

    if !isempty(frequencies)

        println()

        @printf(
            "Mean candidate frequency: %.10f\n",
            mean(frequencies)
        )

        @printf(
            "Frequency standard deviation: %.10f\n",
            std(frequencies)
        )

        @printf(
            "Minimum candidate frequency: %.10f\n",
            minimum(frequencies)
        )

        @printf(
            "Maximum candidate frequency: %.10f\n",
            maximum(frequencies)
        )

    end

    if !isempty(shifts)

        println()

        @printf(
            "Mean frequency shift: %.10f\n",
            mean(shifts)
        )

        @printf(
            "Maximum |frequency shift|: %.10f\n",
            maximum(abs.(shifts))
        )

    end


    # ========================================================
    # λ ANALYSIS
    # ========================================================

    println()
    println("----------------------------------------------")
    println(" DEPENDENCE ON NONLINEAR COUPLING λ")
    println("----------------------------------------------")

    λ_values =
        sort(unique(r.λ for r in results))

    println()
    println(
        "λ          Mean ω          Std ω        Robust fraction"
    )

    println(
        "----------------------------------------------------------"
    )

    for λ in λ_values

        group =
            filter_results(
                results;
                λ=λ
            )

        f =
            [
                r.peak_frequency
                for r in group
                if isfinite(r.peak_frequency)
            ]

        @printf(
            "%.2f       %12.8f   %12.8f      %.4f\n",
            λ,
            safe_mean(f),
            safe_std(f),
            robustness_fraction(group)
        )
    end


    # ========================================================
    # AMPLITUDE ANALYSIS
    # ========================================================

    println()
    println("----------------------------------------------")
    println(" DEPENDENCE ON INITIAL AMPLITUDE")
    println("----------------------------------------------")

    amplitude_values =
        sort(unique(r.amplitude for r in results))

    println()
    println(
        "Amplitude    Mean ω          Std ω        Robust fraction"
    )

    println(
        "-------------------------------------------------------------"
    )

    for amplitude in amplitude_values

        group =
            filter_results(
                results;
                amplitude=amplitude
            )

        f =
            [
                r.peak_frequency
                for r in group
                if isfinite(r.peak_frequency)
            ]

        @printf(
            "%8.3f     %12.8f   %12.8f      %.4f\n",
            amplitude,
            safe_mean(f),
            safe_std(f),
            robustness_fraction(group)
        )
    end


    # ========================================================
    # GRID RESOLUTION ANALYSIS
    # ========================================================

    println()
    println("----------------------------------------------")
    println(" DEPENDENCE ON SPATIAL RESOLUTION")
    println("----------------------------------------------")

    grid_values =
        sort(unique(r.N for r in results))

    println()
    println(
        "N          Mean ω          Std ω        Robust fraction"
    )

    println(
        "----------------------------------------------------------"
    )

    for N in grid_values

        group =
            filter_results(
                results;
                N=N
            )

        f =
            [
                r.peak_frequency
                for r in group
                if isfinite(r.peak_frequency)
            ]

        @printf(
            "%4d       %12.8f   %12.8f      %.4f\n",
            N,
            safe_mean(f),
            safe_std(f),
            robustness_fraction(group)
        )
    end


    # ========================================================
    # TIME ANALYSIS
    # ========================================================

    println()
    println("----------------------------------------------")
    println(" DEPENDENCE ON SIMULATION TIME")
    println("----------------------------------------------")

    time_values =
        sort(unique(r.T for r in results))

    println()
    println(
        "T          Mean ω          Std ω        Robust fraction"
    )

    println(
        "----------------------------------------------------------"
    )

    for T in time_values

        group =
            filter_results(
                results;
                T=T
            )

        f =
            [
                r.peak_frequency
                for r in group
                if isfinite(r.peak_frequency)
            ]

        @printf(
            "%6.1f     %12.8f   %12.8f      %.4f\n",
            T,
            safe_mean(f),
            safe_std(f),
            robustness_fraction(group)
        )
    end


    # ========================================================
    # BEST-CONSTRAINED CASE
    # ========================================================

    println()
    println("----------------------------------------------")
    println(" HIGH-QUALITY CASES")
    println("----------------------------------------------")

    # Prefer highest resolution and longest simulation time.
    max_N = maximum(grid_values)
    max_T = maximum(time_values)

    high_quality =
        filter_results(
            results;
            N=max_N,
            T=max_T
        )

    if !isempty(high_quality)

        println()
        println(
            "Using N = ",
            max_N,
            " and T = ",
            max_T
        )

        println()

        println(
            "λ       A       ω_candidate       shift          relative"
        )

        println(
            "-------------------------------------------------------------"
        )

        for r in high_quality

            @printf(
                "%.2f    %.2f    %14.9f    %11.7f    %10.6f\n",
                r.λ,
                r.amplitude,
                r.peak_frequency,
                r.frequency_shift,
                r.relative_amplitude
            )

        end

    end


    # ========================================================
    # FREQUENCY RANGE
    # ========================================================

    println()
    println("----------------------------------------------")
    println(" FREQUENCY CONSISTENCY")
    println("----------------------------------------------")

    if !isempty(frequencies)

        μ = mean(frequencies)
        σ = std(frequencies)

        @printf(
            "Mean frequency: %.10f\n",
            μ
        )

        @printf(
            "Standard deviation: %.10f\n",
            σ
        )

        @printf(
            "Coefficient of variation: %.6e\n",
            σ / abs(μ)
        )

    end


    # ========================================================
    # WRITE SUMMARY CSV
    # ========================================================

    mkpath("data")

    output_file =
        "data/robustness_summary.csv"

    open(output_file, "w") do io

        println(
            io,
            "lambda,mean_frequency,std_frequency,robust_fraction"
        )

        for λ in λ_values

            group =
                filter_results(
                    results;
                    λ=λ
                )

            f =
                [
                    r.peak_frequency
                    for r in group
                    if isfinite(r.peak_frequency)
                ]

            @printf(
                io,
                "%.8f,%.12f,%.12f,%.8f\n",
                λ,
                safe_mean(f),
                safe_std(f),
                robustness_fraction(group)
            )
        end
    end


    # ========================================================
    # INTERPRETATION
    # ========================================================

    println()
    println("----------------------------------------------")
    println(" AUTOMATED INTERPRETATION")
    println("----------------------------------------------")

    if fraction >= 0.90

        println(
            "STRONG ROBUSTNESS:"
        )

        println(
            "The candidate survives at least 90% of"
        )

        println(
            "the tested parameter combinations."
        )

    elseif fraction >= 0.70

        println(
            "MODERATE ROBUSTNESS:"
        )

        println(
            "The candidate survives most parameter"
        )

        println(
            "variations, but requires further testing."
        )

    else

        println(
            "WEAK ROBUSTNESS:"
        )

        println(
            "The candidate does not consistently survive"
        )

        println(
            "the tested parameter variations."
        )

    end

    println()
    println(
        "IMPORTANT:"
    )

    println(
        "Robustness does NOT establish a new particle."
    )

    println(
        "It establishes whether the spectral feature"
    )

    println(
        "is numerically reproducible under the"
    )

    println(
        "tested conditions."
    )


    # ========================================================
    # FINISH
    # ========================================================

    println()
    println("----------------------------------------------")
    println("Summary saved to:")
    println(output_file)
    println("----------------------------------------------")

    println()
    println("==============================================")
    println(" Robustness analysis completed.")
    println("==============================================")
    println()

end


main()