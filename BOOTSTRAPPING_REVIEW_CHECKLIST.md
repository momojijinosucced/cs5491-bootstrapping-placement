# Bootstrapping Placement: Reproducibility and Evaluation Checklist

This document proposes a review checklist for the CS5491 project on
bootstrapping placement for encrypted Transformer inference. It is an
independent documentation contribution, not a report of completed experiments.
The repository implementation and experiment outputs have not been verified
against this checklist. All items are initially unchecked; mark an item complete
only when supporting evidence is available, and mark inapplicable items N/A with
a short explanation.

## 1. Record the experiment configuration

- [ ] Record the source commit, operating system, CPU/GPU, RAM, thread count,
      language/runtime version, and encryption-library version.
- [ ] Record the model or Transformer block, input shape, sequence length, batch
      size, numerical precision, and dataset or synthetic-input generation method.
- [ ] Record the encryption scheme and all parameters needed to reproduce the
      run. For CKKS, if used, include polynomial modulus degree, coefficient
      modulus chain, and scale configuration.
- [ ] Record random seeds where supported and the exact commands/configuration
      files used. Do not publish secret keys, credentials, or private input data.
- [ ] State whether measurements come from actual encrypted execution, a
      simulator, or an analytical cost model.

## 2. Define a fair placement comparison

- [ ] Describe every compared placement strategy and identify the baseline.
- [ ] List the exact operation boundaries at which each strategy bootstraps;
      a count alone does not specify a placement.
- [ ] Keep model, inputs, encryption parameters, hardware, and thread settings
      fixed across strategies, or explicitly explain any differences.
- [ ] Check each strategy's feasibility under the implementation's supported
      operations and remaining level/noise/precision constraints.
- [ ] Identify infeasible strategies explicitly. Do not treat a failed execution
      as a fast result or silently omit it from the comparison.
- [ ] If placement is chosen using an estimated cost, report both the estimate
      and measured runtime where actual execution is available.

## 3. Check numerical correctness

- [ ] Compare decrypted outputs with a plaintext reference on identical inputs.
- [ ] If encrypted execution uses approximated operations, distinguish a
      matching plaintext approximation reference from the original model.
- [ ] Define error measures and tolerances before comparing strategies. For
      example, report maximum absolute error and RMSE over output elements.
- [ ] Define handling for near-zero reference values if reporting relative error.
- [ ] Include representative inputs and relevant boundary cases supported by
      the implementation; record the input shapes and seeds.
- [ ] Record decryption failures, invalid values, and tolerance violations,
      together with the strategy and configuration that produced them.

## 4. Measure performance consistently

- [ ] Define the timing boundary: state whether setup, key generation,
      encryption, data transfer, and decryption are included.
- [ ] Separate warm-up from measured runs. Use the same documented repetition
      count for each strategy and retain all measured samples.
- [ ] Report end-to-end latency, bootstrap count, and bootstrap time when
      separately measurable. Label unmeasured quantities as unavailable.
- [ ] Report a typical latency and variability, such as median and min/max,
      together with the number of measured runs.
- [ ] Calculate speedup as baseline latency divided by candidate latency using
      the same timing boundary and summary statistic for both.
- [ ] Interpret runtime alongside numerical error and feasibility; fewer
      bootstraps alone do not establish a better overall result.

## 5. Keep a traceable result record

Suggested fields for each configuration:

| Field | What to record |
| --- | --- |
| Run identity | Commit, configuration, seed, and input identifier |
| Strategy | Name and exact bootstrap boundaries |
| Execution type | Actual encrypted run, simulation, or analytical estimate |
| Feasibility | Pass/fail and failure reason where applicable |
| Correctness | Reference definition, error metrics, and tolerance outcome |
| Performance | Timing scope, units, repetitions, latency summary, bootstrap count |
| Evidence | Commands/configuration and paths to logs or raw measurements |

No measurements or performance claims are supplied by this document. Complete
the checklist using the project's actual implementation and collected evidence.
