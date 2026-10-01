# Alignment with the FedSOAR manuscript

This release accompanies **Federated Learning under Client Heterogeneity via Soft Overlap-Aware Relations**, by Wooseok Shin, Janghoon Yang, Zhiqiang Shen, and Jitae Shin. FedSOAR expands to *Federated Learning via Soft Overlap-Aware Relations*. The earlier project name was FedHyDRA; its GitHub URL is retained for compatibility.

The supplied manuscript is prepared for Applied Soft Computing. Neither the citation metadata nor this repository asserts acceptance, publication, a DOI, or independent reproduction of its reported tables.

## What the repository implements

| Manuscript component | Repository support |
|---|---|
| Smoothed label histograms and shared RFF means, Eqs. (3)-(7) | Implemented |
| Maximum-based calibration and projected adaptive JSD-MMD weight, Eqs. (8)-(12) | Implemented |
| Symmetric weighted top-k relation graph, Eqs. (13)-(15) | Implemented over cached population summaries |
| VGAE with weighted message passing and binary-support reconstruction, Eqs. (16)-(18) | Implemented |
| Full-covariance GMM, soft memberships, and WSS, Eqs. (19)-(21) | Implemented |
| Cluster-to-global update, Eqs. (22)-(23) | Implemented with explicit partial-participation scopes; see below |
| Refresh order and default intervals (5, 10, 20) | Implemented |
| CIFAR-100, Tiny-ImageNet, STL-10; random and structured splits | Implemented with the partition conventions below |
| Fixed hybrid, JSD-only, MMD-only, no-VGAE, hard GMM ablations | Implemented; no-VGAE uses direct summary clustering |
| Beta, graph-neighbor, and refresh-cadence sensitivity | Supported through configuration overrides |
| FedAvg and FedProx controls | Implemented |
| FedGCD, KL-FedDis, FedWaD, FedAF, FedDNA; Local-only and Centralized references | Not included as experiment runners |
| Section 5.4 dynamic population, temporal-shift, and OOD scenarios | Protocols documented below; runners not included |

## Aggregation scopes need to be explicit

Equation (20) defines the mixture share using all N clients. Equation (22) also writes its numerator and denominator over N clients, whereas Section 5.1 selects 20 participants from a population of 200. The manuscript does not explicitly state the active-set version of Eq. (22).

This implementation retains the existing `mixture_share_scope: all_clients` convention:

```text
pi_k = (1/N) sum over population i of Gamma_ik
cluster_k = sum over participating i of Gamma_ik Delta_i
            / (sum over participating i of Gamma_ik + epsilon)
global_delta = sum over k of pi_k cluster_k
```

The population memberships and mixture shares are cached between GMM refreshes. Nonparticipating clients retain their most recent summaries. A summary-only bootstrap initializes all clients before the first communication round. This convention is an explicit implementation interpretation, not a detail fully specified by the manuscript equations.

If both sums instead use the same client set, memberships sum to one per client, sample weighting is disabled, and epsilon is zero, the weights cancel:

```text
sum_k [(sum_i Gamma_ik)/N]
      [(sum_i Gamma_ik Delta_i)/(sum_i Gamma_ik)]
  = (1/N) sum_i Delta_i
```

The `participants` scope exposes that invariant and is covered by tests. Full participation with population shares has the same limit. A small positive epsilon only adds cluster-dependent shrinkage; it does not establish a general benefit from overlapping memberships. Any claim connecting the manuscript's ablations to an aggregation implementation needs the actual participant and population scopes and original experiment records.

## Partitions and evaluation conventions

The supplied structured generator allocates numerical class-ID anchor blocks, adjacent overlap, boundary-client mixtures, and deterministic label-preserving color shifts. It uses the manuscript's overlap ratios of 20%, 20%, 30%, and 10%, but these alone do not determine the background/object/animal templates in Tables 1 and 2. The original class mappings and sample-index assignments are not supplied with this repository. See [structured partitions](STRUCTURED_PARTITIONS.md) for the exact generator and saved metadata.

The random partition uses Dirichlet concentration 0.3. Main profiles use 200 clients, 20 participants, 300 rounds, five local epochs, batch size 32, and the stated SGD settings. The implementation chooses MobileNetV2 with width 0.5 and a 512-dimensional penultimate layer; the manuscript specifies a lightweight MobileNet without all architectural details.

For the no-VGAE ablation, Section 5.6 describes direct clustering. The updated profile fits the GMM to the concatenated histograms and RFF summaries from Eq. (15), bypassing graph embedding. This choice of direct input is explicit because the manuscript does not give a separate no-VGAE feature definition. The legacy spectral-embedding option remains available but is not used by that profile.

Balanced accuracy, the 90%-of-final convergence threshold, CoV, and Min/Mean are implemented. By default, stability uses every recorded evaluation point and testing uses the clean held-out split. The simulator measures sequential local computation and server work without network-transfer emulation, so its runtime is not directly comparable to the manuscript's wall-clock values including simulated transfer. Hardware, timing boundaries, feature summaries, sample allocations, and random seeds must accompany comparisons.

## Reported stress protocols

Section 5.4 defines the following scenarios. They are retained here as specifications, not executable claims about the current static-partition runner:

- **Client population variation:** generate 240 client partitions, keep 200 active, and replace 40 active clients every 20 rounds using inactive clients with matched group proportions.
- **Mild temporal shift:** starting at round 101, transfer 5% of each client's local sampling mass to an adjacent group's label pool every 25 rounds, reaching 20% at round 176 while preserving local sample counts.
- **OOD injection:** from round 151, replace 5% of each minibatch for a seed-fixed 20% of clients with resized SVHN images given uniformly sampled in-domain labels. The affected clients, replacement indices, and draws must be reused across compared methods.

Implementing these scenarios also requires preserving changing-client summary caches and defining integer rounding for the minibatch replacement count. No static run, smoke test, or placeholder metric is presented as reproducing these stress results.

## Version and migration

Version 0.2.0 introduces FedSOAR naming, revised manuscript metadata, bilingual documentation, direct-summary no-VGAE ablation, and compatibility checks. Use `fedsoar`, `python -m fedsoar`, `federated.method: fedsoar`, and the `fedsoar:` configuration section for new work. Existing `fedhydra` entry points and configuration spellings remain accepted. Legacy checkpoints are normalized for naming differences while retaining the other resume compatibility checks; keep the original experiment name and run directory when resuming an existing experiment.

Reported manuscript accuracies in the READMEs are transcribed from Table 5. Unit, integration, packaging, and CPU smoke checks verify software behavior; they do not substitute for the five-seed image-benchmark experiments.
