# Project Plan

## Completed

1. ✅ Download and process French election data (municipales, legislatives, presidentielles, etc.) for the last 20+ years → `data/elections/` (documented in `election_data.md`)
2. ✅ Download and process polling data for all French elections → `data/polls/` (documented in `polls_data.md`)
3. ✅ Download and process demographic data (INSEE census + BPE) → `data/demographics/` (documented in `demographic_data.md`)
4. ✅ Build geo-mapping: lat/lon coordinates for all locations → `data/geo/` (documented in `mapping.md`)
5. ✅ Implement Universal Masked Set Transformer with learnable full-pool router (documented in `archi.md`, `sampling_plan.md`)
6. ✅ Implement KL divergence loss over softmax-normalized election groups
7. ✅ Implement data loading pipeline with unified token pool + GPU PoolCache (documented in `dataloading.md`)
8. ✅ Implement training loop with EMA, cosine annealing, entropy regularization
9. ✅ Implement evaluation pipeline with per-split metric breakdown (documented in `eval.md`)
10. ✅ Implement trajectory visualization pipeline (`src/visualize_trajectories.py`)
11. ✅ Integrate BPE equipment density indicators (médecins, pharmacies, postes, supermarchés per 1k pop)
12. ✅ Implement `availability_date` causality mechanism: separate reference dates from publication dates to prevent temporal leakage

## In Progress

13. 🔄 Training with full-pool router architecture (first run underway)

## Future

14. ⬜ Load multi-vintage demographic data (Census 2006–2021, BPE 2014–2024) for demographic trajectories
15. ⬜ Add more demographic sources: Census LOG (housing), État civil, Sirene (see `demographic_data.md`)
16. ⬜ Implement counterfactual scenario simulations (documented conceptually in `vizs.md`)
17. ⬜ Build interactive visualization dashboard
