# RAG assignment: baseline versus re-ranking

Open **RAG_Baseline_vs_Reranking.ipynb** for the step-by-step implementation.
The equivalent executable implementation is `experiment.py`.

## Run locally

```powershell
python -m pip install -r requirements.txt
python experiment.py
```

The downloaded PDF must be at `data/catan_cities_knights.pdf`. The notebook includes a download
cell. Model downloads require internet and several GB of available disk space. CPU execution
works but generation may take a while. No paid API or secret is required.

Official document: https://www.catan.com/sites/default/files/2021-06/catan_c_k_2020_rule_book_200708.pdf

## What each step does

1. Extract one PDF and keep page numbers so claims can be checked.
2. Split into 160-word chunks with 40-word overlap so adjacent clauses have some continuity.
3. Embed all chunks once and normalize vectors to use cosine similarity via dot product.
4. Retrieve ten candidates. Baseline passes the first three to the generator.
5. Cross-encode the same ten pairs, reorder them, and pass the first three to the same generator.
6. Run seventeen document-specific stress-test questions. They are hypotheses, not promised wins.
7. Measure five warm retrieval repetitions. Exclude model loading, indexing, and generation.
8. Export all passages, ranks, scores, prompts, timing samples, and run metadata. Generate
   paired answers for two verified retrieval recoveries and one regression; other queries
   remain explicitly marked as retrieval-only diagnostics.
9. Verify at least two true recoveries against the PDF. Check if both answers reflect the rule.
10. Write the report in your own words using REPORT_WORKSHEET.md.

## Results

- `results/chunks.json`: corpus and page IDs.
- `results/retrieval.json`: rankings and latency, saved before generation starts.
- `results/experiment.json`: both generated answers plus retrieval results.
- `results/evidence_review.md`: readable evidence for manual verification.
- `results/metadata.json`: source hash, environment, and experiment parameters.
- `results/LAB_NOTES.md`: verified cases, actual answers, timings, and learning explanations.

The notebook includes saved measurement output computed from the completed script run.
Earlier notebook cells remain available to reproduce the experiment. Use
`requirements-lock.txt` for the exact installed core package versions used in this run.

The document and models are ignored by Git; download them to reproduce the experiment.
Do not interpret score changes as correctness without reading the source.
No commit, push, or repository publishing is performed by these scripts.
