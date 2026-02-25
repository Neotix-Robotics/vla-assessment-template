# Submission — VLA Foundation Models Technical Assessment

## Candidate Information

- **Name:**
- **Email:**
- **GitHub Username:**
- **Date Submitted:**

---

## Summary

_In 3-5 sentences, summarize what you accomplished and your key findings._



---

## Links

- **Wandb Project:** `https://wandb.ai/<your-username>/smolvla-libero`
- **Checkpoint (if hosted):** _(HuggingFace Hub link, or "local in repo")_

---

## Environment

- **GPU(s) used:**
- **CUDA version:**
- **Python version:**
- **PyTorch version:**
- **LeRobot version:**
- **OS / Platform:** (e.g., Google Colab, Kaggle, local)
- **Total compute time (approx):**

---

## Checklist

_Mark each item with [x] when complete._

### Part A: Fine-tuning
- [ ] Environment set up and verified (LeRobot + LIBERO + MuJoCo headless)
- [ ] Wandb logging enabled — training curves visible in dashboard
- [ ] SmolVLA fine-tuned on LIBERO-Spatial (>= 20K steps)
- [ ] Checkpoints saved at regular intervals
- [ ] Reproducible training script in `scripts/`

### Part B: Evaluation & Video
- [ ] Fine-tuned model evaluated (10 tasks x 10 episodes = 100 total)
- [ ] Reference model evaluated (10 tasks x 10 episodes = 100 total)
- [ ] `results/smolvla_finetuned_results.json` — valid JSON, all fields present
- [ ] `results/smolvla_reference_results.json` — valid JSON, all fields present
- [ ] Inference latency and GPU memory reported in results JSONs
- [ ] Simulation videos recorded (minimum 3 tasks, both success and failure)
- [ ] Videos saved in `videos/` folder

### Part C: Ablation & Report
- [ ] Ablation: chose one design dimension (chunk size / VLM depth / data efficiency / training duration)
- [ ] Ablation: trained and evaluated >= 2 variants
- [ ] Ablation: all variant results saved as `results/ablation_*.json`
- [ ] Ablation: all runs logged to Wandb (same project, different run names)
- [ ] `REPORT.pdf` — paper-format technical report (4-8 pages)
- [ ] Report includes: Abstract, Intro, Setup, Results, Analysis, Proposed Improvement, References
- [ ] Report includes comparison tables (fine-tuned vs reference, per-task)
- [ ] Report includes Wandb figures (loss curves, LR schedule)
- [ ] Report includes video analysis with screenshots and failure mode discussion
- [ ] Wandb project set to public (or link shared with reviewer)

### Final
- [ ] Git history shows iterative progress (multiple commits, not a single dump)
- [ ] Repository follows the required structure
- [ ] All scripts are reproducible (seeds, exact commands, environment details)

---

## Results Summary

### Fine-tuned vs Reference

| Task | Fine-tuned (yours) | Reference (HF) | Delta |
|:----:|:------------------:|:---------------:|:-----:|
| 0    |                    |                 |       |
| 1    |                    |                 |       |
| 2    |                    |                 |       |
| 3    |                    |                 |       |
| 4    |                    |                 |       |
| 5    |                    |                 |       |
| 6    |                    |                 |       |
| 7    |                    |                 |       |
| 8    |                    |                 |       |
| 9    |                    |                 |       |
| **Overall** | | | |

### Ablation Results

| Variant | Description | Aggregate Success Rate |
|---------|-------------|:----------------------:|
| Baseline (default) | _e.g., chunk_size=50_ | |
| Variant 1 | _e.g., chunk_size=25_ | |
| Variant 2 | _e.g., chunk_size=10_ | |

---

## Ablation Details

- **Dimension chosen:** _(e.g., action chunk size)_
- **Hypothesis:** _What did you expect to happen and why?_
- **Variables held constant:** _List everything you kept the same across variants._
- **Key finding:** _One sentence summary of your ablation result._

---

## Issues Encountered

_Describe any setup issues, bugs, or challenges you faced and how you resolved them. This helps us calibrate the assessment difficulty._

1.
2.
3.

---

## Notes

_Any additional context, observations, or things you'd like the reviewer to know._
