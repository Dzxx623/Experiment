# MuJoCo teleoperation semantic-refresh experiment

This experiment tests a low-latency teleoperation hypothesis: under a constrained
visual/semantic channel, refresh the visual entity whose update most changes the
operator's control action, rather than simply refreshing the oldest or fastest-moving
entity.

Stage 1 uses oracle entity states from MuJoCo. This deliberately isolates the
communication/scheduling mechanism from detector/VLM errors. A later vision stage can
replace oracle states with detections from rendered RGB frames without changing the
network/scheduler interface.

Baselines:
- round_robin
- AoI
- entity-wise K-AoI = AoI * entity speed (an adaptation of Tian et al.'s K-AoI)
- state error
- AoI * state error
- counterfactual action impact (proposed)
- action impact * state error

The original K-AoI paper uses K-AoI as an end-effector digital-twin synchronization
metric; it does not propose per-object visual scheduling. The entity-wise scheduler here
is therefore a deliberately strong adaptation for comparison.

Run:
```bash
pip install -r requirements.txt
python smoke_test.py
python experiment.py --episodes 80 --profiles wifi_congested poor_4g --distractors 0 3 5 --out results/raw_results.csv
python analyze.py results/raw_results.csv --outdir results
```

The experiment uses paired random seeds and identical packet-loss/delay traces across
methods for each seed. The distractor ablation tests whether motion-weighted freshness
wastes scarce updates on dynamic but task-irrelevant scene content.
