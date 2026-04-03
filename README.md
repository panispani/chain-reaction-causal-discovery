# [CLeaR 2026] Causal Discovery in Action: Learning Chain-Reaction Mechanisms from Interventions

[![arXiv](https://img.shields.io/badge/arXiv-2603.22620-b31b1b.svg)](https://arxiv.org/abs/2603.22620)

## Installation

```bash
# Install dependencies
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

## How to reproduce all results of the paper

After installing the dependencies in `requirements.txt`, run the following:

```bash
mkdir logs
nohup bash synth_tree.sh > logs/nohup.out 2>&1 &
nohup bash synth_t2.sh > logs/nohup.out 2>&1 &
nohup bash synth_t4_big.sh > logs/nohup.out 2>&1 &
nohup bash physical_t0.sh > logs/nohup.out 2>&1 &
nohup bash physical_t1.sh > logs/nohup.out 2>&1 &
nohup bash physical_t2.sh > logs/nohup.out 2>&1 &
nohup bash physical_t3.sh > logs/nohup.out 2>&1 &
nohup bash physical_t4.sh > logs/nohup.out 2>&1 &
nohup bash physical_t4_big.sh > logs/nohup.out 2>&1 &
```

This will run all the experiments of the paper and include in the logs the latex tables and the commands to replot the Figures of the experiments section.

## Quick Start

### Scaling experiment

Run sample scaling experiments to see how success varies with sample size (this is what is done in the `.sh` files above):

```bash
python sample_scaling_experiment.py t0.yaml \
  --seeds 0 1 2 3 4 \
  --displacement 0.1 \
  --max-samples 100 --evaluate-baselines
```

This tests increasing numbers of samples per variable (1, 2, 3, ...) until achieving 100% success for 2 consecutive sample sizes.
Then it chooses the first sample size that was above 95% and evaluates the method, PC and the other two heuristic observational methods.
For details on the command-line parameters take a look at `sample_scaling_experiment.py`.

### Running a Simulation

One can also run a simulation manually and inspect the samples (This is done inside the `sample_scaling_experiment.py` script).

```bash
python run_simulation.py levels/ball_domino_chain.yaml \
    --seed 42 \
    --max-steps 800 \
    -M 10 \
    -K 10 \
    --displacement 0.05
```

### Creating Levels

Find examples in `levels/`. The environment supports balls, dominos, ramps, buttons, and walls.

```yaml
id: "example_level"
cell_size: 1.0

layout: |
  WWWWWWWWW
  W.......W
  W.B.....W
  W.......W
  WDD.R.T.W
  WWWWWWWWW

legend:
  "W":
    type: wall_tile
  ".":
    type: empty
  "B":
    type: ball
    radius: 0.3
    mass: 1.0
    elasticity: 0.7
    initial_velocity: [3, 0] # Optional initial velocity [vx, vy]
  "D":
    type: domino
    width: 0.2
    height: 1.0
    mass: 0.5
  "R":
    type: ramp
    angle: 30
    direction: up # or "down"
  "T":
    type: button
    width: 0.6
    height: 0.15
    triggers:
      - action: open_walls
        wall_cells:
          - [1, 4]
```

## Output Files

Each simulation run creates a timestamped directory in `runs/` with:

1. **object_info.csv**: Object metadata (id, type, color)
2. **intervention_samples.csv/.npy**: Binary movement states with interventions
3. **no_intervention_samples.csv/.npy**: Binary movement states without interventions
4. **intervention_action_N.gif**: Animation with intervention on object N
5. **no_intervention.gif**: Animation without intervention
6. **initial_motion.png**: Screenshot with motion arrows
7. **ad_hoc_collisions.txt**: Collision graph (simple)
8. **ad_hoc_collisions_with_time.txt**: Collision graph with temporal info

## Physics Parameters

The simulation uses the following physics constants (see physics_env.py):

- **Gravity**: 980 pixels/s<sup>2</sup> (downward)
- **Time Step**: 0.02s per simulation step
- **Friction**: 0.5
- **Scaling**: 100 pixels per grid cell

## Actions

- **Action 0**: No-op (let everything move naturally)
- **Action 1..N**: Hold object N in place (like grabbing it)

## Testing

Run the tests:

```bash
pytest tests/ -v
```

## Examples

See the `levels/` directory for example level configurations:

- `ball_domino_chain.yaml`: Ball hits domino chain

Level breakdown:

- Minimal Chain: t1.yaml (simple) - for debugging mostly
- Sequential Chain: t0.yaml (super sequential)
- Parallel Triggers: t2.yaml (parallel buttons, need interventions)
- Intertwined Mechanisms: t3.yaml (difficult, intertwined mechanisms, moving at the same time)
- Linear Slot-Machine: t4.yaml (slot machine, large number of N, need interventions)
- Large Slot-Machine: t4_big.yaml (more variables)

## Citation

If you find this work useful, please cite our paper:

```bibtex
@article{panayiotou2026causal,
  title={Causal Discovery in Action: Learning Chain-Reaction Mechanisms from Interventions},
  author={Panayiotou, Panayiotis and {\\c{S}}im{\\c{s}}ek, {\\"O}zg{\\"u}r},
  journal={arXiv preprint arXiv:2603.22620},
  year={2026}
}
```
