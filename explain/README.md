# Explainability: Traffic→Delay Sensitivity (Single Path)

This experiment produces a 2D curve showing how predicted delay (loc output) changes as we vary the traffic on a single path while keeping all other paths fixed. It highlights the non-linear congestion behavior learned by the KAN-based RouteNet model.

## Key Idea
Hold the network state constant except for one chosen path P. Sweep that path's raw traffic demand from 0 up to a multiple of its baseline value. For each modified input, run the trained delay model and collect the predicted delay (mean). Plot delay vs traffic and annotate the point of maximum curvature to illustrate emerging congestion.

## Requirements
- Trained KAN delay model weights located at: `./kan_model/137/kan_bspline/best_delay_kan_model.weights.h5`
- A TFRecord file from the dataset (training or evaluation) with the same schema used in training.

## Generated Artifacts (in `./explain_result` by default)
- `traffic_delay_curve.png`: Main sensitivity curve with baseline and max-curvature annotations.
- `traffic_delay_curve.csv`: Numeric data (traffic_raw, predicted_delay_loc).
- `metadata.json`: Run metadata (path id, baseline values, curvature point, etc.).
- `sample_snapshot.json`: Basic stats about the baseline sample.
- `traffic_delay_curve_derivative.png` (optional if `--save_derivative`): First and second derivatives.

## Usage
Example command:

```
python explain/run_sensitivity.py \
  --tfrecord_path data/routenet/gbnbw/train_0000.tfrecords \
  --sample_index 0 \
  --num_steps 60 \
  --max_multiplier 2.0 \
  --save_derivative
```

Optional: specify a concrete path id:
```
python explain/run_sensitivity.py --tfrecord_path data/routenet/gbnbw/train_0000.tfrecords --path_id 123
```

## Path Selection Logic
If `--path_id` is not provided, the script automatically:
1. Computes the length of each path.
2. Filters for paths with length ≥ 2.
3. Chooses the median-length path (index at the middle of the valid set) for stability.

## Scaling Conventions
- Traffic scaling (training): `scaled = (raw - 0.18) / 0.15` → inverted for sweep updates.
- Capacity scaling: `scaled = raw / 10.0` (only used for snapshot stats; capacities remain fixed).

## Interpreting the Curve
- Initial linear/near-linear region: uncongested regime.
- Increasing slope: onset of congestion.
- Marked "Max curvature" point: where second derivative (numeric) peaks—often near a transition into heavier congestion.

## Notes & Limitations
- Extrapolating far beyond the training traffic distribution may yield unreliable predictions.
- The curvature point is purely numerical (argmax of second derivative); it is a heuristic, not a formal threshold.
- Delay (loc) is assumed to be on the same physical scale as training labels.

## Potential Extensions
- Batch analysis over multiple paths (aggregate statistics).
- Confidence intervals via multiple baseline samples.
- Comparing MLP vs KAN sensitivity on identical samples.

## Troubleshooting
- If weights file not found: verify path `./kan_model/137/kan_bspline/best_delay_kan_model.weights.h5`.
- If no path with length ≥ 2: choose a different sample index or TFRecord file.

---
Maintainer: (auto-generated script)