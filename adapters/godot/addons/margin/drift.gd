## Drift classification: typed trajectory states for value histories.
##
## Health tells you WHERE a value is. Drift tells you WHERE IT'S HEADED.
## Given a history of values, classifies the trajectory shape.
class_name MarginDrift


## Drift states — matches Python margin.DriftState
enum State {
	STABLE,       ## Not changing meaningfully
	DRIFTING,     ## Consistent trend in one direction
	ACCELERATING, ## Rate of change increasing
	DECELERATING, ## Rate of change decreasing
	REVERTING,    ## Heading back toward baseline
	OSCILLATING,  ## Periodic fluctuation
}

## Direction — polarity-aware
enum Direction {
	IMPROVING,  ## Moving toward healthier
	WORSENING,  ## Moving toward unhealthier
	NEUTRAL,    ## No meaningful direction
}

const STATE_NAMES := {
	State.STABLE: "STABLE",
	State.DRIFTING: "DRIFTING",
	State.ACCELERATING: "ACCELERATING",
	State.DECELERATING: "DECELERATING",
	State.REVERTING: "REVERTING",
	State.OSCILLATING: "OSCILLATING",
}

const DIRECTION_NAMES := {
	Direction.IMPROVING: "IMPROVING",
	Direction.WORSENING: "WORSENING",
	Direction.NEUTRAL: "NEUTRAL",
}


## Result of drift classification.
class Result:
	var state: State
	var direction: Direction
	var rate: float          ## slope per step (polarity-normalised)
	var r_squared: float     ## goodness of fit [0, 1]
	var n_samples: int

	func _init(
		p_state: State = State.STABLE,
		p_direction: Direction = Direction.NEUTRAL,
		p_rate: float = 0.0,
		p_r_squared: float = 0.0,
		p_n_samples: int = 0,
	) -> void:
		state = p_state
		direction = p_direction
		rate = p_rate
		r_squared = p_r_squared
		n_samples = p_n_samples

	func to_string_() -> String:
		return "%s(%s, rate=%.4f)" % [
			MarginDrift.STATE_NAMES[state],
			MarginDrift.DIRECTION_NAMES[direction],
			rate,
		]

	func to_dict() -> Dictionary:
		return {
			"state": MarginDrift.STATE_NAMES[state],
			"direction": MarginDrift.DIRECTION_NAMES[direction],
			"rate": rate,
			"r_squared": r_squared,
			"n_samples": n_samples,
		}


## Classify drift from a value history.
## values: Array[float] in chronological order
## baseline: the healthy value
## higher_is_better: polarity
## min_samples: minimum history length (default 3)
static func classify(
	values: Array,
	baseline: float,
	higher_is_better: bool = true,
	min_samples: int = 3,
) -> Result:
	var n := values.size()
	if n < min_samples:
		return Result.new()

	# Linear regression: y = slope * x + intercept
	var xs: Array[float] = []
	var ys: Array[float] = []
	for i in range(n):
		xs.append(float(i))
		ys.append(float(values[i]))

	var mean_x := 0.0
	var mean_y := 0.0
	for i in range(n):
		mean_x += xs[i]
		mean_y += ys[i]
	mean_x /= n
	mean_y /= n

	var ss_xx := 0.0
	var ss_yy := 0.0
	var ss_xy := 0.0
	for i in range(n):
		ss_xx += (xs[i] - mean_x) * (xs[i] - mean_x)
		ss_yy += (ys[i] - mean_y) * (ys[i] - mean_y)
		ss_xy += (xs[i] - mean_x) * (ys[i] - mean_y)

	if ss_xx == 0:
		return Result.new()

	var slope := ss_xy / ss_xx
	var intercept := mean_y - slope * mean_x

	# R-squared
	var ss_res := 0.0
	for i in range(n):
		var predicted := intercept + slope * xs[i]
		ss_res += (ys[i] - predicted) * (ys[i] - predicted)
	var r_sq := 0.0
	if ss_yy > 0:
		r_sq = maxf(1.0 - ss_res / ss_yy, 0.0)

	# Standard error of slope
	var se_slope := 0.0
	if n > 2 and ss_xx > 0:
		se_slope = sqrt(ss_res / float(n - 2) / ss_xx)

	# Normalise slope for polarity
	var norm_slope := slope if higher_is_better else -slope

	# --- Classification ---

	# Slope significance
	var slope_significant: bool
	if se_slope == 0:
		slope_significant = absf(slope) > 0
	else:
		slope_significant = absf(slope) > 1.5 * se_slope

	if not slope_significant:
		# Check oscillation: count sign changes in residuals
		var crossings := 0
		var residuals: Array[float] = []
		for i in range(n):
			residuals.append(ys[i] - (intercept + slope * xs[i]))
		var max_res := 0.0
		for r in residuals:
			max_res = maxf(max_res, absf(r))
		var tol := 0.01 * max_res
		for i in range(1, residuals.size()):
			if residuals[i - 1] * residuals[i] < 0:
				if absf(residuals[i - 1]) > tol and absf(residuals[i]) > tol:
					crossings += 1
		var amplitude := 0.0
		var val_min := ys[0]
		var val_max := ys[0]
		for v in ys:
			val_min = minf(val_min, v)
			val_max = maxf(val_max, v)
		amplitude = val_max - val_min
		var mean_abs := 0.0
		for v in ys:
			mean_abs += absf(v)
		mean_abs /= n
		var rel_amp := amplitude / maxf(mean_abs, 0.0000000001)

		if crossings >= 2 and rel_amp > 0.02 and n >= 5:
			return Result.new(State.OSCILLATING, Direction.NEUTRAL, norm_slope, r_sq, n)
		else:
			return Result.new(State.STABLE, Direction.NEUTRAL, norm_slope, r_sq, n)

	# Direction
	var direction := Direction.IMPROVING if norm_slope > 0 else Direction.WORSENING

	# Reversion check
	var first_val: float = ys[0]
	var current_val: float = ys[n - 1]
	var was_unhealthy: bool
	if higher_is_better:
		was_unhealthy = first_val < baseline
	else:
		was_unhealthy = first_val > baseline
	var now_closer := absf(current_val - baseline) < absf(first_val - baseline)

	if was_unhealthy and now_closer:
		return Result.new(State.REVERTING, direction, norm_slope, r_sq, n)

	return Result.new(State.DRIFTING, direction, norm_slope, r_sq, n)
