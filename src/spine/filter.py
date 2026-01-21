import math

from .metrics import MetricValues


class EMAFilter:
    def __init__(self, alpha):
        self.alpha = float(alpha)
        self.value = None

    def update(self, value, _dt):
        if self.value is None:
            self.value = value
        else:
            self.value = self.alpha * value + (1.0 - self.alpha) * self.value
        return self.value


class OneEuroFilter:
    def __init__(self, min_cutoff, beta, d_cutoff):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.value = None
        self.derivative = 0.0

    def _alpha(self, cutoff, dt):
        if dt <= 0:
            return 1.0
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def update(self, value, dt):
        if self.value is None:
            self.value = value
            return value
        dx = (value - self.value) / max(dt, 1e-6)
        alpha_d = self._alpha(self.d_cutoff, dt)
        self.derivative = alpha_d * dx + (1.0 - alpha_d) * self.derivative
        cutoff = self.min_cutoff + self.beta * abs(self.derivative)
        alpha = self._alpha(cutoff, dt)
        self.value = alpha * value + (1.0 - alpha) * self.value
        return self.value


class MetricSmoother:
    def __init__(self, config):
        smoothing = config.get("smoothing", {})
        filter_type = smoothing.get("type", "ema").lower()
        self.filters = {}
        for name in ("lean_deg", "neck_deg", "tilt_deg"):
            if not smoothing.get("enabled", True):
                self.filters[name] = None
            elif filter_type == "one_euro":
                self.filters[name] = OneEuroFilter(
                    smoothing.get("one_euro_min_cutoff", 1.2),
                    smoothing.get("one_euro_beta", 0.02),
                    smoothing.get("one_euro_d_cutoff", 1.0),
                )
            else:
                self.filters[name] = EMAFilter(smoothing.get("alpha", 0.2))

        self.hold_seconds = float(config.get("hold_seconds", 0.5))
        self.decay_seconds = float(config.get("decay_seconds", 1.5))
        self.last_values = {"lean_deg": 0.0, "neck_deg": 0.0, "tilt_deg": 0.0}
        self.last_valid = {"lean_deg": None, "neck_deg": None, "tilt_deg": None}
        self.last_time = None

    def update(self, metrics, now):
        if self.last_time is None:
            dt = 0.0
        else:
            dt = max(0.0, now - self.last_time)
        self.last_time = now

        output = {}
        for name in ("lean_deg", "neck_deg", "tilt_deg"):
            value = getattr(metrics, name)
            if value is not None:
                filtered = value
                if self.filters[name] is not None:
                    filtered = self.filters[name].update(value, dt)
                self.last_values[name] = filtered
                self.last_valid[name] = now
                output[name] = filtered
            else:
                last_seen = self.last_valid[name]
                if last_seen is None:
                    output[name] = 0.0
                    self.last_values[name] = 0.0
                else:
                    age = now - last_seen
                    if age <= self.hold_seconds:
                        output[name] = self.last_values[name]
                    else:
                        if self.decay_seconds > 0:
                            decay = math.exp(-dt / self.decay_seconds)
                            self.last_values[name] *= decay
                            output[name] = self.last_values[name]
                        else:
                            self.last_values[name] = 0.0
                            output[name] = 0.0

        return MetricValues(
            lean_deg=output["lean_deg"],
            neck_deg=output["neck_deg"],
            tilt_deg=output["tilt_deg"],
        )
