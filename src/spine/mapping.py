import math


def _clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))


class SpineMapper:
    def __init__(self, config):
        self.config = config
        self.segments = int(config.get("segments", 1))
        self.neutral = [float(v) for v in config.get("neutral_angles", [])]
        self.min_angles = [float(v) for v in config.get("min_angles", [])]
        self.max_angles = [float(v) for v in config.get("max_angles", [])]
        self.weights = [float(v) for v in config.get("weights", [])]
        self.upper_weights = [float(v) for v in config.get("upper_weights", [])]
        self.directions = [float(v) for v in config.get("directions", [])]
        self.upper_start = int(config.get("upper_segment_start", max(0, self.segments // 2)))
        self.gain = config.get("gain", {})
        self.deadband = float(config.get("deadband_deg", 0.0))
        self.max_deg_per_sec = float(config.get("max_deg_per_sec", 120.0))
        self.last_angles = None
        self.last_time = None

    def map_metrics(self, metrics, now):
        raw = []
        for i in range(self.segments):
            direction = self.directions[i] if i < len(self.directions) else 1.0
            weight = self.weights[i] if i < len(self.weights) else 1.0
            upper_weight = self.upper_weights[i] if i < len(self.upper_weights) else weight
            angle = self.neutral[i] if i < len(self.neutral) else 90.0
            angle += direction * self.gain.get("lean", 0.0) * weight * metrics.lean
            if i >= self.upper_start:
                angle += direction * self.gain.get("neck", 0.0) * upper_weight * metrics.neck
            angle += direction * self.gain.get("tilt", 0.0) * weight * metrics.tilt
            min_angle = self.min_angles[i] if i < len(self.min_angles) else 0.0
            max_angle = self.max_angles[i] if i < len(self.max_angles) else 180.0
            raw.append(_clamp(angle, min_angle, max_angle))

        if self.last_angles is None:
            self.last_angles = raw
            self.last_time = now
            return raw

        dt = max(0.0, now - self.last_time) if self.last_time else 0.0
        self.last_time = now
        if dt <= 0:
            self.last_angles = raw
            return raw

        limited = []
        max_delta = self.max_deg_per_sec * dt
        for prev, target in zip(self.last_angles, raw):
            delta = target - prev
            if abs(delta) > max_delta:
                target = prev + math.copysign(max_delta, delta)
            limited.append(target)

        self.last_angles = limited
        return limited
