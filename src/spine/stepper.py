import logging
import time


class DryRunStepperController:
    def __init__(self, config):
        self.motors = list(config.get("motors", []))
        self.steps_per_rev = int(config.get("steps_per_rev", 200))
        self.microstep = int(config.get("microstep", 1))
        self._initialized = False
        self._positions = [0] * len(self.motors)

    def set_angles(self, angles):
        if not self.motors:
            return
        steps_per_deg = (self.steps_per_rev * self.microstep) / 360.0
        targets = []
        for i, motor in enumerate(self.motors):
            angle_index = int(motor.get("angle_index", i))
            if angle_index < 0 or angle_index >= len(angles):
                targets.append(None)
                continue
            target_steps = int(round(angles[angle_index] * steps_per_deg))
            targets.append(target_steps)
        if not self._initialized:
            self._positions = [t if t is not None else 0 for t in targets]
            self._initialized = True
            logging.info("Dry-run stepper initialized at targets: %s", targets)
            return
        logging.debug("Dry-run stepper targets: %s", targets)

    def close(self):
        pass


class Drv8825StepperController:
    def __init__(self, config):
        import RPi.GPIO as GPIO

        self.GPIO = GPIO
        self.GPIO.setwarnings(False)
        self.GPIO.setmode(GPIO.BCM)

        self.steps_per_rev = int(config.get("steps_per_rev", 200))
        self.microstep = int(config.get("microstep", 1))
        self.max_steps_per_sec = float(config.get("max_steps_per_sec", 800.0))
        self.min_pulse_us = float(config.get("min_pulse_us", 2.0))

        self.motors = []
        for i, motor in enumerate(config.get("motors", [])):
            step_pin = int(motor["step_pin"])
            dir_pin = int(motor["dir_pin"])
            enable_pin = motor.get("enable_pin")
            enable_pin = int(enable_pin) if enable_pin is not None else None
            enable_active_low = bool(motor.get("enable_active_low", True))
            angle_index = int(motor.get("angle_index", i))

            self.GPIO.setup(step_pin, GPIO.OUT)
            self.GPIO.setup(dir_pin, GPIO.OUT)
            if enable_pin is not None:
                self.GPIO.setup(enable_pin, GPIO.OUT)

            self.GPIO.output(step_pin, GPIO.LOW)

            self.motors.append(
                {
                    "step_pin": step_pin,
                    "dir_pin": dir_pin,
                    "enable_pin": enable_pin,
                    "enable_active_low": enable_active_low,
                    "angle_index": angle_index,
                }
            )

        self._positions = [0] * len(self.motors)
        self._initialized = False
        self._last_update = time.time()

        for motor in self.motors:
            self._set_enabled(motor, True)

    def _set_enabled(self, motor, enabled):
        pin = motor["enable_pin"]
        if pin is None:
            return
        active_low = motor["enable_active_low"]
        if active_low:
            level = self.GPIO.LOW if enabled else self.GPIO.HIGH
        else:
            level = self.GPIO.HIGH if enabled else self.GPIO.LOW
        self.GPIO.output(pin, level)

    def set_angles(self, angles):
        if not self.motors:
            return

        steps_per_deg = (self.steps_per_rev * self.microstep) / 360.0
        now = time.time()
        dt = max(0.0, now - self._last_update)
        self._last_update = now

        if self.max_steps_per_sec > 0:
            max_steps = int(self.max_steps_per_sec * dt)
        else:
            max_steps = 0

        pulse_delay = 0.0
        if self.max_steps_per_sec > 0:
            pulse_delay = max(self.min_pulse_us / 1_000_000.0, 0.5 / self.max_steps_per_sec)
        else:
            pulse_delay = self.min_pulse_us / 1_000_000.0

        targets = []
        for i, motor in enumerate(self.motors):
            angle_index = motor["angle_index"]
            if angle_index < 0 or angle_index >= len(angles):
                targets.append(None)
                continue
            target_steps = int(round(angles[angle_index] * steps_per_deg))
            targets.append(target_steps)

        if not self._initialized:
            self._positions = [t if t is not None else 0 for t in targets]
            self._initialized = True
            logging.info("Stepper initialized at targets: %s", targets)
            return

        for idx, motor in enumerate(self.motors):
            target = targets[idx]
            if target is None:
                continue
            delta = target - self._positions[idx]
            if delta == 0:
                continue

            step_limit = abs(delta)
            if max_steps > 0:
                step_limit = min(step_limit, max_steps)
            if step_limit == 0:
                continue

            direction = self.GPIO.HIGH if delta > 0 else self.GPIO.LOW
            self.GPIO.output(motor["dir_pin"], direction)

            for _ in range(step_limit):
                self.GPIO.output(motor["step_pin"], self.GPIO.HIGH)
                time.sleep(pulse_delay)
                self.GPIO.output(motor["step_pin"], self.GPIO.LOW)
                time.sleep(pulse_delay)

            self._positions[idx] += step_limit if delta > 0 else -step_limit

    def close(self):
        for motor in self.motors:
            self._set_enabled(motor, False)
        try:
            self.GPIO.cleanup()
        except Exception:
            pass


def create_stepper_controller(config, dry_run=False):
    if dry_run or not config.get("enabled", False):
        return DryRunStepperController(config)
    try:
        return Drv8825StepperController(config)
    except Exception as exc:
        logging.error("Failed to initialize DRV8825: %s", exc)
        return DryRunStepperController(config)
