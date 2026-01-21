import logging


class DryRunServoController:
    def __init__(self, channels):
        self.channels = channels
        self.last_logged = None

    def set_angles(self, angles):
        rounded = [round(float(a), 1) for a in angles]
        logging.debug("Dry-run angles: %s", rounded)

    def close(self):
        pass


class Pca9685ServoController:
    def __init__(self, config):
        import board
        import busio
        from adafruit_motor import servo as adafruit_servo
        from adafruit_pca9685 import PCA9685

        self.config = config
        i2c = busio.I2C(board.SCL, board.SDA)
        self.pca = PCA9685(i2c, address=int(config.get("i2c_address", 0x40)))
        self.pca.frequency = int(config.get("frequency_hz", 50))

        pulse_min = int(config.get("pulse_min_us", 500))
        pulse_max = int(config.get("pulse_max_us", 2500))
        actuation = int(config.get("actuation_range", 180))

        self.servos = []
        for channel in config.get("channels", []):
            servo = adafruit_servo.Servo(
                self.pca.channels[int(channel)],
                min_pulse=pulse_min,
                max_pulse=pulse_max,
                actuation_range=actuation,
            )
            self.servos.append(servo)

    def set_angles(self, angles):
        for servo, angle in zip(self.servos, angles):
            try:
                servo.angle = float(angle)
            except OSError as exc:
                logging.error("I2C error while setting servo angle: %s", exc)
            except Exception as exc:
                logging.error("Servo error: %s", exc)

    def close(self):
        try:
            self.pca.deinit()
        except Exception:
            pass


def create_servo_controller(config, dry_run=False):
    if dry_run or not config.get("enabled", True):
        return DryRunServoController(config.get("channels", []))
    try:
        return Pca9685ServoController(config)
    except Exception as exc:
        logging.error("Failed to initialize PCA9685: %s", exc)
        return DryRunServoController(config.get("channels", []))
