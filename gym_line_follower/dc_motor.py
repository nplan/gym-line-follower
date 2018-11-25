class DCMotor:

    def __init__(self, nominal_voltage, no_load_speed, stall_torque):
        self.constant, self.resistance = self.get_motor_parameters(nominal_voltage, no_load_speed, stall_torque)

    def get_torque(self, supply_voltage, w):
        return (supply_voltage - w * self.constant) / self.resistance * self.constant

    @staticmethod
    def get_motor_parameters(nominal_voltage, no_load_speed, stall_torque):
        motor_constant = nominal_voltage / no_load_speed
        resistance = nominal_voltage / stall_torque * motor_constant
        return motor_constant, resistance


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np

    v = np.linspace(0, 6, 100)
    motor = DCMotor(6., 105., 0.057)
    T = [motor.get_torque(vs, 105.) for vs in v]
    plt.plot(v, T)
    plt.grid()
    plt.show()
