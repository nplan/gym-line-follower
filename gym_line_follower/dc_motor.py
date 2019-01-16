class DCMotor:
    """
    Simple DC motor model.
    """

    def __init__(self, nominal_voltage, no_load_speed, stall_torque):
        """
        Create a motor with parameters:
        :param nominal_voltage: [V]
        :param no_load_speed: [rad/s]
        :param stall_torque: [Nm]
        """
        self.constant, self.resistance = self.get_motor_parameters(nominal_voltage, no_load_speed, stall_torque)

    def get_torque(self, supply_voltage, w):
        """
        Calculate instant torque from supply voltage and rotation speed.
        :param supply_voltage: [V]
        :param w: [rad/s
        :return: shaft torque [Nm]
        """
        return (supply_voltage - w * self.constant) / self.resistance * self.constant

    @staticmethod
    def get_motor_parameters(nominal_voltage, no_load_speed, stall_torque):
        """
        Calculate motor constant and resistance from parameters:
        :param nominal_voltage: [V]
        :param no_load_speed: [rad/s]
        :param stall_torque: [Nm]
        :return: tuple of: motor constant, resistance
        """
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
