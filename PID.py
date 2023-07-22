import time
import numpy as np
class PID:
    def __init__(self, Kp=0.6, Ki=0.03, Kd=0.3, alpha=0.95, sample_time=0.03):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.alpha = alpha
        self.sample_time = sample_time

        self.min_integral = -5
        self.max_integral = 5
        self.min_correction = -60
        self.max_correction = 60
        self.dead_zone = 2

        self.last_input = 0
        self.previous_error = 0
        self.integral = 0
        self.previous_output = 0
        self.last_time = 0

    def set_output_limit(self, min_limit=-60, max_limit=60):
        self.min_correction = min_limit
        self.max_correction = max_limit

    def set_integral_limit(self, min_limit=-10, max_limit=10):
        self.min_integral = min_limit
        self.max_integral = max_limit

    def set_deadzone(self, deadzone=2):
        self.dead_zone = deadzone

    def set_sample_time(self, sample_time=0.04):
        self.sample_time = sample_time
    def clear(self):
        self.last_input = 0
        self.previous_error = 0
        self.integral = 0
        self.previous_output = 0
        self.last_time = 0
    def set_tunings(self,Kp,Ki,Kd):
        self.Kp = Kp
        self.Ki = Ki * self.sample_time
        self.Kd = Kd / self.sample_time 
    
    def compute(self, setpoint, targetpoint):
        now = time.time()
        if now - self.last_time >= self.sample_time:
            # Compute error
            error = targetpoint - setpoint
           
            self.integral += error
            if self.integral > self.max_integral:
                self.integral = self.max_integral
            elif self.integral < self.min_integral:
                self.integral = self.min_integral

            d_input = setpoint - self.last_input

            # Compute correction
            correction = self.Kp * error + self.Ki * self.integral - self.Kd * d_input

            # Low pass filter on correction
            correction = (1 - self.alpha) * self.previous_output + self.alpha * correction

            # Output limitation
            if correction > self.max_correction:
                correction = self.max_correction
            elif correction < self.min_correction:
                correction = self.min_correction

            if abs(correction) < self.dead_zone:
                correction = 0

            # Update previous values
            self.previous_error = error
            self.previous_output = correction
            self.last_time = now
            self.last_input = setpoint

            return correction

        return 0 

class CascadePIDController:
    def __init__(self, Kp1, Ki1, Kd1, Kp2, Ki2, Kd2):
        # 外环PID控制器
        self.pid_outer = PID(Kp=Kp1, Ki=Ki1, Kd=Kd1)
        self.pid_outer.set_output_limit(-25, 25)
        self.pid_outer.set_integral_limit(-5, 5)
        #self.pid_outer.set_sample_time(0.05)
        # 内环PID控制器
        self.pid_inner = PID(Kp=Kp2, Ki=Ki2, Kd=Kd2)
        self.pid_inner.set_output_limit(-5, 5)
        self.pid_inner.set_integral_limit(-5, 5)		
		
    def clear(self):
        self.pid_inner.clear()
        self.pid_outer.clear()
    def compute(self, setpoint1,targetpoint1,setpoint2,targetpoint2):
        output1 = self.pid_outer.compute(setpoint1,targetpoint1)
        output2 = self.pid_inner.compute(setpoint2,targetpoint2)
        output = output1 - output2
        if output > 30:
            output = 30
        elif output < -30:
            output = -30
        return output
class CascadePIDController2:
    def __init__(self, Kp1, Ki1, Kd1, Kp2, Ki2, Kd2):
        # 外环PID控制器
        self.pid_outer = PID(Kp=Kp1, Ki=Ki1, Kd=Kd1)
        self.pid_outer.set_output_limit(-2, 2)
        self.pid_outer.set_integral_limit(-1, 1)
        self.pid_outer.set_sample_time(0.05)
        # 内环PID控制器
        self.pid_inner = PID(Kp=Kp2, Ki=Ki2, Kd=Kd2)
        self.pid_inner.set_output_limit(-30, 30)
        self.pid_inner.set_integral_limit(-5, 5)		
		
    def clear(self):
        self.pid_inner.clear()
        self.pid_outer.clear()
    def compute(self, setpoint1,targetpoint1,setpoint2):
        output1 = self.pid_outer.compute(setpoint1,targetpoint1)
        output = self.pid_inner.compute(setpoint2,output1)
        return output
