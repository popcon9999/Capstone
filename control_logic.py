import time

# OCV-SoC 룩업 테이블 및 ChargingProfile, PIDController 클래스는 기존과 동일
# (생략)
OCV_SOC_TABLE = [
    (4.20, 1.00),
    (4.15, 0.90),
    (4.08, 0.80),
    (3.98, 0.60),
    (3.85, 0.40),
    (3.75, 0.20),
    (3.60, 0.00)
]

def get_soc_from_ocv(voltage):
    if voltage >= OCV_SOC_TABLE[0][0]: return 1.0
    if voltage <= OCV_SOC_TABLE[-1][0]: return 0.0
    for i in range(len(OCV_SOC_TABLE) - 1):
        v_high, soc_high = OCV_SOC_TABLE[i]
        v_low, soc_low = OCV_SOC_TABLE[i+1]
        if v_low <= voltage < v_high:
            return soc_low + (voltage - v_low) * (soc_high - soc_low) / (v_high - v_low)
    return 0.0

class ChargingProfile:
    def __init__(self, target_soc=1.0, max_current=1.1):
        self.target_soc = target_soc
        self.max_current = max_current

class PIDController:
    def __init__(self, Kp, Ki, Kd, setpoint, output_min=0.0, output_max=100.0):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.setpoint = setpoint
        self.output_min = output_min
        self.output_max = output_max
        self.prev_error = 0.0
        self.integral = 0.0
        self.output = 0.0

    def calculate(self, measured, delta_t):
        if delta_t <= 0: return self.output
        error = self.setpoint - measured
        derivative = (error - self.prev_error) / delta_t
        self.integral += error * delta_t
        output = (self.Kp * error) + (self.Ki * self.integral) + (self.Kd * derivative)
        if output > self.output_max or output < self.output_min:
            self.integral -= error * delta_t
            output = max(self.output_min, min(self.output_max, output))
        self.output = output
        self.prev_error = error
        return self.output

    def reset(self):
        self.prev_error = 0.0
        self.integral = 0.0
        self.output = 0.0

class BatteryChannel:
    """로직 순서가 수정된 최종 제어기"""
    def __init__(self, channel_id, hw_controller):
        # (기존 __init__ 과 동일)
        self.id = channel_id
        self.hw = hw_controller
        self.state = 'IDLE'
        self.profile = ChargingProfile()
        self.pid_current = PIDController(Kp=0.8, Ki=0.1, Kd=0.01, setpoint=self.profile.max_current)
        self.pid_voltage = PIDController(Kp=0.2, Ki=0.2, Kd=0.01, setpoint=4.2)
        self.i_cc_nominal = self.profile.max_current
        self.termination_current = 0.022
        self.cv_entry_soc = 0.80
        self.slew_up = 1.5
        self.slew_down = 0.6
        self.transition_duration = 2.0
        self.transition_elapsed = 0.0
        self.cv_target_pwm = 0.0
        self.estimated_soc = 0.0
        self.capacity_ah = 2.2
        # 🎯 이 줄이 누락되었을 가능성이 높습니다. 아래 라인을 추가해주세요.
        self.min_pwm_cv = 0.0  # CV 단계에서 사용할 최소 PWM 값

    def set_charging_profile(self, profile: ChargingProfile):
        # (기존과 동일)
        print(f"[CH] 새 프로파일 수신: 목표 SoC={profile.target_soc*100:.1f}%, 최대 전류={profile.max_current}A")
        self.profile = profile
        self.i_cc_nominal = self.profile.max_current
        self.pid_current.setpoint = self.profile.max_current

    def start_charging(self):
        # (기존과 동일)
        print(f"[CH] 채널 {self.id}: 충전 시작")
        self.state = 'CHARGING_CC'
        initial_voltage = self.hw.get_voltage(self.id)
        self.estimated_soc = get_soc_from_ocv(initial_voltage)
        print(f"[CH] 초기 전압({initial_voltage:.3f}V) 기반 추정 SoC: {self.estimated_soc*100:.1f}%")
        self.pid_current.reset()
        self.pid_voltage.reset()
        self.hw.set_relay_on(self.id, True)
        self.hw.set_pwm_duty_cycle(self.id, 0.0)

    def _update_soc_by_coulomb_counting(self, current, delta_t):
        # (기존과 동일)
        charge_added_ah = current * (delta_t / 3600.0)
        soc_change = charge_added_ah / self.capacity_ah
        self.estimated_soc += soc_change
        self.estimated_soc = max(0.0, min(1.0, self.estimated_soc))
    
    def _apply_slew(self, prev_pwm, target_pwm, dt):
        # (기존과 동일)
        if target_pwm >= prev_pwm:
            max_delta = self.slew_up * dt
            return min(prev_pwm + max_delta, target_pwm)
        else:
            max_delta = self.slew_down * dt
            return max(prev_pwm - max_delta, target_pwm)

    def run_control_loop(self, delta_t):
        if delta_t <= 0: delta_t = 0.01

        raw_v = self.hw.get_voltage(self.id)
        raw_i = self.hw.get_current(self.id)
        pwm_before = self.hw.pwm_duty_cycle  # 🎯 hw_controller를 통해 pwm_before 가져오기
        
        self._update_soc_by_coulomb_counting(raw_i, delta_t)
        
        true_soc = self.hw.battery.soc
        # print(f"True SoC: {true_soc*100:.2f}%, Estimated SoC: {self.estimated_soc*100:.2f}%")

        if self.state in ('IDLE', 'FULL'):
            self.hw.set_pwm_duty_cycle(self.id, 0.0)
            return {'state': self.state, 'voltage': raw_v, 'current': raw_i, 'soc': true_soc, 'estimated_soc': self.estimated_soc}

        # --- 🎯 하드 스톱 가드 (조건 완화) ---
        # 100% 충전 목표일 때만 바로 FULL 종료
        if self.profile.target_soc >= 0.999 and self.estimated_soc >= self.profile.target_soc:
            if self.state != 'FULL':
                print(f"[CH] 목표 SoC {self.profile.target_soc*100:.1f}% 도달 → 충전 종료")
            self.state = 'FULL'
            self.hw.set_pwm_duty_cycle(self.id, 0.0)
            self.hw.set_relay_on(self.id, False)
            return {
                'state': self.state,
                'voltage': raw_v,
                'current': raw_i,
                'soc': true_soc,
                'estimated_soc': self.estimated_soc
            }
        
        # 🎯 1. 상태 전환(State Transition) 로직을 먼저 처리
        if self.state == 'CHARGING_CC':
            v_margin = 0.02
            if self.estimated_soc >= self.cv_entry_soc and raw_v >= (self.pid_voltage.setpoint - v_margin):
                print(f"[CH] CC→TRANSITION 시작 (추정 SoC={self.estimated_soc*100:.1f}%, V={raw_v:.3f}V)")
                self.state = 'TRANSITION'
                self.transition_elapsed = 0.0
                self.cv_target_pwm = max(self.pid_voltage.calculate(raw_v, delta_t), self.min_pwm_cv)
                self.pid_voltage.prev_error = self.pid_voltage.setpoint - raw_v
                if abs(self.pid_voltage.Ki) > 1e-9:
                    self.pid_voltage.integral = (pwm_before - self.pid_voltage.Kp * self.pid_voltage.prev_error) / self.pid_voltage.Ki
                else:
                    self.pid_voltage.integral = 0.0
        
        elif self.state == 'TRANSITION':
            self.transition_elapsed += delta_t
            if self.transition_elapsed >= self.transition_duration:
                print("[CH] TRANSITION 완료 → CHARGING_CV")
                self.state = 'CHARGING_CV'
        
        elif self.state == 'CHARGING_CV':
            true_soc = self.hw.battery.soc
            # 🎯 조건 완화: 추정 SoC가 목표 SoC의 98% 이상이거나,
            # 실제 SoC가 목표 SoC를 초과했을 때 종료
            if (raw_i < self.termination_current * 3 and 
                (self.estimated_soc >= self.profile.target_soc * 0.98 or 
                true_soc >= self.profile.target_soc)):
                print(f"[CH] 충전 완료 "
                    f"(추정 SoC={self.estimated_soc*100:.1f}%, "
                    f"실제 SoC={true_soc*100:.1f}%)")
                self.state = 'FULL'

        # 🎯 2. 현재 상태에 맞는 행동(Action) 로직 처리
        pwm_target = pwm_before
        if self.state == 'CHARGING_CC':
            temperature = self.hw.battery.temperature
            current_setpoint = self.i_cc_nominal

            if temperature > 40.0:
                throttled = self.i_cc_nominal - (temperature - 40.0) * (self.i_cc_nominal - 0.5) / 5.0
                current_setpoint = max(0.5, throttled)
            elif temperature < 5.0:
                current_setpoint = 0.5
            self.pid_current.setpoint = min(current_setpoint, self.i_cc_nominal)
            pwm_target = self.pid_current.calculate(raw_i, delta_t)

        elif self.state == 'TRANSITION':
            pwm_target = self.cv_target_pwm

        elif self.state == 'CHARGING_CV':
            v_err = self.pid_voltage.setpoint - raw_v
            deadband = 0.02
            if abs(v_err) <= deadband:
                pwm_target = pwm_before
            else:
                pwm_target = self.pid_voltage.calculate(raw_v, delta_t)
            pwm_target = max(pwm_target, self.min_pwm_cv)

        # 🔽 목표 SoC 근접 시 전류 상한 축소 (소프트 테이퍼)
            remaining = max(0.0, self.profile.target_soc - self.estimated_soc)
            if remaining < 0.2:  # 남은 용량 20% 이내
                i_soft_limit = max(self.termination_current,
                                   min(self.i_cc_nominal,
                                       (remaining / 0.2) * 0.22))  # 0.22A=0.1C
                if raw_i > i_soft_limit:
                    pwm_target = min(
                        pwm_target,
                        max(0.0, pwm_before - self.slew_down*delta_t*2.0)
                    )

            pwm_target = max(pwm_target, self.min_pwm_cv)
        
        
        # 🎯 3. 최종 PWM 값 계산 및 적용
        pwm_final = self._apply_slew(pwm_before, pwm_target, delta_t)
        self.hw.set_pwm_duty_cycle(self.id, pwm_final)

        return {'state': self.state, 'voltage': raw_v, 'current': raw_i, 'soc': true_soc, 'estimated_soc': self.estimated_soc}