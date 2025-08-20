# simulator.py
import time

class VirtualBattery:
    """간단한 배터리 모델: OCV(soc), internal resistance, coulomb counting"""
    def __init__(self, capacity_ah=2.2, initial_soc=0.2, internal_resistance=0.095, max_charge_current=2.0):
        self.CAPACITY = capacity_ah
        self.soc = initial_soc
        self.resistance = internal_resistance
        self.max_charge_current = max_charge_current
        self.temperature = 25.0
        # initial OCV/voltage
        self.voltage = self.ocv_from_soc()

    def ocv_from_soc(self):
        # 간단한 선형 OCV 모델 (원하면 테이블 기반으로 교체)
        return 3.7 + self.soc * (4.2 - 3.7)

    def charge(self, current, delta_t_seconds):
        """current: charging current (A, 양수 = 충전).
           delta_t_seconds: 경과 시간(초).
        """
        
        # 간단 온도 모델 (주석 해제 후 테스트)
        # 간단 온도 모델의 상수를 더 높게 설정하여 (예: 0.001 -> 0.005)
        # 온도가 35도에 더 빨리 도달하도록 만듭니다.
        self.temperature += 0.005 * abs(current) * (delta_t_seconds / 1.0)
        
        old_soc = self.soc
        old_voltage = self.voltage

        # 충전/방전 한계 체크(안전)
        if current < 0:
            # 방전 허용 범위 처리 (간단히)
            pass

        # Coulomb counting
        added_ah = current * (delta_t_seconds / 3600.0)
        self.soc += added_ah / self.CAPACITY
        if self.soc > 1.0:
            self.soc = 1.0
        if self.soc < 0.0:
            self.soc = 0.0

        # Terminal voltage (단자전압) 모델:
        # 충전시 단자전압 = OCV + I * R_internal
        # (I: 충전 전류 양수일 때 내부저항에 의한 전압 상승을 흉내냄)
        self.voltage = self.ocv_from_soc() + current * self.resistance

        # 간단 온도 모델 (선택적)
        # self.temperature += 0.001 * abs(current) * (delta_t_seconds / 1.0)

        # 디버그 (너무 자주 찍히면 성능저하 가능)
        # print(f"[DEBUG][Battery] dt={delta_t_seconds}s I={current:.3f}A SoC {old_soc:.3f}->{self.soc:.3f} V {old_voltage:.3f}->{self.voltage:.3f}")

class MockHardwareManager:
    """가상 하드웨어: PWM, relay 상태, PWM->current 맵핑"""
    def __init__(self, virtual_battery: VirtualBattery):
        self.battery = virtual_battery
        self.pwm_duty_cycle = 0.0  # 0..100 (%)
        self.relay_state = False
        self.last_update_time = time.time()

    def set_relay_on(self, channel, state: bool):
        self.relay_state = bool(state)
        print(f"[HW] 채널 {channel} 릴레이 {'ON' if self.relay_state else 'OFF'}")

    def set_pwm_duty_cycle(self, channel, duty_cycle):
        # 안전 범위 clamp
        duty = max(0.0, min(100.0, float(duty_cycle)))
        self.pwm_duty_cycle = duty
        # print(f"[HW] PWM set: {duty:.2f}%")

    def get_voltage(self, channel):
        return self.battery.voltage

    def get_current(self, channel):
        # relay가 ON일 때만 PWM에 따라 전류 발생
        if self.relay_state:
            return self.battery.max_charge_current * (self.pwm_duty_cycle / 100.0)
        return 0.0

    def update_simulation(self, delta_t):
        """delta_t: 초 단위. PWM->전류를 배터리에 적용."""
        current = self.get_current(0)
        self.battery.charge(current, delta_t)
        
