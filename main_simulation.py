# main_simulation.py

import time
import random  # 랜덤 로직 추가(실험용)

import pandas as pd
import matplotlib.pyplot as plt

from simulator import VirtualBattery, MockHardwareManager
from control_logic import BatteryChannel, ChargingProfile

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 🎯 랜덤 초기 SoC 설정 (예: 0% ~ 100% 사이의 임의의 값)
random_initial_soc = random.uniform(0.00, 1.00)
print(f"--- 랜덤 초기 SoC: {random_initial_soc*100:.1f}% 에서 시뮬레이션 시작 ---")

# 가상환경
virtual_battery = VirtualBattery(initial_soc=random_initial_soc, max_charge_current=2.0)
mock_hw = MockHardwareManager(virtual_battery)
charge_controller = BatteryChannel(channel_id=0, hw_controller=mock_hw)

# 🎯 80%까지만 충전하도록 프로파일 설정
profile = ChargingProfile(target_soc=0.8, max_current=1.1)
charge_controller.set_charging_profile(profile)


# 시뮬레이션 설정
SIMULATION_DURATION_SECONDS = 3 * 3600  # 3시간
VIRTUAL_TIME_STEP = 0.5  # 초
current_virtual_time = 0.0
log_data = []

# 충전 시작
charge_controller.start_charging()
cc_to_cv_time = None  # 전환 시점 기록

print("\n--- 시뮬레이션 시작 ---")
while current_virtual_time < SIMULATION_DURATION_SECONDS:
    prev_state = charge_controller.state
    status = charge_controller.run_control_loop(delta_t=VIRTUAL_TIME_STEP)

    # hw 상태 업데이트 (PWM에 따라 전류가 배터리에 적용)
    mock_hw.update_simulation(delta_t=VIRTUAL_TIME_STEP)

    if status:
        # 로그 데이터 기록
        status['time'] = current_virtual_time
        status['temperature'] = virtual_battery.temperature
        status['pwm'] = mock_hw.pwm_duty_cycle
        log_data.append(status)

        # CC→CV 전환 시점 기록
        if prev_state == 'CHARGING_CC' and status['state'] == 'CHARGING_CV':
            cc_to_cv_time = current_virtual_time
            print(f"★ CC→CV 전환! "
                  f"t={cc_to_cv_time:.1f}s, "
                  f"SoC={status['soc']*100:.2f}%, "
                  f"V={status['voltage']:.3f}V")

    # 주기적 요약 로그 (매 60초)
    if int(current_virtual_time) % 60 == 0 and status:
        print(f"[MAIN] t={int(current_virtual_time)}s "
              f"V={status['voltage']:.3f}V "
              f"I={status['current']:.3f}A "
              f"SoC(True)={status['soc']*100:.2f}% "
              f"SoC(Est)={status['estimated_soc']*100:.2f}% "
              f"PWM={status['pwm']:.1f}% "
              f"state={status['state']}")

    if charge_controller.state == 'FULL':
        break

    current_virtual_time += VIRTUAL_TIME_STEP

print("--- 시뮬레이션 완료 ---\n")

# 시각화
log_df = pd.DataFrame(log_data)
if log_df.empty:
    print("로그 데이터 없음.")
else:
    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.set_xlabel('시간 (초)')
    ax1.set_ylabel('전압(V) / SoC(%) / 온도(°C)')
    ax1.plot(log_df['time'], log_df['voltage'], label='전압 (V)', color='tab:blue')
    ax1.plot(log_df['time'], log_df['soc'] * 100, label='SoC (%)', linestyle=':', color='tab:green')
    ax1.plot(log_df['time'], log_df['temperature'], label='온도 (°C)', linestyle='-.', color='tab:orange')
    ax1.tick_params(axis='y')
    ax1.grid(True)

    # CC→CV 전환 시점 표시
    if cc_to_cv_time is not None:
        ax1.axvline(cc_to_cv_time, color='red', linestyle='--', label='CC→CV 전환')

    ax2 = ax1.twinx()
    ax2.set_ylabel('전류 (A) / PWM (%)')
    ax2.plot(log_df['time'], log_df['current'], label='전류 (A)', linestyle='--', color='tab:purple')
    ax2.plot(log_df['time'], log_df['pwm'], label='PWM (%)', linestyle='-.', color='tab:brown')
    ax2.tick_params(axis='y')

    # 범례 합치기
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left')

    plt.title('CC/CV 충전 시뮬레이션 (급락 완화 적용)')
    fig.tight_layout()
    plt.show()
