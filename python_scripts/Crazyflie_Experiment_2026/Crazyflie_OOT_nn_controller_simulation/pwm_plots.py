import pandas as pd
import matplotlib.pyplot as plt

firmware_df = pd.read_csv('pwm_log.csv')

gympybullet_pwm0_df = pd.read_csv('gympybullet_data/pwm0-0.csv', header=None, names=['time', 'pwm0'])
gympybullet_pwm1_df = pd.read_csv('gympybullet_data/pwm1-0.csv', header=None, names=['time', 'pwm1'])
gympybullet_pwm2_df = pd.read_csv('gympybullet_data/pwm2-0.csv', header=None, names=['time', 'pwm2'])
gympybullet_pwm3_df = pd.read_csv('gympybullet_data/pwm3-0.csv', header=None, names=['time', 'pwm3'])

firmware_df.columns = ['time', 'pwm0', 'pwm1', 'pwm2', 'pwm3']

figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

gympybullet_dataframes = [
    gympybullet_pwm3_df,
    gympybullet_pwm0_df,
    gympybullet_pwm2_df,
    gympybullet_pwm1_df
]

pwm_columns = ['pwm3', 'pwm0', 'pwm2', 'pwm1']

for idx, ax in enumerate(axes.flatten()):
    col = pwm_columns[idx]
    print(f"Plotting {col}")
    gympybullet_df = gympybullet_dataframes[idx]

    ax.plot(firmware_df['time'].values, firmware_df[col].values, label=f'Firmware {col}')
    ax.plot(gympybullet_df['time'].values, gympybullet_df[col].values, label=f'GymPyBullet {col}', alpha=0.7)

    ax.set_title(f'{col} comparison')
    ax.set_ylabel('PWM Value')
    ax.legend()

# fig, axes2 = plt.subplots(1, 1, figsize=(12, 4), sharex=True)
# axes2.set_title('NN Computation time (ms)')
# axes2.set_ylabel('Activation Time (ms)')
# axes2.plot(firmware_df['time'].values, firmware_df['activationTime'].values, label='Activation Time', color='orange')

for ax in axes[1]:
    ax.set_xlabel('Time')

plt.tight_layout()
plt.show()