import matplotlib.pyplot as plt
import numpy as np
from aquarel import load_theme
from python_scripts.easy_plots.easy_plots import (
    compose_sources,
    moving_average,
    read_from_csv,
    calculate_statistics,
)

theme = (
    load_theme('scientific')
    .set_font(family='serif', size=22)
    .set_axes(bottom=True, top=True, left=True, right=True, xmargin=0, ymargin=0, zmargin=0, width=2)
    .set_grid(style='--', width=1)
    .set_ticks(draw_minor=True, pad_major=10)
    .set_lines(width=2.5)
    .set_legend(location='upper right', alpha=0.75)
)
theme.apply()

ws = 10

acrobatic = [
    'results/save-robust_policy_seed-123-09.28.2026_22.21.39/tb/run-tb_PPO_1-tag-rollout_ep_rew_mean.csv',
    'results/save-robust_policy_seed-12345-09.29.2026_01.12.14/tb/run-tb_PPO_1-tag-rollout_ep_rew_mean.csv',
    'results/save-robust_policy_seed-39-09.28.2026_11.46.35/tb/run-tb_PPO_1-tag-rollout_ep_rew_mean.csv',
    'results/save-robust_policy_seed-42-09.28.2026_10.34.18/tb/run-tb_PPO_1-tag-rollout_ep_rew_mean.csv',
    'results/save-robust_policy_seed-73-09.28.2026_17.16.19/tb/run-tb_PPO_1-tag-rollout_ep_rew_mean.csv'
]
acrobatic_steps, acrobatic_rewards = read_from_csv(acrobatic)
acrobatic_smoothed_mean_rewards = moving_average(calculate_statistics(acrobatic_rewards)[0], ws)
acrobatic_smoothed_std_rewards = moving_average(calculate_statistics(acrobatic_rewards)[1], ws)
acrobatic_smoothed_steps = np.array(acrobatic_steps[ws - 1:])

plt.rcParams['text.usetex'] = True
plt.plot(acrobatic_smoothed_steps, acrobatic_smoothed_mean_rewards, color='sandybrown', label='Acrobatic')
plt.fill_between(acrobatic_smoothed_steps, acrobatic_smoothed_mean_rewards - acrobatic_smoothed_std_rewards,
                 acrobatic_smoothed_mean_rewards + acrobatic_smoothed_std_rewards, alpha=0.35, color='sandybrown')
plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
plt.xlabel('Time Steps', labelpad=15)
plt.xlim(right=6_000_000)
plt.ylim(bottom=-40)
plt.ylabel('Reward', labelpad=15)
plt.legend(loc='upper right', bbox_to_anchor=(1.0, 1.2), borderaxespad=0.0, ncol=2)
plt.show()
