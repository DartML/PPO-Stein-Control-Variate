import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pickle as p
import seaborn as sns
import fnmatch
import json
import glob
import os

set_prefix = ['ppo-gradient_mlp', 'ppo-gradient_state',
              'ppo-deepmind',
              'ppo-reward_mlp', 'ppo-reward_state']

legends = {
    set_prefix[0]:'MinVar-MLP',
    set_prefix[1]:'MinVar-State Baseline',
    set_prefix[2]:'Value',
    set_prefix[3]:'FitQ-MLP',
    set_prefix[4]:'FitQ-State Baseline'
}

flatui = ["#e74c3c", "#2ecc71", sns.xkcd_rgb['periwinkle'],
          sns.xkcd_rgb['dusty purple'],
          sns.xkcd_rgb['amber'], sns.xkcd_rgb['sky blue'], "#95a5a6" ,sns.xkcd_rgb['purplish blue']]

colors = {}
for idx, prefix in enumerate(set_prefix):
    colors[prefix] = flatui[idx]


"""
averaged over seeds = list(range(13, 250, 30))
"""

log_x = np.log10([2500, 5000, 7500, 10000, 12500, 15000])

x='0.487811 -0.19878311 -0.59551305 -0.8650713  -1.0553298  -1.27113008'
log_mc=[float(i) for i in x.split()]

x='0.12229453 -0.53009254 -0.86770993 -1.14170933 -1.32843781 -1.50873899'
log_grad_state=[float(i) for i in x.split()]

x='0.2080559  -0.50720716 -0.57206059 -1.12673986 -1.41672766 -1.49526441'
log_reward_state=[float(i) for i in x.split()]

x='-3.68289948 -5.02937841 -5.52095079 -5.96457911 -4.90065193 -7.27007866'
log_grad_mlp=[float(i) for i in x.split()]

x='-2.44885826 -3.23678112 -3.67908597 -5.45972824 -5.3687458  -7.46356487'
log_reward_mlp=[float(i) for i in x.split()]


sns.set_style('ticks')
plt.figure(figsize=(5.4,6),facecolor='blue')
plt.ylim([-9.0, 1.0])
plt.xticks([np.log10(2500), np.log10(5000), np.log10(7500),np.log10(10000), np.log10(15000),
            np.log10(20000)]
            ,['2.5k', '5k', '7.5k', '10k', '15k', '20k'] ,fontsize=18)
# plt.yticks([-2.5, -2.0, -1.5, -1, -.5], fontsize=18)


# plt.plot(log_x, log_qprop, lw=2, marker='P',markersize =8, linestyle='--', color=colors['ppo-qprop'])
plt.plot(log_x, log_mc, lw=2, marker='P',markersize =8, linestyle='-.', color=colors['ppo-deepmind'], label=legends['ppo-deepmind'])
plt.plot(log_x, log_grad_state, lw=2, marker='^',markersize =8, linestyle='-', color=colors['ppo-gradient_state'], label=legends['ppo-gradient_state'])
# plt.plot(log_x, log_grad_quadratic, lw=2, marker='^',markersize =8, linestyle='-', color=colors['ppo-gradient_quadratic'])
plt.plot(log_x, log_grad_mlp, lw=2, marker='^',markersize =8, linestyle='-', color=colors['ppo-gradient_mlp'],label=legends['ppo-gradient_mlp'])

# plt.plot(log_x, log_qprop, lw=2, marker='P',markersize =8, linestyle='--', color=colors['ppo-qprop'])
# plt.plot(log_x, log_mc, lw=2, marker='P',markersize =8, linestyle='-.', color=colors['ppo-deepmind'])
plt.plot(log_x, log_reward_state, lw=2, marker='s',markersize =8, linestyle='-', color=colors['ppo-reward_state'], label=legends['ppo-reward_state'])
# plt.plot(log_x, log_reward_quadratic, lw=2, marker='s',markersize =8, linestyle='-', color=colors['ppo-reward_quadratic'])
plt.plot(log_x, log_reward_mlp, lw=2, marker='s',markersize =8, linestyle='-', color=colors['ppo-reward_mlp'], label=legends['ppo-reward_mlp'])
plt.legend(loc='lower left')

plt.savefig('Eval_Walker.pdf',  bbox_inches='tight')


import pylab
fig = pylab.figure()
legend_fig = pylab.figure()



# fig.gca().plot(range(10), pylab.randn(10),lw=2, marker='P',markersize =8, linestyle='-.',  color=colors['ppo-deepmind'], label=legends['ppo-deepmind'])
# fig.gca().plot(range(10), pylab.randn(10),  lw=2, marker='^',markersize =8, linestyle='-', color=colors['ppo-gradient_mlp'], label=legends['ppo-gradient_mlp'])
# fig.gca().plot(range(10), pylab.randn(10),  lw=2, marker='^',markersize =8, linestyle='-', color=colors['ppo-gradient_state'], label=legends['ppo-gradient_state'])
# fig.gca().plot(range(10), pylab.randn(10), lw=2, marker='s',markersize =8, linestyle='-',color=colors['ppo-reward_mlp'], label=legends['ppo-reward_mlp'])
# fig.gca().plot(range(10), pylab.randn(10), lw=2, marker='s',markersize =8, linestyle='-',color=colors['ppo-reward_state'], label=legends['ppo-reward_state'])

# legend_properties = {'weight':'normal', 'size':18}
# legend = pylab.figlegend(*fig.gca().get_legend_handles_labels(), loc = 'center', prop=legend_properties)
# # legend.get_frame().set_color('0.70')
# legend_fig.canvas.draw()
# legend_fig.savefig('eval_legend.pdf',
#     bbox_inches=legend.get_window_extent().transformed(legend_fig.dpi_scale_trans.inverted()))



