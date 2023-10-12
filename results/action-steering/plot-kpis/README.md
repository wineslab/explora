## Note

The results in each subfolder are complete, i.e., they prompt the corresponding change on KPIs enforced by the specific AR policy over the baseline for all slices and KPIs. Note that we do not focus on all the combinations, but in the paper we report the main results, i.e., 
- slices 0 and 2 (*eMBB* and *URLLC* respectively because are the slices where the two agents HT and LL act primarily).
- *Tx bitrate* and *DL buffer size* are the main KPIs of reference.

### Organization

- [`min_reward_vs_baseline/`](action-steering/plot-kpis/min_reward_vs_baseline/) produce Fig. 9. Specifically, Fig. 9(a) is derived from `plot_gains_urllc-trf1_exp_SL2` and Fig. 9(b) from `plot_gains_urllc-trf2_exp_SL2` in the corresponding subfolder.
- [`max_reward_vs_baseline/`](action-steering/plot-kpis/max_reward_vs_baseline/) and [`imp_bitrate_vs_baseline/`](action-steering/plot-kpis/imp_bitrate_vs_baseline/) produce Fig. 10. Specifically, Fig. 10(a) is derived from `plot_gains_embb-trf1_exp_SL0` and Fig. 10(b) is derived from `plot_gains_embb-trf2_exp_SL0` in the corresponding subfolders.

In both cases, the plots have been modified acting directly on the TikZ code to produce the corresponding figures. See the final TikZ code in [`/paper-plots/sec-6.3/`](/paper-plots/sec-6.3/).
