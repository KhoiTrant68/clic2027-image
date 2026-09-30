"""Re-render every figure; figures whose results are not there yet are skipped with a message.

    uv run --no-project --with numpy,scipy,matplotlib,torch python figures/make_all.py
(torch is only needed the first time, to compute the cached dither-rate curves.)
"""
import fig_dither_rate
import fig_learnability
import fig_pending
import fig_toy_dp

if __name__ == "__main__":
    fig_toy_dp.main()
    fig_dither_rate.main()
    fig_learnability.main()
    fig_pending.fig_toyB()
    fig_pending.fig_gonogo1()
    fig_pending.fig_quantstat()
