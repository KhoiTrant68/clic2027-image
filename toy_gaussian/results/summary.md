### d2_delta4  (d=2, Δ=4.0, rate=0.98 bits total)
D*=0.7448  W²=0.2211  W=0.4702  D*+W²=0.9659  2D*=1.4895  sample-W2 floor=0.0862

| decoder | D | D / (D*+W²) | P (sample) | P marg. LB | ‖v(X*,0)‖² |
|---|---|---|---|---|---|
| MMSE X* | 0.7320 | 0.758 | 0.4632 | 0.4657 | |
| exact T(X*) | 0.9411 | 0.974 | 0.0743 | 0.0407 | |
| natural, 1 Euler | 0.7352 | 0.761 | 0.4400 | 0.4428 | 0.0043 |
| natural, ODE 50 RK2 | 0.9360 | 0.969 | 0.0850 | 0.0583 | 0.0043 |
| independent, 1 Euler | 1.2678 | 1.313 | 1.0528 | 1.0590 | 0.5290 |
| independent, ODE 50 RK2 | 0.9337 | 0.967 | 0.0883 | 0.0605 | 0.5290 |
| minibatch_ot, 1 Euler | 0.9279 | 0.961 | 0.0994 | 0.0802 | 0.2083 |
| minibatch_ot, ODE 50 RK2 | 0.9359 | 0.969 | 0.0835 | 0.0563 | 0.2083 |
| exact_ot, 1 Euler | 0.9386 | 0.972 | 0.0885 | 0.0641 | 0.2210 |
| exact_ot, ODE 50 RK2 | 0.9387 | 0.972 | 0.0782 | 0.0490 | 0.2210 |

Exact OT path check (Thm 3): t, D measured vs D*+t²W², P_marg_lb vs (1-t)W

- t=0.0: D=0.7320 (theory 0.7448); P=0.4632, P_lb=0.4657 (theory 0.4702)
- t=0.2: D=0.7380 (theory 0.7536); P=0.3702, P_lb=0.3715 (theory 0.3762)
- t=0.4: D=0.7619 (theory 0.7801); P=0.2781, P_lb=0.2776 (theory 0.2821)
- t=0.6: D=0.8037 (theory 0.8244); P=0.1885, P_lb=0.1845 (theory 0.1881)
- t=0.8: D=0.8634 (theory 0.8863); P=0.1077, P_lb=0.0943 (theory 0.0940)
- t=1.0: D=0.9411 (theory 0.9659); P=0.0743, P_lb=0.0407 (theory 0.0000)

### d64_delta2  (d=64, Δ=2.0, rate=19.08 bits total)
D*=2.7825  W²=1.1233  W=1.0598  D*+W²=3.9058  2D*=5.5650  sample-W2 floor=1.9332

| decoder | D | D / (D*+W²) | P (sample) | P marg. LB | ‖v(X*,0)‖² |
|---|---|---|---|---|---|
| MMSE X* | 2.7826 | 0.712 | 1.6310 | 1.0703 | |
| exact T(X*) | 3.9013 | 0.999 | 1.9280 | 0.0797 | |
| natural, 1 Euler | 2.7911 | 0.715 | 1.6463 | 1.0200 | 0.0079 |
| natural, ODE 50 RK2 | 3.6337 | 0.930 | 1.9319 | 0.7773 | 0.0079 |
| independent, 1 Euler | 4.5327 | 1.161 | 2.0989 | 2.0027 | 1.7422 |
| independent, ODE 50 RK2 | 3.6702 | 0.940 | 1.9499 | 0.7885 | 1.7422 |
| minibatch_ot, 1 Euler | 2.8094 | 0.719 | 1.6082 | 1.0886 | 0.0285 |
| minibatch_ot, ODE 50 RK2 | 3.6591 | 0.937 | 1.9484 | 0.7979 | 0.0285 |
| exact_ot, 1 Euler | 3.3610 | 0.861 | 1.8326 | 0.7271 | 0.5764 |
| exact_ot, ODE 50 RK2 | 3.6486 | 0.934 | 1.9235 | 0.7266 | 0.5764 |

Exact OT path check (Thm 3): t, D measured vs D*+t²W², P_marg_lb vs (1-t)W

- t=0.0: D=2.7826 (theory 2.7825); P=1.6310, P_lb=1.0703 (theory 1.0598)
- t=0.2: D=2.8274 (theory 2.8274); P=1.6487, P_lb=0.8595 (theory 0.8479)
- t=0.4: D=2.9617 (theory 2.9622); P=1.6877, P_lb=0.6492 (theory 0.6359)
- t=0.6: D=3.1854 (theory 3.1869); P=1.7483, P_lb=0.4400 (theory 0.4239)
- t=0.8: D=3.4986 (theory 3.5014); P=1.8291, P_lb=0.2351 (theory 0.2120)
- t=1.0: D=3.9013 (theory 3.9058); P=1.9280, P_lb=0.0797 (theory 0.0000)