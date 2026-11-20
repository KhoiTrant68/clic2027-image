"""GC: perceptual (GAN) learned image codec for CLIC 2027 with bit-exact entropy decoding.

Layout
- layers.py   float building blocks (residual blocks, FiLM conditioning, sub-pixel upsampling)
- model.py    GCodec: ELIC-style analysis/synthesis, integer hyper-synthesis and checkerboard x channel-slice
              entropy-parameter networks (clic27.entropy.intnet), multi-realism beta, rate levels, quality maps
- disc.py     conditional multi-scale PatchGAN (HiFiC-style)
- losses.py   rate control, MSE, LPIPS, Wasserstein distortion, GAN losses
- reward.py   Q-hat v2: differentiable preference model fitted on CLIC human pairwise ratings (Bradley-Terry)
- codec.py    bitstream: header, quality map, z, 10 y blobs; compress / decompress
- rans_c.py   C rANS (same stream format as clic27.entropy.rans), numpy fallback
- alloc.py    per-image quality-map search (worst-crop objective) + corpus byte allocation (multiple-choice knapsack)
- train.py    training stages rd -> gan -> rate fine-tune -> reward alignment
- pipeline.py end-to-end driver (python -m clic27.gc.pipeline --config ... --stages ...)
- decode_main.py  CLIC decoder entry point (bs.zip -> images/*.png)
"""
