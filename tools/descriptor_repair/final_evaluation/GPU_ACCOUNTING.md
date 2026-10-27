# Allocated GPU-hours

Allocated GPU-hours = ElapsedRaw x gres/gpu, read from `sacct AllocTRES`,
not from what was requested.  Both aborted runs are included.

```
JobID             State ElapsedRaw                                                    AllocTRES 
------------ ---------- ---------- ------------------------------------------------------------ 
60298211      COMPLETED      14391 billing=128,cpu=128,energy=6549047,gres/gpu:a100=4,gres/gpu+ 
60298314      COMPLETED       8019 billing=128,cpu=128,energy=2812633,gres/gpu:a100=4,gres/gpu+ 
60302080      COMPLETED       2611 billing=128,cpu=128,energy=898773,gres/gpu:a100=4,gres/gpu=+ 
60305203      COMPLETED      14399 billing=128,cpu=128,energy=5489529,gres/gpu:a100=4,gres/gpu+ 
60334084      COMPLETED         85 billing=32,cpu=32,energy=37986,gres/gpu:a100=1,gres/gpu=1,m+ 
60335321      COMPLETED         52 billing=32,cpu=32,energy=14069,gres/gpu:a100=1,gres/gpu=1,m+ 
60335538      COMPLETED         69 billing=32,cpu=32,energy=25407,gres/gpu:a100=1,gres/gpu=1,m+ 
60335858      COMPLETED        263 billing=32,cpu=32,energy=116041,gres/gpu:a100=1,gres/gpu=1,+ 
60336194      COMPLETED       3606 billing=32,cpu=32,energy=1692101,gres/gpu:a100=1,gres/gpu=1+ 
60348595      COMPLETED        487 billing=32,cpu=32,energy=208249,gres/gpu:a100=1,gres/gpu=1,+ 
60348920         FAILED        309 billing=32,cpu=32,energy=186437,gres/gpu:a100=1,gres/gpu=1,+ 
60348997     CANCELLED+        254 billing=128,cpu=128,energy=92516,gres/gpu:a100=4,gres/gpu=4+ 
60349296        RUNNING       2068 billing=128,cpu=128,gres/gpu:a100=4,gres/gpu=4,mem=487802M,+ 
```

| job | elapsed s | GPUs | GPU-h | what |
|---|---|---|---|---|
| 60302080 | 2611 | 4 | 2.9011 | first session, whole-node trap |
| 60334084 | 85 | 1 | 0.0236 | GPU physics |
| 60335321 | 52 | 1 | 0.0144 | failed probe |
| 60335538 | 69 | 1 | 0.0192 | feature-rate probe |
| 60335858 | 263 | 1 | 0.0731 | smoke test |
| 60336194 | 3606 | 1 | 1.0017 | pilot training |
| 60348595 | 487 | 1 | 0.1353 | evaluation stages 1-2, patch, abort |
| 60348920 | 309 | 1 | 0.0858 | evaluation stages 3-6 |
| 60348997 | 254 | 4 | 0.2822 | whole-node allocation, cancelled |
| | | | **4.5364** | **of 8.000; 3.4636 remain** |

Correction: the pilot job was previously reported at 0.6314 GPU-h from its
elapsed time while still running.  Final `ElapsedRaw` is 3606 s, giving
1.0017 GPU-h, so the earlier total of 3.6619 was understated.

Three further four-GPU jobs ran the same day -- 60298211 (14391 s),
60298314 (8019 s), 60305203 (14399 s), about 40.9 GPU-h together.  They are
excluded as unrelated work.

The statistical analyses ran on a login node at no GPU cost.
