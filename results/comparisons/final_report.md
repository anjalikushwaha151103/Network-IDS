# XAI-NIDS Model & Feature Subset Comparison Report

Generated at: 2026-11-01 17:23:47

### Consolidated Performance Matrix

| Model    |   Feature Tier | Task       |   Accuracy |   Macro F1 |   Weighted F1 |   Precision |   Recall |   Fit Time (s) |   Inference Latency (ms/sample) |
|:---------|---------------:|:-----------|-----------:|-----------:|--------------:|------------:|---------:|---------------:|--------------------------------:|
| lightgbm |             48 | multiclass |   0.7688   |   0.453157 |      0.779711 |    0.458788 | 0.468718 |           3.87 |                          0.0365 |
| lightgbm |             30 | multiclass |   0.756279 |   0.478745 |      0.766874 |    0.501523 | 0.566724 |          18.36 |                          0.0312 |
| lightgbm |             20 | multiclass |   0.7774   |   0.481859 |      0.784946 |    0.483422 | 0.518494 |           5.88 |                          0.0305 |
| lightgbm |             15 | multiclass |   0.775    |   0.492341 |      0.780298 |    0.495164 | 0.520164 |           6.82 |                          0.0392 |

### Key Findings
- **Best Performing Model**: lightgbm (Weighted F1: 0.7849)
- **Fastest Model**: lightgbm (0.0305 ms/sample)
- **Feature Degradation**: Performance comparison across 49, 30, 20, and 15 feature subsets.
