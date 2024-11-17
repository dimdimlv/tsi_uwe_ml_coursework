# Logistic Regression Experiment Report

## Objective

- Optimize logistic regression for detecting defaults in an imbalanced dataset. 
- Evaluate various configurations to address class imbalance, optimize hyperparameters, and adjust decision thresholds.

## Experiment Configurations

|Experiment ID	|Resampling	|Class Weight	|Regularization (C)	|Threshold	|Description |
|---------------|------------|--------------|--------------------|-----------|-------------|
| 1             | None       | Balanced     | 1.0                | 0.5       | Baseline logistic regression with balanced class weight. |
| 2             | SMOTE      | Balanced     | 1.0                | 0.5       | Applied SMOTE to address class imbalance. |
| 3             | None       | Custom (0:1, 1:3) | 0.1          | 0.5       | Introduced custom weights for the default class. |
| 4             | SMOTE      | Custom (0:1, 1:3) | 0.1          | 0.647     | Combined SMOTE with custom weights and optimized the decision threshold. |
| 5             | SMOTE      | Balanced     | 0.01               | 0.5       | Applied stronger regularization with SMOTE and balanced class weight. |
| 6             | SMOTE      | Custom (0:1, 1:5) | 0.01          | 0.768     | Combined SMOTE with heavier custom weights and optimized the decision threshold. |

## Key Metrics


| Experiment ID | Accuracy | F1-Score (Default) | ROC-AUC | CV ROC-AUC |
|---------------|----------|--------------------|---------|------------|
| 1             | 73.84%   | 0.512              | 0.743   | 0.749      |
| 2             | 69.87%   | 0.674              | 0.751   | 0.752      |
| 3             | 76.14%   | 0.517              | 0.743   | 0.749      |
| 4             | 66.27%   | 0.706              | 0.752   | 0.752      |
| 5             | 69.30%   | 0.675              | 0.749   | 0.749      |
| 6             | 66.60%   | 0.705              | 0.750   | 0.750      |

## Insights

1.	**Baseline Performance**: Experiment 1 performed reasonably well but struggled with default detection, as indicated by a low F1-Score for the default class (0.512).
2. **Impact of SMOTE**: Experiments 2, 4, 5, and 6 demonstrate that SMOTE significantly improves the F1-Score for defaults (up to 0.706 in Experiment 4).
3.	**Threshold Optimization**:
   - Experiments 4 and 6 show the importance of adjusting the decision threshold, leading to higher F1-Scores for the minority class.
   - Experiment 4 achieved the best overall balance with an F1-Score of 0.706 and ROC-AUC of 0.752.
4.	**Class Weighting**: Custom weights ({0:1, 1:3} and {0:1, 1:5}) further improved performance for the default class, especially when combined with SMOTE and threshold optimization.
5.	**Regularization Strength (C)**: Stronger regularization (lower C) improved generalization slightly, but the effect on F1-Score and ROC-AUC was minimal compared to SMOTE and weighting.

## Conclusion
Best Configuration:

Experiment 4: 
- SMOTE + Custom Weight {0:1, 1:3} with Threshold Optimization.
- Score (Default): 0.706
- AUC: 0.752
Best for prioritizing default detection.