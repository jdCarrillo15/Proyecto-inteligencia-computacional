# Model Performance Requirements and Limitations

## 📋 Project Context

**System Type:** Plant Disease Detection System  
**Total Classes:** 15 (4 crops: Apple, Corn, Potato, Tomato)  
**Dataset Size:** 28,428 images  
**Class Balance:** 1.18:1 ratio (well-balanced dataset)  
**Image Resolution:** 224x224 pixels

---

## 🎯 Critical Decision: False Negatives vs False Positives

### Stakeholder Requirements Analysis

#### ❓ Key Questions

1. **What is more critical: Detecting diseases or avoiding false alarms?**
   - **Answer:** Detecting diseases is MORE critical
   - **Rationale:** Missing a disease can lead to crop loss and spread to other plants

2. **Do we accept 60% accuracy on minority classes?**
   - **Answer:** NO - Too risky for agricultural applications
   - **Minimum threshold:** 70% recall per class

3. **Or do we prefer 85% accuracy with false positives?**
   - **Answer:** YES - False positives are acceptable
   - **Rationale:** Better to treat unnecessarily than miss a disease

4. **Which classes are unacceptable to fail on?**
   - **Critical classes (high economic impact):**
     - `Potato___Late_blight` (causes total crop loss)
     - `Tomato___Late_blight` (highly contagious)
     - `Corn_(maize)___Northern_Leaf_Blight` (rapid spread)
   - **Target recall for critical classes:** ≥ 80%

---

## 🎯 Performance Targets

### Priority 1: Minimize False Negatives
**Objective:** Never tell a farmer their diseased plant is healthy

- **Recall (Sensitivity) Target:** ≥ 70% on ALL classes
- **Critical Disease Recall:** ≥ 80%
- **Healthy Plant Recall:** ≥ 75% (can be slightly lower)

### Priority 2: Overall Model Quality
**Objective:** Maintain balanced performance across all disease types

- **Macro F1-Score Target:** ≥ 75%
- **Weighted F1-Score Target:** ≥ 78%
- **Overall Accuracy Target:** ≥ 80%

### Priority 3: Precision (Acceptable Range)
**Objective:** Minimize false alarms while prioritizing disease detection

- **Minimum Precision:** ≥ 65% per class
- **Average Precision:** ≥ 73%
- **Acceptable false positive rate:** ≤ 30%

---

## 📊 Acceptable Performance Ranges

### Class-Level Metrics

| Metric | Minimum | Target | Ideal |
|--------|---------|--------|-------|
| **Recall (per class)** | 70% | 75% | 85% |
| **Precision (per class)** | 65% | 73% | 80% |
| **F1-Score (per class)** | 67% | 74% | 82% |

### Overall Metrics

| Metric | Minimum | Target | Ideal |
|--------|---------|--------|-------|
| **Macro F1-Score** | 75% | 78% | 85% |
| **Weighted F1-Score** | 78% | 82% | 88% |
| **Overall Accuracy** | 80% | 84% | 90% |
| **Macro Recall** | 70% | 75% | 82% |
| **Macro Precision** | 68% | 73% | 80% |

---

## ⚠️ Critical Failure Thresholds

### Unacceptable Scenarios (Model Rejection)

1. **Any class with Recall < 60%**
   - Risk: Too many missed diseases
   - Action: Retrain with class weighting or data augmentation

2. **Critical disease classes with Recall < 75%**
   - Classes: Late blight (Potato & Tomato), Northern Leaf Blight (Corn)
   - Action: Apply targeted augmentation or cost-sensitive learning

3. **Macro F1-Score < 70%**
   - Risk: Poor overall model quality
   - Action: Revise architecture or training strategy

4. **More than 3 classes below target thresholds**
   - Risk: Systemic model weakness
   - Action: Increase model capacity or improve data quality

---

## 🔄 Class Imbalance Handling Strategy

### Current Status
- **Balance Ratio:** 1.18:1 (excellent)
- **Strategy:** Preventive approach for future scalability

### Implemented Techniques

1. **Class Weights (Mandatory)**
   - Apply inverse frequency weighting
   - Formula: `weight = total_samples / (num_classes * class_samples)`
   - Ensures minority classes contribute equally to loss

2. **Data Augmentation (Aggressive for minority classes)**
   - Rotation: ±20°
   - Zoom: 0.8-1.2x
   - Horizontal flip: Yes
   - Brightness adjustment: ±20%
   - Target: 2,500+ augmented samples per class

3. **Stratified Splitting**
   - Train/validation split maintains class distribution
   - Prevents validation bias toward majority classes

---

## 🎓 Model Training Priorities

### Training Strategy

1. **Loss Function:** Categorical Crossentropy with Class Weights
2. **Optimization Goal:** Maximize Macro F1-Score (not just accuracy)
3. **Early Stopping:** Monitor validation F1-score (patience: 10 epochs)
4. **Learning Rate:** Adaptive with ReduceLROnPlateau (patience: 5 epochs)

### Validation Approach

- **Validation Split:** 20% stratified
- **Primary Metric:** Macro F1-Score
- **Secondary Metrics:** Recall per class, Confusion Matrix
- **Monitoring:** Track recall for critical disease classes separately

---

## 📈 Success Criteria

### Definition of Success (All must be met)

✅ **Macro F1-Score ≥ 75%**  
✅ **All classes: Recall ≥ 70%**  
✅ **Critical diseases: Recall ≥ 80%**  
✅ **No class with F1-Score < 67%**  
✅ **Overall Accuracy ≥ 80%**

### Definition of Failure (Any single criterion)

❌ **Any class with Recall < 60%**  
❌ **Macro F1-Score < 70%**  
❌ **More than 2 critical diseases with Recall < 75%**  
❌ **Overall Accuracy < 75%**

---

## 🚀 Implementation Recommendations

### Phase 1: Baseline Model
- Train with class weights
- Target: Meet minimum thresholds
- Expected: Macro F1 ~73-76%

### Phase 2: Optimization
- Fine-tune hyperparameters
- Adjust augmentation intensity
- Target: Reach target thresholds
- Expected: Macro F1 ~76-80%

### Phase 3: Production Readiness
- Ensemble methods if needed
- Test time augmentation (TTA)
- Target: Reach ideal thresholds
- Expected: Macro F1 ~80-85%

---

## 📝 Stakeholder Agreement

**Approved by:** Project Team  
**Date:** 2025-11-19  
**Review Cycle:** After each training iteration

### Key Agreements

1. ✅ **False negatives are worse than false positives**
2. ✅ **70% recall minimum is acceptable for minority classes**
3. ✅ **85% accuracy with some false positives is preferred over 80% with missed diseases**
4. ✅ **Critical diseases (blights) require extra attention (≥80% recall)**
5. ✅ **Macro F1-Score ≥ 75% is the primary success metric**

---

## 🔧 Configuration Integration

These requirements are enforced in `backend/config.py`:

- Class weights calculation
- Data augmentation parameters
- Training callbacks (F1-score monitoring)
- Early stopping criteria
- Model evaluation thresholds

**Next Steps:**
1. Implement class weight calculation in training script
2. Add F1-score callback for monitoring
3. Generate detailed per-class metrics reports
4. Create confusion matrix visualization for critical classes
