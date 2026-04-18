# 🧠 Predicting Working Memory using Multimodal Fusion of fMRI & Behavioral Data with XAI Interpretation

<div align="center">

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![SHAP](https://img.shields.io/badge/XAI-SHAP-blueviolet?style=for-the-badge)
![HCP](https://img.shields.io/badge/Dataset-HCP--YA-informational?style=for-the-badge)

**🎓 BRAINEEWHA 딥러닝 분과 · 이화여자대학교**  
**📢 K-BIOX 포스터 발표 @ SYNAPSE — 8개 대학 연합 뇌인지과학 학부생 심포지엄**

</div>

---

## 👥 팀원

| 이름 | 전공 |
|------|------|
| 배주원 | 컴퓨터공학 |
| 김다영 | 뇌인지과학 |
| 김서영 | 통계학 |
| 유지우 | 컴퓨터공학 |
| 최서연 | 중어중문학 |
| 최연재 | 기계·바이오공학 |

---

## 📌 프로젝트 개요

신경영상 딥러닝 모델의 **블랙박스 문제**로 인한 임상 신뢰도 한계를 극복하고자,  
**fMRI + 행동 데이터**를 융합한 멀티모달 모델로 **작업 기억(Working Memory)** 수행 능력을 예측하고,  
**SHAP 기반 XAI**로 해석 가능성을 확보한 프로젝트입니다.

> 💡 작업 기억은 fMRI의 공간적 활성 패턴과 개인의 행동 특성 모두의 영향을 받습니다.

---

## 🗂️ 데이터셋

- **HCP-YA** (Human Connectome Project - Young Adult)
- N-back task fMRI 스캔 + 행동 데이터
- 6명이 데이터를 분할하여 로컬 처리 (1인당 약 400MB)

| 구분 | 내용 |
|------|------|
| 입력 특징 | Age, Gender, Flanker Inhibitory Control, Fluid Intelligence, Processing Speed |
| 타겟 변수 | Working Memory Task Accuracy (회귀) |

---

## 🏗️ 모델 아키텍처

```
┌─────────────────────────────┐    ┌──────────────────────────┐
│       fMRI Branch           │    │    Behavioral Branch     │
│                             │    │                          │
│  Input: (N,1,91,109,91)     │    │  Input: (N, num_features)│
│  Conv3D × 3                 │    │  Linear → 32             │
│  ReLU + MaxPool             │    │  ReLU + Dropout(0.3)     │
│  Flatten                    │    │  Linear → 64             │
│  Output: (N, 50336)         │    │  Output: (N, 64)         │
└────────────┬────────────────┘    └────────────┬─────────────┘
             │                                  │
             └──────────────┬───────────────────┘
                            ▼
                    [ Concatenation ]
                    Linear → 128
                    ReLU + Dropout(0.5)
                    Linear → 1
                            ▼
              🎯 Working Memory Task Accuracy
```

---

## 📊 Ablation Study 결과

| 모델 | Modality | Architecture | MSE | RMSE |
|------|----------|-------------|-----|------|
| fMRI Only | fMRI | 3D-CNN | 283.51 | 16.84 |
| Behavioral Only | Behavioral | MLP | 7455.57 | 86.35 |
| **Multimodal Fusion** | **fMRI + Behavioral** | **Fusion** | **257.82** | **26.06** |

> ✅ **Multimodal Fusion이 가장 낮은 MSE 달성** — 두 모달리티 융합의 유효성 검증

---

## 🔍 XAI — SHAP 분석

SHAP을 활용하여 **뇌 복셀별 기여도**를 시각화했습니다.

```
Subject 108828
├── 실제값:  68.15
└── 예측값:  67.91  ← 높은 정밀도
```

| 색상 | 의미 | 뇌 영역 |
|------|------|---------|
| 🔴 빨간 영역 (양수 기여) | 예측 점수를 높이는 복셀 | **DLPFC (배외측 전전두피질)** 중심 |
| 🔵 파란 영역 (음수 기여) | 예측 점수를 낮추는 복셀 | — |

> 🧬 DLPFC는 작업 기억의 핵심 영역으로 신경과학 문헌과 일치 → **생물학적 타당성 검증**

---

## ⚠️ 한계 및 Future Direction

- 샘플 수 부족 → cross-validation fold마다 검증 손실 분산이 크게 나타남
- 더 가벼운 모델 아키텍처로 일반화 성능 개선 실험 진행 중
- 두 모달리티 출력 차원 불균형 문제 (fMRI: 50,336차원 vs 행동: 64차원)  
  → Concatenation 시 fMRI 기여 편향 가능성 존재

---

## 🛠️ 사용 기술

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-XAI-blueviolet?style=flat-square)
![HCP](https://img.shields.io/badge/HCP--YA-Dataset-lightgrey?style=flat-square)

- **Deep Learning**: 3D-CNN, MLP, Feature Fusion
- **XAI**: SHAP (SHapley Additive exPlanations)
- **Dataset**: HCP-YA (Human Connectome Project)
