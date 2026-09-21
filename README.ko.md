# FedHyDRA

**Federated Learning with Dual-Scale Hybrid Divergence and Relation-Aware Embedding for Structured Non-IID Data** 논문의 **저자 공식 구현**입니다.

[English](README.md) | [한국어](README.ko.md)

**저자:** Wooseok Shin, Janghoon Yang, Zhiqiang Shen, Jitae Shin.

FedHyDRA는 라벨 분포와 특징 분포로 클라이언트 관계를 학습하고, 구조적 non-IID 환경에서 연합학습 업데이트를 집계합니다.

## 방법

1. 클라이언트는 Laplace smoothing을 적용한 라벨 히스토그램과 공통 random Fourier feature(RFF) 평균을 계산합니다.
2. 서버는 Jensen–Shannon divergence와 RFF-MMD를 적응형 가중치로 결합합니다.
3. 가중 클라이언트 그래프와 2층 변분 그래프 오토인코더(VGAE)로 관계를 반영한 임베딩을 생성합니다.
4. Full-covariance Gaussian mixture model이 소프트 클러스터 소속도를 계산합니다.
5. 서버는 전체 클라이언트의 저장된 혼합 비율과 현재 참여자의 업데이트를 집계하며, 결합 가중치·임베딩·클러스터를 설정 주기에 따라 갱신합니다.

**FedHyDRA, FedAvg, FedProx**와 고정 결합 가중치, JSD-only, MMD-only, no-VGAE, hard-membership ablation을 제공합니다. No-VGAE 실험은 정규화 인접 행렬의 spectral embedding을 사용합니다.

## 설치

**Python 3.10 이상**과 PyTorch, torchvision, NumPy, scikit-learn, PyYAML, tqdm을 사용합니다. Smoke 실험은 CPU에서 실행할 수 있으며 이미지 데이터셋 실험은 CUDA를 지원합니다.

```bash
git clone https://github.com/wsshinskku/FedHyDRA.git
cd FedHyDRA
python -m venv .venv
```

Linux/macOS에서는 `source .venv/bin/activate`, PowerShell에서는 `.venv\Scripts\Activate.ps1`로 가상환경을 활성화한 뒤 설치합니다.

```bash
python -m pip install -e ".[dev]"
```

## 빠른 시작

```bash
fedhydra train --config configs/smoke.yaml --run-dir runs/smoke-check
pytest -q
```

합성 데이터 기반 smoke 설정은 클라이언트 4개, 라운드별 참여자 2개, 통신 2라운드를 사용합니다. 로컬 학습, 요약 통계 추출, 그래프 학습, 클러스터링, 집계, 평가, 체크포인트 저장까지 수행합니다.

## 데이터셋과 실험

| 데이터셋 | 구조적 분할 | Dirichlet 분할 |
|---|---|---|
| CIFAR-100 | [cifar100.yaml](configs/cifar100.yaml) | [cifar100-random.yaml](configs/cifar100-random.yaml) |
| Tiny-ImageNet | [tiny-imagenet.yaml](configs/tiny-imagenet.yaml) | [tiny-imagenet-random.yaml](configs/tiny-imagenet-random.yaml) |
| STL-10 | [stl10.yaml](configs/stl10.yaml) | [stl10-random.yaml](configs/stl10-random.yaml) |

CIFAR-100과 STL-10은 `data.download: true`일 때 torchvision으로 다운로드합니다. Tiny-ImageNet은 다음 명령으로 준비합니다.

```bash
python scripts/download_tiny_imagenet.py --destination data
```

Tiny-ImageNet 로더는 기본 `train/<class>/images`, `val/images`, `val_annotations.txt` 구조를 읽습니다. 데이터 저장 경로는 `data.root`로 지정합니다.

```bash
fedhydra train --config configs/cifar100.yaml
fedhydra inspect-partition --config configs/cifar100.yaml
fedhydra train --config configs/cifar100.yaml --set experiment.seed=1
fedhydra train --config configs/cifar100.yaml --set federated.method=fedavg --set experiment.name=cifar100-structured-fedavg
fedhydra train --config configs/fedprox-cifar100.yaml
```

CIFAR-100 설정은 클라이언트 200개, 라운드별 참여자 20개, 통신 300라운드, 로컬 5에포크, 너비 배율 0.5의 MobileNetV2를 사용합니다. 구조적 분할은 인접 클래스 그룹의 중첩, 경계 클라이언트, 라벨을 유지하는 색상 변환을 결합합니다. 샘플 인덱스와 소속도는 `partition.json`에 저장합니다.

Ablation 설정: [고정 결합 가중치](configs/ablation-fixed-hybrid.yaml), [JSD only](configs/ablation-jsd-only.yaml), [MMD only](configs/ablation-mmd-only.yaml), [no VGAE](configs/ablation-no-vgae.yaml), [hard memberships](configs/ablation-hard-gmm.yaml).

## 여러 시드 실행과 체크포인트

```bash
python scripts/run_five_seeds.py --config configs/cifar100.yaml
python scripts/summarize_runs.py --root runs --experiment cifar100-structured-fedhydra --output runs/cifar100-structured-summary.json
fedhydra train --config configs/cifar100.yaml --resume runs/cifar100-structured-fedhydra/seed-0/checkpoints/round-0020.pt
```

시드 실행 스크립트는 0–4를 순차 실행합니다. 요약 스크립트는 평균, 표본 표준편차, 정규근사 95% 신뢰구간을 JSON과 Markdown으로 저장합니다.

## 출력 파일

기본 경로는 `runs/<experiment>/seed-<seed>/`이며 `--run-dir`로 변경할 수 있습니다.

| 파일 | 내용 |
|---|---|
| `config.json` | 실제 적용된 실험 설정 |
| `implementation_manifest.json` | 패키지 버전, 장치, 모델 크기, 알고리즘 설정 |
| `partition.json` | 샘플 인덱스와 구조적 소속도 |
| `metrics.csv` | 평가 지표와 서버 진단값 |
| `round_times.csv` | 각 통신 라운드의 실행 시간 |
| `summary.json` | Balanced accuracy, 수렴, 안정성, 평균 라운드 시간 |
| `checkpoints/` | 학습 재개용 상태 |

Balanced top-1 accuracy는 클래스별 recall의 평균입니다. 수렴 시점은 최종 정확도의 90%에 처음 도달한 평가 라운드이며, CoV와 Min/Mean은 전체 평가 시점으로 계산합니다. 라운드 시간에는 로컬 학습, 요약 통계 생성, 서버 갱신, 집계가 포함되고 평가는 별도 구간에서 수행합니다. 구조적 색상 변환은 학습 데이터에 적용하며 평가는 원본 테스트 분할을 사용합니다.

## 저장소 구성

| 경로 | 역할 |
|---|---|
| [src/fedhydra](src/fedhydra) | 데이터, 모델, 클라이언트/서버 학습, 핵심 방법, 지표 |
| [configs](configs) | 데이터셋, 비교 방법, ablation 설정 |
| [scripts](scripts) | 데이터 준비, 다중 시드 실행, 결과 요약 |
| [tests](tests) | 수식, 데이터, 체크포인트, 통합 검증 |
| [구현 상세](docs/IMPLEMENTATION_NOTES.md) | 수식 대응, 갱신 주기, 집계 방식 |
| [구조적 데이터 분할](docs/STRUCTURED_PARTITIONS.md) | 분할 생성과 사용자 지정 API |
| [재현성 가이드](docs/REPRODUCIBILITY.md) | 시드 설정과 결과 보고 방법 |

## 인용

```bibtex
@unpublished{shin2026fedhydra,
  title  = {Federated Learning with Dual-Scale Hybrid Divergence and Relation-Aware Embedding for Structured Non-IID Data},
  author = {Shin, Wooseok and Yang, Janghoon and Shen, Zhiqiang and Shin, Jitae},
  note   = {Manuscript},
  year   = {2026}
}
```

기계 판독용 인용 정보는 [CITATION.cff](CITATION.cff)에 있습니다. 소스 코드는 [MIT License](LICENSE)로 배포합니다.
