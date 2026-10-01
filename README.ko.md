# FedSOAR

**Federated Learning under Client Heterogeneity via Soft Overlap-Aware Relations** 논문의 **저자 공식 구현**입니다.

[English](README.md) | [한국어](README.ko.md)

**저자:** Wooseok Shin, Janghoon Yang, Zhiqiang Shen, Jitae Shin.

FedSOAR(Federated Learning via Soft Overlap-Aware Relations)는 라벨·특징 분포가 여러 잠재 그룹에 걸쳐 겹치는 클라이언트 관계를 모델링합니다. 적응형 JSD-MMD 관계, VGAE 임베딩, 소프트 GMM 소속도를 결합해 하나의 공유 글로벌 모델을 학습합니다.

기존 FedHyDRA를 수정된 Applied Soft Computing 원고에 맞춰 갱신한 저장소입니다. 기존 링크를 유지하도록 저장소 주소는 `wsshinskku/FedHyDRA`로 유지합니다. 게재 승인이나 출판을 의미하지 않습니다.

## 방법

1. 클라이언트는 Laplace smoothing을 적용한 라벨 히스토그램과 공통 random Fourier feature(RFF) 평균을 계산합니다.
2. 서버는 Jensen–Shannon divergence와 RFF-MMD를 적응형 가중치로 결합합니다.
3. 가중 클라이언트 그래프와 2층 변분 그래프 오토인코더(VGAE)로 관계를 반영한 임베딩을 생성합니다.
4. Full-covariance Gaussian mixture model이 소프트 클러스터 소속도를 계산합니다.
5. 서버는 전체 클라이언트의 저장된 혼합 비율과 현재 참여자의 업데이트를 집계하며, 결합 가중치·임베딩·클러스터를 설정 주기에 따라 갱신합니다.

**FedSOAR, FedAvg, FedProx**와 고정 결합 가중치, JSD-only, MMD-only, no-VGAE, hard-membership ablation을 제공합니다. No-VGAE 실험은 라벨 히스토그램과 RFF 요약을 연결한 특징에 GMM을 직접 적용합니다. 기존 spectral embedding 옵션은 별도 구현 선택지로 유지합니다.

## 원고의 보고 결과와 구현 범위

수정 원고의 Table 5에 보고된 balanced top-1 accuracy입니다. 이번 smoke test나 새 재현 실험에서 얻은 값이 아닙니다.

| 데이터셋 | Random non-IID (%) | Structured non-IID (%) | 구조적 분할에서 가장 높은 다른 비교 방법 대비 향상 (%p) |
|---|---:|---:|---:|
| CIFAR-100 | 89.2 | 89.0 | 1.7 |
| Tiny-ImageNet | 89.0 | 88.6 | 2.0 |
| STL-10 | 89.7 | 89.5 | 2.2 |

원고의 주된 이점은 구조적 non-IID 분할에서 나타납니다. Random 분할에서는 FedWaD가 CIFAR-100에서 0.1%p 높고, Tiny-ImageNet에서는 동률이며, STL-10에서는 더 일찍 수렴합니다.

주요 학습 파이프라인, FedAvg/FedProx 비교, 구성요소 ablation과 smoothing·그래프 밀도·갱신 주기 설정을 제공합니다. 원 실험의 정확한 샘플 인덱스, 모든 비교 방법의 구현, Section 5.4의 동적 스트레스 실험 실행기는 포함되어 있지 않습니다. 집계식의 구현 규칙과 재현 범위는 [원고 대응 문서](docs/MANUSCRIPT_ALIGNMENT.md)에 정리했습니다.

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
fedsoar train --config configs/smoke.yaml --run-dir runs/smoke-check
pytest -q
```

기본 명령은 `fedsoar`이며 `python -m fedsoar`도 지원합니다. 기존 `fedhydra` 명령·import·YAML 키·방법 이름도 계속 사용할 수 있습니다. 새 설정은 `fedsoar:`와 `federated.method: fedsoar`를 사용합니다.

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
fedsoar train --config configs/cifar100.yaml
fedsoar inspect-partition --config configs/cifar100.yaml
fedsoar train --config configs/cifar100.yaml --set experiment.seed=1
fedsoar train --config configs/cifar100.yaml --set federated.method=fedavg --set experiment.name=cifar100-structured-fedavg
fedsoar train --config configs/fedprox-cifar100.yaml
```

CIFAR-100 설정은 클라이언트 200개, 라운드별 참여자 20개, 통신 300라운드, 로컬 5에포크, 너비 배율 0.5의 MobileNetV2를 사용합니다. 구조적 분할은 인접 클래스 그룹의 중첩, 경계 클라이언트, 라벨을 유지하는 색상 변환을 결합합니다. 샘플 인덱스와 소속도는 `partition.json`에 저장합니다.

Ablation 설정: [고정 결합 가중치](configs/ablation-fixed-hybrid.yaml), [JSD only](configs/ablation-jsd-only.yaml), [MMD only](configs/ablation-mmd-only.yaml), [no VGAE](configs/ablation-no-vgae.yaml), [hard memberships](configs/ablation-hard-gmm.yaml).

## 여러 시드 실행과 체크포인트

```bash
python scripts/run_five_seeds.py --config configs/cifar100.yaml
python scripts/summarize_runs.py --root runs --experiment cifar100-structured-fedsoar --output runs/cifar100-structured-summary.json
fedsoar train --config configs/cifar100.yaml --resume runs/cifar100-structured-fedsoar/seed-0/checkpoints/round-0020.pt
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
| [src/fedsoar](src/fedsoar) | FedSOAR import 및 명령 진입점 |
| [src/fedhydra](src/fedhydra) | 공통 구현과 기존 API 호환 |
| [configs](configs) | 데이터셋, 비교 방법, ablation 설정 |
| [scripts](scripts) | 데이터 준비, 다중 시드 실행, 결과 요약 |
| [tests](tests) | 수식, 데이터, 체크포인트, 통합 검증 |
| [구현 상세](docs/IMPLEMENTATION_NOTES.md) | 수식 대응, 갱신 주기, 집계 방식 |
| [구조적 데이터 분할](docs/STRUCTURED_PARTITIONS.md) | 분할 생성과 사용자 지정 API |
| [재현성 가이드](docs/REPRODUCIBILITY.md) | 시드 설정과 결과 보고 방법 |

기본 갱신 주기는 결합 가중치 5라운드, 그래프/VGAE 10라운드, GMM 소속도 20라운드입니다.

## 인용

```bibtex
@unpublished{shin2026fedsoar,
  title  = {Federated Learning under Client Heterogeneity via Soft Overlap-Aware Relations},
  author = {Shin, Wooseok and Yang, Janghoon and Shen, Zhiqiang and Shin, Jitae},
  note   = {Manuscript prepared for Applied Soft Computing},
  year   = {2026}
}
```

기계 판독용 인용 정보는 [CITATION.cff](CITATION.cff)에 있습니다. 소스 코드는 [MIT License](LICENSE)로 배포합니다.
