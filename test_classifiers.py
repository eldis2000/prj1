"""
GPU 경량 이미지 분류 모델 벤치마크 스크립트
=========================================
대표적인 경량 이미지 분류 모델들의 GPU 추론 성능을 비교 테스트합니다.

테스트 모델:
  1. MobileNet V2   - Google의 모바일 최적화 모델
  2. MobileNet V3 Small - MobileNet V2 후속, 초경량
  3. EfficientNet B0 - 효율적인 스케일링 기반 모델
  4. ShuffleNet V2   - 채널 셔플 기반 경량 모델
  5. SqueezeNet 1.1  - Fire 모듈 기반 초경량 모델
  6. ResNet-18       - 잔차 연결 기반 경량 모델 (기준선)

사용법:
  python test_classifiers.py                    # 랜덤 이미지로 테스트
  python test_classifiers.py --image photo.jpg  # 특정 이미지로 테스트
  python test_classifiers.py --warmup 20 --runs 100  # 벤치마크 횟수 조정
  python test_classifiers.py --cpu              # CPU 강제 사용
"""

import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from tabulate import tabulate

# ─────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────
IMAGENET_CLASSES_URL = (
    "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
)
SAMPLE_IMAGE_URL = (
    "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/"
    "Cat_November_2010-1a.jpg/1200px-Cat_November_2010-1a.jpg"
)

# ImageNet 표준 전처리
PREPROCESS = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


# ─────────────────────────────────────────────
# 모델 정의
# ─────────────────────────────────────────────
def get_model_configs():
    """테스트할 모델 목록과 설정을 반환합니다."""
    return [
        {
            "name": "MobileNet V2",
            "builder": lambda: models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT),
            "description": "Google의 inverted residual 기반 모바일 최적화 모델",
        },
        {
            "name": "MobileNet V3 Small",
            "builder": lambda: models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT),
            "description": "NAS + NetAdapt 기반 초경량 모델",
        },
        {
            "name": "EfficientNet B0",
            "builder": lambda: models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT),
            "description": "compound scaling 기반 효율적 모델",
        },
        {
            "name": "ShuffleNet V2 x1.0",
            "builder": lambda: models.shufflenet_v2_x1_0(weights=models.ShuffleNet_V2_X1_0_Weights.DEFAULT),
            "description": "채널 셔플 기반 경량 모델",
        },
        {
            "name": "SqueezeNet 1.1",
            "builder": lambda: models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.DEFAULT),
            "description": "Fire 모듈 기반 초경량 모델",
        },
        {
            "name": "ResNet-18",
            "builder": lambda: models.resnet18(weights=models.ResNet18_Weights.DEFAULT),
            "description": "잔차 연결 기반 경량 모델 (기준선)",
        },
    ]


# ─────────────────────────────────────────────
# 유틸리티 함수
# ─────────────────────────────────────────────
def get_device(force_cpu=False):
    """사용 가능한 디바이스를 감지합니다."""
    if force_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    print("⚠️  CUDA GPU를 사용할 수 없습니다. CPU로 대체합니다.")
    return torch.device("cpu")


def print_device_info(device):
    """디바이스 정보를 출력합니다."""
    print("\n" + "=" * 60)
    print("🖥️  시스템 정보")
    print("=" * 60)
    print(f"  PyTorch 버전  : {torch.__version__}")
    print(f"  CUDA 사용 가능: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA 버전     : {torch.version.cuda}")
        print(f"  GPU 이름      : {torch.cuda.get_device_name(0)}")
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"  GPU 메모리    : {gpu_mem:.1f} GB")
    print(f"  사용 디바이스  : {device}")
    print("=" * 60)


def count_parameters(model):
    """모델의 전체 파라미터 수를 반환합니다."""
    return sum(p.numel() for p in model.parameters())


def get_model_size_mb(model):
    """모델의 메모리 크기(MB)를 반환합니다."""
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024**2)


def load_imagenet_classes():
    """ImageNet 클래스 이름을 로드합니다."""
    cache_path = Path(__file__).parent / "imagenet_classes.txt"

    if cache_path.exists():
        with open(cache_path, "r") as f:
            return [line.strip() for line in f.readlines()]

    try:
        print("📥 ImageNet 클래스 목록 다운로드 중...")
        urllib.request.urlretrieve(IMAGENET_CLASSES_URL, str(cache_path))
        with open(cache_path, "r") as f:
            return [line.strip() for line in f.readlines()]
    except Exception as e:
        print(f"⚠️  클래스 목록 다운로드 실패: {e}")
        return None


def load_image(image_path=None):
    """이미지를 로드하고 전처리합니다."""
    if image_path and os.path.exists(image_path):
        print(f"\n📷 이미지 로드: {image_path}")
        img = Image.open(image_path).convert("RGB")
        return PREPROCESS(img).unsqueeze(0), img
    else:
        # 샘플 이미지 다운로드
        cache_path = Path(__file__).parent / "sample_image.jpg"
        if not cache_path.exists():
            try:
                print("📥 샘플 이미지 다운로드 중...")
                urllib.request.urlretrieve(SAMPLE_IMAGE_URL, str(cache_path))
            except Exception:
                print("⚠️  샘플 이미지 다운로드 실패. 랜덤 텐서를 사용합니다.")
                dummy = torch.randn(1, 3, 224, 224)
                return dummy, None

        img = Image.open(cache_path).convert("RGB")
        return PREPROCESS(img).unsqueeze(0), img


# ─────────────────────────────────────────────
# 벤치마크 함수
# ─────────────────────────────────────────────
def benchmark_model(model, input_tensor, device, warmup_runs=10, benchmark_runs=50):
    """
    모델의 추론 속도를 벤치마크합니다.

    Returns:
        dict: 벤치마크 결과 (평균, 최소, 최대, 중앙값 시간)
    """
    model.eval()

    with torch.no_grad():
        # 워밍업: GPU 커널 초기화 및 캐시 준비
        for _ in range(warmup_runs):
            _ = model(input_tensor)

        if device.type == "cuda":
            torch.cuda.synchronize()

        # 벤치마크 실행
        times = []
        for _ in range(benchmark_runs):
            if device.type == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()
            output = model(input_tensor)

            if device.type == "cuda":
                torch.cuda.synchronize()

            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms 단위

    times_tensor = torch.tensor(times)
    return {
        "mean_ms": times_tensor.mean().item(),
        "std_ms": times_tensor.std().item(),
        "min_ms": times_tensor.min().item(),
        "max_ms": times_tensor.max().item(),
        "median_ms": times_tensor.median().item(),
        "fps": 1000.0 / times_tensor.mean().item(),
        "output": output,
    }


def get_top_predictions(output, classes, top_k=5):
    """상위 k개의 예측 결과를 반환합니다."""
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top_prob, top_catid = torch.topk(probabilities, top_k)

    results = []
    for i in range(top_k):
        idx = top_catid[i].item()
        class_name = classes[idx] if classes and idx < len(classes) else f"class_{idx}"
        results.append(
            {
                "rank": i + 1,
                "class": class_name,
                "probability": top_prob[i].item() * 100,
            }
        )
    return results


# ─────────────────────────────────────────────
# 메인 테스트 루프
# ─────────────────────────────────────────────
def run_tests(args):
    """전체 테스트를 실행합니다."""
    device = get_device(force_cpu=args.cpu)
    print_device_info(device)

    # 이미지 로드
    input_tensor, raw_image = load_image(args.image)
    input_tensor = input_tensor.to(device)

    # ImageNet 클래스 로드
    classes = load_imagenet_classes()

    model_configs = get_model_configs()
    all_results = []

    print(f"\n{'=' * 60}")
    print(f"🚀 벤치마크 시작 (워밍업: {args.warmup}회, 측정: {args.runs}회)")
    print(f"{'=' * 60}")

    for i, config in enumerate(model_configs, 1):
        name = config["name"]
        print(f"\n{'─' * 60}")
        print(f"[{i}/{len(model_configs)}] 📦 {name}")
        print(f"  {config['description']}")
        print(f"{'─' * 60}")

        # 모델 로드
        print(f"  ⏳ 모델 로딩 중...")
        load_start = time.perf_counter()
        model = config["builder"]()
        model = model.to(device)
        model.eval()
        load_time = (time.perf_counter() - load_start) * 1000

        # 모델 정보
        param_count = count_parameters(model)
        model_size = get_model_size_mb(model)
        print(f"  ✅ 로드 완료 ({load_time:.0f} ms)")
        print(f"  📊 파라미터: {param_count:,} ({param_count / 1e6:.2f}M)")
        print(f"  💾 모델 크기: {model_size:.1f} MB")

        # 벤치마크
        print(f"  ⏱️  벤치마크 실행 중...")
        result = benchmark_model(
            model, input_tensor, device, args.warmup, args.runs
        )
        print(f"  ⚡ 평균 추론 시간: {result['mean_ms']:.2f} ms ({result['fps']:.1f} FPS)")

        # Top-5 예측 결과
        if classes:
            predictions = get_top_predictions(result["output"], classes)
            print(f"  🏷️  Top-5 예측 결과:")
            for pred in predictions:
                bar = "█" * int(pred["probability"] / 2)
                print(f"     {pred['rank']}. {pred['class']:<25} {pred['probability']:6.2f}% {bar}")

        # 결과 저장
        all_results.append(
            {
                "name": name,
                "params_m": param_count / 1e6,
                "size_mb": model_size,
                "load_ms": load_time,
                "mean_ms": result["mean_ms"],
                "std_ms": result["std_ms"],
                "min_ms": result["min_ms"],
                "median_ms": result["median_ms"],
                "fps": result["fps"],
                "top1_class": predictions[0]["class"] if classes else "N/A",
                "top1_prob": predictions[0]["probability"] if classes else 0,
            }
        )

        # GPU 메모리 정리
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ─────────────────────────────────────────
    # 요약 테이블 출력
    # ─────────────────────────────────────────
    print(f"\n\n{'=' * 80}")
    print("📊 종합 벤치마크 결과 요약")
    print(f"{'=' * 80}")

    # 속도 순으로 정렬
    all_results.sort(key=lambda x: x["mean_ms"])

    table_data = []
    for rank, r in enumerate(all_results, 1):
        table_data.append(
            [
                rank,
                r["name"],
                f"{r['params_m']:.2f}M",
                f"{r['size_mb']:.1f} MB",
                f"{r['mean_ms']:.2f}",
                f"±{r['std_ms']:.2f}",
                f"{r['min_ms']:.2f}",
                f"{r['fps']:.1f}",
                f"{r['top1_class']} ({r['top1_prob']:.1f}%)",
            ]
        )

    headers = [
        "순위",
        "모델",
        "파라미터",
        "크기",
        "평균(ms)",
        "표준편차",
        "최소(ms)",
        "FPS",
        "Top-1 예측",
    ]

    print(tabulate(table_data, headers=headers, tablefmt="grid", stralign="center"))

    # 최고 성능 모델
    fastest = all_results[0]
    smallest = min(all_results, key=lambda x: x["params_m"])
    print(f"\n🏆 가장 빠른 모델: {fastest['name']} ({fastest['mean_ms']:.2f} ms, {fastest['fps']:.1f} FPS)")
    print(f"🪶 가장 가벼운 모델: {smallest['name']} ({smallest['params_m']:.2f}M 파라미터)")

    # GPU 메모리 사용량 요약 (CUDA인 경우)
    if device.type == "cuda":
        print(f"\n💾 GPU 메모리 사용량:")
        print(f"  최대 할당: {torch.cuda.max_memory_allocated() / 1024**2:.1f} MB")
        print(f"  최대 예약: {torch.cuda.max_memory_reserved() / 1024**2:.1f} MB")

    # 결과를 JSON으로 저장
    output_path = Path(__file__).parent / "benchmark_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "device": str(device),
                "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "N/A",
                "pytorch_version": torch.__version__,
                "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
                "warmup_runs": args.warmup,
                "benchmark_runs": args.runs,
                "results": all_results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\n💾 결과가 {output_path}에 저장되었습니다.")
    print("=" * 80)


# ─────────────────────────────────────────────
# 엔트리포인트
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="GPU 경량 이미지 분류 모델 벤치마크",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python test_classifiers.py                       # 기본 실행 (GPU, 샘플 이미지)
  python test_classifiers.py --image cat.jpg       # 특정 이미지로 테스트
  python test_classifiers.py --warmup 20 --runs 100  # 정밀 벤치마크
  python test_classifiers.py --cpu                 # CPU 모드로 실행
        """,
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="테스트에 사용할 이미지 파일 경로 (없으면 샘플 이미지 다운로드)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="워밍업 실행 횟수 (기본값: 10)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=50,
        help="벤치마크 측정 횟수 (기본값: 50)",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="CPU 모드 강제 사용",
    )

    args = parser.parse_args()
    run_tests(args)


if __name__ == "__main__":
    main()
