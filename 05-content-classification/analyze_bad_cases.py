# -*- coding: utf-8 -*-
"""找出误判图片，并生成可以直接打开的 Bad Case HTML 报告。"""

import argparse
import base64
import html
import io
from pathlib import Path

from PIL import Image

from data import load_datasets
from inference import predict_pil
from model import DEFAULT_CHECKPOINT, load_model
from train import DEFAULT_DATA, choose_device


DEFAULT_REPORT = Path(__file__).parent / "reports" / "bad_cases.html"


def image_data_url(image, size=(220, 220)):
    thumbnail = image.copy()
    thumbnail.thumbnail(size)
    buffer = io.BytesIO()
    thumbnail.save(buffer, format="JPEG", quality=85)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def collect_bad_cases(dataset, model, metadata, device):
    cases = []
    for path, label_index in dataset.samples:
        actual = dataset.classes[label_index]
        with Image.open(path) as image:
            rgb_image = image.convert("RGB")
            result = predict_pil(model, rgb_image, metadata, device=device)
            thumbnail = image_data_url(rgb_image)

        predicted = result["decision"]
        if predicted == actual:
            continue

        error_type = "FP（误杀）" if predicted == "unsafe" else "FN（漏放）"
        wrong_confidence = result["probabilities"][predicted]
        cases.append(
            {
                "path": str(path),
                "actual": actual,
                "predicted": predicted,
                "error_type": error_type,
                "wrong_confidence": wrong_confidence,
                "unsafe_probability": result["probabilities"]["unsafe"],
                "thumbnail": thumbnail,
            }
        )

    return sorted(cases, key=lambda case: case["wrong_confidence"], reverse=True)


def render_report(cases, output, split, total, threshold, limit):
    visible_cases = cases[:limit]
    fp_count = sum(case["error_type"].startswith("FP") for case in cases)
    fn_count = sum(case["error_type"].startswith("FN") for case in cases)

    if visible_cases:
        cards = "\n".join(
            f"""
            <article class="case">
              <img src="{case['thumbnail']}" alt="Bad Case">
              <div class="case-body">
                <strong>{html.escape(case['error_type'])}</strong>
                <dl>
                  <div><dt>真实标签</dt><dd>{html.escape(case['actual'])}</dd></div>
                  <div><dt>模型判断</dt><dd>{html.escape(case['predicted'])}</dd></div>
                  <div><dt>unsafe 概率</dt><dd>{case['unsafe_probability']:.3f}</dd></div>
                  <div><dt>判断阈值</dt><dd>{threshold:.2f}</dd></div>
                </dl>
                <p title="{html.escape(case['path'])}">{html.escape(Path(case['path']).name)}</p>
              </div>
            </article>
            """
            for case in visible_cases
        )
    else:
        cards = """
        <div class="empty">
          当前 split 没有误判。真实项目中仍要继续补充困难样本，而不是据此认定模型已经完美。
        </div>
        """

    document = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>内容分类 Bad Case 报告</title>
  <style>
    :root {{ color-scheme: light; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; color: #17212b; background: #f4f6f8; }}
    header {{ padding: 32px max(24px, calc((100vw - 1120px) / 2)); background: #fff; border-bottom: 1px solid #dfe4e8; }}
    h1 {{ margin: 0 0 8px; font-size: 28px; letter-spacing: 0; }}
    header p {{ margin: 0; color: #5c6873; }}
    .stats {{ max-width: 1120px; margin: 0 auto; padding: 24px; display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }}
    .stat {{ background: #fff; border: 1px solid #dfe4e8; border-radius: 6px; padding: 18px; }}
    .stat span {{ display: block; color: #64717d; font-size: 13px; }}
    .stat strong {{ display: block; margin-top: 6px; font-size: 25px; }}
    main {{ max-width: 1120px; margin: 0 auto; padding: 0 24px 40px; }}
    h2 {{ font-size: 18px; letter-spacing: 0; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 16px; }}
    .case {{ overflow: hidden; background: #fff; border: 1px solid #dfe4e8; border-radius: 6px; }}
    .case img {{ width: 100%; aspect-ratio: 4 / 3; object-fit: contain; background: #edf1f3; }}
    .case-body {{ padding: 16px; }}
    .case-body > strong {{ color: #b42318; }}
    dl {{ margin: 14px 0; }}
    dl div {{ display: flex; justify-content: space-between; gap: 16px; padding: 5px 0; }}
    dt {{ color: #64717d; }}
    dd {{ margin: 0; font-variant-numeric: tabular-nums; }}
    .case-body p {{ margin: 0; color: #7b8791; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
    .empty {{ padding: 28px; background: #fff; border: 1px solid #dfe4e8; border-radius: 6px; line-height: 1.7; }}
    @media (max-width: 680px) {{ .stats {{ grid-template-columns: repeat(2, 1fr); }} }}
  </style>
</head>
<body>
  <header>
    <h1>内容分类 Bad Case 报告</h1>
    <p>数据集：{html.escape(split)} · threshold={threshold:.2f} · 按错误判断的置信度排序</p>
  </header>
  <section class="stats">
    <div class="stat"><span>样本总数</span><strong>{total}</strong></div>
    <div class="stat"><span>误判总数</span><strong>{len(cases)}</strong></div>
    <div class="stat"><span>FP 误杀</span><strong>{fp_count}</strong></div>
    <div class="stat"><span>FN 漏放</span><strong>{fn_count}</strong></div>
  </section>
  <main>
    <h2>误判样本（最多展示 {limit} 个）</h2>
    <div class="grid">{cards}</div>
  </main>
</body>
</html>
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="生成图片分类 Bad Case 报告")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    args = parser.parse_args()

    device = choose_device(args.device)
    dataset = load_datasets(args.data)[args.split]
    model, metadata = load_model(args.checkpoint, device=device)
    cases = collect_bad_cases(dataset, model, metadata, device)
    render_report(
        cases,
        args.output,
        split=args.split,
        total=len(dataset),
        threshold=float(metadata["threshold"]),
        limit=args.limit,
    )
    print(
        f"{args.split}: total={len(dataset)} bad_cases={len(cases)} "
        f"report={args.output}"
    )


if __name__ == "__main__":
    main()
