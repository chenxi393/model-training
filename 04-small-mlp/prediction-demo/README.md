# 训练效果验证

这个目录只放训练完成后的效果验证内容：

```text
mnist_mlp.pt         训练好的模型参数，本地生成，不提交 Git
predict.html         图片上传和概率可视化页面
serve_predict.py     本地页面和预测 API
predict_image.py     命令行单图预测
test_images/         数字 0～9 的 MNIST 测试图片
export_test_images.py 重新导出测试图片
test_mlp.py          模型、checkpoint 和图片预处理测试
```

## 启动可视化页面

先在仓库根目录训练模型：

```bash
uv run python 04-small-mlp/train_mlp.py
```

再启动页面：

```bash
uv run python 04-small-mlp/prediction-demo/serve_predict.py
```

浏览器访问：

```text
http://127.0.0.1:8004
```

## 命令行预测

```bash
uv run python 04-small-mlp/prediction-demo/predict_image.py \
  04-small-mlp/prediction-demo/test_images/digit_7.png
```

## 运行测试

```bash
uv run python 04-small-mlp/prediction-demo/test_mlp.py
```
