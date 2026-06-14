# Diffusion-DICE Real Benchmark Demo

Demo này thuộc **Cách 2: tự xây dựng demo từ đầu** theo yêu cầu CSC14005. Nhóm tự implement một phần cơ chế **Diffusion-DICE** để minh họa **value-function error exploitation** và **in-sample guidance** trong offline RL / contextual bandit setting.

Demo chính **không dùng synthetic dataset**. Dataset chính là real benchmark datasets có sẵn trong scikit-learn:

- **Breast Cancer Wisconsin Diagnostic**
- **Wine Recognition**

Các file `toycase_*` là bản thử nghiệm toy trước đó, chỉ giữ để tham khảo. Deliverable chính nên dùng các file `real_*`.

## Cách tiếp cận

Nhóm chọn **Cách 2: tự xây dựng demo từ đầu**, không kế thừa official implementation hoặc third-party implementation của Diffusion-DICE, QGPO hay IDQL.

Các thành phần được tự cài đặt trong `real_benchmark_diffusion_dice.py`:

- Learned critic trên không gian PCA 2D của real dataset.
- Behavior diffusion model sinh action theo phân phối offline data.
- DICE dual weighting để xác định vùng in-sample có reward cao.
- In-sample guidance score dùng trong reverse diffusion.
- IDQL-style select-only baseline.
- QGPO-style guide-only baseline.
- Metric định lượng và visualize định tính.

## Notebook chính

Notebook chính:

```text
demo/real_benchmark_diffusion_dice_demo.ipynb
```

Notebook chạy đầy đủ:

1. Breast Cancer benchmark comparison.
2. Breast Cancer guidance-scale sweep.
3. Wine benchmark comparison.
4. Wine guidance-scale sweep.

## Cài đặt môi trường

Từ thư mục gốc project:

```bash
python -m pip install -r demo/requirements.txt
```

Demo chạy bằng CPU, không cần GPU/cloud.

## Chạy notebook

```bash
jupyter notebook demo/real_benchmark_diffusion_dice_demo.ipynb
```

Chạy lần lượt các cell từ trên xuống. Notebook sẽ tạo hình và CSV metric trong thư mục `demo/`.

Nếu cần execute notebook bằng command line trên Windows và gặp lỗi quyền `SetFileSecurity`, dùng:

```powershell
$env:JUPYTER_ALLOW_INSECURE_WRITES='1'
python -m jupyter nbconvert --to notebook --execute --inplace demo\real_benchmark_diffusion_dice_demo.ipynb
```

## Chạy bằng script

Notebook dùng lại backend trong `real_benchmark_diffusion_dice.py`. Có thể chạy toàn bộ bằng script:

```bash
python demo/real_benchmark_diffusion_dice.py --dataset all --experiment all
```

Smoke test nhanh:

```bash
python demo/real_benchmark_diffusion_dice.py --dataset breast_cancer --experiment all --quick --output-suffix _quick
```

## Output chính từ notebook

Sau khi chạy notebook, các file chính được tạo:

| File | Nội dung |
|---|---|
| `real_breast_cancer_results_notebook.png` | Visualize comparison trên Breast Cancer |
| `real_breast_cancer_metrics_notebook.csv` | Metric comparison trên Breast Cancer |
| `real_breast_cancer_tuning_notebook.png` | Guidance-scale sweep trên Breast Cancer |
| `real_breast_cancer_tuning_metrics_notebook.csv` | Metric sweep trên Breast Cancer |
| `real_wine_results_notebook.png` | Visualize comparison trên Wine |
| `real_wine_metrics_notebook.csv` | Metric comparison trên Wine |
| `real_wine_tuning_notebook.png` | Guidance-scale sweep trên Wine |
| `real_wine_tuning_metrics_notebook.csv` | Metric sweep trên Wine |

## Metric

Các CSV output có các metric:

- `id_ratio`: tỉ lệ generated actions nằm gần support của real dataset.
- `ood_ratio`: tỉ lệ generated actions nằm ngoài support.
- `oracle_reward_mean`: reward trung bình suy ra từ nhãn thật bằng KNN smoothing.
- `critic_reward_mean`: score trung bình từ learned critic.
- `target_region_ratio`: tỉ lệ generated actions nằm trong vùng target class.
- `mean_nearest_distance`: khoảng cách trung bình tới điểm dữ liệu thật gần nhất.

## Cách đọc hình

Mỗi hình comparison có 6 panel:

- Real dataset support after PCA: phân phối dữ liệu thật sau khi chiếu PCA 2D.
- KNN oracle reward from real labels: vùng reward cao suy ra từ nhãn thật.
- Learned critic score: score học được từ critic.
- QGPO-style guide-only: actions sinh bằng critic guidance.
- IDQL-style select-only: actions sinh rồi chọn bằng critic.
- Diffusion-DICE: actions sinh bằng in-sample guidance rồi select.

Ý nghĩa chính:

- Nếu `critic_reward_mean` cao nhưng `id_ratio` thấp, method đang có dấu hiệu critic exploitation.
- Nếu `id_ratio` cao và `oracle_reward_mean` cao, method vừa bám data support vừa hướng tới vùng reward tốt.
- Diffusion-DICE kỳ vọng ổn định hơn khi tăng guidance scale vì có in-sample guidance.

## Hạn chế

- Đây là demo rút gọn của Diffusion-DICE, không phải official implementation đầy đủ.
- Baseline là phiên bản **style/proxy** của IDQL và QGPO để minh họa cơ chế, không phải full implementation của hai paper.
- PCA 2D giúp visualize rõ hơn nhưng làm mất một phần thông tin từ dữ liệu gốc.
- Kết quả có thể dao động nhẹ theo seed do diffusion sampling và neural network training.
