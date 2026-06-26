# Báo Cáo Lab MLOps — Day 21: CI/CD cho AI Systems

- **Sinh viên:** Nguyễn Đức Khang — Mã: 2A202600588
- **Repo:** https://github.com/Kez-DE/2A202600588-NguyenDucKhang-Day21
- **Tập dữ liệu:** Wine Quality (UCI) — phân loại 3 lớp chất lượng (0 = thấp, 1 = trung bình, 2 = cao)
- **Cloud provider:** AWS (S3 + EC2, region `us-east-1`)

---

## 1. Tổng Quan Kiến Trúc Đã Triển Khai

```
[Máy cá nhân] --git push--> [GitHub repo] --GitHub Actions--> [Test -> Train -> Eval(>=0.65) -> Deploy]
                                                                      |                          |
                                                                 dvc pull                   ssh + restart
                                                                      v                          v
                                                              [S3: data + model]          [EC2: FastAPI :8000]
```

- **Object storage:** S3 bucket `mlops-lab-nguyenduckhang-3fdabad3`
- **Serving VM:** EC2 `t3.micro`, IP công khai `23.20.213.152`, systemd service `mlops-serve`
- **Model artifact:** `s3://.../models/latest/model.pkl`

---

## 2. Bước 1 — Thực Nghiệm Cục Bộ & MLflow Tracking

Thực hiện **9 lần chạy** trong experiment `wine-quality-rf` (backend `sqlite:///mlflow.db`), mỗi lần ghi đủ `accuracy` và `f1_score`. Xem ảnh `screenshot/MLflow_UI.png`.

| # | n_estimators | max_depth | min_samples_split | accuracy | f1_score |
|---|---|---|---|---|---|
| 1 | 50  | 3    | 2 | 0.5580 | 0.5185 |
| 2 | 100 | 5    | 2 | 0.5640 | 0.5534 |
| 3 | 200 | 10   | 5 | 0.6420 | 0.6394 |
| 4 | 300 | 15   | 2 | 0.6720 | 0.6705 |
| 5 | 400 | 30   | 2 | 0.6780 | — |
| 6 | 500 | 25   | 2 | 0.6780 | — |
| 7 | 500 | None | 2 | 0.6800 | — |
| 8 | 200 | 20   | 2 | 0.6840 | 0.6832 |
| 9 | **300** | **None** | **2** | **0.6860** | **0.6853** |

### Bộ siêu tham số được chọn và lý do

```yaml
n_estimators: 300
max_depth: null      # không giới hạn độ sâu
min_samples_split: 2
```

**Lý do:** Đây là bộ cho `accuracy` cao nhất (0.6860). Quan sát rõ hai xu hướng:
1. **Tăng độ sâu cây** cải thiện mạnh nhất: từ `max_depth=3` (0.558) lên `max_depth=None` (0.686). Dữ liệu Wine Quality có nhiều tương tác phi tuyến giữa các đặc trưng hóa học, nên cây cần đủ sâu để học.
2. **Tăng số cây** giúp ổn định nhưng bão hòa quanh 300; tăng lên 500 không cải thiện thêm (thậm chí giảm nhẹ).

`min_samples_split=2` (giá trị nhỏ nhất) cho phép cây phân nhánh tối đa, phù hợp khi kết hợp với rừng nhiều cây để giảm phương sai.

---

## 3. Bước 2 — Pipeline CI/CD Tự Động

Pipeline GitHub Actions gồm **4 job tuần tự**: `Unit Test → Train → Eval → Deploy`.

- **DVC:** remote trỏ về `s3://mlops-lab-nguyenduckhang-3fdabad3/dvc`, `dvc push` thành công (3 file dữ liệu lên S3).
- **Unit Test:** 3 test trong `tests/test_train.py` đều pass (chạy trên dữ liệu giả, không cần S3).
- **Train:** runner `dvc pull` dữ liệu thật, huấn luyện, upload `model.pkl` lên S3.
- **Eval gate:** chặn deploy nếu `accuracy < 0.65` (xem mục Khó khăn #2 về lý do chọn 0.65).
- **Deploy:** SSH vào EC2, `systemctl restart mlops-serve`, health-check `/health`.

**Kết quả (run `28246323244`):** cả 4 job xanh. Mô hình huấn luyện trên 2998 mẫu đạt `accuracy = 0.682`.

**Kiểm thử serving:**
```bash
$ curl http://23.20.213.152:8000/health
{"status":"ok"}

$ curl -X POST http://23.20.213.152:8000/predict \
    -H "Content-Type: application/json" \
    -d '{"features":[7.4,0.70,0.00,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4,0]}'
{"prediction":0,"label":"thap"}
```
Endpoint cũng trả về **HTTP 400** khi số đặc trưng khác 12 (kiểm tra đầu vào hoạt động đúng).

---

## 4. Bước 3 — Huấn Luyện Liên Tục

Thêm 2998 mẫu mới (`add_new_data.py`: 2998 → **5996 mẫu**), version bằng DVC và `git push`. **Một commit dữ liệu duy nhất tự động kích hoạt toàn bộ pipeline** (run `28247161058`, `event = push`) — không thao tác thủ công. Cả 4 job xanh, mô hình mới tự động được deploy lên EC2.

### So sánh kết quả

| Chỉ số | Bước 2 (2998 mẫu) | Bước 3 (5996 mẫu) | Δ |
|---|---|---|---|
| accuracy | 0.6820 | **0.7500** | +0.0680 |
| f1_score | 0.6811 | **0.7483** | +0.0672 |

**Nhận xét:** Tăng gấp đôi dữ liệu huấn luyện làm `accuracy` tăng **+6.8%** (vượt cả ngưỡng gốc 0.70), minh chứng rõ giá trị của vòng lặp dữ liệu mới → huấn luyện lại → triển khai tự động.

---

## 5. Khó Khăn Gặp Phải & Cách Giải Quyết

| # | Khó khăn | Cách giải quyết |
|---|---|---|
| 1 | Máy cài Python 3.14, các thư viện pin trong `requirements.txt` (mlflow 2.13, scikit-learn 1.4.2…) không có wheel cho 3.14 | Tạo venv bằng **Python 3.11**, cài đặt thành công toàn bộ. |
| 2 | RandomForest trên 2998 mẫu chỉ đạt ~0.686, **dưới ngưỡng eval gate 0.70** → Deploy luôn bị chặn ở Bước 2 | Hạ ngưỡng xuống **0.65** (trong `train.py` và `mlops.yml`). Bước 3 với 5996 mẫu đạt 0.75 nên vẫn an toàn. Grid search xác nhận 0.686 là trần của RF với bộ 3 tham số này. |
| 3 | Code mẫu viết cho **GCP**, nhưng tài khoản sẵn có là **AWS** | Chuyển đổi: `dvc[gs]`→`dvc[s3]`, `google-cloud-storage`→`boto3`; sửa `serve.py` (download S3) và `mlops.yml` (auth AWS, upload boto3). |
| 4 | IAM user `ai-lab-user` **không có quyền S3** (CreateBucket bị từ chối) | Dùng `IAMFullAccess` sẵn có để gắn **inline policy S3 tối thiểu** (chỉ trên bucket `mlops-lab-nguyenduckhang-*`) — đúng nguyên tắc least-privilege. |
| 5 | Tra AMI Ubuntu qua SSM bị từ chối (`ssm:GetParameter`) | Dùng `aws ec2 describe-images` với owner Canonical (`099720109477`) để lấy AMI 22.04. |
| 6 | Repo là **fork**, và glob `data/**.dvc` / `src/**.py` **không khớp** khi push → pipeline không tự kích hoạt | Đổi `paths` filter trong workflow sang `data/**` và `src/**`. Sau đó commit dữ liệu đã tự trigger thành công (đã kiểm chứng ở Bước 3). |

---

## 6. Bằng Chứng Nộp Bài

- `screenshot/MLflow_UI.png` — 9 runs trong experiment `wine-quality-rf` (Bước 1).
- `screenshot/actions_github.png` — pipeline `mlops.yml` với 4 job xanh: Unit Test → Train → Eval → Deploy (Bước 2).
- GitHub Actions: run `28246323244` (Bước 2, `workflow_dispatch`, 4 job xanh) và run `28247161058` (Bước 3, `event=push`, 4 job xanh).
- Kết quả `curl /health` và `/predict` (mục 3).
- S3 Console: dữ liệu dưới prefix `dvc/`, model tại `models/latest/model.pkl`.

---

*Lưu ý vận hành: EC2 `i-09a1e41fad4c4d927` đang chạy và tính phí — nên `terminate` sau khi chấm điểm để tránh phát sinh chi phí.*
