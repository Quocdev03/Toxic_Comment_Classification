# Fake News Detection (TF-IDF + SMOTE + LinearSVC)

Dự án phân loại bình luận **Toxic / Non-Toxic** bằng tiếng Anh, sử dụng:

- Tiền xử lý văn bản (chuẩn hoá chat-words, loại bỏ HTML/URL/email/số/dấu câu, loại stopwords, stemming)
- TF-IDF Vectorizer
- SMOTE để cân bằng nhãn (toxic chiếm tỉ lệ rất nhỏ so với non-toxic)
- LinearSVC để huấn luyện mô hình phân loại

## 1. Cấu trúc thư mục

```
project/
├── main_fixed.py        # Code đã sửa lỗi (chạy file này)
├── requirements.txt
├── data/
│   ├── train.csv                  # đã có sẵn
│   ├── test.csv                   # đã có sẵn
│   ├── test_labels.csv            # CẦN TỰ TẢI (xem mục 3)
│   └── youtoxic_english_1000.csv  # CẦN TỰ TẢI (xem mục 3)
└── toxic_model.pkl       # sẽ được tạo ra sau khi chạy xong (model đã train)
```

## 2. Cài đặt

```bash
pip install -r requirements.txt
```

## 3. Dữ liệu

### a) Đã có sẵn trong thư mục `data/`

- **train.csv** và **test.csv**: lấy từ bộ dữ liệu _Jigsaw Toxic Comment Classification
  Challenge_ (Kaggle) — các bình luận từ trang thảo luận Wikipedia, gán nhãn
  `toxic, severe_toxic, obscene, threat, insult, identity_hate`.

### b) Cần tự tải thêm (do nguồn yêu cầu đăng nhập, không tải tự động được)

1. **`test_labels.csv`** — nhãn thật của tập `test.csv`, được Kaggle công bố
   sau khi cuộc thi kết thúc (giá trị `-1` = không dùng để tính điểm).
    - Tải tại: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data
    - Hoặc bản mirror: https://www.kaggle.com/datasets/julian3833/jigsaw-toxic-comment-classification-challenge
    - Hoặc Hugging Face (không cần đăng nhập Kaggle):
      https://huggingface.co/datasets/thesofakillers/jigsaw-toxic-comment-classification-challenge

2. **`youtoxic_english_1000.csv`** — 1000 bình luận YouTube được gán nhãn tay,
   gồm cột `Text` và `IsToxic`.
    - Tải tại: https://www.kaggle.com/datasets/reihanenamdari/youtube-toxicity-data

> Cả hai file trên hãy đặt vào thư mục `data/` cùng cấp với `train.csv`, `test.csv`.

> Nếu chưa có `test_labels.csv`, chương trình vẫn chạy được toàn bộ phần
> huấn luyện mô hình — chỉ phần đánh giá cuối cùng trên `test.csv` (so khớp với
> nhãn thật) sẽ được tự động bỏ qua và in ra thông báo nhắc tải file.

## 4. Chạy chương trình

```bash
python main_fixed.py
```

Kết quả:

- Các biểu đồ (phân bố nhãn, Word Cloud, phân bố trước/sau SMOTE, confusion matrix,
  precision/recall/F1) sẽ hiện ra từng cửa sổ (dùng `plt.show()`).
- Mô hình đã train được lưu vào `toxic_model.pkl` (gồm model SVM, vectorizer TF-IDF,
  và hàm tiền xử lý).
- Nếu có `test_labels.csv`, sẽ in ra số dòng khớp/không khớp và độ chính xác trên
  tập `test.csv`.
