# -*- coding: utf-8 -*-
"""
Mô hình phân loại bình luận Toxic / Non-Toxic bằng TF-IDF + SMOTE + LinearSVC
"""

import os
from collections import Counter

import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import nltk

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib

# FIX: tải đủ resource cần dùng của NLTK (chỉ "stopwords" là chưa đủ nếu môi trường
# chưa có sẵn, nhưng SnowballStemmer không cần thêm gói)
nltk.download('stopwords')

# FIX: đưa đường dẫn dữ liệu vào biến cấu hình để dễ thay đổi,
# và để toàn bộ file CSV trong thư mục "data/" cùng cấp với file code này
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
YOUTOXIC_PATH = os.path.join(DATA_DIR, "youtoxic_english_1000.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
TEST_LABELS_PATH = os.path.join(DATA_DIR, "test_labels.csv")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "toxic_model.pkl")


# =====================================================================
# MODULE 1: THU THẬP DỮ LIỆU, XỬ LÝ DỮ LIỆU VÀ NHÃN
# =====================================================================

# Tải data train (Jigsaw Toxic Comment Classification Challenge)
train = pd.read_csv(TRAIN_PATH)

# Tải data toxic youtube
youtoxic = pd.read_csv(YOUTOXIC_PATH)

# FIX: đã xoá dòng đọc lại train.csv lần thứ 2 (bị lặp, không cần thiết)

# Chỉ giữ lại hai cột cần thiết: 'toxic' và 'comment_text'
# FIX: thêm .copy() để tránh SettingWithCopyWarning khi gán cột bên dưới
train_new = train[['toxic', 'comment_text']].copy()

# Đổi tên cột để đồng nhất các bảng:
train_new = train_new.rename(columns={'toxic': 'toxic_binary'})

print(train_new)

# Lưu lại trong dataframe mới gồm bình luận và nhãn
# FIX: thêm .copy() để tránh SettingWithCopyWarning khi gán cột bên dưới
youtoxic_new = youtoxic[['Text', 'IsToxic']].copy()

# Đổi tên cột để đồng nhất hai dữ liệu
youtoxic_new = youtoxic_new.rename(columns={'Text': 'comment_text'})
youtoxic_new = youtoxic_new.rename(columns={'IsToxic': 'toxic_binary'})

# Đổi giá trị cột toxic_binary thành true/false -> 1/0
# FIX: dữ liệu IsToxic khi đọc từ CSV có thể là kiểu bool (True/False) hoặc chuỗi
# "True"/"False". Dùng .map() với cả hai kiểu để tránh lỗi không chuyển đổi được,
# sau đó ép kiểu sang int.
youtoxic_new['toxic_binary'] = (
    youtoxic_new['toxic_binary']
    .map({True: 1, False: 0, 'True': 1, 'False': 0, 1: 1, 0: 0})
    .astype(int)
)

print(youtoxic_new)

# Kết hợp hai tập dữ liệu thành 1 tập dữ liệu mới
train_binary = pd.concat([train_new, youtoxic_new], ignore_index=True)

print(train_binary)

# In số lượng data mỗi nhãn
print(train_binary['toxic_binary'].value_counts())

"""**Xử lý dòng có dữ liệu rỗng hoặc bị lặp**"""

# Check giá trị rỗng
print("Số giá trị rỗng theo từng cột:")
print(train_binary.isnull().sum())

# FIX: trước đây code chỉ "in" số dòng rỗng nhưng KHÔNG xoá -> dữ liệu rỗng
# vẫn đi vào bước tiền xử lý và có thể gây lỗi khi áp dụng các hàm xử lý chuỗi.
# Xoá các dòng có comment_text hoặc toxic_binary bị rỗng:
train_binary = train_binary.dropna(subset=['comment_text', 'toxic_binary'])

# Các dòng bị trùng lặp
print("Số dòng trùng lặp:", train_binary.duplicated().sum())

# Thực hiện xoá các dòng trùng lặp
train_binary.drop_duplicates(keep='first', inplace=True)

# Reset lại index
train_binary.reset_index(drop=True, inplace=True)

# In số lượng data mỗi nhãn
print(train_binary['toxic_binary'].value_counts())

# FIX: ép kiểu nhãn về int (sau khi concat hai nguồn dữ liệu có thể bị thành float)
train_binary['toxic_binary'] = train_binary['toxic_binary'].astype(int)


# =====================================================================
# Biểu đồ cột thống kê
# =====================================================================

plt.figure(figsize=(8, 6))
toxic_counts = train_binary['toxic_binary'].value_counts()
toxic_counts.index = ['Non-Toxic' if x == 0 else 'Toxic' for x in toxic_counts.index]

ax = toxic_counts.plot(kind='bar', color=['green', 'red'])

# Thêm số lượng lên trên từng cột
for p in ax.patches:
    ax.annotate(str(int(p.get_height())),
                (p.get_x() + p.get_width() / 2, p.get_height()),
                ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.title('Toxic vs Non-Toxic Comments')
plt.xlabel('Category')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()


# =====================================================================
# Word Cloud của comment toxic
# =====================================================================

toxic_comments = ' '.join(train_binary.loc[train_binary['toxic_binary'] == 1, 'comment_text'].dropna())
# FIX: nếu chuỗi rỗng (không có comment toxic nào) WordCloud sẽ báo lỗi
# -> kiểm tra trước khi generate
if toxic_comments.strip():
    wordcloud_toxic = WordCloud(width=800, height=400, background_color='white').generate(toxic_comments)
    plt.figure()
    plt.imshow(wordcloud_toxic, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud for Toxic Comments')
    plt.show()
else:
    print("Không có dữ liệu toxic để vẽ Word Cloud.")


# =====================================================================
# Word Cloud của comment non_toxic
# =====================================================================

non_toxic_comments = ' '.join(train_binary.loc[train_binary['toxic_binary'] == 0, 'comment_text'].dropna())
if non_toxic_comments.strip():
    wordcloud_non_toxic = WordCloud(width=800, height=400, background_color='white').generate(non_toxic_comments)
    plt.figure()
    plt.imshow(wordcloud_non_toxic, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud for Non-Toxic Comments')
    plt.show()
else:
    print("Không có dữ liệu non-toxic để vẽ Word Cloud.")


# =====================================================================
# MODULE 2: XÂY DỰNG HÀM VÀ TIỀN XỬ LÝ DỮ LIỆU
# (Làm sạch, chuẩn hóa, stemming, ...)
# =====================================================================

chat_words_str = [
    ("afaik", "as far as i know"),
    ("afk", "away from keyboard"),
    ("asap", "as soon as possible"),
    ("atk", "at the keyboard"),
    ("atm", "at the moment"),
    ("a3", "anytime, anywhere, anyplace"),
    ("bak", "back at keyboard"),
    ("bbl", "be back later"),
    ("bbs", "be back soon"),
    ("bfn", "bye for now"),
    ("b4n", "bye for now"),
    ("brb", "be right back"),
    ("brt", "be right there"),
    ("btw", "by the way"),
    ("b4", "before"),
    ("cu", "see you"),
    ("cul8r", "see you later"),
    ("cya", "see you"),
    ("faq", "frequently asked questions"),
    ("fc", "fingers crossed"),
    ("fwiw", "for what it's worth"),
    ("fyi", "for your information"),
    ("gal", "get a life"),
    ("gg", "good game"),
    ("gn", "good night"),
    ("gmta", "great minds think alike"),
    ("gr8", "great!"),
    ("g9", "genius"),
    ("ic", "i see"),
    ("icq", "i seek you (also a chat program)"),
    ("ilu", "i love you"),
    ("imho", "in my honest/humble opinion"),
    ("imo", "in my opinion"),
    ("iow", "in other words"),
    ("irl", "in real life"),
    ("kiss", "keep it simple, stupid"),
    ("ldr", "long distance relationship"),
    ("lmao", "laugh my a.. off"),
    ("lol", "laughing out loud"),
    ("ltns", "long time no see"),
    ("l8r", "later"),
    ("mte", "my thoughts exactly"),
    ("m8", "mate"),
    ("nrn", "no reply necessary"),
    ("oic", "oh i see"),
    ("pita", "pain in the a.."),
    ("prt", "party"),
    ("prw", "parents are watching"),
    ("rofl", "rolling on the floor laughing"),
    ("rofllol", "rolling on the floor laughing out loud"),
    ("rotflmao", "rolling on the floor laughing my a.. off"),
    ("sk8", "skate"),
    ("stats", "your sex and age"),
    ("asl", "age, sex, location"),
    ("thx", "thank you"),
    ("ttfn", "ta-ta for now!"),
    ("ttyl", "talk to you later"),
    ("u", "you"),
    ("u2", "you too"),
    ("u4e", "yours for ever"),
    ("wb", "welcome back"),
    ("wtf", "what the f..."),
    ("wtg", "way to go!"),
    ("wuf", "where are you from?"),
    ("w8", "wait..."),
    ("7k", "sick:-d laugher"),
    ("tfw", "that feeling when"),
    ("mfw", "my face when"),
    ("mrw", "my reaction when"),
    ("ifyp", "i feel your pain"),
    ("tntl", "trying not to laugh"),
    ("jk", "just kidding"),
    ("idc", "i don't care"),
    ("ily", "i love you"),
    ("imu", "i miss you"),
    ("adih", "another day in hell"),
    ("zzz", "sleeping, bored, tired"),
    ("wywh", "wish you were here"),
    ("time", "tears in my eyes"),
    ("bae", "before anyone else"),
    ("fimh", "forever in my heart"),
    ("bsaaw", "big smile and a wink"),
    ("bwl", "bursting with laughter"),
    ("bff", "best friends forever"),
    ("csl", "can't stop laughing"),
    ("i'm", "i am"),
    ("you're", "you are"),
    ("he's", "he is"),
    ("she's", "she is"),
    ("it's", "it is"),
    ("we're", "we are"),
    ("they're", "they are"),
    ("i've", "i have"),
    ("you've", "you have"),
    ("they've", "they have"),
    ("we've", "we have"),
    ("isn't", "is not"),
    ("aren't", "are not"),
    ("wasn't", "was not"),
    ("weren't", "were not"),
    ("haven't", "have not"),
    ("hasn't", "has not"),
    ("hadn't", "had not"),
    ("won't", "will not"),
    ("wouldn't", "would not"),
    ("don't", "do not"),
    ("doesn't", "does not"),
    ("didn't", "did not"),
    ("can't", "cannot"),
    ("couldn't", "could not"),
    ("shouldn't", "should not"),
    ("mightn't", "might not"),
    ("mustn't", "must not"),
    ("what's", "what is "),
]

"""## Các Hàm Tiền Xử Lý"""

stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer("english")

chat_words_list = dict(chat_words_str)


def chat_words_conversion(text):
    words = text.split()
    converted_words = [chat_words_list.get(word.lower(), word) for word in words]
    return " ".join(converted_words)


def preprocess_text(text):
    """
    Hàm tiền xử lý văn bản gồm:
    - Chuyển đổi từ viết tắt/chat words
    - Làm sạch dữ liệu (loại bỏ HTML, email, URL, số, dấu câu, khoảng trắng thừa)
    - Loại bỏ stopwords
    - Stemming
    """
    # FIX: trong file gốc, docstring trên được đặt SAU dòng "return" của hàm
    # chat_words_conversion() (thụt sai cấp độ thu lề) nên không hề thuộc về
    # hàm preprocess_text và chỉ là một chuỗi "chết" không có tác dụng.
    # Đã chuyển docstring vào đúng vị trí làm docstring của preprocess_text.

    # Chuyển đổi từ viết tắt/chat words
    text = chat_words_conversion(text)

    # Làm sạch dữ liệu
    text = str(text).lower().strip()
    text = re.sub(r'\[.*?\]', '', text)            # Xóa nội dung trong []
    text = re.sub(r'\S+@\S+', '', text)            # Xóa email
    text = re.sub(r'<.*?>+', '', text)             # Xóa thẻ HTML
    text = re.sub(r'http\S+|www\S+', '', text)     # Xóa URL
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)  # Xóa dấu câu
    text = re.sub(r'\d+', '', text)                # Xóa số
    text = re.sub(r'\s+', ' ', text).strip()       # Xóa khoảng trắng thừa

    # Tokenization
    words = text.split()

    # Loại bỏ stopwords
    words = [word for word in words if word not in stop_words]

    # Stemming
    words = [stemmer.stem(word) for word in words]

    return " ".join(words)


"""**Tiền xử lý dữ liệu**"""

train_binary['clean_text'] = train_binary['comment_text'].map(lambda com: preprocess_text(com))

print(train_binary)  # Kiểm tra dữ liệu sau khi tiền xử lý


# =====================================================================
# MODULE 3: TRÍCH XUẤT ĐẶC TRƯNG VÀ CÂN BẰNG NHÃN
# =====================================================================

vect = TfidfVectorizer(max_features=5000, stop_words='english')
X = vect.fit_transform(train_binary['clean_text'])
Y = train_binary['toxic_binary']

"""Hiển thị để kiểm tra"""

feature_names = vect.get_feature_names_out()

# FIX: nếu vì lý do nào đó số đặc trưng < 20 thì np.random.choice(..., 20, ...) sẽ lỗi
n_sample_words = min(20, len(feature_names))
random_words = np.random.choice(feature_names, n_sample_words, replace=False)

word_indices = [np.where(feature_names == word)[0][0] for word in random_words]
df_tfidf_sample = pd.DataFrame(X[:, word_indices].toarray(), columns=random_words)

print(df_tfidf_sample.head())

"""Module SMOTE cân bằng nhãn"""

# Initialize SMOTE
smote = SMOTE(random_state=42)

# Apply SMOTE for oversampling
X_resampled, y_resampled = smote.fit_resample(X, Y)

# FIX: bản gốc gọi X_resampled.todense() để tạo DataFrame chỉ nhằm mục đích
# in ra số lượng nhãn sau SMOTE. Với ma trận TF-IDF max_features=5000 và
# hàng trăm nghìn dòng dữ liệu, việc chuyển sang ma trận dense có thể tốn
# hàng chục GB RAM và làm crash chương trình (MemoryError).
# -> Thay bằng cách đếm trực tiếp trên y_resampled (mảng 1D, rất nhẹ).
toxic_distribution_before = train_binary['toxic_binary'].value_counts()
toxic_distribution_after = pd.Series(Counter(y_resampled))

print("\nDistribution trước SMOTE:")
print(toxic_distribution_before)
print("\nDistribution sau SMOTE:")
print(toxic_distribution_after)

"""Vẽ biểu đồ để kiểm tra"""

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.barplot(x=toxic_distribution_before.index, y=toxic_distribution_before.values, ax=axes[0])
axes[0].set_title("Trước khi SMOTE")
axes[0].set_xlabel("Nhãn (toxic_binary)")
axes[0].set_ylabel("Số lượng")

sns.barplot(x=toxic_distribution_after.index, y=toxic_distribution_after.values, ax=axes[1])
axes[1].set_title("Sau khi SMOTE")
axes[1].set_xlabel("Nhãn (toxic_binary)")
axes[1].set_ylabel("Số lượng")

plt.tight_layout()
plt.show()


# =====================================================================
# MODULE 4: HUẤN LUYỆN MÔ HÌNH VÀ DỰ ĐOÁN NHÃN
# =====================================================================

"""## Chia tập dữ liệu đi train"""

x_train, x_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42
)

"""## Dùng SVM để train"""

svm_model = LinearSVC(class_weight='balanced', random_state=42)
svm_model.fit(x_train, y_train)

# Predict
y_pred = svm_model.predict(x_test)

# Evaluate
print(classification_report(y_test, y_pred))
print("Mô hình đã được huấn luyện xong!")

model_package = {
    "model": svm_model,
    "vectorizer": vect,
    "preprocess": preprocess_text,
}

# Lưu tất cả vào file
joblib.dump(model_package, MODEL_PATH)
print(f"Mô hình đã được lưu xong tại: {MODEL_PATH}")

# Tính confusion matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Non-Toxic", "Toxic"], yticklabels=["Non-Toxic", "Toxic"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Lấy kết quả từ classification_report
report = classification_report(y_test, y_pred, output_dict=True)

# Chuyển kết quả thành DataFrame và bỏ "accuracy", "macro avg"
df_report = pd.DataFrame(report).T.drop(["accuracy", "macro avg"], errors="ignore").iloc[:, :3]

ax = df_report.plot(kind='bar', figsize=(10, 5), colormap='coolwarm')

for p in ax.patches:
    ax.annotate(f"{p.get_height():.2f}",
                (p.get_x() + p.get_width() / 2, p.get_height()),
                ha='center', va='bottom', fontsize=8)

plt.title("So sánh Precision, Recall, F1-score của SVM", fontsize=12)
plt.xticks(rotation=0, fontsize=10)
plt.yticks(fontsize=10)
plt.xlabel("Loại nhãn", fontsize=10)
plt.ylabel("Giá trị", fontsize=10)
plt.ylim(0, 1)
plt.legend(loc='lower right', fontsize=9)
plt.show()


"""## Hàm dự đoán

Hàm dự đoán chạy chung với code train (model + vectorizer trong bộ nhớ)
"""


def predict_toxic_comment(text, model, vectorizer):
    """Dự đoán nhãn Toxic / Not Toxic cho một đoạn văn bản, dùng model & vectorizer
    đang có sẵn trong bộ nhớ (vừa train ở trên)."""
    clean_text = preprocess_text(text)
    text_vectorized = vectorizer.transform([clean_text])
    prediction = model.predict(text_vectorized)
    return "Toxic" if prediction[0] == 1 else "Not Toxic"


"""Ví dụ về Hàm dự đoán"""

# Dữ liệu bình luận mẫu để kiểm tra mô hình
demo_data = {
    "comment_text": [
        # Toxic comments (1)
        "You are so stupid, no one cares about your opinion!",
        "This is the worst comment I've ever read, you are completely useless!",
        "Shut up and get lost, nobody wants you here!",
        "You are such an idiot, go educate yourself before talking nonsense.",
        "This is the dumbest thing I have ever heard, stop wasting my time.",
        "You're a complete failure, just delete your account.",
        "No one likes you, go cry somewhere else.",
        "This is absolute garbage, only a fool would believe this.",
        "Go back to where you came from, we don't need people like you here.",
        "Your comment is as worthless as you are.",
        "I can't believe how dumb you are, just quit talking.",
        "You must be the most ignorant person I've ever seen.",
        "Stop spreading your stupidity, it's embarrassing.",
        "How can someone be this clueless? Unbelievable.",
        "You're nothing but a joke, nobody respects you.",
        "This is why no one likes you, you keep saying dumb things.",
        "I bet your family is ashamed of you.",
        "You're too dumb to even argue with, just stop.",
        "Do us all a favor and stay silent forever.",
        "You ruin every discussion with your nonsense.",

        # Non-toxic comments (0)
        "That's an interesting perspective, thanks for sharing!",
        "I appreciate your opinion, but I have a different point of view.",
        "This discussion is really insightful, I've learned a lot.",
        "Can you elaborate more on your point? I'd love to understand better.",
        "That's a valid argument, but have you considered this perspective?",
        "I disagree, but I respect your opinion.",
        "Your explanation was really clear, thanks for breaking it down.",
        "Let's try to keep the conversation respectful and constructive.",
        "Great points! I think there's value in both sides of the argument.",
        "I see what you mean, and I appreciate your input.",
        "That's a really thoughtful response, I never saw it that way before.",
        "I like how you presented your argument, very well structured!",
        "It's okay to have different opinions, that's how we learn.",
        "Thank you for keeping the discussion polite and meaningful.",
        "I appreciate the facts you brought up, they add depth to the topic.",
        "Your comment made me think about this in a new way.",
        "This is a great debate, thanks for sharing your insights!",
        "You make some really good points, let's continue discussing.",
        "That was a very respectful disagreement, I appreciate it.",
        "Your argument is well-thought-out and presented professionally.",
    ]
}

# Tạo DataFrame
df = pd.DataFrame(demo_data)

# Thêm cột nhãn: Toxic cho 20 dòng đầu, Not Toxic cho 20 dòng sau
# FIX: dùng len() thay vì số "20" cứng để tránh lệch nhãn nếu danh sách thay đổi
n_toxic = 20
n_non_toxic = len(df) - n_toxic
df['label'] = ['Toxic'] * n_toxic + ['Not Toxic'] * n_non_toxic

print(df)

"""Chạy dự đoán mà không dùng mô hình lưu (dùng svm_model, vect vừa train ở trên)"""

df['toxic_binary'] = df['comment_text'].apply(lambda x: predict_toxic_comment(x, svm_model, vect))
print(df)


"""## Tải data và xử lý data

# Dùng hàm dự đoán trên tập dữ liệu lớn (test.csv)
"""

test = pd.read_csv(TEST_PATH)

# Áp dụng hàm dự đoán toxicity cho từng dòng
test['toxic_binary'] = test['comment_text'].apply(lambda x: predict_toxic_comment(x, svm_model, vect))

columns_to_display = ['comment_text', 'toxic_binary']
print(test[columns_to_display])

test['toxic_binary'] = test['toxic_binary'].map({'Toxic': 1, 'Not Toxic': 0})
print(test)

"""# Tải và xử lý file nhãn đúng"""

# FIX: kiểm tra sự tồn tại của test_labels.csv trước khi đọc, vì đây là file
# chỉ được Kaggle công bố SAU khi cuộc thi kết thúc và người dùng cần tự tải về.
if os.path.exists(TEST_LABELS_PATH):
    label = pd.read_csv(TEST_LABELS_PATH)

    # FIX: thêm .copy() để tránh SettingWithCopyWarning ở dòng gán 'match' bên dưới
    test_binary = label[['toxic']].copy()
    print(test_binary)

    """## Đánh giá trên tập dữ liệu lớn"""

    test_binary['match'] = test_binary['toxic'] == test['toxic_binary']

    num_matches = test_binary['match'].sum()                       # Số dòng giống
    count_negative_labels = (test_binary['toxic'] == -1).sum()     # Số lượng nhãn không xác định
    num_mismatches = len(test_binary) - num_matches - count_negative_labels

    denom = len(test_binary) - count_negative_labels
    # FIX: tránh lỗi chia cho 0 nếu toàn bộ nhãn test đều là -1
    accuracy = (num_matches / denom * 100) if denom > 0 else float('nan')

    print(f"Số dòng khớp: {num_matches}")
    print(f"Số dòng không khớp: {num_mismatches}")
    print(f"Độ chính xác: {accuracy:.2f}%")

    """Vẽ biểu đồ để biểu diễn"""

    labels = ['Khớp', 'Không khớp']
    sizes = [num_matches, num_mismatches]
    colors = ['#4CAF50', '#FF5733']
    explode = (0.1, 0)

    plt.figure(figsize=(6, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, explode=explode, shadow=True, startangle=140)
    plt.title('Tỷ lệ Khớp/Không khớp')
    plt.show()
else:
    print(f"Không tìm thấy {TEST_LABELS_PATH} - bỏ qua bước đánh giá trên test.csv.")
    print("Tải file test_labels.csv từ trang Kaggle của cuộc thi Jigsaw Toxic Comment "
          "Classification Challenge để chạy phần đánh giá này.")


# =====================================================================
# Code mở rộng: Dự đoán dùng mô hình đã lưu (toxic_model.pkl)
# =====================================================================

def predict_toxic_comment_from_package(model_package_dict, text):
    """Dự đoán Toxic/Not Toxic dùng model_package đã load từ file .pkl."""
    model = model_package_dict["model"]
    vectorizer = model_package_dict["vectorizer"]
    preprocess = model_package_dict["preprocess"]

    clean_text = preprocess(text)
    text_vectorized = vectorizer.transform([clean_text])
    prediction = model.predict(text_vectorized)
    return "Toxic" if prediction[0] == 1 else "Not Toxic"


# Load mô hình đã lưu
# FIX: bản gốc đặt tên hàm "predict_toxic_comment" giống hàm ở trên (2 hàm trùng tên,
# hàm sau ghi đè hàm trước với chữ ký tham số khác nhau) -> dễ gây nhầm lẫn / lỗi
# khi gọi sai thứ tự tham số. Đã đổi tên hàm thứ hai thành
# predict_toxic_comment_from_package() để rõ ràng và không bị đè lẫn nhau.
model_package = joblib.load(MODEL_PATH)

df['toxic_binary'] = df['comment_text'].apply(lambda x: predict_toxic_comment_from_package(model_package, x))
print(df)

test = pd.read_csv(TEST_PATH)
test['clean_text'] = test['comment_text'].map(lambda com: preprocess_text(com))
print(test)

test['toxic_binary'] = test['clean_text'].apply(lambda x: predict_toxic_comment_from_package(model_package, x))
print(test)

test['toxic_binary'] = test['toxic_binary'].map({'Toxic': 1, 'Not Toxic': 0})
print(test)

if os.path.exists(TEST_LABELS_PATH):
    label = pd.read_csv(TEST_LABELS_PATH)
    test_binary = label[['toxic']].copy()
    print(test_binary)

    test_binary['match'] = test_binary['toxic'] == test['toxic_binary']

    num_matches = test_binary['match'].sum()
    count_negative_labels = (test_binary['toxic'] == -1).sum()
    num_mismatches = len(test_binary) - num_matches - count_negative_labels

    denom = len(test_binary) - count_negative_labels
    accuracy = (num_matches / denom * 100) if denom > 0 else float('nan')

    print(f"Số dòng khớp: {num_matches}")
    print(f"Số dòng không khớp: {num_mismatches}")
    print(f"Độ chính xác: {accuracy:.2f}%")
else:
    print(f"Không tìm thấy {TEST_LABELS_PATH} - bỏ qua bước đánh giá trên test.csv.")
