# ===================================================================
# THÊM MỚI: TĂNG CƯỜNG DANH MỤC BẰNG MACHINE LEARNING (WORD2VEC + K-MEANS)
# ===================================================================
from copyreg import pickle
import datetime

from gensim.models import Word2Vec
from sklearn.cluster import KMeans
import numpy as np

print("-- Bắt đầu quá trình tạo Danh mục ẩn (Latent Categories) @ %ss" % datetime.datetime.now())

# Bước 1: Thu thập tất cả các chuỗi hợp lệ (đã lọc các item xuất hiện < 5 lần)
all_valid_sequences = []
for s, seq in sess_clicks.items():
    # Chuyển ID thành chuỗi (string) vì Word2Vec yêu cầu đầu vào là văn bản
    str_seq = [str(item) for item in seq]
    all_valid_sequences.append(str_seq)

# Bước 2: Huấn luyện Word2Vec để hiểu ngữ nghĩa của từng Item
# vector_size: Độ dài vector nhúng (có thể chọn 64 hoặc 100)
# window: Khoảng cách nhìn trước/nhìn sau trong session
# min_count: Bỏ qua các item hiếm (ta đã lọc bằng 5 ở trên rồi nên để 1)
print("   + Đang huấn luyện Word2Vec để tìm mối quan hệ sản phẩm...")
w2v_model = Word2Vec(sentences=all_valid_sequences, vector_size=64, window=5, min_count=1, workers=4, epochs=10)

# Bước 3: Lấy tất cả vector của các Item đã được học
vocab_items = list(w2v_model.wv.key_to_index.keys())
item_vectors = np.array([w2v_model.wv[item] for item in vocab_items])

# Bước 4: Chạy K-Means để chia sản phẩm thành N danh mục mới
# Bạn có thể điều chỉnh số lượng danh mục mong muốn ở biến n_clusters
NUM_NEW_CATEGORIES = 300 
print(f"   + Đang phân cụm sản phẩm thành {NUM_NEW_CATEGORIES} danh mục mới...")
kmeans = KMeans(n_clusters=NUM_NEW_CATEGORIES, random_state=42, n_init='auto')
cluster_labels = kmeans.fit_predict(item_vectors)

# Bước 5: Cập nhật lại từ điển item2cat_raw bằng các danh mục mới này
item2cat_raw = {}
for i, item_str in enumerate(vocab_items):
    original_item_id = int(item_str)
    new_category_id = int(cluster_labels[i]) + 1 # Cộng 1 để tránh trùng với số 0 (padding)
    item2cat_raw[original_item_id] = new_category_id

print(f"-- Đã tạo xong {NUM_NEW_CATEGORIES} danh mục mới tinh xảo hơn! @ %ss" % datetime.datetime.now())

# Lưu lại từ điển danh mục mới để sau này Module 2 dùng
pickle.dump(item2cat_raw, open('item2cat_internal.pkl', 'wb'))
# ===================================================================