import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------
# PHẦN 1: CHUẨN BỊ "CỬA HÀNG" VÀ "KHÁCH HÀNG"
# ---------------------------------------------------------

# Giả sử cửa hàng của bạn chỉ có đúng 3 món đồ:
# ID 0: Điện thoại iPhone
# ID 1: Ốp lưng
# ID 2: Củ sạc nhanh
NUM_ITEMS = 3
EMBEDDING_DIM = 2 # Mỗi món đồ được đại diện bằng 1 vector có 2 chữ số

# 1. Khởi tạo Ma trận hàng hóa (Bảng Embedding). PyTorch sẽ random các con số lúc đầu.
item_embeddings = nn.Embedding(num_embeddings=NUM_ITEMS, embedding_dim=EMBEDDING_DIM)

# 2. Khởi tạo "Ông thầy giáo"
optimizer = torch.optim.SGD(item_embeddings.parameters(), lr=0.5)

# 3. Lịch sử lướt web (Đề bài)
# Khách đã click xem Điện thoại (0) và Ốp lưng (1)
session_items = torch.tensor([0, 1]) 

# Món đồ khách thực sự sẽ mua tiếp theo (Đáp án cần AI đoán trúng): Củ sạc (2)
target_item = torch.tensor([2])      

print("--- TRƯỚC KHI HỌC ---")
# Chưa học nên AI sẽ tính điểm random, đoán bừa bãi
initial_vectors = item_embeddings(session_items)
initial_intent = torch.sum(initial_vectors, dim=0, keepdim=True)
initial_scores = torch.matmul(initial_intent, item_embeddings.weight.T)
print(f"Điểm số đoán mò lúc đầu cho 3 món [Điện thoại, Ốp, Sạc]:\n{initial_scores.data[0]}\n")

# ---------------------------------------------------------
# PHẦN 2: VÒNG LẶP HUẤN LUYỆN (MÔ PHỎNG SR-GNN)
# ---------------------------------------------------------

for epoch in range(15):
    optimizer.zero_grad()
    
    # BƯỚC 1: RÚT ĐẶC TRƯNG HÀNG HÓA
    # Bốc vector của Điện thoại và Ốp lưng từ trong kho ra
    vectors_in_session = item_embeddings(session_items) 
    
    # BƯỚC 2: TỔNG HỢP Ý ĐỊNH PHIÊN (Giả lập mạng GNN)
    # Trong SR-GNN, tác giả dùng Gated GNN và Attention phức tạp.
    # Ở đây ta tối giản hóa: Lấy vector Điện thoại CỘNG với vector Ốp lưng thành 1 vector "Ý định"
    session_intent = torch.sum(vectors_in_session, dim=0, keepdim=True)
    
    # BƯỚC 3: PHÉP CHỐT SALE (Dot Product)
    # Lấy Ý định nhân vô hướng với Toàn bộ 3 món đồ trong cửa hàng
    all_items = item_embeddings.weight
    scores = torch.matmul(session_intent, all_items.T) # Ra 3 điểm số
    
    # BƯỚC 4: TÍNH LỖI VÀ CHỮA BÀI
    # Hàm CrossEntropy: Ép AI phải đẩy điểm của món số 2 (Củ sạc) lên cao nhất
    loss = F.cross_entropy(scores, target_item)
    
    loss.backward()  # Tự động tính toán xem vector nào đang bị lệch hướng
    optimizer.step() # Vặn lại các con số trong Bảng Embedding
    
    # Trích xuất xem AI đang ưu tiên món nào nhất
    predicted_item = torch.argmax(scores).item()
    
    print(f"Epoch {epoch+1:2d} | Lỗi (Loss): {loss.item():.4f} | AI dự đoán khách sẽ mua món ID: {predicted_item}")

print("\n--- KẾT LUẬN SAU KHI HỌC ---")
print(f"Điểm số cuối cùng [Điện thoại, Ốp, Sạc]:\n{scores.data[0]}")
print("Thành công! Vector của Củ sạc đã bị kéo giãn ra để khớp hoàn hảo với Ý định của phiên, khiến điểm của Củ sạc vọt lên cao nhất!")