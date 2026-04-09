import numpy as np

def build_basic_session_graph(session_sequence):
    """
    Xây dựng đồ thị cho 1 phiên mua sắm (Bản GỐC - Không có Danh mục)
    Đầu vào: Mảng chứa ID các món đồ khách hàng đã lướt xem
    """
    # 1. TÌM CÁC ĐỈNH (NODES): Lọc ra các món đồ duy nhất trong phiên
    unique_nodes = np.unique(session_sequence)
    n_nodes = len(unique_nodes)
    
    # 2. TẠO KHUNG ĐỒ THỊ: Khởi tạo Ma trận vuông toàn số 0
    A = np.zeros((n_nodes, n_nodes))
    
    # 3. KẾT NỐI CẠNH (EDGES): Đọc lịch sử để vẽ hướng đi
    for i in range(len(session_sequence) - 1):
        u = session_sequence[i]     # Món hiện tại
        v = session_sequence[i + 1] # Món tiếp theo
            
        # Tìm tọa độ của u và v trong ma trận
        u_idx = np.where(unique_nodes == u)[0][0]
        v_idx = np.where(unique_nodes == v)[0][0]
        
        # Đánh dấu có 1 cú click (bước nhảy) từ u sang v
        A[u_idx][v_idx] = 1
        
    # 4. CHUẨN HÓA (Để tránh nổ gradient khi tính toán)
    # 4.1. Chuẩn hóa chiều ĐI VÀO (In-degree)
    sum_in = np.sum(A, axis=0)
    sum_in[sum_in == 0] = 1 # Chỗ nào tổng = 0 thì ép thành 1 để tránh lỗi chia cho 0
    A_in = np.divide(A, sum_in)
    
    # 4.2. Chuẩn hóa chiều ĐI RA (Out-degree)
    sum_out = np.sum(A, axis=1)
    sum_out[sum_out == 0] = 1
    A_out = np.divide(A.transpose(), sum_out)
    
    # 4.3. Ghép 2 chiều lại thành ma trận GNN chuẩn mực
    A_final = np.concatenate([A_in, A_out]).transpose()
    
    return unique_nodes, A_final

# ==========================================
# CHẠY THỬ NGHIỆM ĐỂ XEM AI "MÙ MỜ" THẾ NÀO
# ==========================================

# Khách xem: Áo (15) -> Quần (230) -> Quay lại nhìn Áo (15) -> Mua Mũ (4)
session = [15, 230, 15, 4]

nodes, matrix = build_basic_session_graph(session)

print(f"1. Các Nút (Nodes) đại diện cho sản phẩm: {nodes}")
print("\n2. Ma trận kề A (Đã chuẩn hóa In/Out) mà mạng GNN sẽ học:")
print(np.round(matrix, 2))