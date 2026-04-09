import pickle
import random

# 1. Đọc cuốn từ điển gốc bạn vừa tạo
item2cat = pickle.load(open('item2cat_internal.pkl', 'rb'))

# 2. Đảo ngược từ điển
cat2items = {}
for item, cat in item2cat.items():
    if cat not in cat2items:
        cat2items[cat] = []
    cat2items[cat].append(item)

# 3. Lưu lại để dùng cho vòng lặp train
pickle.dump(cat2items, open('cat2items_dict.pkl', 'wb'))
print(f"Đã tạo xong rổ hàng hóa cho {len(cat2items)} danh mục!")

import pickle

# 1. Đọc cuốn từ điển vừa tạo
with open('cat2items_dict.pkl', 'rb') as f:
    cat2items = pickle.load(f)

# 2. Tính toán các chỉ số thống kê
total_categories = len(cat2items)
items_per_cat = {cat: len(items) for cat, items in cat2items.items()}

# Lấy ra danh sách số lượng để dễ tính toán
lengths = list(items_per_cat.values())

total_items = sum(lengths)
max_items = max(lengths)
min_items = min(lengths)
avg_items = total_items / total_categories

# 3. Phân tích độ khả thi cho Module 2 (Augmentation)
# Danh mục "Cô đơn": Chỉ có 1 sản phẩm -> KHÔNG THỂ tráo đổi
lonely_cats = {cat: count for cat, count in items_per_cat.items() if count < 2}
num_lonely = len(lonely_cats)

# Sắp xếp để xem Top danh mục lớn nhất và nhỏ nhất
sorted_cats = sorted(items_per_cat.items(), key=lambda x: x[1], reverse=True)
top_5_largest = sorted_cats[:5]
top_5_smallest = sorted_cats[-5:]

# ==========================================
# IN KẾT QUẢ BÁO CÁO
# ==========================================
print("="*50)
print("📊 BÁO CÁO THỐNG KÊ TỪ ĐIỂN DANH MỤC")
print("="*50)
print(f"- Tổng số Danh mục hiện có: {total_categories:,}")
print(f"- Tổng số Sản phẩm (Items) đã ánh xạ: {total_items:,}")
print(f"- Trung bình: {avg_items:.2f} sản phẩm / danh mục")
print(f"- Danh mục LỚN NHẤT chứa: {max_items:,} sản phẩm")
print(f"- Danh mục NHỎ NHẤT chứa: {min_items:,} sản phẩm")
print("-" * 50)
print("🔍 ĐÁNH GIÁ SỨC KHỎE CHO MODULE 2 (AUGMENTATION):")
print(f"- Số danh mục 'Cô đơn' (<= 1 sản phẩm): {num_lonely:,} ({num_lonely/total_categories*100:.2f}%)")
if num_lonely > 0:
    print("  -> LƯU Ý: Khi thuật toán bốc trúng sản phẩm thuộc các danh mục này,")
    print("     nó sẽ bỏ qua và giữ nguyên phiên gốc vì không có gì để tráo.")
print("-" * 50)
print("🏆 TOP 5 DANH MỤC CHỨA NHIỀU SẢN PHẨM NHẤT:")
for cat, count in top_5_largest:
    print(f"  + Danh mục ID {cat}: {count:,} sản phẩm")
print("="*50)