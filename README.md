# Hanoi House Demo V8.5 — Notebook Exact + Flexible Auto Mode

## Hai chế độ tự động
1. **Notebook Exact**: chỉ kích hoạt khi file có đầy đủ 13 cột nhận dạng gốc của notebook.
   Pipeline được chép đúng từ `Phan_tich_du_bao_gia_nha_HaNoi(3).ipynb`.
2. **Flexible**: schema khác dùng Column Intelligence + Strict Hanoi Scope + Adaptive Cleaning.

## Regression test bắt buộc đã chạy
Với `HN_Houseprice(1).csv`:
- Raw: 13,535
- Loại trùng: 13,535 → 10,761 (loại 2,774)
- Lọc Hà Nội + BĐS để ở + có Price/Area: 10,761 → 8,233
- IQR tuần tự:
  - Price_ty: -504
  - Area_m2: -111
  - Bedrooms_n: -386
  - Bathrooms_n: -2
  - Floors_n: -1
- Price < 0.3 tỷ: -4
- **Clean cuối: 7,225 × 27**
- Từ 8,233 → 7,225: loại 1,008 = **12.2%**

Với `dataset.csv` schema khác:
- Raw: 81,162
- Không kích hoạt Notebook Exact
- Flexible pipeline clean: 81,081

Backend compile và train() đã được kiểm tra cho cả hai chế độ.

## V8.5.1 — EDA frontend fix
- Sửa lỗi `renderEDA is not defined`.
- Nguyên nhân: code `renderEDA`, `hist`, `chart`, `medianChart` nằm nhầm bên trong `<script src="...">`; trình duyệt bỏ qua phần inline đó.
- Tách đúng thành `<script src="..."></script>` và một `<script>` riêng.
- Bổ sung fallback cho các trường EDA tùy chọn.
- Đã chạy `node --check` cho toàn bộ inline JavaScript và `py_compile` cho backend.
- Giữ nguyên Notebook Exact regression: 13,535 → 10,761 → 8,233 → 7,225.

## V8.5.2 — Prediction imputation
- Chỉ thay đổi logic ở màn hình dự báo; không thay đổi Notebook Exact training pipeline.
- Căn hộ/chung cư/condotel: UI khóa `Số tầng`; backend nhận `Floors_n=null` và tự lấy median `Floors_n` theo `PropertyType` từ dữ liệu sạch dùng huấn luyện.
- Nhà riêng/biệt thự/nhà mặt phố: vẫn cho nhập số tầng.
- Các predictor bổ sung: numeric dùng median, categorical dùng mode như trước.
- Đầu vào ngoài khoảng train được cảnh báo, không tự clip.
- Regression test: Notebook Exact vẫn 8,233 → 7,225 (1,008 = 12.2%); flexible `dataset.csv` vẫn 81,081.

## V8.5.2 + Default Dataset
- Giữ nguyên cơ chế tự nhận diện/chuẩn hóa cột của V8.5.2; không có Mapping Wizard.
- `Unnamed: 0` không được hard-code thành một trường nghiệp vụ.
- Khi khởi động, tự nạp `data/HN_Houseprice_default.csv`, chạy Notebook Exact và train model.
- Người dùng có thể vào Dự báo ngay mà không cần upload.
- Upload CSV khác vẫn đi qua pipeline V8.5.2 như cũ.
- Có nút `Dùng lại dữ liệu mặc định`.

## CSV Review before training
- Upload CSV no longer replaces the active model immediately.
- Shows up to 20 preview rows and non-destructive suggestions.
- User may rename columns, edit visible cells, or mark columns to drop.
- Only `Xác nhận và xử lý dữ liệu` invokes the original V8.5.2 preprocessing/training pipeline.
- Default notebook model remains active while reviewing/cancelling a new CSV.
