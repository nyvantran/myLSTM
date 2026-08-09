
# 1. Mục đích của nhánh LSTM:
Trong khi mạng MCNN (Multi-granularity CNN) có khả năng trích xuất các đặc trưng không gian (spatial representations) tĩnh trên từng khung hình, nhánh LSTM được áp dụng trên chuỗi các đặc trưng này nhằm tìm ra **mối quan hệ phụ thuộc dài hạn (long-term temporal dependencies) với độ dài thay đổi** trên video. Việc học các manh mối thời gian này cho phép mô hình dễ dàng phân biệt các trạng thái hành vi dễ nhầm lẫn nếu chỉ nhìn vào không gian tĩnh như chớp mắt so với nhắm mắt khi buồn ngủ, hoặc ngáp so với cười.

# 2. Cấu trúc chi tiết của khối LSTM:
*   **Đầu vào:** Đầu vào của nhánh bộ nhớ thời gian là 1 batch các vector đặc trưng không gian $x^t$ có kích thước ([640, 15, 15]), được sinh ra từ đầu ra của MCNN. 
*   **Kiến trúc mạng (Deep LSTM Layers):** Nhánh này sử dụng một kiến trúc mạng sâu gồm **3 lớp LSTM được xếp chồng lên nhau** (three layers LSTMs) để tăng cường khả năng trích xuất. Mỗi khối LSTM đơn lẻ trong mạng được cấu tạo từ các thành phần tiêu chuẩn: một cổng đầu vào (input gate), một cổng quên (forget gate), một cổng đầu ra (output gate) và một ô nhớ (memory cell) để có thể ghi đè hoặc lưu trữ thông tin thời gian dài hạn.
*   **Đơn vị ẩn (Hidden units):** **Số lượng các đơn vị ẩn (hidden units) trong mỗi khối LSTM được thiết lập là 256**, tương ứng với số chiều kích thước của vector đầu vào $x^t$.
*   **Độ dài bộ nhớ (Memory length):** Quá trình xử lý kích hoạt cổng quên (forget gate) với bước nhớ tối đa (max memory step) thường được đặt là **60 khung hình** (trên tập dữ liệu FI-DDD). Khi huấn luyện đánh giá trên tập NTHU-DDD (chứa các chuỗi có đặc trưng nhớ rất dài), độ dài bước nhớ này được mở rộng lên **120 khung hình**.

# 3. Phân loại ở tầng đầu ra (Output & Classification):
*   Sau khi đi qua 3 lớp, trạng thái ẩn của khối LSTM thứ ba ($h^t_3$) sẽ hội tụ đầy đủ các chuỗi phụ thuộc thời gian cần thiết. 
*   Trạng thái này tiếp tục được chiếu (project) qua **một lớp kết nối đầy đủ (fully connected layer)** mang theo bộ trọng số $W^R$ và vector độ lệch $b^R$ để hạ chiều xuống thành một vector 2 chiều.
*   Cuối cùng, **hàm kích hoạt Softmax** giải mã vector 2 chiều này thành xác suất của 2 danh mục (buồn ngủ hoặc bình thường) để từ đó dự đoán khung hình hiện tại thuộc lớp nào.
