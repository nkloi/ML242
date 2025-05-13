Phần tiền xử lý :extract_emails()
    - xử lý subject
    load các email , xử lý file .mbox gộp các file mail lại 
    add các file có liên quan như từ điển ,spamword,stopwords
    loại bỏ nội dung html và các dấu câu, đổi chữ in hoa thành in thường
    ----------------------------------------------------
    - xử lý body
    loại bỏ các tệp đính kèm 
    clean test giống phần subject để lấy text
    -----------------------------------------------------
    đọc file spamword.txt để kiểm tra với hệ thống file có chấm điểm 
    với các từ có khả năng là spam khi xuất hiện trong thư thì điểm càng cao
    nếu tổng điểm của mail đó lớn hơn 10 thì gắn mác spam lên đầu mail
    xuất file ra file output.csv 
Phần traning với tập dữ liệu có sẵn
    Logistic regression:
        với file có sẵn output trên lấy 80% là để tranning và 20% để test 
        vecto hóa và đưa 2 phần là mail và label vào tranning sau đó predic 
        dùng confusion matrix để kiểm tra số lượng mail bị dự đoán sai
        đưa ra đánh giá cho phần này
    K-mean 
        add file output (do K-mean là đọc toàn bộ chữ cái của văn bản nên)
        tạo file dictionary bao gồm tiếng ANH lẫn tiếng VIỆT
        lọc qua chỉ giữ lại những từ có trong từ điển
        bắt đầu vecto hóa mail giữ lại 70% những từ có trong văn bản để loại bỏ những từ hiển thị quá nhiều như là "là ,tôi ,tớ, me,you,..."
        sau đó có 2 mảng là những từ được sử dụng nhiều và số lần sử dụng
        sắp xếp theo tăng dần
        tạo điểm K ngẫu nhiên
        chọn các điểm ở gần 1 trong 2 điểm 
        sau đó hiệu chỉnh tính trung bình khoảng cách lấy điểm K mới 
        lấy các danh sách các từ ở phần đa số ở bảng trên 
        lấy danh sách này đánh giá lại file output.csv 
        lấy các số liệu ở trên và đưa ra đánh giá mô hình
