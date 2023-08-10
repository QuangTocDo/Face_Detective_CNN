## Giới thiệu chương trình:

- Chương trình sử dụng flask để chạy server trên python.
- Khi chạy chương trình trang web sẽ hiển thị và khi ấn vào đăng nhập sẽ có 2 bước xác thực là nhập mật khẩu bình thường sau đó là nhận diện khuôn mặt thành công thì mới trả về kết quả.

## Các chức năng của:

- Vì mục đích chính của chương trình là nhận diện xác thực khuôn mặt nên tài khoản login sẽ được set cứng trong file app.py
- Sau khi đã nhập đúng username và password thì người dùng sẽ phải qua bước xác thực khuôn mặt nếu đúng là người đó thì mới cho phép đăng nhập.

## Thực hiện chương trình
- Để chạy chương trình thì đầu tiên phải có một bộ dữ liệu ảnh dataset của người dùng cung cấp
- sử dụng lệnh:
    - py NewUser.py để thêm bộ ảnh của người dùng mới

- Sau khi đã thêm người dùng mới, ta tiếp tục chạy chương trình train mô hình với lệnh sau:
    - py TrainCNN.py: chương trình này có thể năng và tốn nhiều tài nguyên của máy

- Sau khi đã train mô hình xong thì vào app.py:
    - thêm người dùng vừa tạo với id vào biến users

- Cuối cùng là chạy file app.py: py app.py