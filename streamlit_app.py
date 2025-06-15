import os
from dotenv import load_dotenv
import google.generativeai as genai
import streamlit as st

load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=google_api_key)


model = genai.GenerativeModel("models/gemini-2.5-flash-preview-04-17-thinking",
                              system_instruction="""
                              Bạn là một chuyên gia tư vấn nhạc cụ, đặc biệt là đàn guitar. 
                              1. Nhiệm vụ của bạn là giúp khách hàng chọn cây đàn phù hợp nhất dựa trên trình độ, nhu cầu, và ngân sách của họ.
                              2. Hãy đặt thêm câu hỏi ngắn gọn, bao hàm khi cần để hiểu rõ khách hàng hơn. Đừng hỏi quá dông dài, chỉ cần đúng trọng tâm 
                              3. Đưa lý do cụ thể tại sao nên chọn cây đàn đó, và nếu cần thì giải thích ngắn gọn xúc tích , không sử dụng từ ngữ chuyên môn quá phức tạp
                              4. Hãy đưa ra các lựa chọn phù hợp kèm theo mô tả chi tiết và so sánh giữa các loại đàn (classic, acoustic, electric,...)
                              5. Đừng hỏi liền 1 mạch tất cả câu hỏi, hãy hỏi theo thứ tự từng câu: trình độ chơi rồi đến lối chơi, thể loại muốn chơi rồi cuối cùng là giá tiền
""")

chat = model.start_chat()

# Dùng session state để lưu trạng thái người dùng
if 'page' not in st.session_state:
    st.session_state.page = 'home'

# Hàm quay lại trang chủ
def go_home():
    st.session_state.page = 'home'

# Hàm chuyển trang
def set_page(page_name):
    st.session_state.page = page_name

# Trang chủ
def home():
    st.title("Tư vấn & Dự đoán Đàn Guitar")
    st.write("Chọn một chức năng bên dưới để bắt đầu:")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Chatbot tư vấn chọn đàn"):
            set_page('chatbot')
    with col2:
        if st.button("Dự đoán thể loại đàn theo hình ảnh"):
            set_page('predict')


def chatbot():
    st.title("Chatbot tư vấn chọn đàn")
    # Khởi tạo session state nếu chưa có
    if "messages" not in st.session_state:
        st.session_state.messages = []  # Danh sách để lưu các tin nhắn

    # Hiển thị lịch sử chat
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Gửi tin nhắn mới
    if prompt := st.chat_input("Nhập tin nhắn của bạn ở đây..."):
        # Lưu tin nhắn người dùng
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Gửi prompt tới Gemini
        response = chat.send_message(prompt)
        bot_reply = response.candidates[0].content.parts[0].text

        # Lưu và hiển thị tin nhắn của bot
        st.session_state.messages.append({"role": "assistant", "content": bot_reply})
        with st.chat_message("assistant"):
            st.markdown(bot_reply)
            # Nút back nằm riêng ở đầu, không bị ảnh hưởng bởi chat
        with st.container():
            if st.button("⬅ Quay lại trang chủ", key="back_btn_chatbot"):
                go_home()
                st.stop()  # Dừng render tiếp nếu quay lại

def predict():
    st.title("Dự đoán thể loại đàn theo hình ảnh")
    st.write("Chức năng này sẽ được cập nhật trong tương lai.")
    # Nút back nằm riêng ở đầu, không bị ảnh hưởng bởi chat
    with st.container():
        if st.button("⬅ Quay lại trang chủ", key="back_btn_predict"):
            go_home()
            st.stop()  # Dừng render tiếp nếu quay lại
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras import layers, models
    
    # Đường dẫn tới thư mục chứa dữ liệu
    train_dir = 'Chatbot/train'
    val_dir = 'Chatbot/validation'
    
    # Tiền xử lý và tăng cường dữ liệu
    train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, zoom_range=0.2, horizontal_flip=True)
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(128, 128),
        batch_size=32,
        class_mode='binary'
    )
    
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(128, 128),
        batch_size=32,
        class_mode='binary'
    )
    
    # Xây dựng mô hình CNN đơn giản
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Huấn luyện mô hình
    model.fit(
        train_generator,
        epochs=10,
        validation_data=val_generator
    )
    
    # Lưu mô hình
    model.save('guitar_classifier.h5')
# Router
if st.session_state.page == 'home':
    home()
elif st.session_state.page == 'chatbot':
    chatbot()
# elif st.session_state.page == 'predict':
#     predict()
