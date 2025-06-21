import os
from dotenv import load_dotenv
import google.generativeai as genai
import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.saving import load_model

load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=google_api_key)

# Dùng session state để lưu trạng thái người dùng
if 'page' not in st.session_state:
    st.session_state.page = 'home'

# Hàm quay lại trang chủ
def go_home():
    st.session_state.page = "home"

# Hàm chuyển trang
def set_page(page_name):
    st.session_state.page = page_name

# Trang chủ
def home():
    st.title("Guitarbot")
    st.header("Nơi tư vấn và dự đoán thể loại đàn guitar")
    st.title("")
    st.subheader("Chọn một chức năng bên dưới để bắt đầu:")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Chatbot tư vấn chọn đàn"):
            set_page('chatbot')
    with col2:
        if st.button("Dự đoán thể loại đàn theo hình ảnh"):
            set_page('predict')


def chatbot():
    st.title("🎸 Chatbot tư vấn chọn đàn guitar")

    # Khởi tạo chat history nếu chưa có
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Tạo system_instruction từ lịch sử chat
    def create_system_instruction():
        formatted = ""
        for msg in st.session_state.messages:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                formatted += f"Khách hàng: {content}\n"
            elif role == "assistant":
                formatted += f"Chatbot: {content}\n"
        return formatted

    # Gọi hàm để lấy nội dung hội thoại
    formatted_history = create_system_instruction()

    # Tạo mô hình Gemini với system_instruction đã chèn nội dung lịch sử
    model = genai.GenerativeModel(
        "models/gemini-2.5-flash-preview-04-17-thinking",
        system_instruction=f"""
        Bạn là một chuyên gia tư vấn nhạc cụ, đặc biệt là đàn guitar.
        1. Nhiệm vụ của bạn là giúp khách hàng chọn cây đàn phù hợp nhất dựa trên trình độ, nhu cầu, và ngân sách của họ.
        2. Đưa lý do cụ thể tại sao nên chọn cây đàn đó, và nếu cần thì giải thích ngắn gọn, xúc tích, không sử dụng từ ngữ chuyên môn quá phức tạp.
        3. Đừng hỏi liền một mạch tất cả câu hỏi, hãy hỏi từng câu một và chờ khách hàng trả lời trước khi hỏi tiếp.
        4. Sau khi khách hàng trả lời một câu hỏi, hãy lưu lại thông tin đó và KHÔNG hỏi lại cùng một câu hỏi. Chỉ chuyển sang câu hỏi tiếp theo trong thứ tự nếu chưa có thông tin. Nếu đã có đủ thông tin, hãy tư vấn đàn phù hợp.
        5. Nếu khách hàng hỏi về các vấn đề khác không liên quan đến đàn guitar, hãy trả lời ngắn gọn và chuyển hướng về guitar.
        6. Không đưa ra ví dụ mỗi câu hỏi
        7. Nếu khách hàng hỏi về các vấn đề khác không liên quan đến đàn guitar, hãy trả lời ngắn gọn và chuyển hướng về guitar.
        
        Dưới đây là hội thoại trước đó với khách hàng:
        {formatted_history}
        
        Dựa trên nội dung trước, hãy tiếp tục phản hồi thông minh và đúng nhu cầu.
        """
    )

    chat = model.start_chat()

    # Hiển thị lịch sử chat
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Nhập câu mới
    if prompt := st.chat_input("Nhập tin nhắn của bạn ở đây..."):
        # Lưu tin nhắn người dùng
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Gửi tin nhắn đến Gemini
        response = chat.send_message(prompt)
        bot_reply = response.candidates[0].content.parts[0].text

        # Lưu và hiển thị phản hồi của bot
        st.session_state.messages.append({"role": "assistant", "content": bot_reply})
        with st.chat_message("assistant"):
            st.markdown(bot_reply)

    # Nút quay lại trang chủ (nằm cố định cuối trang)
    with st.container():
        if st.button("⬅ Quay lại trang chủ", key="back_btn_chatbot"):
            go_home()
            st.stop()

# Danh sách nhãn (label) tương ứng với output của model
labels = ["Double Neck Guitar", "Electric Guitar", "Acoustic Guitar", "Triple Neck Guitar"]

# Hàm tiền xử lý ảnh
def preprocess_PIL(image):
    image = image.convert("RGB")
    image = image.resize((128, 128))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@st.cache_resource
def load_guitar_model(model_path="guitar_type_classifier.keras"):
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Không thể load model: {e}")
        return None

def predict():
    st.title("Dự đoán thể loại đàn bằng hình ảnh")
    with st.container():
        if st.button("⬅ Quay lại trang chủ", key="back_btn_predict"):
            go_home()
            st.stop()  # Dừng render tiếp nếu quay lại

    model = load_guitar_model("guitar_type_classifier.keras")
    if model is None:
        st.warning("Model chưa sẵn sàng.")
        return

    uploaded_file = st.file_uploader("Chọn ảnh đàn guitar...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Ảnh đã tải lên", use_container_width=True)
        if st.button("Dự đoán"):
            img_array = preprocess_PIL(image)
            pred = model.predict(img_array)
            pred_index = np.argmax(pred)
            pred_label = labels[pred_index]
            confidence = np.max(pred)
            st.success(f"Kết quả: **{pred_label}** ({confidence*100:.2f}%)")

# Router
if st.session_state.page == 'home':
    home()
elif st.session_state.page == 'chatbot':
    chatbot()
elif st.session_state.page == 'predict':
    predict()
