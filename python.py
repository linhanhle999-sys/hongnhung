import streamlit as st
import numpy as np
import json
import io
import time
import math

# --- CÁC HẰNG SỐ VÀ CẤU HÌNH API ---
# KHÔNG CẦN CHỈ ĐỊNH API KEY, canvas sẽ tự động cung cấp
API_KEY = ""
GEMINI_MODEL = "gemini-2.5-flash-preview-05-20"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={API_KEY}"

# --- HÀM HỖ TRỢ PHÂN TÍCH VÀ API ---

# 1. Hàm API: Gọi Gemini để trích xuất dữ liệu tài chính có cấu trúc
def call_gemini_extraction(text_content):
    """Gọi Gemini API để trích xuất các thông số tài chính dưới dạng JSON."""
    
    # Định nghĩa Schema JSON mong muốn
    response_schema = {
        "type": "OBJECT",
        "properties": {
            "investment_capital": {"type": "NUMBER", "description": "Tổng vốn đầu tư ban đầu (năm 0), đơn vị tiền tệ, không cần đơn vị nghìn/triệu."},
            "project_lifespan_years": {"type": "INTEGER", "description": "Dòng đời dự án theo năm (ví dụ: 5 hoặc 10)."},
            "annual_revenue": {"type": "ARRAY", "items": {"type": "NUMBER"}, "description": "Doanh thu dự kiến hàng năm (từ năm 1 đến hết đời dự án)."},
            "annual_cost": {"type": "ARRAY", "items": {"type": "NUMBER"}, "description": "Chi phí hoạt động hàng năm (từ năm 1 đến hết đời dự án)."},
            "wacc_percent": {"type": "NUMBER", "description": "Chi phí sử dụng vốn (WACC) tính theo phần trăm (ví dụ: 10.5 cho 10.5%)."},
            "tax_rate_percent": {"type": "NUMBER", "description": "Thuế suất doanh nghiệp tính theo phần trăm (ví dụ: 20)."}
        },
        "required": ["investment_capital", "project_lifespan_years", "annual_revenue", "annual_cost", "wacc_percent", "tax_rate_percent"]
    }
    
    # Hướng dẫn hệ thống (System Instruction)
    system_prompt = (
        "Bạn là một trợ lý trích xuất dữ liệu tài chính cấp cao. "
        "Nhiệm vụ của bạn là đọc nội dung kế hoạch kinh doanh được cung cấp, "
        "xác định chính xác các con số sau: Vốn đầu tư ban đầu (năm 0), Dòng đời dự án (năm), "
        "Doanh thu và Chi phí hàng năm (phải có đủ số năm theo dòng đời dự án), "
        "WACC (%), và Thuế suất (%). "
        "Nếu các giá trị Doanh thu/Chi phí không được đề cập chi tiết qua từng năm, "
        "hãy nhân bản giá trị trung bình tìm được cho đủ số năm của dự án. "
        "Đảm bảo đầu ra là một đối tượng JSON TUYỆT ĐỐI tuân thủ schema đã cung cấp."
    )

    user_query = f"Hãy trích xuất các thông số tài chính sau từ kế hoạch kinh doanh này:\n\n---\n{text_content}"

    payload = {
        "contents": [{"parts": [{"text": user_query}]}],
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": response_schema
        }
    }
    
    # Thực hiện gọi API với cơ chế retry (exponential backoff)
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = st.session_state.fetch(
                API_URL, 
                method='POST', 
                headers={'Content-Type': 'application/json'}, 
                body=json.dumps(payload)
            )
            
            if response.status_code != 200:
                raise Exception(f"API returned status code {response.status_code}")
                
            result = response.json()
            
            # Kiểm tra và phân tích kết quả JSON
            candidate = result.get('candidates', [{}])[0]
            if candidate and candidate.get('content') and candidate['content'].get('parts'):
                json_string = candidate['content']['parts'][0].get('text')
                if json_string:
                    return json.loads(json_string)

        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                time.sleep(wait_time)
            else:
                st.error(f"Lỗi API sau {max_retries} lần thử: {e}")
                return None
    return None

# 2. Hàm API: Gọi Gemini để phân tích kết quả tài chính
def call_gemini_analysis(metrics_report):
    """Gọi Gemini API để phân tích các chỉ số hiệu quả dự án."""
    
    system_prompt = (
        "Bạn là một chuyên gia phân tích tài chính đầu tư. "
        "Hãy đọc báo cáo chỉ số hiệu quả dự án (NPV, IRR, PP, DPP) và cung cấp một phân tích chuyên sâu, "
        "tập trung vào tính khả thi, độ hấp dẫn của dự án (so sánh IRR với WACC), "
        "và rủi ro thanh khoản (dựa trên PP và DPP). "
        "Sử dụng ngôn ngữ chuyên nghiệp, dễ hiểu. Phân tích nên dài khoảng 3-4 đoạn văn."
    )
    
    user_query = f"Hãy phân tích báo cáo hiệu quả dự án sau:\n\n---\n{metrics_report}"

    payload = {
        "contents": [{"parts": [{"text": user_query}]}],
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "tools": [{"google_search": {}}] # Kích hoạt Google Search để tăng tính chính xác
    }
    
    # Thực hiện gọi API với cơ chế retry
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = st.session_state.fetch(
                API_URL, 
                method='POST', 
                headers={'Content-Type': 'application/json'}, 
                body=json.dumps(payload)
            )
            
            if response.status_code != 200:
                raise Exception(f"API returned status code {response.status_code}")
                
            result = response.json()
            
            candidate = result.get('candidates', [{}])[0]
            if candidate and candidate.get('content') and candidate['content'].get('parts'):
                analysis_text = candidate['content']['parts'][0].get('text')
                return analysis_text

        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                time.sleep(wait_time)
            else:
                st.error(f"Lỗi API sau {max_retries} lần thử: {e}")
                return None
    return None

# 3. Hàm tính toán chỉ số tài chính
def calculate_metrics(data):
    """
    Xây dựng dòng tiền và tính toán các chỉ số NPV, IRR, PP, DPP.
    """
    
    # 1. Chuẩn bị dữ liệu
    T = data['project_lifespan_years']
    I = data['investment_capital']
    R_array = data['annual_revenue']
    C_array = data['annual_cost']
    k = data['wacc_percent'] / 100.0
    tax_rate = data['tax_rate_percent'] / 100.0

    # Đảm bảo R và C có đủ T phần tử
    if len(R_array) < T:
        R_array.extend([R_array[-1]] * (T - len(R_array)))
    if len(C_array) < T:
        C_array.extend([C_array[-1]] * (T - len(C_array)))

    # 2. Xây dựng dòng tiền (Cash Flow)
    cash_flows = [-I]  # Năm 0: Vốn đầu tư ban đầu
    for i in range(T):
        # Giả định CF = (Doanh thu - Chi phí) * (1 - Thuế suất). 
        # (Để đơn giản, bỏ qua chi phí không tiền mặt như Khấu hao, vì không có trong input của AI)
        net_income = R_array[i] - C_array[i]
        ncf = net_income * (1 - tax_rate)
        cash_flows.append(ncf)

    np_cash_flows = np.array(cash_flows)

    # 3. Tính toán NPV và IRR
    npv = np.npv(k, np_cash_flows)
    
    try:
        irr = np.irr(np_cash_flows)
    except:
        irr = np.nan

    # 4. Tính PP (Payback Period - Thời gian hoàn vốn) & DPP (Discounted Payback Period)
    
    # Tính PP
    cumulative_cf = np.cumsum(np_cash_flows)
    pp = None
    for i, cum_cf in enumerate(cumulative_cf):
        if i == 0: continue
        if cum_cf >= 0:
            # Hoàn vốn trong năm i. Tính phần thập phân
            prev_cum_cf = cumulative_cf[i-1]
            cf_i = np_cash_flows[i]
            # Công thức: i - 1 + |CF tích lũy trước đó| / CF năm i
            pp = (i - 1) + abs(prev_cum_cf) / cf_i
            break
    
    # Tính DPP
    discounted_cf = [np_cash_flows[0]] 
    for i in range(1, len(np_cash_flows)):
        # Công thức Dòng tiền chiết khấu: CF_t / (1+k)^t
        dc_flow = np_cash_flows[i] / ((1 + k) ** i)
        discounted_cf.append(dc_flow)

    np_discounted_cf = np.array(discounted_cf)
    cumulative_dcf = np.cumsum(np_discounted_cf)
    dpp = None

    for i, cum_dcf in enumerate(cumulative_dcf):
        if i == 0: continue
        if cum_dcf >= 0:
            prev_cum_dcf = cumulative_dcf[i-1]
            dcf_i = np_discounted_cf[i]
            # Công thức: i - 1 + |CF chiết khấu tích lũy trước đó| / CF chiết khấu năm i
            dpp = (i - 1) + abs(prev_cum_dcf) / dcf_i
            break
            
    # Xây dựng bảng dòng tiền chi tiết
    cash_flow_df = {
        'Năm': list(range(T + 1)),
        'Vốn đầu tư (I)': [I] + [0] * T,
        'Doanh thu (R)': [0] + R_array,
        'Chi phí (C)': [0] + C_array,
        'Lợi nhuận ròng (NCF)': np_cash_flows,
        'Dòng tiền chiết khấu (DCF)': np_discounted_cf,
        'CF Tích lũy': cumulative_cf,
        'DCF Tích lũy': cumulative_dcf
    }

    return {
        'npv': npv, 
        'irr': irr, 
        'pp': pp, 
        'dpp': dpp,
        'cash_flow_data': cash_flow_df,
        'T': T,
        'I': I,
        'WACC': k
    }

# 4. Hàm đọc nội dung file DOCX (Giả lập)
def read_uploaded_file(uploaded_file):
    """
    Đọc nội dung của file đã tải lên.
    LƯU Ý: Đây là một hàm đơn giản để đọc file buffer. 
    Để xử lý file .docx phức tạp, cần dùng thư viện python-docx hoặc docx2txt trong môi trường thực tế.
    """
    try:
        # Giả định file là text hoặc có thể đọc buffer đơn giản
        file_buffer = uploaded_file.getvalue()
        
        # Nếu là file Word (.docx), thư viện docx2txt sẽ được dùng trong môi trường thực tế
        if uploaded_file.name.endswith('.docx'):
            st.warning("Đang sử dụng trình đọc nội dung cơ bản cho file .docx. Để trích xuất chính xác hơn, môi trường thực tế cần cài đặt thư viện 'docx2txt' hoặc 'python-docx'.")
            
            # Giả lập đọc nội dung file Word (chỉ hoạt động nếu file là văn bản thuần)
            # Trong môi trường thực, code sẽ là: text = docx2txt.process(io.BytesIO(file_buffer))
            return file_buffer.decode('utf-8', errors='ignore')

        # Đối với các loại file text khác (.txt, etc.)
        return file_buffer.decode('utf-8', errors='ignore')
        
    except Exception as e:
        st.error(f"Không thể đọc nội dung file. Vui lòng kiểm tra định dạng: {e}")
        return None

# --- GIAO DIỆN STREAMLIT CHÍNH ---

st.set_page_config(layout="wide", page_title="Đánh giá Phương án Kinh doanh bằng AI")

st.title("💰 Ứng dụng Đánh giá Phương án Kinh doanh bằng AI")
st.markdown("---")

# Khởi tạo state để lưu dữ liệu
if 'financial_data' not in st.session_state:
    st.session_state.financial_data = None
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None

# Nếu fetch không có trong session_state, hãy thêm nó vào (cần thiết cho API calls)
if 'fetch' not in st.session_state:
    st.session_state.fetch = st.runtime.scriptrunner.add_script_run_ctx(st.runtime.scriptrunner.RerunData(st.runtime.scriptrunner.RerunData.MAIN_SCRIPT_NAME)).get_callback_handle('fetch')


# 1. Tải file và Lọc thông tin

col1, col2 = st.columns([2, 1])

with col1:
    uploaded_file = st.file_uploader(
        "Tải lên File Kế hoạch Kinh doanh (.docx hoặc .txt)", 
        type=['docx', 'txt']
    )

with col2:
    st.text("") # Đảm bảo căn chỉnh nút bấm
    st.text("")
    extract_button = st.button("🚀 Lọc Dữ liệu Tài chính bằng AI", use_container_width=True, type="primary")

if extract_button and uploaded_file:
    with st.spinner("AI đang đọc và trích xuất dữ liệu từ Kế hoạch Kinh doanh..."):
        file_content = read_uploaded_file(uploaded_file)
        
        if file_content:
            extracted_data = call_gemini_extraction(file_content)
            
            if extracted_data:
                # Lưu dữ liệu thô đã trích xuất vào session state
                st.session_state.extracted_data_raw = extracted_data
                st.success("Trích xuất dữ liệu thành công!")

                # Tiến hành tính toán
                try:
                    metrics = calculate_metrics(extracted_data)
                    st.session_state.financial_data = metrics
                    st.session_state.analysis_result = None # Reset phân tích cũ
                except Exception as e:
                    st.error(f"Lỗi tính toán chỉ số tài chính: {e}. Vui lòng kiểm tra lại dữ liệu trích xuất.")
                    st.session_state.financial_data = None
            else:
                st.error("Không thể trích xuất dữ liệu. Vui lòng thử lại hoặc đảm bảo file Word có chứa đủ các thông tin cần thiết.")
elif extract_button and not uploaded_file:
    st.warning("Vui lòng tải lên một file Kế hoạch Kinh doanh.")

st.markdown("---")

# 2 & 3. Hiển thị bảng dòng tiền và các chỉ số đánh giá

if st.session_state.financial_data:
    metrics = st.session_state.financial_data
    
    st.header("📊 Kết quả Đánh giá Hiệu quả Dự án")

    # Hiển thị các chỉ số tài chính
    m_col1, m_col2, m_col3, m_col4, m_col5, m_col6 = st.columns(6)
    
    # NPV
    m_col1.metric("NPV (Giá trị hiện tại ròng)", 
                  f"{metrics['npv']:,.0f}", 
                  "Dương: Khả thi", 
                  delta_color="normal")
    
    # IRR
    m_col2.metric("IRR (Tỷ suất sinh lời nội bộ)", 
                  f"{metrics['irr'] * 100:,.2f}%" if not math.isnan(metrics['irr']) else "Không xác định",
                  f"WACC: {metrics['WACC'] * 100:,.2f}%")
    
    # PP
    m_col3.metric("PP (Thời gian hoàn vốn)", 
                  f"{metrics['pp']:,.2f} năm" if metrics['pp'] else "Không hoàn vốn")
    
    # DPP
    m_col4.metric("DPP (Thời gian hoàn vốn chiết khấu)", 
                  f"{metrics['dpp']:,.2f} năm" if metrics['dpp'] else "Không hoàn vốn")

    # WACC
    m_col5.metric("WACC", 
                  f"{metrics['WACC'] * 100:,.2f}%")
    
    # Dòng đời dự án
    m_col6.metric("Dòng đời dự án", 
                  f"{metrics['T']} năm")

    st.subheader("Bảng Dòng tiền (Cash Flow Statement)")
    
    # Tạo DataFrame để hiển thị
    cf_data = st.session_state.financial_data['cash_flow_data']
    import pandas as pd
    cf_df = pd.DataFrame(cf_data).set_index('Năm').style.format({
        'Vốn đầu tư (I)': "{:,.0f}",
        'Doanh thu (R)': "{:,.0f}",
        'Chi phí (C)': "{:,.0f}",
        'Lợi nhuận ròng (NCF)': "{:,.0f}",
        'Dòng tiền chiết khấu (DCF)': "{:,.0f}",
        'CF Tích lũy': "{:,.0f}",
        'DCF Tích lũy': "{:,.0f}"
    })
    
    st.dataframe(cf_df, use_container_width=True)

    st.markdown("---")
    
    # 4. Yêu cầu AI phân tích

    st.subheader("🧠 Phân tích Chuyên sâu từ AI")
    
    analysis_button = st.button("Phân tích Chỉ số Hiệu quả Dự án", use_container_width=False)
    
    if analysis_button:
        # Chuẩn bị báo cáo tóm tắt để gửi cho AI
        summary_report = (
            f"Báo cáo tài chính dự án:\n"
            f"- Vốn đầu tư (I): {metrics['I']:,.0f}\n"
            f"- Dòng đời dự án (T): {metrics['T']} năm\n"
            f"- Chi phí sử dụng vốn (WACC): {metrics['WACC'] * 100:,.2f}%\n"
            f"- NPV (Giá trị hiện tại ròng): {metrics['npv']:,.0f}\n"
            f"- IRR (Tỷ suất sinh lời nội bộ): {metrics['irr'] * 100:,.2f}%\n"
            f"- Thời gian hoàn vốn (PP): {metrics['pp']:,.2f} năm\n"
            f"- Thời gian hoàn vốn có chiết khấu (DPP): {metrics['dpp']:,.2f} năm"
        )
        
        with st.spinner("AI đang tổng hợp và phân tích báo cáo tài chính..."):
            analysis_text = call_gemini_analysis(summary_report)
            
            if analysis_text:
                st.session_state.analysis_result = analysis_text
            else:
                st.error("Không thể nhận được kết quả phân tích từ AI.")

    if st.session_state.analysis_result:
        st.markdown(st.session_state.analysis_result)

else:
    st.info("Vui lòng tải lên Kế hoạch Kinh doanh và nhấn nút Lọc Dữ liệu để bắt đầu đánh giá.")
