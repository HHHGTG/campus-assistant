import streamlit as st
import os
import re
import requests
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from prompt_templates import RAG_PROMPT
from tools import get_current_week, calculate_gpa

load_dotenv()

# ------------------- 页面配置 -------------------
st.set_page_config(
    page_title="校园百事通",
    page_icon="🏫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------- 自定义 CSS（仅调整背景和布局，不强制文字颜色） -------------------
st.markdown("""
<style>
    /* 标题样式 - 使用 Streamlit 默认颜色变量 */
    .main-title {
        font-size: 2.8rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.2rem;
        color: var(--text-color); /* 自适应主题 */
    }
    .sub-title {
        font-size: 1.2rem;
        text-align: center;
        margin-bottom: 2rem;
        color: var(--text-color);
        opacity: 0.8;
    }
    /* 聊天消息 - 只设置背景，文字颜色自动适应 */
    .stChatMessage.user {
        background-color: var(--primary-color) !important;
        border-radius: 12px;
        padding: 12px;
        margin-bottom: 8px;
    }
    .stChatMessage.assistant {
        background-color: var(--secondary-background-color) !important;
        border-radius: 12px;
        padding: 12px;
        margin-bottom: 8px;
    }
    /* 侧边栏美化 */
    .sidebar-content {
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ------------------- 缓存资源 -------------------
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh",
        model_kwargs={"trust_remote_code": True}
    )

@st.cache_resource
def load_vector_db():
    embeddings = load_embeddings()
    return Chroma(persist_directory="./vector_db", embedding_function=embeddings)

embeddings = load_embeddings()
vector_db = load_vector_db()

APIPASSWORD = os.getenv("SPARK_APIPASSWORD")
if not APIPASSWORD:
    st.error("❌ 请在 .env 文件中设置 SPARK_APIPASSWORD")
    st.stop()

# ------------------- RAG 问答 -------------------
def rag_retrieve_answer(question):
    docs = vector_db.similarity_search(question, k=3)
    context = "\n\n".join([d.page_content for d in docs])
    prompt_text = RAG_PROMPT.format(context=context, question=question)

    url = "https://spark-api-open.xf-yun.com/x2/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {APIPASSWORD}"
    }
    payload = {
        "model": "spark-x",
        "messages": [{"role": "user", "content": prompt_text}],
        "temperature": 0.3
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"]
        else:
            return f"❌ API 错误：{resp.status_code} - {resp.text}"
    except Exception as e:
        return f"⚠️ 请求异常：{e}"

# ------------------- 智能体路由 -------------------
def agent_answer(question):
    if re.search(r'第.*周|校历|本周|几周', question):
        return get_current_week()
    if re.search(r'绩点|GPA|平均分', question):
        nums = re.findall(r'\d+', question)
        if nums:
            return calculate_gpa(','.join(nums))
        else:
            return "请提供您的各科分数，例如：85,90,78"
    return rag_retrieve_answer(question)

# ------------------- 侧边栏 -------------------
with st.sidebar:
    st.markdown("## 🏫 校园百事通")
    st.markdown("---")
    st.markdown("### ✨ 我能做什么？")
    st.markdown("""
    - 📚 **校园规则查询**（请假、奖学金、报修等）
    - 📅 **校历周数查询**
    - 🎓 **绩点计算器**
    """)
    st.markdown("---")
    st.markdown("### 💡 示例问题")
    examples = [
        "怎么请病假？",
        "奖学金需要什么条件？",
        "现在是第几周？",
        "绩点计算 85,90,78"
    ]
    for ex in examples:
        if st.button(ex, key=ex, use_container_width=True):
            st.session_state["input_example"] = ex
    st.markdown("---")
    st.markdown("### 🗑️ 管理对话")
    if st.button("清空聊天记录", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.markdown("---")
    st.markdown("### 📌 关于")
    st.caption("基于 RAG + 讯飞星火 | 实训项目")

# ------------------- 主界面 -------------------
st.markdown('<div class="main-title">🏫 校园生活百事通助手</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">智能问答 · 校历查询 · 绩点计算</div>', unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "input_example" in st.session_state and st.session_state["input_example"]:
    prompt = st.session_state["input_example"]
    st.session_state["input_example"] = ""
else:
    prompt = None

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt is None:
    prompt = st.chat_input("请输入你的校园问题...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🤔 思考中..."):
            answer = agent_answer(prompt)
        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
