import streamlit as st
import os
import re
import requests
import pandas as pd
import json
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

# ------------------- 自定义 CSS -------------------
st.markdown("""
<style>
    .main-title {
        font-size: 2.8rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.2rem;
        color: var(--text-color);
    }
    .sub-title {
        font-size: 1.2rem;
        text-align: center;
        margin-bottom: 2rem;
        color: var(--text-color);
        opacity: 0.8;
    }
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
</style>
""", unsafe_allow_html=True)

# ------------------- 缓存资源（路径修正） -------------------
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh",
        model_kwargs={"trust_remote_code": True}
    )

@st.cache_resource
def load_vector_db():
    embeddings = load_embeddings()
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_path = os.path.join(base_dir, "vector_db")
    csv_path = os.path.join(base_dir, "data", "campus_data.csv")

    if not os.path.exists(db_path) or not os.listdir(db_path):
        df = pd.read_csv(csv_path)
        texts = df['answer'].tolist()
        metadatas = df[['id', 'category', 'question', 'source']].to_dict('records')
        vector_db = Chroma.from_texts(
            texts=texts,
            embedding=embeddings,
            metadatas=metadatas,
            persist_directory=db_path
        )
        return vector_db
    else:
        return Chroma(persist_directory=db_path, embedding_function=embeddings)

embeddings = load_embeddings()
vector_db = load_vector_db()

APIPASSWORD = os.getenv("SPARK_APIPASSWORD")
if not APIPASSWORD:
    st.error("❌ 请在 .env 文件中设置 SPARK_APIPASSWORD")
    st.stop()

# ------------------- 获取对话历史 -------------------
def get_conversation_history(max_turns=5):
    if "messages" not in st.session_state:
        return ""
    recent = st.session_state.messages[-max_turns*2:]
    history_text = ""
    for msg in recent:
        role = "用户" if msg["role"] == "user" else "助手"
        history_text += f"{role}: {msg['content']}\n"
    return history_text

# ------------------- RAG 问答 -------------------
def rag_retrieve_answer(question):
    docs = vector_db.similarity_search(question, k=5)
    context = "\n\n".join([d.page_content for d in docs])
    history = get_conversation_history(max_turns=5)

    if not context.strip() and not history.strip():
        return "知识库和对话历史都没有相关信息，请提供更多内容。"

    if history:
        prompt_text = f"""你是校园生活助手。请结合对话历史和知识库回答用户问题。

【重要】如果问题涉及用户个人信息（如姓名、身份等），请优先从对话历史中查找。
如果知识库中有相关信息，请结合知识库回答。
如果两者都没有，请回答"我不清楚，建议咨询辅导员"。

【对话历史】
{history}

【知识库参考】
{context}

【当前用户问题】
{question}

【回答】"""
    else:
        prompt_text = RAG_PROMPT.format(context=context, question=question)

    url = "https://spark-api-open.xf-yun.com/x2/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {APIPASSWORD}"
    }
    payload = {
        "model": "spark-x",
        "messages": [{"role": "user", "content": prompt_text}],
        "temperature": 0.1
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
    - 💬 **多轮对话记忆**
    - 🔊 **自动语音播报**（沉稳风格）
    """)
    st.markdown("---")
    st.markdown("### 💡 示例问题")
    examples = [
        "怎么请病假？",
        "奖学金需要什么条件？",
        "宿舍灯坏了怎么报修？",
        "现在是第几周？",
        "绩点计算 85,90,78"
    ]
    for ex in examples:
        if st.button(ex, key=ex, use_container_width=True):
            st.session_state["example_question"] = ex
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
st.markdown('<div class="sub-title">智能问答 · 校历查询 · 绩点计算 · 多轮对话 · 语音播报</div>', unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("请输入你的校园问题...")

if "example_question" in st.session_state and st.session_state["example_question"]:
    prompt = st.session_state["example_question"]
    st.session_state["example_question"] = ""

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🤔 思考中..."):
            answer = agent_answer(prompt)
        st.markdown(answer)

        # ------------------- 语音播报（沉稳风格） -------------------
        if answer and len(answer.strip()) > 0:
            safe_answer = json.dumps(answer)
            st.components.v1.html(f"""
            <script>
            (function() {{
                var text = {safe_answer};
                if (!window.speechSynthesis) {{
                    console.warn('浏览器不支持语音合成');
                    return;
                }}
                window.speechSynthesis.cancel();

                var utterance = new SpeechSynthesisUtterance(text);
                utterance.lang = 'zh-CN';
                utterance.rate = 0.8;    // 语速稍慢，沉稳
                utterance.pitch = 0.7;   // 音调低沉
                utterance.volume = 1;

                // 优先选择较自然的语音
                var voices = window.speechSynthesis.getVoices();
                var preferredVoice = null;
                for (var i = 0; i < voices.length; i++) {{
                    var name = voices[i].name;
                    if (name.includes('Huihui') || name.includes('Microsoft') || name.includes('Google 普通话')) {{
                        preferredVoice = voices[i];
                        break;
                    }}
                }}
                if (preferredVoice) {{
                    utterance.voice = preferredVoice;
                }}

                window.speechSynthesis.speak(utterance);
                console.log('✅ 语音播报已触发（语速0.8，音调0.7）');
            }})();
            </script>
            """, height=0)

        st.session_state.messages.append({"role": "assistant", "content": answer})
