"""
app.py — Streamlit UI для демонстрации дообученной модели деловых писем
Запуск: streamlit run app.py
"""

import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import time

# ─── Настройки страницы ───────────────────────────────────────────
st.set_page_config(
    page_title="Генератор деловых писем",
    page_icon="✉️",
    layout="wide",
)

# ─── CSS ──────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500&display=swap');

/* ─── Emerald Editorial Theme ─────────────────────────── */

html,
body,
[data-testid="stAppViewContainer"],
[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stSidebar"],
section[data-testid="stSidebar"],
.main,
.block-container,
.stApp {
    background-color: #0F1A17 !important;
    color: #DCE8E2 !important;
}

/* Remove white blocks */
div[data-testid="stVerticalBlock"],
div[data-testid="column"] {
    background: transparent !important;
}

/* Typography */
html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    color: #DCE8E2 !important;
}

h1, h2, h3 {
    font-family: 'Playfair Display', serif !important;
    color: #8FE3B0 !important;
    letter-spacing: -0.02em;
}

/* ─── Letter Output ───────────────────────────────────── */
.letter-output {
    background: #152320;
    border: 1px solid #27443C;
    border-left: 4px solid #56C596;
    border-radius: 4px;
    padding: 2rem 2.5rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    line-height: 1.8;
    color: #EAF5EF;
    white-space: pre-wrap;
    min-height: 200px;
    box-shadow: 0 6px 20px rgba(0,0,0,0.25);
}

/* ─── Sample Cards ────────────────────────────────────── */
.sample-card {
    background: #162723;
    border: 1px solid #27443C;
    border-radius: 4px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
    cursor: pointer;
    transition: all 0.2s ease;
    font-size: 0.82rem;
    color: #D7E7DF;
    line-height: 1.6;
}

.sample-card:hover {
    border-color: #56C596;
    background: #1B302B;
    transform: translateY(-2px);
    box-shadow: 0 4px 14px rgba(0,0,0,0.2);
}

/* ─── Tags ────────────────────────────────────────────── */
.sample-tag {
    display: inline-block;
    background: #56C596;
    color: #0F1A17;
    font-size: 0.68rem;
    font-weight: 600;
    padding: 3px 9px;
    border-radius: 999px;
    margin-bottom: 8px;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    font-family: 'IBM Plex Mono', monospace;
}

/* ─── Metrics ─────────────────────────────────────────── */
.metric-box {
    background: #152320;
    border: 1px solid #27443C;
    color: #DCE8E2;
    padding: 1rem 1.25rem;
    border-radius: 4px;
    text-align: center;
    font-family: 'IBM Plex Mono', monospace;
}

.metric-val {
    font-size: 1.7rem;
    font-weight: 600;
    color: #7BE0A8;
    display: block;
}

.metric-label {
    font-size: 0.72rem;
    opacity: 0.75;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* ─── Header Stamp ────────────────────────────────────── */
.header-stamp {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: #7BE0A8;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    border: 1px solid #3E6E60;
    display: inline-block;
    padding: 4px 12px;
    margin-bottom: 0.75rem;
    border-radius: 999px;
    background: rgba(86, 197, 150, 0.08);
}

/* ─── Text Areas ──────────────────────────────────────── */
.stTextArea textarea {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.85rem !important;
    background: #162723 !important;
    border: 1px solid #2F5248 !important;
    border-radius: 4px !important;
    color: #EAF5EF !important;
}

/* ─── Selectbox ───────────────────────────────────────── */
.stSelectbox > div > div {
    background: #162723 !important;
    border: 1px solid #2F5248 !important;
    border-radius: 4px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.85rem !important;
    color: #EAF5EF !important;
}

/* Dropdown */
div[data-baseweb="select"] > div {
    background-color: #162723 !important;
    color: #EAF5EF !important;
}

/* ─── Buttons ─────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #56C596, #3FAF7F) !important;
    color: #08110E !important;
    border: none !important;
    border-radius: 4px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em !important;
    padding: 0.7rem 2rem !important;
    width: 100% !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 12px rgba(86,197,150,0.25);
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    filter: brightness(1.05);
    box-shadow: 0 6px 18px rgba(86,197,150,0.35);
}

/* ─── Text Colors ─────────────────────────────────────── */
label,
p,
span,
div {
    color: #DCE8E2 !important;
}

/* ─── Divider ─────────────────────────────────────────── */
hr {
    border-color: #27443C;
}

/* ─── Footer ──────────────────────────────────────────── */
.footer-note {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: #8FB2A4;
    text-align: center;
    margin-top: 2rem;
    padding-top: 1rem;
    border-top: 1px solid #27443C;
}
</style>
""", unsafe_allow_html=True)

# ─── Примеры писем ────────────────────────────────────────────────
SAMPLES = [
    {
        "tag": "Запрос информации",
        "instruction": "Составьте вежливый профессиональный ответ на входящее деловое письмо.",
        "text": "Уважаемые коллеги,\nПрошу предоставить финансовый отчёт за Q3 2024 в срок до 15.02.2025.\nС уважением,\nА.В. Петрова, ООО «ТехноСервис»",
    },
    {
        "tag": "Жалоба",
        "instruction": "Подготовьте настойчивый, но вежливый ответ на деловое письмо.",
        "text": "Добрый день,\nВыражаем обеспокоенность в связи с нарушением сроков поставки по договору №123-2024. Просим принять меры в срок до 20.02.2025.\nС уважением,\nИ.И. Иванов, АО «РосИнвест»",
    },
    {
        "tag": "Коммерческое предложение",
        "instruction": "Составьте ответное письмо в соответствии с деловым этикетом.",
        "text": "Уважаемые партнёры,\nПредлагаем рассмотреть сотрудничество в сфере IT-аутсорсинга. Готовы обсудить детали на встрече 25.02.2025.\nС уважением,\nМ.С. Сидорова, ЗАО «АльфаГрупп»",
    },
    {
        "tag": "Уведомление",
        "instruction": "Напишите краткий деловой ответ на следующее письмо.",
        "text": "Уважаемые коллеги,\nУведомляем об изменении тарифов на услуги, вступающих в силу с 01.03.2025.\nС уважением,\nВ.П. Козлов, ИП Козлов В.П.",
    },
    {
        "tag": "Согласование",
        "instruction": "Сформируйте официальный ответ на входящее письмо в деловом стиле.",
        "text": "Добрый день,\nНаправляем на согласование бюджет на 2025 год. Ждём подтверждения до 28.02.2025.\nС уважением,\nЕ.А. Смирнова, ПАО «МегаТрейд»",
    },
]

STOP_PATTERNS = [
    "### ", "==", "\nA:", "\nQ:",
    "ПАО «", "ООО «", "Агентство", "Партнёрство",
    "г. Санкт", "г. Москва", "Финансовый блок",
    "Нашими основными", "Тел.:", "Факс:", "…\n", "\n\n\n",
]

# ─── Загрузка модели ──────────────────────────────────────────────
@st.cache_resource
def load_model():
    adapter_path = "./lora_adapter"
    base = AutoModelForCausalLM.from_pretrained(
        "ai-forever/rugpt3large_based_on_gpt2",
        torch_dtype=torch.float32,
    )
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    return model, tokenizer

def clean_output(text: str) -> str:
    for stop in STOP_PATTERNS:
        if stop in text:
            text = text[:text.index(stop)]
    # Убираем незакрытые скобки в конце
    text = text.rstrip("( \n")
    return text.strip()

def generate(model, tokenizer, letter: str, instruction: str) -> tuple[str, float]:
    prompt = f"""### Инструкция:
{instruction}

### Входящее письмо:
{letter}

### Ответ:
"""
    inputs = tokenizer(prompt, return_tensors="pt")
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=220,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.3,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.perf_counter() - t0
    full = tokenizer.decode(out[0], skip_special_tokens=True)
    answer = clean_output(full[len(prompt):])
    return answer, elapsed

# ─── UI ───────────────────────────────────────────────────────────

# Заголовок
st.markdown('<div class="header-stamp">Вариант 12 — NLP · LoRA Fine-tuning</div>', unsafe_allow_html=True)
st.markdown("# Генератор деловых писем")
st.markdown("*Дообученная модель RuGPT-3 Large · LoRA адаптер · Русский деловой стиль*")
st.markdown("---")

# Метрики
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown('<div class="metric-box"><span class="metric-val">4.61</span><span class="metric-label">Экспертная оценка / 5</span></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="metric-box"><span class="metric-val">5 000</span><span class="metric-label">Обучающих пар</span></div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="metric-box"><span class="metric-val">LoRA r=8</span><span class="metric-label">Метод дообучения</span></div>', unsafe_allow_html=True)
with col4:
    st.markdown('<div class="metric-box"><span class="metric-val">0.15%</span><span class="metric-label">Обучаемых параметров</span></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Основные колонки
left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown("### ✉️ Входящее письмо")

    # Примеры
    st.markdown("**Примеры для демонстрации:**")
    for i, s in enumerate(SAMPLES):
        if st.button(f"[{s['tag']}] {s['text'][:55]}...", key=f"sample_{i}"):
            st.session_state["input_text"] = s["text"]
            st.session_state["instruction"] = s["instruction"]

    st.markdown("<br>", unsafe_allow_html=True)

    # Инструкция
    instruction = st.selectbox(
        "Стиль ответа:",
        [
            "Составьте вежливый профессиональный ответ на входящее деловое письмо.",
            "Напишите краткий деловой ответ на следующее письмо.",
            "Подготовьте настойчивый, но вежливый ответ на деловое письмо.",
            "Сформируйте официальный ответ на входящее письмо в деловом стиле.",
            "Составьте ответное письмо в соответствии с деловым этикетом.",
        ],
        index=0,
        key="instruction",
    )

    # Текстовое поле
    input_text = st.text_area(
        "Или напишите своё письмо:",
        value=st.session_state.get("input_text", ""),
        height=200,
        placeholder="Уважаемые коллеги,\nПрошу предоставить...",
        key="input_text",
    )

    generate_btn = st.button("⟶ Сгенерировать ответ", type="primary")

with right:
    st.markdown("### 📄 Сгенерированный ответ")

    if generate_btn:
        if not input_text.strip():
            st.warning("Введите текст входящего письма.")
        else:
            with st.spinner("Генерация..."):
                try:
                    model, tokenizer = load_model()
                    result, elapsed = generate(model, tokenizer, input_text, instruction)
                    st.session_state["result"] = result
                    st.session_state["elapsed"] = elapsed
                except Exception as e:
                    st.error(f"Ошибка: {e}")
                    st.info("Убедитесь что папка ./lora_adapter существует и модель обучена.")

    if "result" in st.session_state and st.session_state["result"]:
        st.markdown(
            f'<div class="letter-output">{st.session_state["result"]}</div>',
            unsafe_allow_html=True,
        )
        elapsed = st.session_state.get("elapsed", 0)
        tokens_approx = len(st.session_state["result"].split())
        st.caption(f"⏱ {elapsed:.1f} сек · ~{tokens_approx} слов · RuGPT-3 Large + LoRA")

        st.download_button(
            label="⬇ Скачать ответ (.txt)",
            data=st.session_state["result"],
            file_name="business_reply.txt",
            mime="text/plain",
        )
    else:
        st.markdown(
            '<div class="letter-output" style="color:#B8A98A; font-style:italic;">Ответ появится здесь...</div>',
            unsafe_allow_html=True,
        )

# Подвал
st.markdown("""
<div class="footer-note">
    Вариант 12 · Дообучение LLM · RuGPT-3 Large + LoRA (PEFT) · 5 000 деловых писем · M1 Pro
</div>
""", unsafe_allow_html=True)