"""ai_chat.py – Ask AI chat backend powered by Google Gemini."""

from google import genai
from google.genai import types
import os, json, re, time

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable is not set")

client = genai.Client(api_key=GEMINI_API_KEY)
MODELS = ['gemini-2.5-flash']
_model_cooldown = {}


def chat_with_ai(user_message, conversation_history, summary_context=None):
    """
    Send a message to Gemini with conversation history and optional summary context.
    Returns the AI response as an HTML string.
    """
    system_prompt = """You are StudySnap AI, a friendly and expert study assistant.
You help students understand ANY topic they ask about — whether from their video study material or general knowledge.

RULES:
• Answer ANY question the student asks. You are NOT limited to only the study material.
• If the question relates to the study material, give context-aware answers.
• If the question is about general knowledge, answer it fully and accurately.
• Give clear, concise answers with examples and analogies.
• Format your answer as valid HTML for display.
• Use <strong> for emphasis, <ul>/<ol> for lists, <table class="study-table"> for comparisons.
• Use <div class="formula-box">...</div> for formulas.
• Use <div class="concept-card"><h5>Title</h5><p>text</p></div> for concept highlights.
• Use <mark>keyword</mark> for important terms.
• If asked for a quiz, provide questions with options.
• Use emoji (📌 💡 ✅ 🎯) to make responses scannable.
• NEVER say "this topic is not covered" or "information not covered" — always provide a helpful answer."""

    context_block = system_prompt
    if summary_context:
        title = summary_context.get('video_title', '')
        chapter = summary_context.get('chapter_title', '')
        summary_text = summary_context.get('main_summary', '')
        key_terms = ', '.join(summary_context.get('key_terms', []))
        context_block += f"""

The student is studying: "{title}" – Chapter: "{chapter}"
Summary: {summary_text[:2000]}
Key terms: {key_terms}

Use this context when relevant, but also answer questions beyond this material."""

    # Build conversation for Gemini
    full_prompt = context_block + "\n\n"
    for msg in conversation_history[-10:]:
        role = msg.get('role', 'user')
        content = msg.get('content', '')
        if role == 'assistant':
            full_prompt += f"Assistant: {content}\n\n"
        else:
            full_prompt += f"User: {content}\n\n"
    full_prompt += f"User: {user_message}\n\nAssistant:"

    text = ''
    last_err = None
    now = time.time()
    available = [m for m in MODELS if _model_cooldown.get(m, 0) <= now]
    if not available:
        soonest = min(_model_cooldown.values())
        wait = max(0, soonest - now) + 1
        print(f'[chat] All models on cooldown, waiting {wait:.0f}s...')
        time.sleep(wait)
        available = list(MODELS)
    for model in available:
        for attempt in range(2):
            try:
                resp = client.models.generate_content(
                    model=model,
                    contents=full_prompt,
                    config=types.GenerateContentConfig(temperature=0.7),
                )
                text = (resp.text or '').strip()
                if text:
                    break
            except Exception as e:
                last_err = e
                err_str = str(e)
                if '429' in err_str or 'RESOURCE_EXHAUSTED' in err_str:
                    _model_cooldown[model] = time.time() + 65
                    print(f'[chat] {model}: 429 quota exhausted, cooldown 65s, trying next...')
                    break
                if '503' in err_str or 'UNAVAILABLE' in err_str:
                    print(f'[chat] {model} attempt {attempt+1}: 503, retrying...')
                    time.sleep(3)
                    continue
                raise
        if text:
            break
    if not text and last_err:
        raise last_err
    # Clean up markdown if model wraps in code blocks
    if text.startswith('```html'):
        text = text[7:]
    if text.startswith('```'):
        text = text[3:]
    if text.endswith('```'):
        text = text[:-3]
    return text.strip()
