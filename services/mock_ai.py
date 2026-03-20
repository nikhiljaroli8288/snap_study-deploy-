"""mock_ai.py – Generate mock test MCQs using Google Gemini."""

from google import genai
from google.genai import types
import json, re, os, time

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable is not set")

client = genai.Client(api_key=GEMINI_API_KEY)
MODELS = ['gemini-2.5-flash']
_model_cooldown = {}


def _sanitize_control_chars(text):
    """Escape literal control characters inside JSON string values."""
    result = []
    in_string = False
    escape_next = False
    for ch in text:
        if escape_next:
            result.append(ch)
            escape_next = False
            continue
        if ch == '\\' and in_string:
            result.append(ch)
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            result.append(ch)
            continue
        if in_string:
            code = ord(ch)
            if ch == '\n':
                result.append('\\n')
            elif ch == '\r':
                result.append('\\r')
            elif ch == '\t':
                result.append('\\t')
            elif code < 0x20:
                result.append(f'\\u{code:04x}')
            else:
                result.append(ch)
        else:
            result.append(ch)
    return ''.join(result)


def _parse_json_text(text):
    """Parse JSON from text, handling markdown fences, control chars and truncation."""
    text = re.sub(r'^```(?:json)?\s*\n?', '', text)
    text = re.sub(r'\n?\s*```$', '', text.strip()).strip()
    if not text.startswith(('{', '[')):
        m = re.search(r'([\{\[]\s\S]*[\}\]])', text)
        if m:
            text = m.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    try:
        return json.loads(_sanitize_control_chars(text))
    except json.JSONDecodeError:
        pass
    sanitized = _sanitize_control_chars(text)
    for end_char in ['}', ']']:
        idx = sanitized.rfind(end_char)
        if idx > 0:
            try:
                return json.loads(sanitized[:idx + 1])
            except json.JSONDecodeError:
                continue
    raise json.JSONDecodeError('Could not parse JSON after all recovery attempts', text, 0)


def _call_gemini_json(prompt):
    """Call Gemini with model fallback on 429 and retry on 503/empty/bad JSON."""
    last_err = None
    now = time.time()
    available = [m for m in MODELS if _model_cooldown.get(m, 0) <= now]
    if not available:
        soonest = min(_model_cooldown.values())
        wait = max(0, soonest - now) + 1
        print(f'[mock] All models on cooldown, waiting {wait:.0f}s...')
        time.sleep(wait)
        available = list(MODELS)
    for model in available:
        for attempt in range(2):
            try:
                create_kwargs = {
                    'model': model,
                    'contents': prompt,
                    'config': types.GenerateContentConfig(
                        temperature=0.3,
                        response_mime_type='application/json',
                    ),
                }
                resp = client.models.generate_content(**create_kwargs)
                text = (resp.text or '').strip()
                # Strip any think tags if present
                text = re.sub(r'<think>[\s\S]*?</think>', '', text).strip()
                if not text:
                    print(f'[mock] {model} attempt {attempt+1}: empty response, retrying...')
                    time.sleep(2)
                    continue
                return _parse_json_text(text)
            except json.JSONDecodeError as e:
                last_err = e
                print(f'[mock] {model} attempt {attempt+1}: JSONDecodeError \u2013 {e}')
                time.sleep(2)
                continue
            except Exception as e:
                last_err = e
                err_str = str(e)
                if '429' in err_str or 'RESOURCE_EXHAUSTED' in err_str:
                    _model_cooldown[model] = time.time() + 65
                    print(f'[mock] {model}: 429 quota exhausted, cooldown 65s, trying next...')
                    break
                if '503' in err_str or 'UNAVAILABLE' in err_str:
                    print(f'[mock] {model} attempt {attempt+1}: 503/UNAVAILABLE, retrying...')
                    time.sleep(3)
                    continue
                raise
    raise last_err or RuntimeError('All Gemini models exhausted. Please try again later.')


def _strip_html(html):
    """Remove HTML tags from a string."""
    return re.sub(r'<[^>]+>', '', str(html))


def _build_context(summary_data):
    """Build a text context string from summary data."""
    title = summary_data.get('video_title', 'Unknown Topic')
    chapter = summary_data.get('chapter_title', '')
    main_summary = _strip_html(summary_data.get('main_summary', ''))
    key_terms = ', '.join(summary_data.get('key_terms', []))

    important = ''
    if isinstance(summary_data.get('important_points'), dict):
        important = _strip_html(
            summary_data['important_points'].get('detailed', '') or
            summary_data['important_points'].get('medium', '')
        )

    priority = ''
    if isinstance(summary_data.get('priority_topics'), dict):
        priority = _strip_html(
            summary_data['priority_topics'].get('detailed', '') or
            summary_data['priority_topics'].get('medium', '')
        )

    conclusion = _strip_html(summary_data.get('conclusion', ''))

    return f"""Topic: {title} – {chapter}
Summary: {main_summary[:3000]}
Key Terms: {key_terms}
Important Points: {important[:3000]}
Priority Topics: {priority[:2000]}
Conclusion: {conclusion[:1000]}"""


def generate_mock_test(summary_data):
    """Generate 25 MCQs from the summary content."""
    context = _build_context(summary_data)
    title = summary_data.get('video_title', 'Unknown Topic')

    prompt = f"""You are an expert exam question creator. You MUST generate EXACTLY 25 multiple choice questions (MCQs) based on this study material.

Study Material:
{context[:12000]}

━━━ ABSOLUTE RULES ━━━
1. You MUST generate EXACTLY 25 MCQs. Count them: 1,2,3,...,25. NOT 18, NOT 20, EXACTLY 25.
2. Each question MUST have exactly 4 options: A, B, C, D.
3. Exactly ONE option must be correct per question.
4. Cover ALL topics mentioned in the material — distribute questions evenly across all topics.
5. Mix difficulty: ~8 easy, ~10 medium, ~7 hard questions.
6. Include conceptual, factual, and application-based questions.
7. Questions must be clear, unambiguous, and exam-realistic.
8. Provide a 1-2 sentence explanation for each correct answer.
9. Options should be plausible — no obviously wrong distractors.
10. The "questions" array MUST contain exactly 25 objects with ids 1 through 25.

━━━ JSON OUTPUT ━━━
{{
  "title": "Mock Test: {title}",
  "total_questions": 25,
  "questions": [
    {{
      "id": 1,
      "question": "Clear question text?",
      "options": {{
        "A": "First option",
        "B": "Second option",
        "C": "Third option",
        "D": "Fourth option"
      }},
      "correct": "B",
      "explanation": "Brief explanation of why B is correct."
    }}
  ]
}}

IMPORTANT: The questions array MUST have EXACTLY 25 items (id 1 to 25). Double-check the count before responding. Return ONLY the JSON object."""

    result = _call_gemini_json(prompt)
    if 'questions' not in result:
        result = {'title': f'Mock Test: {title}', 'total_questions': 25, 'questions': []}
    return result
