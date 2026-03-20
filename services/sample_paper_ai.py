"""sample_paper_ai.py – Generate sample question paper using Google Gemini."""

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
        print(f'[sample-paper] All models on cooldown, waiting {wait:.0f}s...')
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
                    print(f'[sample-paper] {model} attempt {attempt+1}: empty response, retrying...')
                    time.sleep(2)
                    continue
                return _parse_json_text(text)
            except json.JSONDecodeError as e:
                last_err = e
                print(f'[sample-paper] {model} attempt {attempt+1}: JSONDecodeError \u2013 {e}')
                time.sleep(2)
                continue
            except Exception as e:
                last_err = e
                err_str = str(e)
                if '429' in err_str or 'RESOURCE_EXHAUSTED' in err_str:
                    _model_cooldown[model] = time.time() + 65
                    print(f'[sample-paper] {model}: 429 quota exhausted, cooldown 65s, trying next...')
                    break
                if '503' in err_str or 'UNAVAILABLE' in err_str:
                    print(f'[sample-paper] {model} attempt {attempt+1}: 503/UNAVAILABLE, retrying...')
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


def generate_sample_paper(summary_data, counts=None):
    """Generate a sample paper with configurable question counts."""
    if counts is None:
        counts = {}
    n_mcq     = max(1, min(20, int(counts.get('mcq',     5))))
    n_fillups = max(1, min(15, int(counts.get('fillups', 5))))
    n_short   = max(1, min(15, int(counts.get('short',   5))))
    n_long    = max(1, min(10, int(counts.get('long',    5))))
    total_marks = n_mcq * 1 + n_fillups * 1 + n_short * 3 + n_long * 5

    context = _build_context(summary_data)
    title = summary_data.get('video_title', 'Unknown Topic')

    prompt = f"""You are an expert exam paper creator. Create a sample question paper from this study material. Generate ONLY questions — NO answers.

Study Material:
{context[:8000]}

━━━ PAPER STRUCTURE ━━━
Section A: {n_mcq} MCQs (1 mark each = {n_mcq} marks)
Section B: {n_fillups} Fill in the Blanks (1 mark each = {n_fillups} marks)
Section C: {n_short} Short Answer Questions (3 marks each = {n_short * 3} marks)
Section D: {n_long} Long Answer Questions (5 marks each = {n_long * 5} marks)
Total: {total_marks} marks

━━━ RULES ━━━
1. Cover ALL topics from the material.
2. MCQs: 4 options (A-D), one correct answer letter.
3. Fill-ups: Use "______" for the blank.
4. Short/Long: Just the question text. No answers.
5. Make questions exam-realistic and progressively harder.

━━━ JSON OUTPUT ━━━
{{
  "title": "Sample Paper: {title}",
  "total_marks": {total_marks},
  "sections": {{
    "mcq": {{
      "title": "Section A: Multiple Choice Questions",
      "marks_each": 1,
      "questions": [
        {{
          "id": 1,
          "question": "Question text?",
          "options": {{"A": "opt1", "B": "opt2", "C": "opt3", "D": "opt4"}}
        }}
      ]
    }},
    "fillups": {{
      "title": "Section B: Fill in the Blanks",
      "marks_each": 1,
      "questions": [
        {{
          "id": 1,
          "question": "Sentence with ______ blank."
        }}
      ]
    }},
    "short": {{
      "title": "Section C: Short Answer Questions",
      "marks_each": 3,
      "questions": [
        {{
          "id": 1,
          "question": "Question text?"
        }}
      ]
    }},
    "long": {{
      "title": "Section D: Long Answer Questions",
      "marks_each": 5,
      "questions": [
        {{
          "id": 1,
          "question": "Question text?"
        }}
      ]
    }}
  }}
}}

Return ONLY the JSON object with all sections and questions."""

    result = _call_gemini_json(prompt)
    if 'sections' not in result:
        result = {'title': f'Sample Paper: {title}', 'total_marks': total_marks, 'sections': {}}
    return result
