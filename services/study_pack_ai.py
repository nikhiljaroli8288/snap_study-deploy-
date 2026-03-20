"""study_pack_ai.py – Generate comprehensive study pack PDF content using Google Gemini."""

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
        print(f'[study-pack] All models on cooldown, waiting {wait:.0f}s...')
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
                    print(f'[study-pack] {model} attempt {attempt+1}: empty response, retrying...')
                    time.sleep(2)
                    continue
                return _parse_json_text(text)
            except json.JSONDecodeError as e:
                last_err = e
                print(f'[study-pack] {model} attempt {attempt+1}: JSONDecodeError \u2013 {e}')
                time.sleep(2)
                continue
            except Exception as e:
                last_err = e
                err_str = str(e)
                if '429' in err_str or 'RESOURCE_EXHAUSTED' in err_str:
                    _model_cooldown[model] = time.time() + 65
                    print(f'[study-pack] {model}: 429 quota exhausted, cooldown 65s, trying next...')
                    break
                if '503' in err_str or 'UNAVAILABLE' in err_str:
                    print(f'[study-pack] {model} attempt {attempt+1}: 503/UNAVAILABLE, retrying...')
                    time.sleep(3)
                    continue
                raise
    raise last_err or RuntimeError('All Gemini models exhausted. Please try again later.')


def _strip_html(html):
    return re.sub(r'<[^>]+>', '', str(html))


def _build_context(summary_data):
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


def generate_study_pack(summary_data):
    """Generate a comprehensive study pack with notes, FAQ topics, 5 sample papers, and key points."""
    context = _build_context(summary_data)
    title = summary_data.get('video_title', 'Unknown Topic')

    prompt = f"""You are an expert academic study material creator. Generate a COMPREHENSIVE study pack based on this study material.

Study Material:
{context[:10000]}

Generate ALL of the following sections in ONE JSON response:

━━━ SECTION 1: DETAILED NOTES ━━━
6-10 topic sections with complete, detailed study notes covering every topic and subtopic. Include definitions, explanations, formulas, examples.

━━━ SECTION 2: FREQUENTLY ASKED TOPICS ━━━
8-10 most frequently asked exam topics/questions with easy-to-understand solutions, step-by-step examples, and tips.

━━━ SECTION 3: FIVE SAMPLE PAPERS ━━━
Exactly 5 different sample question papers. Each paper must have:
- 5 MCQs (1 mark each, 4 options A-D, include correct answer letter)
- 5 Fill in the Blanks (1 mark each, use "______" for blank)
- 5 Short Answer Questions (3 marks each)
- 5 Long Answer Questions (5 marks each)
Total per paper: 20 questions, 45 marks
Each paper should cover different aspects/combinations of topics. Do NOT repeat questions across papers.

━━━ SECTION 4: POINTS TO REMEMBER ━━━
15-20 crucial points, formulas, facts, and mnemonics that students must remember.

━━━ JSON OUTPUT FORMAT ━━━
{{
  "title": "{title}",
  "detailed_notes": [
    {{
      "heading": "Topic heading",
      "content": "Detailed explanation with all concepts, definitions, formulas, examples"
    }}
  ],
  "faq_topics": [
    {{
      "question": "Frequently asked question/topic",
      "solution": "Easy step-by-step solution with examples",
      "exam_tip": "How to answer this in exam"
    }}
  ],
  "sample_papers": [
    {{
      "paper_number": 1,
      "mcq": [
        {{
          "id": 1,
          "question": "Question text?",
          "options": {{"A": "opt1", "B": "opt2", "C": "opt3", "D": "opt4"}},
          "correct": "B"
        }}
      ],
      "fillups": [
        {{
          "id": 1,
          "question": "Sentence with ______ blank."
        }}
      ],
      "short": [
        {{
          "id": 1,
          "question": "Short answer question?"
        }}
      ],
      "long": [
        {{
          "id": 1,
          "question": "Long answer question?"
        }}
      ]
    }}
  ],
  "points_to_remember": [
    "Important point or formula to remember"
  ]
}}

RULES:
1. detailed_notes must cover ALL topics comprehensively with 6-10 sections.
2. Each of the 5 sample papers must have EXACTLY 5 MCQs, 5 fill-ups, 5 short, 5 long questions.
3. Papers should vary — don't repeat the same questions across papers.
4. faq_topics: give practical, exam-oriented solutions with worked examples.
5. points_to_remember: include formulas, key facts, definitions, mnemonics. 15-20 items.
6. sample_papers array must have exactly 5 objects with paper_number 1 through 5.
7. Return ONLY the JSON object."""

    return _call_gemini_json(prompt)
