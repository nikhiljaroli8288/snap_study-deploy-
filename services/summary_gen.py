"""summary_gen.py – Generates rich study material using Google Gemini."""

from google import genai
from google.genai import types
import json, re, os, time
from html import escape

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable is not set")

client = genai.Client(api_key=GEMINI_API_KEY)
MODELS = ['gemini-2.5-flash']
_model_cooldown = {}  # model -> timestamp when cooldown expires


def _sanitize_control_chars(text):
    """
    Walk the JSON text and escape any literal control characters that appear
    inside string values (raw newlines, tabs, etc. returned by the LLM).
    This is the most reliable fix for 'Invalid control character' JSON errors.
    """
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


def _has_devanagari(text):
    """Return True if text contains Devanagari (Hindi) characters."""
    return bool(re.search(r'[\u0900-\u097F]', text))


def _translate_to_english(html_text):
    """Ask Gemini to translate Hindi HTML content into English, preserving HTML tags."""
    prompt = (
        "Translate the following HTML content from Hindi to English. "
        "Keep ALL HTML tags exactly as they are — only translate the visible text. "
        "Do NOT add any markdown fences or extra text. Return ONLY the translated HTML.\n\n"
        + html_text
    )
    try:
        cfg = types.GenerateContentConfig(temperature=0.2)
        resp = client.models.generate_content(
            model=MODELS[0], contents=prompt, config=cfg
        )
        result = (resp.text or '').strip()
        result = re.sub(r'^```(?:html)?\s*\n?', '', result)
        result = re.sub(r'\n?\s*```$', '', result.strip()).strip()
        return result
    except Exception as e:
        print(f'[summary] translation fallback failed: {e}')
        return html_text


def _parse_json_text(text):
    """Parse JSON from text, handling markdown fences, control chars and truncation."""
    if text is None:
        raise json.JSONDecodeError('Empty response', '', 0)

    text = re.sub(r'^```(?:json)?\s*\n?', '', text)
    text = re.sub(r'\n?\s*```$', '', text.strip()).strip()
    if not text:
        raise json.JSONDecodeError('Empty response', '', 0)

    if not text.startswith(('{', '[')):
        m = re.search(r'([\{\[][\s\S]*[\}\]])', text)
        if m:
            text = m.group(1)
        else:
            first_obj = text.find('{')
            last_obj = text.rfind('}')
            first_arr = text.find('[')
            last_arr = text.rfind(']')
            if first_obj != -1 and last_obj > first_obj:
                text = text[first_obj:last_obj + 1]
            elif first_arr != -1 and last_arr > first_arr:
                text = text[first_arr:last_arr + 1]

    # Attempt 1: parse as-is
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Attempt 2: sanitize control characters inside strings, then parse
    try:
        return json.loads(_sanitize_control_chars(text))
    except json.JSONDecodeError:
        pass

    # Attempt 3: sanitize + truncation repair
    sanitized = _sanitize_control_chars(text)
    for end_char in ['}', ']']:
        idx = sanitized.rfind(end_char)
        if idx > 0:
            try:
                return json.loads(sanitized[:idx + 1])
            except json.JSONDecodeError:
                continue
    raise json.JSONDecodeError('Could not parse JSON after all recovery attempts', text, 0)


def _call_gemini(prompt):
    """Call Gemini with model fallback on 429 and retry on 503/empty/bad JSON."""
    last_err = None
    now = time.time()
    available = [m for m in MODELS if _model_cooldown.get(m, 0) <= now]
    if not available:
        # All on cooldown — wait for the soonest one
        soonest = min(_model_cooldown.values())
        wait = max(0, soonest - now) + 1
        print(f'[summary] All models on cooldown, waiting {wait:.0f}s...')
        time.sleep(wait)
        available = list(MODELS)
    for model in available:
        for attempt in range(2):
            try:
                cfg_kwargs = {'temperature': 0.3, 'response_mime_type': 'application/json'}
                if hasattr(types, 'ThinkingConfig'):
                    cfg_kwargs['thinking_config'] = types.ThinkingConfig(thinking_budget=1024)
                create_kwargs = {
                    'model': model,
                    'contents': prompt,
                    'config': types.GenerateContentConfig(**cfg_kwargs),
                }
                resp = client.models.generate_content(**create_kwargs)
                text = (resp.text or '').strip()
                # Strip any think tags if present
                text = re.sub(r'<think>[\s\S]*?</think>', '', text).strip()
                if not text:
                    print(f'[summary] {model} attempt {attempt+1}: empty response, retrying...')
                    time.sleep(2)
                    continue
                return _parse_json_text(text)
            except json.JSONDecodeError as e:
                last_err = e
                preview = text[:300] if text else '(empty)'
                print(f'[summary] {model} attempt {attempt+1}: JSONDecodeError \u2013 {e}')
                print(f'[summary]   raw response preview: {preview!r}')
                time.sleep(2)
                continue
            except Exception as e:
                last_err = e
                err_str = str(e)
                if '429' in err_str or 'RESOURCE_EXHAUSTED' in err_str:
                    _model_cooldown[model] = time.time() + 65
                    print(f'[summary] {model}: 429 quota exhausted, cooldown 65s, trying next...')
                    break
                if '503' in err_str or 'UNAVAILABLE' in err_str:
                    print(f'[summary] {model} attempt {attempt+1}: 503/UNAVAILABLE, retrying...')
                    time.sleep(3)
                    continue
                raise
    raise last_err or RuntimeError('All Gemini models exhausted. Please try again later.')


def _extract_sentences(transcript, max_items=8):
    """Pick concise, non-trivial sentences from transcript for fallback notes."""
    chunks = re.split(r'(?<=[.!?])\s+|\n+', transcript or '')
    cleaned = []
    seen = set()
    for raw in chunks:
        s = re.sub(r'\s+', ' ', raw).strip()
        if len(s) < 35:
            continue
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(s)
        if len(cleaned) >= max_items:
            break
    return cleaned


def _fallback_html_list(items):
    if not items:
        return '<ul><li>No key points could be extracted automatically.</li></ul>'
    return '<ul>' + ''.join(f'<li>{escape(item)}</li>' for item in items) + '</ul>'


def _build_fallback_study_material(transcript, exam_mode, language):
    """Return a safe, template-compatible summary payload when AI JSON fails."""
    points = _extract_sentences(transcript, max_items=10)
    short_points = points[:4]
    medium_points = points[:7]
    detailed_points = points[:10]
    short_summary = ' '.join(short_points[:2]) if short_points else 'Auto summary could not be generated from this transcript.'
    medium_summary = ' '.join(medium_points[:4]) if medium_points else short_summary
    detailed_summary = ' '.join(detailed_points) if detailed_points else medium_summary

    terms = []
    for sentence in points:
        for token in re.findall(r"[A-Za-z][A-Za-z0-9\-/+]{3,}", sentence):
            token_l = token.lower()
            if token_l not in terms:
                terms.append(token_l)
            if len(terms) >= 10:
                break
        if len(terms) >= 10:
            break

    summary_html_short = f'<p>{escape(short_summary)}</p>'
    summary_html_medium = f'<p>{escape(medium_summary)}</p>'
    summary_html_detailed = f'<p>{escape(detailed_summary)}</p>'

    important_short = "<h5>Key Concepts</h5>" + _fallback_html_list(short_points)
    important_medium = "<h5>Topic Overview</h5>" + _fallback_html_list(medium_points)
    important_detailed = "<h5>Detailed Notes</h5>" + _fallback_html_list(detailed_points)

    priority_sections = []
    for idx, point in enumerate(short_points or ['Review the source transcript to extract exam questions.'], start=1):
        priority_sections.append(
            "<div class='concept-card'>"
            f"<h5>Topic {idx}</h5>"
            f"<p><strong>Definition:</strong> {escape(point)}</p>"
            f"<p><strong>Why Important for {escape(exam_mode)}:</strong> Revise this concept for direct and short-answer questions.</p>"
            "<ul>"
            "<li><strong>Q:</strong> What is the main idea? <strong>A:</strong> Refer to this topic statement.</li>"
            "<li><strong>Q:</strong> Why is it tested? <strong>A:</strong> It captures a core concept from the source.</li>"
            "</ul>"
            "</div>"
        )

    return {
        'video_title': 'Auto-generated notes (fallback)',
        'chapter_title': 'Transcript Analysis',
        'main_summary': {
            'short': summary_html_short,
            'medium': summary_html_medium,
            'detailed': summary_html_detailed,
        },
        'important_points': {
            'short': important_short,
            'medium': important_medium,
            'detailed': important_detailed,
        },
        'priority_topics': {
            'short': ''.join(priority_sections[:2]),
            'medium': ''.join(priority_sections[:3]),
            'detailed': ''.join(priority_sections),
        },
        'hindi_notes': {
            'short': '<h5>मुख्य बिंदु</h5><ul><li>ऑटो-जनरेट नोट्स उपलब्ध नहीं हो सके।</li></ul>',
            'medium': '<h5>संक्षिप्त नोट्स</h5><ul><li>कृपया थोड़ी देर बाद पुनः प्रयास करें।</li></ul>',
            'detailed': '<h5>विस्तृत नोट्स</h5><ul><li>सिस्टम ने ट्रांसक्रिप्ट सेव कर लिया है, आप पुनः जनरेट कर सकते हैं।</li></ul>',
        },
        'core_observations': [
            {'title': 'Fallback Mode', 'description': 'The AI returned invalid JSON, so a safe summary was generated from transcript text.'}
        ],
        'highlights': [
            {'type': 'insight', 'title': 'Recovery Applied', 'description': 'StudySnap generated a reliable fallback to keep the summary page available.'},
            {'type': 'exam', 'title': 'Exam Relevance', 'description': f'Use the important points to prepare for {exam_mode} questions.'}
        ],
        'key_terms': terms,
        'conclusion': f'<p>Fallback summary generated in {escape(language)}. Regenerate later for richer AI output.</p>',
    }


def generate_study_material(transcript, exam_mode, language, difficulty, study_depth):
    """Generate comprehensive study material from a transcript using Gemini."""

    # Limit transcript to keep generation fast
    transcript_text = transcript[:20000]

    # Adjust note depth based on study_depth selection
    if study_depth == '20':
        depth_instruction = "Depth=QUICK: keep output concise. Short ~100w, Medium ~200w, Detailed ~400w. Priority: top 6-8 topics."
    elif study_depth == 'full':
        depth_instruction = "Depth=FULL: be comprehensive. Short ~150w, Medium ~350w, Detailed ~800w. Priority: cover ALL topics."
    else:  # 40 min smart
        depth_instruction = "Depth=SMART: balanced depth. Short ~120w, Medium ~300w, Detailed ~600w. Priority: all major topics."

    prompt = f"""You are an expert study note generator. Analyze the transcript and produce structured study material.

IMPORTANT LANGUAGE RULE: The transcript may be in Hindi/Hinglish but ALL output fields MUST be written in {language} ONLY. The ONLY exception is the "hindi_notes" field which must be in Devanagari Hindi. Every other field — main_summary, important_points, priority_topics, core_observations, highlights, conclusion, key_terms — must contain ZERO Hindi/Devanagari characters. Translate everything to {language}.

Config: Exam={exam_mode}, Language={language}, Difficulty={difficulty}, {depth_instruction}

Transcript:
{transcript_text}

Return ONLY a raw JSON object (no markdown fences, no extra text). Use exactly this structure:
{{
  "video_title": "Precise descriptive title of the video topic",
  "chapter_title": "Chapter or unit name",
  "main_summary": {{
    "short": "<p>2-3 sentences covering all main topics. Use <mark>keywords</mark>.</p>",
    "medium": "<p>3-4 paragraphs, one per major topic. Use <mark>keywords</mark>. ~250 words.</p>",
    "detailed": "<p>5-6 comprehensive paragraphs covering every concept. Use <mark>keywords</mark>. ~600 words.</p>"
  }},
  "important_points": {{
    "short": "[MUST BE WRITTEN IN {language} ONLY — NO HINDI] HTML (80-120 words): <h5>📋 Key Concepts</h5><ul><li>one-line bullet per concept — cover ALL topics</li></ul>. Add <h5>📌 Important Definitions</h5><ul>...</ul> if any. Add <h5>🔢 Formulas / Facts</h5><ul>...</ul> if any.",
    "medium": "[MUST BE WRITTEN IN {language} ONLY — NO HINDI] HTML (200-350 words): <h5>📖 Topic Overview</h5><p>short intro</p> <h5>🔑 Key Concepts</h5><ul>bullets with 1-2 sentence explanations for ALL topics</ul> <h5>📚 Important Definitions</h5><ul>term: meaning</ul> <h5>⚙️ Key Processes</h5><ul>step-by-step if applicable</ul> <h5>📐 Formulas / Facts</h5><ul>if any</ul> <h5>✅ Quick Revision</h5><ul>5 must-remember bullet points</ul>",
    "detailed": "[MUST BE WRITTEN IN {language} ONLY — ABSOLUTELY NO HINDI WORDS] HTML (500-800 words) ENTIRELY IN {language}: <h5>📖 Topic Introduction</h5><p>overview and importance</p> <h5>🧠 Core Concepts</h5><ul>detailed explanations with examples</ul> <h5>📚 Definitions & Terminology</h5><ul>all key terms defined</ul> <h5>⚙️ Key Processes / Workflows</h5><ul>step-by-step</ul> <h5>💡 Examples & Applications</h5><ul>concrete examples</ul> <h5>📐 Formulas / Rules</h5><ul>all formulas</ul> <h5>⚖️ Advantages / Disadvantages</h5><ul>if applicable</ul> <h5>🌍 Real-world Applications</h5><ul>practical uses</ul> <h5>🎯 Top 10 Exam Points</h5><ul>10 must-know bullets for {exam_mode} — ALL IN {language}</ul>"
  }},
  "priority_topics": "FLAT HTML STRING (NOT a dict). For EVERY major topic create: <div class='concept-card'><h5>🎯 [Topic Name]</h5><p><strong>Definition:</strong> clear explanation. <strong>Why Important for {exam_mode}:</strong> significance.</p><ul><li><strong>Q:</strong> common exam question? <strong>A:</strong> concise answer.</li><li><strong>Q:</strong> another question? <strong>A:</strong> answer.</li><li><strong>Q:</strong> third question? <strong>A:</strong> answer.</li></ul></div>. Cover ALL major topics with 2-3 FAQs each.",
  "hindi_notes": {{
    "short": "HTML: <h5>📌 मुख्य बिंदु</h5><ul>8-10 <li> in Devanagari Hindi (one topic per bullet)</ul>",
    "medium": "HTML: <h5>📝 विस्तृत नोट्स</h5><ul>15-20 <li> in Devanagari Hindi with brief explanation per topic</ul>",
    "detailed": "HTML: Full notes in Devanagari Hindi. Use <h5> section headers for each major topic with definitions and examples."
  }},
  "core_observations": [
    {{"title": "Observation 1", "description": "Specific factual insight from the transcript"}},
    {{"title": "Observation 2", "description": "Another specific insight"}},
    {{"title": "Observation 3", "description": "Another insight"}},
    {{"title": "Observation 4", "description": "Another insight"}},
    {{"title": "Observation 5", "description": "Another insight"}}
  ],
  "highlights": [
    {{"type": "insight", "title": "Key Insight", "description": "Most important conceptual insight from the material"}},
    {{"type": "mastery", "title": "Mastery Checkpoint", "description": "What the student masters after studying this"}},
    {{"type": "exam", "title": "Exam Relevance", "description": "How these topics appear in {exam_mode} exams and what to focus on"}}
  ],
  "key_terms": ["term1","term2","term3","term4","term5","term6","term7","term8","term9","term10"],
  "conclusion": "<p>Revision strategy: topics to prioritize, likely {exam_mode} exam questions, and 3 practical study tips.</p>"
}}

CRITICAL RULES:
1. Return ONLY raw JSON — no markdown fences, no extra text.
2. hindi_notes MUST use Devanagari script (हिंदी में लिखें).
3. priority_topics MUST be a single flat HTML string — NOT a dict with short/medium/detailed.
4. ALL content (main_summary, important_points, priority_topics, core_observations, highlights, conclusion, key_terms) MUST be written EXCLUSIVELY in {language}. Zero Hindi words allowed outside hindi_notes.
5. important_points, main_summary, hindi_notes values must be HTML strings (not arrays or dicts).
6. Cover ALL topics from the transcript.
7. LANGUAGE ENFORCEMENT: Every single word in main_summary, important_points (including the "🎯 Top 10 Exam Points" section), priority_topics, core_observations, highlights, and conclusion MUST be in {language} only. If the transcript is in Hindi, still write all output (except hindi_notes) in {language}.
8. IMPORTANT_POINTS LANGUAGE RULE (HIGHEST PRIORITY): The important_points.short, important_points.medium, and important_points.detailed fields MUST ALL be written 100% in {language}. Do NOT write any Devanagari (Hindi) characters in important_points under any circumstances. important_points is NOT hindi_notes — keep them completely separate.
"""

    try:
        result = _call_gemini(prompt)
    except Exception as e:
        print(f'[summary] Falling back to transcript-based summary due to AI error: {type(e).__name__}: {e}')
        result = _build_fallback_study_material(transcript_text, exam_mode, language)

    # Ensure result is a dict
    if not isinstance(result, dict):
        result = {}

    # Ensure top-level keys have safe defaults
    result.setdefault('video_title', 'Untitled')
    result.setdefault('chapter_title', '')
    result.setdefault('main_summary', {})
    result.setdefault('conclusion', '')
    result.setdefault('key_terms', [])
    if not isinstance(result.get('core_observations'), list):
        result['core_observations'] = []
    if not isinstance(result.get('highlights'), list):
        result['highlights'] = []
    if not isinstance(result.get('key_terms'), list):
        result['key_terms'] = []

    # Normalise nested structures
    for key in ['important_points', 'priority_topics', 'hindi_notes', 'main_summary']:
        if key not in result or not isinstance(result[key], dict):
            result[key] = {'short': '', 'medium': str(result.get(key, '')), 'detailed': ''}
        for level in ['short', 'medium', 'detailed']:
            val = result[key].get(level)
            if isinstance(val, list):
                result[key][level] = '<ul>' + ''.join(f'<li>{item}</li>' for item in val) + '</ul>'
            elif isinstance(val, dict):
                parts = []
                for v in val.values():
                    if isinstance(v, list):
                        parts.extend(v)
                    else:
                        parts.append(str(v))
                result[key][level] = '<ul>' + ''.join(f'<li>{item}</li>' for item in parts) + '</ul>'
            elif not val:
                result[key][level] = result[key].get('medium', '') or result[key].get('short', '')

    # Post-processing: translate important_points if Hindi leaked in
    for level in ['short', 'medium', 'detailed']:
        text = result.get('important_points', {}).get(level, '')
        if text and _has_devanagari(text):
            print(f'[summary] Hindi detected in important_points.{level}, translating to English...')
            result['important_points'][level] = _translate_to_english(text)

    return result
