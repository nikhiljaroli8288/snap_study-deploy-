"""Test transcript extraction functionality"""
import sys
import re

# Test the video ID extraction
def extract_video_id(url):
    """Extract YouTube video ID from various URL formats."""
    patterns = [
        r'(?:v=)([0-9A-Za-z_-]{11})',
        r'(?:youtu\.be\/)([0-9A-Za-z_-]{11})',
        r'(?:embed\/)([0-9A-Za-z_-]{11})',
        r'(?:shorts\/)([0-9A-Za-z_-]{11})',
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

# Test various URL formats
test_urls = [
    'https://www.youtube.com/watch?v=dQw4w9WgXcQ',
    'https://youtu.be/dQw4w9WgXcQ',
    'https://www.youtube.com/embed/dQw4w9WgXcQ',
    'https://www.youtube.com/shorts/dQw4w9WgXcQ',
]

print("Testing video ID extraction:")
for url in test_urls:
    video_id = extract_video_id(url)
    print(f"  {url} -> {video_id}")
    assert video_id == 'dQw4w9WgXcQ', f"Failed to extract video ID from {url}"

print("\n[PASS] All URL extraction tests passed!")

# Test transcript extraction (requires youtube-transcript-api)
try:
    from youtube_transcript_api import YouTubeTranscriptApi

    # Test with a well-known video (Rick Astley - Never Gonna Give You Up)
    test_video_id = 'dQw4w9WgXcQ'
    print(f"\nTesting transcript extraction for video: {test_video_id}")

    try:
        transcript = YouTubeTranscriptApi.get_transcript(test_video_id)
        if transcript and len(transcript) > 0:
            print(f"[PASS] Successfully extracted transcript with {len(transcript)} entries")
            # Show first few words
            text = ' '.join([entry.get('text', '') for entry in transcript[:5]])
            print(f"  First words: {text[:100]}...")
        else:
            print("[FAIL] Transcript extraction returned empty result")
    except Exception as e:
        print(f"[FAIL] Transcript extraction failed: {type(e).__name__}: {e}")
        print("  This may be expected if the video doesn't have captions")

except ImportError:
    print("\n[WARN] youtube-transcript-api not installed, skipping transcript tests")
    print("  Install with: pip install youtube-transcript-api")

print("\n[PASS] Test script completed!")
