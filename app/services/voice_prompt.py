from __future__ import annotations


def build_voice_prompt(values: dict[str, str]) -> str:
    """Build a rich, descriptive voice prompt for VoxCPM2.

    VoxCPM2 responds best to natural-language voice descriptions
    that paint a complete picture — like a director describing a character.
    Short comma-separated keywords dilute the signal and produce flat audio.
    """
    parts: list[str] = []

    # Base voice character
    gender = values.get("gender", "").strip() or "female"
    age = values.get("age", "").strip() or "adult"
    tone = values.get("tone", "").strip()

    # Build rich character descriptors
    descriptors = []

    # Tone is the primary character descriptor
    if tone:
        descriptors.append(tone)
    else:
        descriptors.append("warm")

    # Add emotion descriptors
    emotion = values.get("emotion", "").strip()
    if emotion:
        # Map common emotions to richer descriptions
        emotion_map = {
            "happy": "cheerful and uplifting",
            "sad": "soft and melancholic",
            "angry": "intense and powerful",
            "calm": "serene and composed",
            "serious": "authoritative and confident",
            "playful": "fun and playful",
            "confident": "confident and assured",
        }
        descriptors.append(emotion_map.get(emotion.lower(), emotion))

    # Add style descriptors
    style = values.get("style", "").strip()
    if style:
        style_map = {
            "professional": "polished and professional",
            "natural": "natural and authentic",
            "conversational": "natural and conversational",
            "casual": "natural and approachable",
            "dramatic": "expressive and dynamic",
            "gentle": "soft and soothing",
            "energetic": "lively and energetic",
        }
        descriptors.append(style_map.get(style.lower(), style))

    # Main voice character sentence
    parts.append(f"A {gender} {age} voice that is {_join_natural(descriptors)}.")

    # Speaking pace
    speed = values.get("speed", "").strip()
    if speed:
        speed_map = {
            "slow": "speaks slowly and deliberately",
            "slower": "speaks slowly and deliberately",
            "medium": "speaks at a natural, conversational pace",
            "normal": "speaks at a natural, conversational pace",
            "fast": "speaks with energetic, lively pacing",
            "faster": "speaks with energetic, lively pacing",
        }
        speed_phrase = speed_map.get(speed.lower(), f"speaks at a {speed} pace")
        parts.append(f"The speaker {speed_phrase}.")

    # Clarity / pronunciation note
    clarity = values.get("clarity", "").strip()
    if clarity:
        parts.append(f"{clarity}.")
    else:
        parts.append("Clear native pronunciation with natural emotional expression.")

    # Use case context based on style/emotion
    emotion_lower = emotion.lower() if emotion else ""
    style_lower = style.lower() if style else ""

    if emotion_lower in ("happy", "playful", "confident") or style_lower in ("professional", "energetic"):
        parts.append(
            "The tone should be friendly, engaging, and trustworthy, "
            "suitable for commercial advertising or social media content."
        )
    elif emotion_lower in ("serious", "calm") or style_lower == "professional":
        parts.append(
            "The tone should be authoritative and trustworthy, "
            "suitable for informative content or announcements."
        )
    elif emotion_lower == "calm" or style_lower in ("gentle", "casual"):
        parts.append(
            "The tone should be pleasant and reassuring, "
            "suitable for lifestyle or relaxation content."
        )

    # Technical delivery note
    parts.append(
        "Smooth delivery with natural pauses, excellent articulation, "
        "and memorable commercial quality."
    )

    # Custom note at the end
    custom = values.get("custom", "").strip()
    if custom:
        parts.append(custom)

    return " ".join(parts)


def _join_natural(items: list[str], conjunction: str = "and") -> str:
    """Join a list into natural English: 'a, b, and c'."""
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])}, and {items[-1]}"
