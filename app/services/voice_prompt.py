from __future__ import annotations


def build_voice_prompt(values: dict[str, str]) -> str:
    """Build a concise, high-impact voice prompt for VoxCPM2.

    VoxCPM2 responds best to short natural-language style instructions.
    Long comma-separated attribute lists dilute the signal and consume
    KV cache tokens that would otherwise be available for content.
    Priority: speed > clarity > style > emotion > tone > gender/age > custom.
    """
    # Speed is the highest-impact attribute — put it first, phrased naturally
    speed = _format_speed(values.get("speed", "").strip())

    # Collect other attributes in priority order
    clarity = values.get("clarity", "").strip()
    style = values.get("style", "").strip()
    emotion = values.get("emotion", "").strip()
    tone = values.get("tone", "").strip()
    gender = values.get("gender", "").strip()
    age = values.get("age", "").strip()
    custom = values.get("custom", "").strip()

    # Build a natural-language phrase instead of comma-separated list
    parts = []
    if speed:
        parts.append(speed)
    if clarity:
        parts.append(clarity)
    if style:
        parts.append(style)
    if emotion:
        parts.append(emotion)
    if tone:
        parts.append(tone)
    if gender and age:
        parts.append(f"{gender} {age} voice")
    elif gender:
        parts.append(f"{gender} voice")
    elif age:
        parts.append(f"{age} voice")
    if custom:
        parts.append(custom)

    # Join with periods for clearer sentence boundaries (better than commas)
    return ". ".join(parts)


def _format_speed(speed: str) -> str:
    """Convert speed value to a clear natural-language phrase.

    VoxCPM2 has no direct speed parameter — it interprets style text.
    Short imperative phrases work better than descriptive labels.
    """
    if not speed:
        return ""
    speed_lower = speed.lower()
    # Already a phrase with speed/rate keyword — keep as-is
    if "speed" in speed_lower or "rate" in speed_lower:
        return speed
    # Map common values to natural imperative phrases
    mapping = {
        "slow": "speak slowly",
        "slower": "speak slowly",
        "fast": "speak quickly",
        "faster": "speak quickly",
        "medium": "speak at a normal pace",
        "normal": "speak at a normal pace",
    }
    return mapping.get(speed_lower, f"speak at {speed} speed")
