import asyncio

from shifa.voice import stt, tts
from shifa.voice.turns import looks_incomplete


def test_incomplete_sentences_wait_for_the_rest():
    assert looks_incomplete("مجھے ڈاکٹر عائشہ کے ساتھ اور")  # "...with Dr Ayesha and"
    assert looks_incomplete("I want to book an appointment with")
    assert looks_incomplete("kal subah ke liye")


def test_complete_sentences_answer_now():
    assert not looks_incomplete("مجھے کل صبح اپائنٹمنٹ چاہیے۔")
    assert not looks_incomplete("I want to book an appointment.")
    assert not looks_incomplete("جی ہاں")


def test_filler_sounds_are_ignored():
    assert stt.is_filler("ہم")
    assert stt.is_filler("Hmm.")
    assert not stt.is_filler("جی")  # can be a real "yes" to "should I book it?"


def test_known_hallucination_needs_high_confidence():
    seg = lambda text, nsp, lp: {"text": text, "no_speech_prob": nsp, "avg_logprob": lp}  # noqa: E731

    class Res:
        def __init__(self, segments):
            self.segments, self.text = segments, " ".join(s["text"] for s in segments)

    # Whisper's classic noise output, with middling confidence: dropped.
    assert stt._speech_text(Res([seg("ڈاکٹر صاحب", 0.35, -0.6)])) == ""
    # A real short answer with good confidence: kept.
    assert stt._speech_text(Res([seg("ڈاکٹر عائشہ", 0.05, -0.2)])) == "ڈاکٹر عائشہ"


def test_tts_stream_keeps_sentence_order(monkeypatch):
    async def fake_produce(text, voice, queue):
        await asyncio.sleep(0.02 if "first" in text else 0.0)  # the 2nd sentence finishes FIRST
        await queue.put(text.encode())
        await queue.put(None)

    monkeypatch.setattr(tts, "_produce", fake_produce)

    async def collect():
        return [c async for c in tts.stream("This is the first one. Then the second.", "en")]

    assert asyncio.run(collect()) == [b"This is the first one.", b"Then the second."]


def test_long_urdu_sentences_split_at_commas_for_fast_first_audio():
    long_ur = ("ڈاکٹر عثمان طارق کے ساتھ کل شام پانچ بج کر بیس منٹ، پانچ بج کر چالیس منٹ یا چھ بجے کا وقت "
               "دستیاب ہے، آپ کون سا وقت پسند کریں گے۔")
    parts = tts.pieces(long_ur, "ur")
    assert len(parts) >= 3 and all(len(p) <= URDU_LIMIT for p in parts)
    assert "".join(parts).replace(" ", "") == long_ur.replace(" ", "")  # nothing lost


def test_english_and_short_urdu_stay_whole():
    en = "We have Dr. Ayesha Malik, a general physician, and Dr. Usman Tariq, also a general physician."
    assert tts.pieces(en, "en") == [en]
    assert tts.pieces("جی ضرور، کل صبح۔", "ur") == ["جی ضرور، کل صبح۔"]


URDU_LIMIT = tts.URDU_PIECE_CHARS + tts.MIN_PIECE_CHARS


def test_long_unpunctuated_urdu_split_at_word_boundaries():
    # Verbatim shape from the voice test: ~100 chars, no comma.
    text = "ڈاکٹر عائشہ ملک کی فیس دو ہزار روپے ہے اور وہ پیر سے ہفتے تک صبح نو بجے سے دوپہر ایک بجے تک بیٹھتی ہیں۔"
    parts = tts.pieces(text, "ur")
    assert len(parts) >= 2 and all(len(p) <= URDU_LIMIT for p in parts)
    assert " ".join(parts).split() == text.split()  # same words, same order



def test_background_media_in_other_languages_is_dropped(monkeypatch):
    class Res:
        text, language, segments = "¡Ay, Dios, qué bonita era!", "spanish", []

    class Client:
        class audio:
            class transcriptions:
                @staticmethod
                def create(**kw):
                    return Res()

    monkeypatch.setattr(stt, "_client", lambda: Client)
    assert stt.transcribe(b"wav") == ("", "spanish")
