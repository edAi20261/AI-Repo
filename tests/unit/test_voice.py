"""Voice: a recorded question becomes text, and a reply can be heard."""

from __future__ import annotations

from contextlib import asynccontextmanager

import anyio
import httpx
import pytest

from agent.credits import CreditCatalog, Credits, InMemoryCreditStore
from agent.voice.service import SpeakingDisabled, cap, hear, script, speak
from agent.voice.spoken import has_notation, readable
from shared import speech
from shared.config import get_config
from shared.errors import UnprocessableError


class Reply:
    def __init__(self, status: int = 200, json: dict | None = None, content: bytes = b"",
                 text: str = "") -> None:
        self.status_code = status
        self._json = json or {}
        self.content = content
        self.text = text

    def json(self):
        return self._json


class Provider:
    """Stands in for httpx.post and remembers what was sent."""

    def __init__(self, reply: Reply) -> None:
        self.reply = reply
        self.calls: list[dict] = []

    def __call__(self, url, **kwargs):
        self.calls.append({"url": url, **kwargs})
        return self.reply


SCRIBE = {"language_code": "ara", "language_probability": 0.98,
          "text": "  ليه   الإشارة بتتقلب  ",
          "words": [{"text": "ليه", "start": 0.1, "end": 0.4, "type": "word"},
                    {"text": "بتتقلب", "start": 1.2, "end": 3.0, "type": "word"}]}


@pytest.fixture
def config(monkeypatch):
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("ELEVENLABS_API_KEY", "el-key")
    return get_config(reload=True)


@pytest.fixture
def speaking(config):
    config._data["voice"]["speak"]["enabled"] = True
    return config


def _provider(monkeypatch, reply: Reply) -> Provider:
    provider = Provider(reply)
    monkeypatch.setattr(httpx, "post", provider)
    return provider


# --- what a voice is given to say ---------------------------------------------

class Writer:
    """A model that rewrites notation; records the prompt it was given."""

    def __init__(self, reply: str = "", fail: bool = False) -> None:
        self.reply, self.fail, self.prompts = reply, fail, []

    def invoke(self, messages):
        self.prompts.append(messages[-1].content)
        if self.fail:
            raise RuntimeError("gateway down")
        return type("R", (), {"content": self.reply})()


def test_structure_is_removed_and_notation_is_left_for_the_model():
    """Markdown, citation markers, links and the sources block are syntax, the
    same in every language. Notation is language, so it is not touched here."""
    reply = ("### The roots\n\n**Roots** of $ax^2 + bx + c = 0$ come from the "
             "discriminant [1, 2].\n\n- See [the lesson](https://x.y/z).\n\n📚\n[1] Lesson 4\n\nTry it.")
    assert readable(reply) == ("The roots\nRoots of $ax^2 + bx + c = 0$ come from the "
                               "discriminant.\nSee the lesson.\nTry it.")


def test_a_table_is_read_row_by_row_and_pictographs_are_silent():
    reply = "| x | y |\n|---|---|\n| 1 | 2 |\nVoilà 📊 [📊](s3://bucket/c.png)"
    assert readable(reply) == "x, y\n1, 2\nVoilà"


def test_notation_is_found_by_what_it_is_not_by_a_word_list():
    assert has_notation("مساحة الدائرة $\\pi r^2$")
    assert has_notation("المساحة = الطول × العرض")
    assert has_notation("b) -12")
    assert not has_notation("Plants make food from sunlight, a well-known rule.")
    assert not has_notation("النباتات تصنع غذاءها من ضوء الشمس.")


def test_a_reply_without_notation_costs_no_model_call(config):
    writer = Writer("should not be used")
    assert script("Plants make food.", "en", config=config, llm=writer) == "Plants make food."
    assert writer.prompts == []


def test_notation_is_worded_by_the_model_in_the_replys_language(config):
    writer = Writer("مساحة الدائرة تساوي باي في نق تربيع.")
    spoken = script("مساحة الدائرة تساوي $\\pi r^2$.", "ar", config=config, llm=writer)
    assert spoken == "مساحة الدائرة تساوي باي في نق تربيع."
    assert "Arabic" in writer.prompts[0] and "$\\pi r^2$" in writer.prompts[0]


def test_a_failed_rewrite_reads_the_text_as_it_is(config, caplog):
    text = "If $x = 2$, then $x^2 = 4$."
    assert script(text, "en", config=config, llm=Writer(fail=True)) == text
    assert "speech script failed" in caplog.text


def test_a_rewrite_that_grew_into_something_else_is_not_spoken(config, caplog):
    """The rewrite may only word the notation. One that comes back as an essay
    has changed what the student hears from what they were shown."""
    text = "If $x = 2$, then $x^2 = 4$."
    assert script(text, "en", config=config, llm=Writer("x " * 200)) == text
    assert "speech script rejected" in caplog.text


def test_a_long_reply_is_cut_at_a_sentence():
    assert cap("One. Two three four. Five six seven eight nine.", 30) == "One. Two three four."


# --- hearing --------------------------------------------------------------------

def test_a_recording_comes_back_as_the_words_and_the_language(config, monkeypatch):
    provider = _provider(monkeypatch, Reply(json=SCRIBE))
    heard = hear(b"OggS...", "audio/webm;codecs=opus", config=config)
    assert heard.text == "ليه الإشارة بتتقلب"
    assert heard.language == "ar" and heard.seconds == 3.0
    sent = provider.calls[0]
    assert sent["headers"]["xi-api-key"] == "el-key"
    assert sent["data"]["model_id"] == "scribe_v2"
    assert "language_code" not in sent["data"]


def test_the_codec_parameter_never_reaches_the_provider(config, monkeypatch):
    """`audio/webm;codecs=opus` is what Chrome records; providers refuse the
    parameter with a 400 that reads like a broken recording."""
    provider = _provider(monkeypatch, Reply(json=SCRIBE))
    hear(b"OggS...", "audio/webm;codecs=opus", config=config)
    assert provider.calls[0]["files"]["file"][2] == "audio/webm"


def test_silence_is_heard_as_nothing_rather_than_an_error(config, monkeypatch):
    _provider(monkeypatch, Reply(json={"language_code": "eng", "text": "", "words": []}))
    heard = hear(b"...", "audio/webm", config=config)
    assert heard.heard is False and heard.text == "" and heard.language is None


def test_an_unreadable_recording_is_the_requests_fault(config, monkeypatch):
    _provider(monkeypatch, Reply(status=400, text="invalid audio"))
    with pytest.raises(speech.SpeechUnreadable) as refused:
        hear(b"not audio", "audio/webm", config=config)
    assert refused.value.status_code == 422


def test_a_rejected_key_is_the_services_fault(config, monkeypatch):
    _provider(monkeypatch, Reply(status=401, text="invalid api key"))
    with pytest.raises(speech.SpeechUnavailable) as refused:
        hear(b"...", "audio/webm", config=config)
    assert refused.value.status_code == 503


def test_voice_without_keys_says_what_is_missing(config, monkeypatch):
    monkeypatch.delenv("ELEVENLABS_API_KEY")
    with pytest.raises(speech.SpeechUnavailable) as refused:
        hear(b"...", "audio/webm", config=config)
    assert refused.value.details == {"missing": "ELEVENLABS_API_KEY"}


# --- speaking -------------------------------------------------------------------

def test_a_reply_is_spoken_in_its_own_language(speaking, monkeypatch):
    """A multilingual voice left to guess reads an Arabic reply with English
    vowels; the reply's language is passed so it never has to guess."""
    provider = _provider(monkeypatch, Reply(content=b"ID3mp3"))
    writer = Writer("الناتج x أصغر من خمسة و صحيح")
    spoken = speak("الناتج $x < 5$ و **صحيح** [1]", "ar", config=speaking, llm=writer)
    sent = provider.calls[0]
    assert spoken.audio == b"ID3mp3" and spoken.content_type == "audio/mpeg"
    assert sent["url"] == f"https://api.elevenlabs.io/v1/text-to-speech/{spoken.voice}"
    assert sent["headers"]["xi-api-key"] == "el-key"
    assert sent["json"] == {"text": "الناتج x أصغر من خمسة و صحيح",
                            "model_id": "eleven_flash_v2_5", "language_code": "ar"}
    assert "**" not in writer.prompts[0] and "[1]" not in writer.prompts[0]


def test_a_reply_with_nothing_to_say_is_refused(speaking, monkeypatch):
    _provider(monkeypatch, Reply(content=b"x"))
    with pytest.raises(UnprocessableError):
        speak("📚\n[1] Lesson 4", "en", config=speaking)


def test_speaking_is_off_unless_a_deployment_turns_it_on(config, monkeypatch):
    """Reading aloud is most of the voice bill, so it ships off. Off means no
    provider call can happen, whoever calls."""
    provider = _provider(monkeypatch, Reply(content=b"ID3mp3"))
    with pytest.raises(SpeakingDisabled) as refused:
        speak("Plants make food.", "en", config=config)
    assert refused.value.status_code == 503 and refused.value.code == "speaking_disabled"
    assert provider.calls == []


# --- the endpoints ----------------------------------------------------------------

def _client(config, credits=None):
    from fastapi.testclient import TestClient

    from system.billing import Billing
    from system.main import create_app
    from system.runtime import AgentRuntime
    from system.settings import Settings

    app = create_app(config)
    app.state.settings = Settings.from_config(config)

    @asynccontextmanager
    async def lifespan(scoped):
        scoped.state.billing = Billing(credits)
        scoped.state.runtime = AgentRuntime(config, None, None, billing=scoped.state.billing,
                                            max_concurrency=2, timeout=5)
        yield

    app.router.lifespan_context = lifespan
    return TestClient(app)


HEADERS = {"x-service-token": "token"}


def test_the_transcribe_endpoint_returns_the_words(config, monkeypatch):
    monkeypatch.setenv("AI_WEBHOOK_SHARED_SECRET", "token")
    _provider(monkeypatch, Reply(json=SCRIBE))
    with _client(get_config(reload=True)) as client:
        response = client.post("/ai/voice/transcribe", params={"billed_user_id": "s-1"},
                               files={"audio": ("q.webm", b"OggS", "audio/webm;codecs=opus")},
                               headers=HEADERS)
    assert response.status_code == 200, response.text
    assert response.json()["text"] == "ليه الإشارة بتتقلب"
    assert response.json()["heard"] is True and response.json()["language"] == "ar"


def test_a_file_that_is_not_audio_is_refused_before_anything_is_paid(config, monkeypatch):
    monkeypatch.setenv("AI_WEBHOOK_SHARED_SECRET", "token")
    provider = _provider(monkeypatch, Reply(json=SCRIBE))
    with _client(get_config(reload=True)) as client:
        response = client.post("/ai/voice/transcribe", params={"billed_user_id": "s-1"},
                               files={"audio": ("q.png", b"\x89PNG", "image/png")},
                               headers=HEADERS)
    assert response.status_code == 422 and provider.calls == []


def test_a_recording_relabelled_by_a_proxy_is_still_accepted(config, monkeypatch):
    """Found by the feature test: a backend that forwards the upload sends it as
    application/octet-stream, and every recording was refused as "not audio"."""
    monkeypatch.setenv("AI_WEBHOOK_SHARED_SECRET", "token")
    provider = _provider(monkeypatch, Reply(json=SCRIBE))
    with _client(get_config(reload=True)) as client:
        response = client.post("/ai/voice/transcribe", params={"billed_user_id": "s-1"},
                               files={"audio": ("blob", b"\x1aE\xdf\xa3webm...",
                                                "application/octet-stream")},
                               headers=HEADERS)
    assert response.status_code == 200, response.text
    assert provider.calls[0]["files"]["file"][2] == "audio/webm"


def test_an_unknown_file_relabelled_as_generic_is_still_refused(config, monkeypatch):
    monkeypatch.setenv("AI_WEBHOOK_SHARED_SECRET", "token")
    provider = _provider(monkeypatch, Reply(json=SCRIBE))
    with _client(get_config(reload=True)) as client:
        response = client.post("/ai/voice/transcribe", params={"billed_user_id": "s-1"},
                               files={"audio": ("blob", b"%PDF-1.4", "application/octet-stream")},
                               headers=HEADERS)
    assert response.status_code == 422 and provider.calls == []


def test_the_speak_endpoint_returns_audio(speaking, monkeypatch):
    monkeypatch.setenv("AI_WEBHOOK_SHARED_SECRET", "token")
    _provider(monkeypatch, Reply(content=b"ID3mp3"))
    with _client(speaking) as client:
        response = client.post("/ai/voice/speak", headers=HEADERS, json={
            "text": "Plants make food [1].", "language": "en", "billed_user_id": "s-1"})
    assert response.status_code == 200, response.text
    assert response.content == b"ID3mp3"
    assert response.headers["content-type"] == "audio/mpeg"
    assert response.headers["x-voice"] == speaking.get("voice.speak.voices.en")


def test_the_frontend_can_ask_which_buttons_to_show(config, monkeypatch):
    monkeypatch.setenv("AI_WEBHOOK_SHARED_SECRET", "token")
    with _client(config) as client:
        response = client.get("/ai/voice/capabilities", headers=HEADERS)
    assert response.status_code == 200
    assert {k: response.json()[k] for k in ("transcribe", "speak")} == {
        "transcribe": True, "speak": False}


def test_a_refused_reply_is_never_charged(config, monkeypatch):
    """With speaking off, a request must be refused before a credit is taken,
    not charged and refunded: the ledger would fill with entries for nothing."""
    from system.billing import Billing
    from system.runtime import AgentRuntime

    credits = Credits(InMemoryCreditStore(), CreditCatalog.from_config(config))
    runtime = AgentRuntime(config, None, None, billing=Billing(credits),
                           max_concurrency=1, timeout=5)

    async def run():
        await runtime.speak("Plants make food.", "en", billed_user_id="s-1")

    with pytest.raises(SpeakingDisabled):
        anyio.run(run)
    entries, _ = credits.history("s-1", page=1, page_size=10)
    assert [e.kind.value for e in entries if e.action] == []


def test_a_recording_with_no_speech_costs_nothing(config, monkeypatch):
    """A student who pressed the mic and said nothing must not pay for silence."""
    from system.billing import Billing
    from system.runtime import AgentRuntime

    _provider(monkeypatch, Reply(json={"language_code": "eng", "text": "", "words": []}))
    credits = Credits(InMemoryCreditStore(), CreditCatalog.from_config(config))
    runtime = AgentRuntime(config, None, None, billing=Billing(credits),
                           max_concurrency=1, timeout=5)

    async def run():
        return await runtime.transcribe(b"...", "audio/webm", language=None,
                                        billed_user_id="s-1")

    anyio.run(run)
    assert credits.summary("s-1").balance == 50
