"""Grounded question answering: every answer is drawn from, and cites, the indexed documents."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .config import Settings
from .stores import VectorIndex

NOT_FOUND = "I couldn't find this in your documents."

SYSTEM = (
    "You answer questions using ONLY the numbered context passages. After every claim add the "
    "passage number in square brackets, like [1]. If the context does not contain the answer, "
    f"reply exactly: {NOT_FOUND} Never use outside knowledge and never guess."
)


@dataclass
class Source:
    n: int
    file: str
    page: int
    score: float
    snippet: str
    cited: bool = False


@dataclass
class Answer:
    question: str
    answer: str
    grounded: bool
    sources: list[Source] = field(default_factory=list)


class OllamaChat:
    """LangChain LCEL chain: prompt | ChatOllama | parser. Runs fully locally."""

    def __init__(self, settings: Settings):
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_ollama import ChatOllama

        prompt = ChatPromptTemplate.from_messages([("system", "{system}"), ("human", "{user}")])
        llm = ChatOllama(model=settings.chat_model, base_url=settings.ollama_host, temperature=0)
        self.chain = prompt | llm | StrOutputParser()

    def generate(self, system: str, user: str) -> str:
        return self.chain.invoke({"system": system, "user": user})


class ExtractiveChat:
    """Offline stand-in: returns the best-matching sentences from the context, with citations."""

    _BLOCK = re.compile(r"\[(\d+)\] \([^)]*\)\n(.*?)(?=\n\n\[\d+\] \(|\n\nQuestion:|\Z)", re.S)

    def generate(self, system: str, user: str) -> str:
        question = user.rsplit("Question:", 1)[-1].lower()
        q_words = set(re.findall(r"[a-z0-9]+", question)) - {"the", "a", "is", "what", "how", "of", "to"}
        scored = []
        for n, body in self._BLOCK.findall(user):
            for sentence in re.split(r"(?<=[.!?])\s+|\n+", body):
                words = set(re.findall(r"[a-z0-9]+", sentence.lower()))
                if sentence.strip() and words & q_words:
                    scored.append((len(words & q_words), sentence.strip(), n))
        scored.sort(key=lambda s: -s[0])
        if not scored:
            return NOT_FOUND
        return " ".join(f"{s} [{n}]" for _, s, n in scored[:2])


def build_chat(settings: Settings):
    return ExtractiveChat() if settings.backend == "offline" else OllamaChat(settings)


class Answerer:
    def __init__(self, index: VectorIndex, chat, settings: Settings):
        self.index, self.chat, self.settings = index, chat, settings

    def ask(self, question: str, k: int | None = None) -> Answer:
        hits = self.index.search(question, k or self.settings.top_k)
        hits = [(d, s) for d, s in hits if s >= self.settings.min_relevance]
        if not hits:  # nothing relevant indexed: refuse without calling the LLM
            return Answer(question, NOT_FOUND, grounded=False)

        sources = [
            Source(n, d.metadata.get("source", "?"), int(d.metadata.get("page", 1)), round(s, 3),
                   d.page_content[:240].replace("\n", " "))
            for n, (d, s) in enumerate(hits, start=1)
        ]
        context = "\n\n".join(
            f"[{src.n}] ({src.file}, page {src.page})\n{d.page_content}"
            for src, (d, _) in zip(sources, hits)
        )
        text = self.chat.generate(SYSTEM, f"Context:\n\n{context}\n\nQuestion: {question}").strip()

        cited = {int(n) for n in re.findall(r"\[(\d+)\]", text)}
        for src in sources:
            src.cited = src.n in cited
        grounded = NOT_FOUND not in text and bool(cited & {s.n for s in sources})
        return Answer(question, text, grounded, sources)
