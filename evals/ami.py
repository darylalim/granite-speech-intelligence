"""Reader for the AMI Meeting Corpus manual annotations (NXT XML, v1.6.2).

Only the layers the transcript analysis evaluation uses: words, dialogue acts,
named entities, and the abstractive summaries' actions and decisions with the
links that tie each summary sentence to the dialogue acts it summarises.
Standard library only, so `ruff` and `ty` can check it in CI without the data.
"""

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

NITE_ID = "{http://nite.sourceforge.net/}id"
NITE_POINTER = "{http://nite.sourceforge.net/}pointer"
NITE_CHILD = "{http://nite.sourceforge.net/}child"

_HREF = re.compile(
    r"^(?P<file>[^#]+)#id\((?P<first>[^)]+)\)(?:\.\.id\((?P<last>[^)]+)\))?$"
)
_TYPE_HREF = re.compile(r"#id\((?P<id>[^)]+)\)$")
_TOKEN = re.compile(r"[a-z0-9']+")


@dataclass(frozen=True)
class Word:
    speaker: str
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class Span:
    """An annotation over words: a dialogue act or a named entity."""

    id: str
    kind: str
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class SummarySentence:
    """An abstractive `actions` or `decisions` sentence, timed by its linked acts."""

    kind: str
    text: str
    times: tuple[float, ...]


class Meeting:
    """One meeting's annotations, loaded from the unpacked manual archive."""

    def __init__(self, root: Path, meeting_id: str) -> None:
        self.root = root
        self.id = meeting_id
        # Every element in a speaker's words file, in file order, because a
        # child range `id(a)..id(b)` spans elements, not just <w> words.
        self._order: dict[str, list[str]] = {}
        self._index: dict[str, int] = {}
        self._words: dict[str, Word] = {}
        for path in sorted((root / "words").glob(f"{meeting_id}.*.words.xml")):
            speaker = path.name.split(".")[1]
            ids: list[str] = []
            for element in ET.parse(path).getroot():
                element_id = element.get(NITE_ID)
                if element_id is None:
                    continue
                self._index[element_id] = len(ids)
                ids.append(element_id)
                start, end = element.get("starttime"), element.get("endtime")
                if (
                    element.tag == "w"
                    and element.get("punc") != "true"
                    and start is not None
                    and end is not None
                    and element.text
                ):
                    self._words[element_id] = Word(
                        speaker, float(start), float(end), element.text
                    )
            self._order[path.name] = ids
        if not self._order:
            raise FileNotFoundError(f"no words files for {meeting_id} under {root}")
        self._ontologies: dict[str, dict[str, str]] = {}

    def words(self) -> list[Word]:
        return sorted(self._words.values(), key=lambda w: (w.start, w.end))

    def transcript(self, start: float, end: float) -> str:
        """Lowercase, unpunctuated words starting in `[start, end)`, ASR-shaped.

        The model under test sees TurboCTC's lowercase unpunctuated output, so the
        human transcript is normalised to the same shape: what differs between the
        two runs is then the words, not the casing or punctuation.
        """
        tokens: list[str] = []
        for word in self.words():
            if start <= word.start < end:
                tokens.extend(_TOKEN.findall(word.text.lower()))
        return " ".join(tokens)

    def _resolve(self, href: str) -> list[Word]:
        match = _HREF.match(href)
        if match is None:
            return []
        order = self._order.get(match["file"])
        if order is None:
            return []
        first = self._index[match["first"]]
        last = self._index[match["last"] or match["first"]]
        return [self._words[i] for i in order[first : last + 1] if i in self._words]

    def _ontology(self, name: str) -> dict[str, str]:
        if name not in self._ontologies:
            tree = ET.parse(self.root / "ontologies" / name)
            self._ontologies[name] = {
                element_id: element.get("name", "")
                for element in tree.iter()
                if (element_id := element.get(NITE_ID)) is not None
            }
        return self._ontologies[name]

    def _spans(self, layer: str, pattern: str, tag: str, ontology: str) -> list[Span]:
        names = self._ontology(ontology)
        spans: list[Span] = []
        for path in sorted((self.root / layer).glob(pattern)):
            for element in ET.parse(path).getroot().iter(tag):
                kind = ""
                for pointer in element.iter(NITE_POINTER):
                    found = _TYPE_HREF.search(pointer.get("href", ""))
                    if found:
                        kind = names.get(found["id"], "")
                words: list[Word] = []
                for child in element.iter(NITE_CHILD):
                    words.extend(self._resolve(child.get("href", "")))
                if not words:
                    continue
                spans.append(
                    Span(
                        id=element.get(NITE_ID, ""),
                        kind=kind,
                        start=min(w.start for w in words),
                        end=max(w.end for w in words),
                        text=" ".join(w.text for w in words),
                    )
                )
        return spans

    def dialogue_acts(self) -> list[Span]:
        return self._spans(
            "dialogueActs", f"{self.id}.*.dialog-act.xml", "dact", "da-types.xml"
        )

    def named_entities(self) -> list[Span]:
        return self._spans(
            "namedEntities", f"{self.id}.*.ne.xml", "named-entity", "ne-types.xml"
        )

    def summary_sentences(self) -> list[SummarySentence]:
        """The `actions` and `decisions` sentences, each timed by its linked acts.

        A sentence with no link (some annotators linked only part of a summary)
        has empty `times` and so falls in no excerpt.
        """
        abstract = self.root / "abstractive" / f"{self.id}.abssumm.xml"
        links = self.root / "extractive" / f"{self.id}.summlink.xml"
        if not abstract.exists() or not links.exists():
            return []
        acts = {act.id: act for act in self.dialogue_acts()}
        linked: dict[str, list[float]] = {}
        for link in ET.parse(links).getroot().iter("summlink"):
            act_id = sentence_id = None
            for pointer in link.iter(NITE_POINTER):
                found = _TYPE_HREF.search(pointer.get("href", ""))
                if found and pointer.get("role") == "extractive":
                    act_id = found["id"]
                elif found and pointer.get("role") == "abstractive":
                    sentence_id = found["id"]
            if act_id in acts and sentence_id is not None:
                linked.setdefault(sentence_id, []).append(acts[act_id].start)
        sentences: list[SummarySentence] = []
        for section in ET.parse(abstract).getroot():
            if section.tag not in ("actions", "decisions"):
                continue
            for sentence in section.iter("sentence"):
                text = " ".join((sentence.text or "").split())
                times = tuple(sorted(linked.get(sentence.get(NITE_ID, ""), [])))
                sentences.append(SummarySentence(section.tag, text, times))
        return sentences
