"""
Claim and evidence parsing with heuristics.

Extracted from legacy factcheck_pipeline.py for Pipeline 2.0.
Provides claim categorization, entity extraction, and evidence parsing.
"""
import re
from typing import List, Literal, Optional

try:
    import spacy  # type: ignore[import]
except ImportError:
    spacy = None

from app.pipeline.models import ParsedClaim, ParsedEvidence
from app.pipeline.utils import infer_tense


# Constants for claim categorization and extraction
ROLE_KEYWORDS = [
    "bundeskanzler",
    "bundeskanzlerin",
    "bundespräsident",
    "bundespräsidentin",
    "präsident",
    "präsidentin",
    "kanzler",
    "kanzlerin",
    "minister",
    "ministerin",
]

PARTY_KEYWORDS = {
    # Germany
    "cdu": "CDU",
    "csu": "CSU",
    "spd": "SPD",
    "afd": "AfD",
    "grüne": "Bündnis 90/Die Grünen",
    "gruene": "Bündnis 90/Die Grünen",
    "fdp": "FDP",
    "linke": "Die Linke",
    
    # USA
    "democrats": "Democratic Party",
    "democratic party": "Democratic Party",
    "republicans": "Republican Party",
    "republican party": "Republican Party",
    "gop": "Republican Party",
    
    # UK
    "conservative party": "Conservative Party",
    "conservatives": "Conservative Party",
    "tories": "Conservative Party",
    "labour party": "Labour Party",
    "labour": "Labour Party",
    "liberal democrats": "Liberal Democrats",
    "lib dems": "Liberal Democrats",
    "snp": "Scottish National Party",
    
    # France
    "renaissance": "Renaissance",
    "lrem": "Renaissance",
    "en marche": "Renaissance",
    "les républicains": "Les Républicains",
    "rassemblement national": "Rassemblement National",
    "front national": "Rassemblement National",
    "rn": "Rassemblement National",
    "fn": "Rassemblement National",
    "la france insoumise": "La France Insoumise",
    "lfi": "La France Insoumise",
    "parti socialiste": "Parti Socialiste",
    
    # Spain
    "partido popular": "Partido Popular",
    "pp": "Partido Popular",
    "psoe": "PSOE",
    "vox": "Vox",
    "podemos": "Podemos",
    "ciudadanos": "Ciudadanos",
    
    # Italy
    "fratelli d'italia": "Fratelli d'Italia",
    "fdi": "Fratelli d'Italia",
    "lega": "Lega",
    "lega nord": "Lega",
    "partito democratico": "Partito Democratico",
    "forza italia": "Forza Italia",
    "movimento 5 stelle": "Movimento 5 Stelle",
    "m5s": "Movimento 5 Stelle",
    
    # Japan
    "liberal democratic party": "Liberal Democratic Party",
    "ldp": "Liberal Democratic Party",
    "constitutional democratic party": "Constitutional Democratic Party",
    "komeito": "Komeito",
    "japanese communist party": "Japanese Communist Party",
}

SPORT_EVENT_MAP = {
    "tour de france": {
        "keywords": ["tour de france"],
    },
    "giro d'italia": {"keywords": ["giro d'italia"]},
    "vuelta a españa": {"keywords": ["vuelta a españa", "vuelta"]},
    "french open": {"keywords": ["french open", "roland garros"]},
    "us open": {"keywords": ["us open"]},
    "australian open": {"keywords": ["australian open"]},
    "wimbledon": {"keywords": ["wimbledon"]},
    "champions league": {"keywords": ["champions league", "uefa champions league", "cl"]},
    "weltmeisterschaft": {"keywords": ["weltmeisterschaft", "wm", "world cup"]},
    "bundesliga": {"keywords": ["bundesliga"]},
}

SPORT_PROFESSION_KEYWORDS = {
    "tennis": ["tennisspieler", "tennisspielerin", "tennisprofi", "tennis-spieler", "tennis"],
    "cycling": ["radrennfahrer", "radrennfahrerin", "radprofi", "radsportler", "radsportlerin", "radfahrer", "radsport"],
    "football": ["fußballer", "fussballer", "fußballerin", "fussballerin", "soccer", "footballer"],
}

SPORTS_EVENT_KEYWORDS = [
    "tour de france",
    "weltmeisterschaft",
    "olympia",
    "champions league",
    "bundesliga",
    "giro",
    "vuelta",
]

ABILITY_KEYWORDS = [
    "kann",
    "können",
    "fähig",
    "vermögen",
    "vermögend",
    "in der lage",
]

MODALITY_WORDS = [
    "kann",
    "können",
    "konnte",
    "könnte",
    "ist",
    "sind",
    "war",
    "waren",
    "wurde",
    "wurden",
    "hat",
    "haben",
    "muss",
    "müssen",
]

NEGATION_WORDS = [
    "nicht",
    "nie",
    "kein",
    "keine",
    "keinen",
    "keiner",
]


class ClaimParser:
    """
    Parser for claims and evidence with heuristic extraction.
    Supports person name extraction, claim categorization, and evidence parsing.
    """
    
    def __init__(self):
        """Initialize ClaimParser. Optionally loads spaCy model if available."""
        self.nlp = None
        if spacy is not None:
            try:
                self.nlp = spacy.load("de_core_news_sm")
            except Exception as e:
                print(f"[CLAIM_PARSER] spaCy load error: {e}")
    
    def is_personal_or_subjective_claim(self, claim: str) -> bool:
        """
        Detect if a claim is personal, subjective, or colloquial and should not be fact-checked.
        
        Also detects clearly invalid/malformed claims (e.g., "meins ist die Hauptstadt").
        
        Args:
            claim: The claim text to check
            
        Returns:
            True if claim is personal/subjective/colloquial/invalid and should be skipped
        """
        if not claim:
            return False
        
        text_lower = claim.lower()
        
        # Pattern 0: Invalid/malformed claims - possessive pronouns as subject with factual predicates
        # "meins ist die Hauptstadt" - "meins" (possessive pronoun) cannot be a city/entity
        invalid_claim_patterns = [
            r'\bmeins\s+ist\s+die\s+hauptstadt\b',  # "meins ist die Hauptstadt"
            r'\bmeins\s+liegt\s+in\b',  # "meins liegt in"
            r'\bmeins\s+ist\s+ein\b',  # "meins ist ein"
            r'\bmeins\s+hat\s+\d+\b',  # "meins hat 1000"
        ]
        if any(re.search(pattern, text_lower) for pattern in invalid_claim_patterns):
            return True
        
        # Pattern 1: Personal pronouns (ich, du, deine, dein, dir, dich, etc.)
        personal_pronouns = [
            r'\bich\b',
            r'\bdu\b',
            r'\bdeine\b',
            r'\bdein\b',
            r'\bdir\b',
            r'\bdich\b',
            r'\bdeiner\b',
            r'\bdeinem\b',
            r'\bdeinen\b',
            r'\bmeine\b',
            r'\bmein\b',
            r'\bmir\b',
            r'\bmich\b',
            r'\bmeiner\b',
            r'\bmeinem\b',
            r'\bmeinen\b',
        ]
        if any(re.search(pattern, text_lower) for pattern in personal_pronouns):
            return True
        
        # Pattern 2: Subjective adjectives (beste, schönste, wunderschön, toll, etc.)
        subjective_adjectives = [
            r'\bbeste\b',
            r'\bschönste\b',
            r'\bwunderschön\b',
            r'\btoll\b',
            r'\bfantastisch\b',
            r'\bwunderbar\b',
            r'\bschlecht\b',
            r'\bschlimm\b',
            r'\bhorrible\b',
            r'\bperfekt\b',
        ]
        if any(re.search(pattern, text_lower) for pattern in subjective_adjectives):
            return True
        
        # Pattern 3: Colloquial expressions
        colloquial_patterns = [
            r'die\s+hölle\s+los',  # "die Hölle los"
            r'\bhalt\b',  # "halt" (colloquial)
            r'\bwirklich\s+(wunderschön|toll|schön|gut|schlecht)',  # "wirklich wunderschön"
            r'\b(echt|total|voll|mega|super)\s+(toll|schön|gut|schlecht|geil)',  # "echt toll"
            r'\b(rewe|ikea|aldi|lidl|edeka)\s+.*\s+(hölle|wirklich|toll)',  # Store names with subjective language
        ]
        if any(re.search(pattern, text_lower) for pattern in colloquial_patterns):
            return True
        
        # Pattern 4: Question formats with personal address
        question_patterns = [
            r'wusstest\s+du',
            r'weißt\s+du',
            r'hast\s+du\s+schon\s+gehört',
            r'kennst\s+du',
            r'wusstest\s+du.*dass',  # "Wusstest du, dass..."
        ]
        if any(re.search(pattern, text_lower) for pattern in question_patterns):
            return True
        
        # Pattern 5: Personal experience markers
        experience_patterns = [
            r'\b(lernt|lerne|gelernt)\s+man\s+halt',
            r'\b(lernt|lerne|gelernt)\s+man\s+nicht\s+in\s+der\s+schule',
            r'\b(war|ist)\s+.*\s+die\s+hölle\s+los',  # "war im Rewe die Hölle los"
            r'\b(war|ist)\s+.*\s+wirklich\s+(wunderschön|toll|schön)',  # "ist wirklich wunderschön"
            r'\b(war|ist)\s+im\s+.*\s+die\s+hölle\s+los',  # "war im Rewe die Hölle los"
        ]
        if any(re.search(pattern, text_lower) for pattern in experience_patterns):
            return True
        
        # Pattern 6: Generic/abstract subjects that are not factual
        generic_subjects = [
            r'^solche\s+dinge',
            r'^das\s+ist\s+der\s+beste\s+lifehack',
            r'^das\s+ist\s+.*\s+für\s+deine',
            r'^die\s+lampe\s+von\s+.*\s+ist\s+wirklich\s+wunderschön',  # "Die Lampe von Ikea ist wirklich wunderschön"
        ]
        if any(re.search(pattern, text_lower) for pattern in generic_subjects):
            return True
        
        # Pattern 7: Lifehack/Tip patterns (subjective recommendations)
        if re.search(r'\b(beste|bester|besten)\s+lifehack', text_lower):
            return True
        
        # Pattern 8: Cooking tips with personal address
        if re.search(r'wusstest\s+du.*dass.*man.*kochen\s+kann', text_lower):
            return True
        
        return False
    
    def parse_claim(self, claim: str) -> ParsedClaim:
        """
        Parse a claim and extract structured information.
        
        Args:
            claim: The claim text to parse
            
        Returns:
            ParsedClaim with extracted subject, predicate, category, relation_type, etc.
        """
        parsed = ParsedClaim(raw=claim)
        text_lower = claim.lower()
        
        # Set coarse tense early so relation logic can use it (present vs past)
        # Use utils.infer_tense (SSOT)
        parsed.tense = infer_tense(claim)
        
        # Extract person name using improved method (for all claims)
        person_name = self._extract_person_name(claim)
        if person_name:
            parsed.subject = person_name
        
        # For general claims, use extracted person name or fallback to dependency parsing
        if not parsed.subject:
            doc = self.nlp(claim) if self.nlp else None
            if doc:
                subject_token = next(
                    (token for token in doc if token.dep_ in {"nsubj", "nsubjpass"}),
                    None,
                )
                if subject_token:
                    # Get full noun phrase (e.g., "Der Zweite Weltkrieg" not just "Der Zweite")
                    # Strategy: Get the full subtree, which includes all dependent tokens
                    subtree = list(subject_token.subtree)
                    if len(subtree) > 1:
                        # Sort by position in sentence to maintain order
                        subtree_sorted = sorted(subtree, key=lambda t: t.i)
                        full_subject = " ".join([t.text for t in subtree_sorted]).strip()
                        parsed.subject = full_subject
                    else:
                        # Single token subject - check if it's part of a compound noun
                        # Look for following nouns that might be part of the subject
                        if subject_token.i < len(doc) - 1:
                            next_token = doc[subject_token.i + 1]
                            # If next token is a noun and not already part of another phrase
                            if next_token.pos_ == "NOUN" and next_token.dep_ not in {"dobj", "pobj", "nmod"}:
                                # Check if it's likely part of the subject (e.g., "Weltkrieg" after "Zweite")
                                full_subject = f"{subject_token.text} {next_token.text}"
                                parsed.subject = full_subject
                            else:
                                parsed.subject = subject_token.text
                        else:
                            parsed.subject = subject_token.text

                root = doc[:].root
                if root:
                    parsed.predicate = root.lemma_
            else:
                tokens = claim.split()
                if tokens:
                    # For multi-word subjects like "Der Zweite Weltkrieg" or "Ein Jahr", try to get full phrase
                    if len(tokens) >= 2 and tokens[0].lower() in ["der", "die", "das", "ein", "eine"]:
                        # Check if second token is capitalized (likely noun after article)
                        if tokens[1][0].isupper():
                            # Take article + noun (e.g., "Ein Jahr", "Der Tag")
                            parsed.subject = f"{tokens[0]} {tokens[1]}"
                        else:
                            parsed.subject = tokens[0]
                    elif len(tokens) >= 3 and tokens[0].lower() in ["der", "die", "das"]:
                        # Check if second token is capitalized (likely adjective + noun)
                        if tokens[1][0].isupper() and tokens[2][0].isupper():
                            parsed.subject = f"{tokens[0]} {tokens[1]} {tokens[2]}"
                        else:
                            parsed.subject = tokens[0]
                    else:
                        parsed.subject = tokens[0]
                    parsed.predicate = tokens[1] if len(tokens) > 1 else ""

        parsed.numbers = self._extract_years(text_lower)
        parsed.year = parsed.numbers[0] if parsed.numbers else None
        parsed.category = self._categorize_claim(text_lower)
        parsed.party = self._extract_party(text_lower)
        parsed.role = self._extract_role(text_lower)
        parsed.event = self._extract_event(text_lower)
        parsed.sport = self._detect_sport(text_lower)
        
        # Normalize subject for country names (remove articles, unify common variants)
        # Helps matching e.g. "Die USA" -> "Vereinigte Staaten", "Vereinigten Staaten" -> "Vereinigte Staaten"
        if parsed.subject:
            subject_normalized = parsed.subject.strip()
            # Remove leading articles
            articles = ["die", "der", "das", "ein", "eine", "einen", "einem", "einer"]
            subject_words = subject_normalized.split()
            if len(subject_words) > 1 and subject_words[0].lower() in articles:
                subject_normalized = " ".join(subject_words[1:])
            
            # Country name mappings (synonyms and inflections -> canonical Wikipedia title)
            country_mappings = {
                "usa": "Vereinigte Staaten",
                "u.s.a": "Vereinigte Staaten",
                "u.s.a.": "Vereinigte Staaten",
                "us": "Vereinigte Staaten",
                "vereinigte staaten": "Vereinigte Staaten",
                "vereinigten staaten": "Vereinigte Staaten",
            }
            subject_lower_normalized = subject_normalized.lower()
            if subject_lower_normalized in country_mappings:
                parsed.subject = country_mappings[subject_lower_normalized]
            else:
                # Heuristic: if subject contains both "vereinigte(n)" and "staaten", canonicalize
                if ("staaten" in subject_lower_normalized and
                    ("vereinigte" in subject_lower_normalized or "vereinigten" in subject_lower_normalized)):
                    parsed.subject = "Vereinigte Staaten"
                else:
                    parsed.subject = subject_normalized
        
        # Extract object using dependency parsing if available
        if not parsed.object and self.nlp:
            try:
                doc = self.nlp(claim)
                # Look for object tokens (dobj, pobj, nmod, obl)
                obj_token = next(
                    (token for token in doc if token.dep_ in {"dobj", "pobj", "nmod", "obl"}),
                    None,
                )
                if obj_token:
                    # Get the full noun phrase if available
                    if obj_token.head.pos_ == "NOUN" or obj_token.pos_ == "NOUN":
                        # Try to get the full noun phrase
                        subtree = list(obj_token.subtree)
                        if len(subtree) > 1:
                            parsed.object = " ".join([t.text for t in subtree]).strip()
                        else:
                            parsed.object = obj_token.text
                else:
                    # Fallback: look for prepositional phrases (e.g., "in Chloroplasten")
                    for token in doc:
                        if token.pos_ == "ADP" and token.dep_ == "case":  # Preposition
                            # Find the noun that follows the preposition
                            for child in token.head.children:
                                if child.pos_ == "NOUN" and child.dep_ in {"nmod", "obl"}:
                                    parsed.object = child.text
                                    break
                            if parsed.object:
                                break
            except Exception:
                pass
        
        # For nationality claims, explicitly extract the demonym as object
        if parsed.category == "nationality" and not parsed.object:
            known_demonyms = [
                "argentinier", "argentinierin", "argentinisch",
                "deutscher", "deutsche", "deutsch",
                "franzose", "französin", "französisch",
                "spanier", "spanierin", "spanisch",
                "brasilianer", "brasilianerin", "brasilianisch",
                "portugiese", "portugiesin", "portugiesisch",
                "italiener", "italienerin", "italienisch",
                "engländer", "engländerin", "englisch",
                "amerikaner", "amerikanerin", "amerikanisch",
            ]
            # Try to extract demonym from claim
            demonym_patterns = [
                r'\bist\s+(?:ein|eine|der|die|das)?\s*([a-zäöüß]+(?:er|in|isch|e))\b',
                r'\bwar\s+(?:ein|eine|der|die|das)?\s*([a-zäöüß]+(?:er|in|isch|e))\b',
            ]
            for pattern in demonym_patterns:
                match = re.search(pattern, text_lower)
                if match:
                    demonym = match.group(1).lower()
                    # Check if it's a known demonym
                    if demonym in known_demonyms:
                        # Find the original capitalization from claim
                        original_demonym = re.search(
                            r'\b(?:ist|war)\s+(?:ein|eine|der|die|das)?\s*(' + re.escape(demonym) + r')\b',
                            claim,
                            re.IGNORECASE
                        )
                        if original_demonym:
                            parsed.object = original_demonym.group(1)
                        else:
                            parsed.object = demonym.capitalize()
                        break
            
            # Fallback: search for any known demonym in claim
            if not parsed.object:
                for demonym in known_demonyms:
                    demonym_pattern = r'\b(?:ist|war)\s+(?:ein|eine|der|die|das)?\s*(' + re.escape(demonym) + r')\b'
                    match = re.search(demonym_pattern, claim, re.IGNORECASE)
                    if match:
                        parsed.object = match.group(1)
                        break
        
        # For sports_title and sports_title_year, ensure event is extracted
        if (parsed.category == "sports_title" or parsed.category == "sports_title_year") and not parsed.event:
            # Try to extract event from claim directly
            for event_key, event_data in SPORT_EVENT_MAP.items():
                if any(kw in text_lower for kw in event_data["keywords"]):
                    parsed.event = event_key
                    break
        
        # Map category to relation_type
        category_to_relation = {
            "politics_role": "political_office",
            "party_membership": "party_membership",
            "nationality": "nationality",
            "sports_title": "sports_title",
            "sports_title_year": "sports_title_year",
            "person_sport": "person_sport",
            "biological_capability": "biological_capability",
            "comparison": "comparison",
        }
        parsed.relation_type = category_to_relation.get(parsed.category, "general")
        
        # STEP 1: Canonical claim normalization
        parsed.canonical_claim = self._normalize_claim(claim)
        
        # STEP 2: Hard-claim detection
        parsed.is_hard_claim = self._is_hard_claim(claim, parsed)
        
        # STEP 3: Claim decomposition (only for hard claims)
        if parsed.is_hard_claim:
            parsed.subclaims = self._decompose_claim(claim, parsed.canonical_claim, parsed)
            # Limit to max 4 subclaims
            parsed.subclaims = parsed.subclaims[:4]

        return parsed
    
    def parse_evidence(self, text: str) -> ParsedEvidence:
        """
        Parse evidence text and extract structured information.
        
        Args:
            text: The evidence text to parse
            
        Returns:
            ParsedEvidence with extracted subject, predicate, tense, negation, etc.
        """
        parsed = ParsedEvidence(raw=text)
        text_lower = text.lower()
        doc = self.nlp(text) if self.nlp else None

        if doc:
            subject_token = next(
                (token for token in doc if token.dep_ in {"nsubj", "nsubjpass"}),
                None,
            )
            if subject_token:
                parsed.subject = subject_token.text

            root = doc[:].root
            if root:
                parsed.predicate = root.lemma_
                parsed.tense = self._detect_tense_from_token(root)

            obj_token = next(
                (
                    token
                    for token in doc
                    if token.dep_ in {"dobj", "attr", "pobj", "acomp", "xcomp"}
                ),
                None,
            )
            if obj_token:
                parsed.object = obj_token.text
        else:
            tokens = text.split()
            if tokens:
                parsed.subject = tokens[0]
                parsed.predicate = tokens[1] if len(tokens) > 1 else ""
                parsed.object = tokens[2] if len(tokens) > 2 else None

        parsed.modality = self._extract_modality(text_lower)
        if parsed.tense == "unknown":
            # Use utils.infer_tense (SSOT)
            parsed.tense = infer_tense(text)
        parsed.numbers = self._extract_years(text_lower)
        parsed.party = self._extract_party(text_lower)
        parsed.role = self._extract_role(text_lower)
        parsed.mentions_negation = any(word in text_lower for word in NEGATION_WORDS)

        return parsed
    
    def _extract_person_name(self, claim: str) -> Optional[str]:
        """
        Extract full person name using spaCy PER entities.
        Fallback: if first two tokens are capitalized → use both, else first token.
        """
        if self.nlp:
            try:
                doc = self.nlp(claim)
                # Look for PER (person) entities
                for ent in doc.ents:
                    if ent.label_ == "PER":
                        return ent.text.strip()
            except Exception as e:
                print(f"[PARSE] spaCy error: {e}")
        
        # Fallback: capitalization-based extraction
        tokens = claim.split()
        if len(tokens) >= 2:
            # If first two tokens are capitalized, use both
            # BUT avoid false positives for generic noun phrases like "Ein Jahr", "Der Krieg", etc.
            if tokens[0][0].isupper() and tokens[1][0].isupper():
                determiners = {"der", "die", "das", "ein", "eine", "einen", "einem", "einer"}
                common_nouns = {
                    "jahr", "monate", "monat", "tag", "tage", "woche", "wochen",
                    "minute", "minuten", "sekunde", "sekunden",
                }
                t0 = re.sub(r"[^\wäöüÄÖÜß-]", "", tokens[0]).lower()
                t1 = re.sub(r"[^\wäöüÄÖÜß-]", "", tokens[1]).lower()
                if t0 in determiners or t1 in common_nouns:
                    return None
                return f"{tokens[0]} {tokens[1]}"
        if tokens:
            return tokens[0]
        return None
    
    def _is_full_person_name(self, name: str) -> bool:
        """Check if name appears to be a full person name (2+ tokens)"""
        if not name:
            return False
        parts = name.strip().split()
        return len(parts) >= 2
    
    def _extract_modality(self, text_lower: str) -> Optional[str]:
        """Extract modality words from text."""
        for word in MODALITY_WORDS:
            if f" {word} " in text_lower or text_lower.startswith(f"{word} "):
                return word
        return None
    
    def _categorize_claim(self, text_lower: str) -> str:
        """Simplified categorization - avoid false positives."""
        # Check comparison patterns FIRST (they have clear indicators like "als")
        comparison_patterns = [
            "länger als", "kürzer als", "größer als", "kleiner als", "mehr als", "weniger als",
            "höher als", "niedriger als", "schneller als", "langsamer als",
        ]
        if any(pattern in text_lower for pattern in comparison_patterns):
            return "comparison"
        
        # Only categorize if we have clear indicators
        # Don't mark as politics_role if it's just "ist Franzose" etc.
        has_role_keyword = any(keyword in text_lower for keyword in ROLE_KEYWORDS)
        # CRITICAL: Use same word-boundary matching as _extract_party to avoid false positives
        # E.g., "Bayern" should NOT match "rn" (Rassemblement National)
        has_party_keyword = False
        for party_key in PARTY_KEYWORDS.keys():
            party_key_len = len(party_key)
            if party_key_len <= 3:
                # Very short key (e.g., "rn", "fn", "pp", "ldp", "fdi", "m5s")
                # Use word-boundary regex and require party context
                pattern = r'\b' + re.escape(party_key) + r'\b'
                if re.search(pattern, text_lower, re.IGNORECASE):
                    # Check for party context keywords (same as in _extract_party)
                    party_context_keywords = [
                        "partei", "party", "mitglied", "member", "wählt", "wahl",
                        "politiker", "politikerin", "politik", "abgeordnete", "kandidat",
                        "fraktion", "faction", "regierung", "opposition", "koalition",
                        "bei der", "von der", "der partei",
                        "wählt die", "wählt", "stimmt für", "unterstützt",
                    ]
                    has_party_context = any(ctx_kw in text_lower for ctx_kw in party_context_keywords)
                    
                    # Check if full party name is present (same as in _extract_party)
                    full_party_names = {
                        "rn": "rassemblement national",
                        "fn": "front national",
                        "pp": "partido popular",
                        "ldp": "liberal democratic party",
                        "fdi": "fratelli d'italia",
                        "m5s": "movimento 5 stelle",
                    }
                    full_name_present = False
                    if party_key in full_party_names:
                        full_name = full_party_names[party_key]
                        if full_name in text_lower:
                            full_name_present = True
                    
                    if has_party_context or full_name_present:
                        has_party_keyword = True
                        break
            else:
                # Longer key: use word-boundary matching
                pattern = r'\b' + re.escape(party_key) + r'\b'
                if re.search(pattern, text_lower, re.IGNORECASE):
                    has_party_keyword = True
                    break
                # Fallback: check if key appears as whole words in sequence (for multi-word keys)
                if ' ' in party_key and party_key in text_lower:
                    party_key_words = party_key.split()
                    text_words = re.findall(r'\b\w+\b', text_lower)
                    text_lower_words = [w.lower() for w in text_words]
                    party_key_words_lower = [w.lower() for w in party_key_words]
                    
                    # Check if key_words appear consecutively in text_words
                    for i in range(len(text_lower_words) - len(party_key_words_lower) + 1):
                        if text_lower_words[i:i+len(party_key_words_lower)] == party_key_words_lower:
                            has_party_keyword = True
                            break
                    if has_party_keyword:
                        break
        # Check for nationality/demonym patterns FIRST (before sport detection)
        # This ensures claims like "Ronaldo ist Portugiese" are recognized as nationality, not person_sport
        # (e.g., "Messi ist Argentinier", "X ist Deutscher", "Ronaldo ist Portugiese")
        # Pattern: "ist/war" + demonym (ends with -er, -in, -isch, -e, or in known list)
        known_demonyms = [
            "argentinier", "argentinierin", "argentinisch",
            "deutscher", "deutsche", "deutsch",
            "franzose", "französin", "französisch",
            "spanier", "spanierin", "spanisch",
            "brasilianer", "brasilianerin", "brasilianisch",
            "portugiese", "portugiesin", "portugiesisch",
            "italiener", "italienerin", "italienisch",
            "engländer", "engländerin", "englisch",
            "amerikaner", "amerikanerin", "amerikanisch",
        ]
        
        # Pattern 1: Standard pattern for demonyms ending with -er, -in, -isch
        nationality_patterns = [
            r'\bist\s+(?:ein|eine|der|die|das)?\s*([a-zäöüß]+(?:er|in|isch))\b',
            r'\bwar\s+(?:ein|eine|der|die|das)?\s*([a-zäöüß]+(?:er|in|isch))\b',
        ]
        for pattern in nationality_patterns:
            match = re.search(pattern, text_lower)
            if match:
                demonym = match.group(1).lower()
                # Check if it's a known demonym or matches pattern
                if demonym in known_demonyms or demonym.endswith(("er", "in", "isch")):
                    # Additional check: avoid false positives like "ist Lehrer" (profession)
                    profession_keywords = ["lehrer", "arzt", "anwalt", "ingenieur", "wissenschaftler"]
                    if demonym not in profession_keywords:
                        return "nationality"
        
        # Pattern 2: Direct check for known demonyms (catches cases like "ist Portugiese" that don't end with -er/-in/-isch)
        # This is a fallback for demonyms that don't match the standard pattern
        # CRITICAL: Check both with and without article to catch all cases
        for demonym in known_demonyms:
            # Pattern A: "ist/war" + optional article + demonym
            demonym_pattern_a = r'\b(?:ist|war)\s+(?:ein|eine|der|die|das)?\s*' + re.escape(demonym) + r'\b'
            # Pattern B: "ist/war" + demonym (without article, more direct)
            demonym_pattern_b = r'\b(?:ist|war)\s+' + re.escape(demonym) + r'\b'
            
            match_a = re.search(demonym_pattern_a, text_lower)
            match_b = re.search(demonym_pattern_b, text_lower)
            
            if match_a or match_b:
                # Additional check: avoid false positives like "ist Lehrer" (profession)
                profession_keywords = ["lehrer", "arzt", "anwalt", "ingenieur", "wissenschaftler"]
                if demonym not in profession_keywords:
                    return "nationality"
        
        # Check for political office patterns (must have role keyword + office context)
        if has_role_keyword and ("ist" in text_lower or "war" in text_lower):
            # Avoid false positives like "ist Franzose", "ist Portugiese" (now handled as nationality)
            known_nationality_demonyms = [
                "franzose", "französin", "französisch",
                "deutscher", "deutsche", "deutsch",
                "amerikaner", "amerikanerin", "amerikanisch",
                "argentinier", "argentinierin", "argentinisch",
                "portugiese", "portugiesin", "portugiesisch",
                "spanier", "spanierin", "spanisch",
                "brasilianer", "brasilianerin", "brasilianisch",
                "italiener", "italienerin", "italienisch",
                "engländer", "engländerin", "englisch",
            ]
            if not any(nationality in text_lower for nationality in known_nationality_demonyms):
                return "politics_role"
        
        if any(keyword in text_lower for keyword in ABILITY_KEYWORDS):
            return "biological_capability"
        if any(keyword in text_lower for keyword in SPORTS_EVENT_KEYWORDS):
            return "sports_title"
        return "general"
    
    def _extract_party(self, text_lower: str) -> Optional[str]:
        """
        Extract political party from text.
        
        Uses word-boundary matching to avoid false positives (e.g., "Bayern" -> "RN").
        For very short abbreviations (2-3 letters), requires additional context checks.
        """
        # CRITICAL: Use word-boundary matching to avoid substring false positives
        # E.g., "Bayern" should NOT match "rn" (Rassemblement National)
        for key, value in PARTY_KEYWORDS.items():
            # For very short abbreviations (2-3 letters), use strict word-boundary matching
            # and require party context keywords to avoid false positives
            key_len = len(key)
            if key_len <= 3:
                # Very short key (e.g., "rn", "fn", "pp", "ldp", "fdi", "m5s")
                # Use word-boundary regex to ensure it's a standalone word
                # Pattern: word boundary + key + word boundary (case-insensitive)
                pattern = r'\b' + re.escape(key) + r'\b'
                if re.search(pattern, text_lower, re.IGNORECASE):
                    # Additional check: require party context keywords for very short abbreviations
                    # This prevents "Bayern" from matching "rn"
                    party_context_keywords = [
                        "partei", "party", "mitglied", "member", "wählt", "wahl",
                        "politiker", "politikerin", "politik", "abgeordnete", "kandidat",
                        "fraktion", "faction", "regierung", "opposition", "koalition",
                        "bei der", "von der", "der partei", "der partei",
                        "wählt die", "wählt", "stimmt für", "unterstützt",
                    ]
                    # Check if text contains party context keywords nearby
                    # If the key appears but no party context is present, it's likely a false positive
                    has_party_context = any(
                        ctx_kw in text_lower for ctx_kw in party_context_keywords
                    )
                    
                    # Exception: If key is already in a multi-word party name (e.g., "Rassemblement National"),
                    # we don't need additional context
                    # Check if full party name is present
                    full_party_names = {
                        "rn": "rassemblement national",
                        "fn": "front national",
                        "pp": "partido popular",
                        "ldp": "liberal democratic party",
                        "fdi": "fratelli d'italia",
                        "m5s": "movimento 5 stelle",
                    }
                    full_name_present = False
                    if key in full_party_names:
                        full_name = full_party_names[key]
                        if full_name in text_lower:
                            full_name_present = True
                    
                    # Only accept if party context is present OR full party name is present
                    if has_party_context or full_name_present:
                        return value
                    # Otherwise, skip this match (likely false positive)
                    continue
            else:
                # Longer key (>= 4 characters): use word-boundary matching but allow substring matches
                # if the key is part of a larger phrase (e.g., "Democratic Party" in "is a member of the Democratic Party")
                pattern = r'\b' + re.escape(key) + r'\b'
                if re.search(pattern, text_lower, re.IGNORECASE):
                    return value
                # Fallback: if word-boundary doesn't match, check if key appears as a whole word
                # (e.g., "conservative party" might not match with word boundaries due to spacing)
                # But still use boundaries to avoid false positives
                if key in text_lower:
                    # Additional safety check: ensure key is not a substring of a longer word
                    # Split text into words and check if key matches any word exactly (case-insensitive)
                    words = re.findall(r'\b\w+\b', text_lower)
                    if any(key.lower() == word.lower() for word in words):
                        return value
                    # Also check if key appears as part of a known phrase pattern
                    # E.g., "conservative party" should match even if word boundaries don't catch it
                    if ' ' in key:
                        # Multi-word key: check if all words appear in sequence
                        key_words = key.split()
                        text_words = re.findall(r'\b\w+\b', text_lower)
                        text_lower_words = [w.lower() for w in text_words]
                        key_words_lower = [w.lower() for w in key_words]
                        
                        # Check if key_words appear consecutively in text_words
                        for i in range(len(text_lower_words) - len(key_words_lower) + 1):
                            if text_lower_words[i:i+len(key_words_lower)] == key_words_lower:
                                return value
        return None
    
    def _extract_role(self, text_lower: str) -> Optional[str]:
        """Extract political role from text."""
        for role in ROLE_KEYWORDS:
            if role in text_lower:
                return role
        return None
    
    def _extract_event(self, text_lower: str) -> Optional[str]:
        """Extract sports event from text."""
        for event, data in SPORT_EVENT_MAP.items():
            if any(keyword in text_lower for keyword in data["keywords"]):
                return event
        return None
    
    def _detect_sport(self, text_lower: str) -> Optional[str]:
        """Detect sport type from text."""
        for sport, keywords in SPORT_PROFESSION_KEYWORDS.items():
            if any(keyword in text_lower for keyword in keywords):
                return sport
        return None
    
    @staticmethod
    def _extract_years(text: str) -> List[int]:
        """Extract years (1900-2099) from text."""
        if not text:
            return []
        return [int(year) for year in re.findall(r"\b((?:19|20)\d{2})\b", text)]
    
    def _detect_tense_from_token(self, token) -> Literal["present", "past", "unknown"]:
        """Detect tense from spaCy token morphology."""
        if not token:
            return "unknown"
        tense = token.morph.get("Tense")
        if "Past" in tense:
            return "past"
        if "Pres" in tense:
            return "present"
        return "unknown"
    
    def _normalize_claim(self, claim: str) -> str:
        """
        Normalize claim text to canonical form.
        
        Minimal rules (deterministic, low-risk):
        - lowercase
        - whitespace normalization (multiple spaces → 1)
        - remove quotes and edge punctuation
        - normalize number formats (basic)
        - remove harmless filler words (small list)
        
        Args:
            claim: Original claim text
            
        Returns:
            Canonical claim string
        """
        if not claim:
            return ""
        
        # Lowercase
        canonical = claim.lower()
        
        # Remove quotes (single and double, German and English)
        canonical = re.sub(r'["""''„"]', '', canonical)
        
        # Remove harmless filler words (small list, without destroying semantics)
        filler_words = ["angeblich", "wohl", "laut", "vermutlich", "möglicherweise", "scheinbar"]
        for filler in filler_words:
            # Remove filler word with surrounding spaces
            canonical = re.sub(rf'\b{filler}\b', '', canonical)
        
        # Normalize whitespace: collapse multiple spaces/tabs/newlines to single space
        canonical = re.sub(r'\s+', ' ', canonical)
        
        # Basic number normalization: ensure consistent spacing around numbers
        # E.g., "1,2" → "1.2" (but keep "1.200.000" as is)
        # This is minimal - more advanced normalization can be added later
        
        # Remove leading/trailing whitespace
        canonical = canonical.strip()
        
        return canonical
    
    def _is_hard_claim(self, claim: str, parsed_claim: ParsedClaim) -> bool:
        """
        Detect if claim is "hard" and requires decomposition.
        
        Triggers (True if at least one applies):
        - Comparison words: "länger", "kürzer", "größer", "kleiner", "mehr", "weniger", "als"
        - Numeric tokens: digits, percentages, units (km, %, mio, million, milliarden, °C, etc.)
        - At least two recognizable entities/Proper Nouns (heuristic)
        - Contains year (4-digit) or explicit time markers ("seit", "im Jahr")
        
        Args:
            claim: Original claim text
            parsed_claim: Parsed claim object
            
        Returns:
            True if claim is hard, False otherwise
        """
        claim_lower = claim.lower()
        
        # Trigger 1: Comparison words
        comparison_words = ["länger", "kürzer", "größer", "kleiner", "mehr", "weniger", "als", "höher", "niedriger"]
        if any(word in claim_lower for word in comparison_words):
            return True
        
        # Trigger 2: Numeric tokens (digits, percentages, units)
        # BUT: Skip if this is a nationality claim (they are factual but not "hard" in the decomposition sense)
        is_nationality = parsed_claim.category == "nationality"
        if not is_nationality:
            numeric_patterns = [
                r'\d+',  # Digits
                r'%',  # Percentage
                r'\b(km|m|cm|mm|kg|g|mg|°c|°f|mio|million|milliarden|milliard|billion)\b',  # Units
            ]
            for pattern in numeric_patterns:
                if re.search(pattern, claim_lower):
                    return True
        
        # Trigger 3: At least two recognizable entities/Proper Nouns (heuristic)
        # Count capitalized words (likely proper nouns)
        capitalized_words = re.findall(r'\b[A-ZÄÖÜ][a-zäöüß]+\b', claim)
        if len(capitalized_words) >= 2:
            return True
        
        # Trigger 4: Year (4-digit) or explicit time markers
        if parsed_claim.year or re.search(r'\b((?:19|20)\d{2})\b', claim):
            return True
        
        time_markers = ["seit", "im jahr", "im monat", "am", "von", "bis"]
        if any(marker in claim_lower for marker in time_markers):
            return True
        
        return False
    
    def _decompose_claim(self, claim: str, canonical: str, parsed_claim: ParsedClaim) -> List[str]:
        """
        Decompose claim into subclaims (max 3-4).
        
        Rules (simple pattern-based):
        A) Comparison claims (X ist länger als Y)
        B) Numeric claims (... beträgt Y Einheit)
        C) Temporal claims (ist/war/seit + Rolle)
        
        Args:
            claim: Original claim text
            canonical: Canonical claim text
            parsed_claim: Parsed claim object
            
        Returns:
            List of subclaims (max 3-4), empty list if decomposition fails
        """
        subclaims = []
        # Use original claim (not canonical) for pattern matching to preserve structure
        # Remove trailing punctuation for cleaner matching
        claim_lower = claim.lower().rstrip('.,!?')
        
        # A) Comparison claims (X ist länger als Y / X größer als Y / X mehr als Y)
        # Try pattern with verb first: "X ist länger als Y"
        comparison_pattern1 = r'(.+?)\s+(ist|war|sind|waren)\s+(länger|kürzer|größer|kleiner|mehr|weniger|höher|niedriger)\s+als\s+(.+)'
        match = re.search(comparison_pattern1, claim_lower)
        
        if match:
            x_entity = match.group(1).strip()
            comparison_word = match.group(3).strip()
            y_entity = match.group(4).strip().rstrip('.,')
        else:
            # Try pattern without verb: "X länger als Y"
            comparison_pattern2 = r'(.+?)\s+(länger|kürzer|größer|kleiner|mehr|weniger|höher|niedriger)\s+als\s+(.+)'
            match = re.search(comparison_pattern2, claim_lower)
            if match:
                x_entity = match.group(1).strip()
                comparison_word = match.group(2).strip()
                y_entity = match.group(3).strip().rstrip('.,')
            else:
                match = None
        
        if match:
            # Clean entities (remove determiners like "der", "die", "das")
            x_entity_cleaned = re.sub(r'^(der|die|das|ein|eine|einen|einem|einer)\s+', '', x_entity, flags=re.IGNORECASE)
            y_entity_cleaned = re.sub(r'^(der|die|das|ein|eine|einen|einem|einer)\s+', '', y_entity, flags=re.IGNORECASE)
            
            if x_entity_cleaned and y_entity_cleaned and comparison_word:
                # Map comparison word to attribute
                attr_map = {
                    "länger": "länge",
                    "kürzer": "länge",
                    "größer": "größe",
                    "kleiner": "größe",
                    "mehr": "anzahl",
                    "weniger": "anzahl",
                    "höher": "höhe",
                    "niedriger": "höhe",
                }
                attribute = attr_map.get(comparison_word, "wert")
                
                # Subclaim 1: Attribute of X
                subclaims.append(f"{x_entity_cleaned} {attribute}")
                
                # Subclaim 2: Attribute of Y
                subclaims.append(f"{y_entity_cleaned} {attribute}")
                
                # Subclaim 3: Comparison phrase
                subclaims.append(f"{x_entity_cleaned} {comparison_word} als {y_entity_cleaned}")
                
                return subclaims[:4]  # Return max 4
        
        # B) Numeric claims (... beträgt Y Einheit / ... ist Y)
        # BUT: Skip if this is a definition-like time-unit claim (e.g., "jahr hat 10 monate")
        # These should not be decomposed as they produce garbage subclaims
        time_unit_keywords = ["jahr", "monat", "woche", "tag", "stunde", "minute", "sekunde"]
        time_unit_objects = ["monat", "monate", "woche", "wochen", "tag", "tage", "stunde", "stunden", "minute", "minuten", "sekunde", "sekunden"]
        is_time_unit_definition = False
        if parsed_claim.subject:
            subject_lower = parsed_claim.subject.lower()
            # Remove articles
            articles = ["der", "die", "das", "ein", "eine", "einen", "einem", "einer"]
            subject_words = subject_lower.split()
            if subject_words and subject_words[0] in articles:
                subject_clean = " ".join(subject_words[1:]) if len(subject_words) > 1 else subject_words[0] if subject_words else ""
            else:
                subject_clean = subject_lower
            # Check if subject is a time unit
            if any(kw in subject_clean for kw in time_unit_keywords):
                # Check if object is also a time unit (e.g., "jahr hat monate")
                if parsed_claim.object:
                    object_lower = parsed_claim.object.lower()
                    if any(kw in object_lower for kw in time_unit_objects):
                        is_time_unit_definition = True
        
        numeric_patterns = [
            r'(.+?)\s+(beträgt|ist|war|sind|waren)\s+(\d+[.,]?\d*)\s*([kmcmg%°cmio|million|milliarden]+)?',
            r'(.+?)\s+(hat|hatte|haben|hatten)\s+(\d+[.,]?\d*)\s*([kmcmg%°cmio|million|milliarden]+)?',
        ]
        
        for pattern in numeric_patterns:
            match = re.search(pattern, claim_lower)
            if match:
                # Skip if this is a time-unit definition (will be handled as definition-like)
                if is_time_unit_definition:
                    return []
                
                main_object = match.group(1).strip()
                number = match.group(3).strip() if len(match.groups()) >= 3 else ""
                # group(4) is optional and can be None even if the group exists
                unit = (match.group(4) or "").strip() if len(match.groups()) >= 4 else ""
                
                if main_object and number:
                    # Extract attribute from claim (heuristic: look for key terms)
                    attribute = ""
                    
                    # Check for population-related keywords in claim
                    population_keywords = ["einwohner", "bevölkerung", "population"]
                    if any(kw in claim_lower for kw in population_keywords):
                        # For population claims, use "Einwohner" or "Bevölkerung" as attribute
                        if "einwohner" in claim_lower:
                            attribute = "Einwohner"
                        elif "bevölkerung" in claim_lower:
                            attribute = "Bevölkerung"
                        else:
                            attribute = "Einwohner"  # Default
                    elif parsed_claim.subject:
                        # Try to extract attribute from predicate or object
                        if parsed_claim.predicate:
                            attribute = parsed_claim.predicate
                        elif parsed_claim.object:
                            attribute = parsed_claim.object
                    
                    # Clean main_object (remove determiners like "der", "die", "das")
                    main_object_clean = re.sub(r'^(der|die|das|ein|eine|einen|einem|einer)\s+', '', main_object, flags=re.IGNORECASE).strip()
                    
                    # For population claims, simplify subclaims to improve Wikipedia search
                    # Instead of "portugal Einwohner 83 millionen", use just "portugal" or "portugal Einwohner"
                    is_population_claim = any(kw in claim_lower for kw in ["einwohner", "bevölkerung", "population"])
                    
                    if is_population_claim:
                        # Subclaim 1: Just the entity (e.g., "portugal") - this will find the main article
                        subclaims.append(main_object_clean)
                        # Subclaim 2: Entity + attribute (e.g., "portugal Einwohner") - for more specific search
                        if attribute:
                            subclaims.append(f"{main_object_clean} {attribute}")
                        # Don't include the number in subclaims - it makes the query too specific
                    else:
                        # For non-population claims, use original logic
                        # Subclaim 1: Main object + attribute (without number) - this is the key query
                        if attribute:
                            subclaims.append(f"{main_object_clean} {attribute}")
                        else:
                            subclaims.append(main_object_clean)
                        
                        # Subclaim 2: Main object + attribute + number + unit (for verification)
                        if unit:
                            subclaims.append(f"{main_object_clean} {attribute} {number} {unit}".strip())
                        elif attribute:
                            subclaims.append(f"{main_object_clean} {attribute} {number}".strip())
                        else:
                            subclaims.append(f"{main_object_clean} {number}".strip())
                    
                    return subclaims[:4]
        
        # C) Temporal claims (ist/war/seit + Rolle)
        if parsed_claim.role and parsed_claim.subject:
            # Subclaim 1: Without time reference
            subclaims.append(f"{parsed_claim.subject} {parsed_claim.role}")
            
            # Subclaim 2: With time anchor if available
            if parsed_claim.year:
                subclaims.append(f"{parsed_claim.subject} {parsed_claim.role} {parsed_claim.year}")
            elif "seit" in claim_lower:
                # Extract "seit" phrase
                seit_match = re.search(r'seit\s+(\d{4})', claim_lower)
                if seit_match:
                    year = seit_match.group(1)
                    subclaims.append(f"{parsed_claim.subject} {parsed_claim.role} seit {year}")
            
            if subclaims:
                return subclaims[:4]
        
        # Fallback: If no rules match, return empty list (no decomposition)
        return []


# Module-level convenience functions
def parse_claim(claim: str) -> ParsedClaim:
    """Convenience function to parse a claim."""
    parser = ClaimParser()
    return parser.parse_claim(claim)


def parse_evidence(evidence_text: str) -> ParsedEvidence:
    """Convenience function to parse evidence."""
    parser = ClaimParser()
    return parser.parse_evidence(evidence_text)

