"""
Async Wikipedia evidence retriever with semantic reranking and query expansion.

Usage example:
    import aiohttp
    from app.pipeline.wikipedia_client import WikipediaClient
    from app.pipeline.retriever import Retriever
    from app.pipeline.models import ParsedClaim
    
    async with aiohttp.ClientSession() as session:
        wiki_client = WikipediaClient(session)
        retriever = Retriever(wiki_client)
        
        parsed_claim = ParsedClaim(raw="Angela Merkel ist Bundeskanzlerin")
        evidences = await retriever.retrieve(
            claim="Angela Merkel ist Bundeskanzlerin",
            parsed_claim=parsed_claim,
            max_pages=3
        )
"""
import asyncio
import re
import logging
from typing import List, Optional, Callable, Any

from app.pipeline.wikipedia_client import WikipediaClient
from app.pipeline.models import Evidence, ParsedClaim
from app.pipeline.semantic_reranker import SemanticReranker
from app.pipeline.query_expander import QueryExpander
from app.pipeline.utils import tokenize_text, split_into_sentences

logger = logging.getLogger(__name__)


class Retriever:
    """
    Async Wikipedia evidence retriever.
    Retrieves article summaries in parallel for fast evidence collection.
    """
    
    def __init__(
        self,
        wikipedia_client: WikipediaClient,
        enable_semantic_reranking: bool = True,
        enable_query_expansion: bool = False,  # Default OFF, only as fallback
        enable_crossencoder_rerank: bool = False,
        crossencoder_prune_top_n: int = 50,
        relevance_scorer: Optional[Callable[[str, List[str]], List[float]]] = None,
        embedding_model: Optional[Any] = None,
    ):
        """
        Initialize retriever with a Wikipedia client.
        
        Args:
            wikipedia_client: WikipediaClient instance for API access
            enable_semantic_reranking: Enable semantic reranking (requires sentence-transformers)
            enable_query_expansion: Enable query expansion with synonyms (default: False, used as fallback)
            enable_crossencoder_rerank: Enable cross-encoder reranking stage (Stage 2)
            crossencoder_prune_top_n: Number of candidates to rerank with cross-encoder (default: 50)
            relevance_scorer: Optional callable for relevance scoring (claim, sentences) -> scores
                             If None and enable_crossencoder_rerank=True, Stage 2 is skipped
            embedding_model: Optional pre-loaded SentenceTransformer model (shared instance)
        """
        self.wikipedia_client = wikipedia_client
        self.enable_semantic_reranking = enable_semantic_reranking
        self.enable_query_expansion = enable_query_expansion  # Only used as fallback flag
        self.enable_crossencoder_rerank = enable_crossencoder_rerank
        self.crossencoder_prune_top_n = crossencoder_prune_top_n
        self.relevance_scorer = relevance_scorer
        
        # Initialize semantic reranker with shared embedding model
        self._reranker: Optional[SemanticReranker] = None
        if enable_semantic_reranking:
            try:
                self._reranker = SemanticReranker(embedding_model=embedding_model)
            except Exception as e:
                logger.warning(f"RETRIEVER: Failed to initialize semantic reranker: {e}")
                self._reranker = None
        
        # Initialize query expander (lazy, only used as fallback)
        self._query_expander: Optional[QueryExpander] = None
    
    def _build_search_query(self, claim: str, parsed_claim: ParsedClaim) -> str:
        """
        Build a clean search query from ParsedClaim.
        
        Args:
            claim: Original claim text (fallback)
            parsed_claim: Parsed claim with structured fields
            
        Returns:
            Clean search query string (max ~8-10 tokens or ~80 chars)
        """
        query_parts = []

        # Special handling: calendar fact "Ein Jahr hat zwölf Monate" (avoid film titles like "Die zwölf Monate (2012)")
        claim_lower_global = claim.lower()
        if ("jahr" in claim_lower_global and ("monat" in claim_lower_global or "monate" in claim_lower_global)
            and ("zwölf" in claim_lower_global or "12" in claim_lower_global)):
            # Prefer encyclopedia articles over phrase-matching titles
            return "Kalenderjahr Monate"
        
        # Early path for population claims: build deterministic "<Subject> Einwohner" query
        claim_lower_global = claim.lower()
        is_population_claim_global = any(kw in claim_lower_global for kw in ["einwohner", "bevölkerung", "population"])
        if is_population_claim_global and parsed_claim.subject:
            subject_clean = re.sub(r"[^\w\s]", "", parsed_claim.subject).strip()
            subject_lower = subject_clean.lower()
            # Normalize common country variants (e.g., "vereinigten staaten" -> "Vereinigte Staaten")
            if ("staaten" in subject_lower and ("vereinigte" in subject_lower or "vereinigten" in subject_lower)):
                subject_clean = "Vereinigte Staaten"
            elif subject_lower in {"usa", "u.s.a", "u.s.a.", "us"}:
                subject_clean = "Vereinigte Staaten"
            # Return strict ordered query to avoid token reordering
            return f"{subject_clean} Einwohner"
        
        # Special handling for comparison subclaims (e.g., "matterhorn höhe")
        # For these, extract just the entity part (remove attribute words)
        claim_lower = claim.lower()
        attribute_words = {"höhe", "höhen", "länge", "läng", "größe", "groß", "anzahl", "alter", "jahr", "tore", "regierungsjahre", "titel"}
        is_comparison_subclaim = (
            parsed_claim.category == "comparison" and 
            any(attr in claim_lower for attr in attribute_words) and
            len(claim.split()) <= 3  # Short subclaims like "matterhorn höhe"
        )
        
        if is_comparison_subclaim:
            # Extract entity part (remove attribute words)
            tokens = claim.split()
            entity_tokens = [t for t in tokens if t.lower() not in attribute_words]
            if entity_tokens:
                # Use just the entity for better Wikipedia search results
                entity_query = " ".join(entity_tokens)
                query_parts.append(entity_query)
            else:
                # Fallback: use first token
                query_parts.append(tokens[0] if tokens else claim)
            
            # For comparison subclaims, return early to avoid adding attribute words back
            # Deduplicate and return the query
            seen_lower = set()
            deduplicated_parts = []
            for part in query_parts:
                part_lower = part.lower().strip()
                if not part_lower:
                    continue
                if part_lower in seen_lower:
                    continue
                seen_lower.add(part_lower)
                deduplicated_parts.append(part)
            
            query = " ".join(deduplicated_parts)
            return query.strip()
        # For comparison claims, extract both entities being compared
        elif parsed_claim.category == "comparison":
            # Extract both entities from claim (e.g., "Der Nil" and "der Rhein")
            # Find "als" to split entities
            if " als " in claim_lower:
                parts = claim.split(" als ")
                if len(parts) == 2:
                    # First entity (subject)
                    entity1 = parts[0].strip()
                    # Second entity (after "als")
                    entity2_parts = parts[1].split()
                    # Remove predicate words
                    entity2_words = [w for w in entity2_parts if w.lower() not in ["ist", "war", "sind", "waren"]]
                    entity2 = " ".join(entity2_words).rstrip(".,")
                    # Add both entities to query
                    query_parts.append(entity1)
                    query_parts.append(entity2)
        
        # CRITICAL: Check for compound terms like "Vitamin D" BEFORE subject processing
        # These are important medical/chemical terms that should be included in the query
        # Pattern: Two consecutive capitalized words (e.g., "Vitamin D", "Vitamin C", "Vitamin B12")
        claim_words = claim.split()
        compound_terms = []
        for i in range(len(claim_words) - 1):
            word1_clean = re.sub(r"[^\w]", "", claim_words[i])
            word2_clean = re.sub(r"[^\w]", "", claim_words[i + 1])
            # Check if both words are capitalized (or first is capitalized and second is single letter/number)
            # Examples: "Vitamin D", "Vitamin C", "Vitamin B12", "Vitamin K"
            if (word1_clean and word1_clean[0].isupper() and len(word1_clean) > 2 and
                word2_clean and (word2_clean[0].isupper() or (len(word2_clean) <= 2 and word2_clean[0].isalnum()))):
                compound_term = f"{word1_clean} {word2_clean}"
                compound_terms.append(compound_term)
        
        # Prefer structured fields in order: subject, role, party, object, event, sport
        if parsed_claim.subject:
            # Check if subject is generic (e.g., "am", "der", "die", "das", "ein", "eine")
            # If so, skip adding it to query and rely on enrichment/compound terms
            subject_lower = parsed_claim.subject.lower().strip()
            generic_subjects = {"am", "der", "die", "das", "ein", "eine", "einen", "einem", "einer", "dem", "den"}
            is_generic_subject = subject_lower in generic_subjects or len(subject_lower) <= 2
            
            if not is_generic_subject:
                # Keep subject as-is (may be multi-word like "Angela Merkel")
                # For single-word subjects that look like surnames, try to get full name from claim
                subject_clean = re.sub(r"[^\w\s]", "", parsed_claim.subject).strip()
                if subject_clean:
                    # If subject is single word and claim contains capitalized words, try to get full name
                    subject_words = subject_clean.split()
                    if len(subject_words) == 1 and len(subject_words[0]) > 3:
                        # Try to find full name in claim (look for capitalized words before/after subject)
                        claim_tokens = claim.split()
                        subject_idx = -1
                        for i, token in enumerate(claim_tokens):
                            if subject_words[0].lower() in token.lower():
                                subject_idx = i
                                break
                        if subject_idx >= 0:
                            # Look for capitalized word before subject (first name)
                            if subject_idx > 0:
                                prev_token = claim_tokens[subject_idx - 1]
                                if prev_token and prev_token[0].isupper() and len(prev_token) > 2:
                                    subject_clean = f"{prev_token} {subject_clean}"
                    query_parts.append(subject_clean)
                
                # Special handling for population claims: simplify query to improve search results
                # Don't add too many keywords - just the subject is usually enough to find the main article
                # The demographic keywords will be found in the article content, not needed in search query
                claim_lower = claim.lower()
                is_population_claim = any(kw in claim_lower for kw in ["einwohner", "bevölkerung", "population"])
                
                # For population claims, we rely on the subject alone to find the main article
                # The article content will contain "Bevölkerung" and "Demografie" sections
                # Adding these to the search query makes it too specific and reduces recall
                # Only add "Einwohner" if it's explicitly in the claim and the query is very short
                if is_population_claim and len(query_parts) <= 1:
                    # If query is just the subject, add "Einwohner" to help find relevant articles
                    if "einwohner" in claim_lower:
                        query_parts.append("Einwohner")
                    elif "bevölkerung" in claim_lower:
                        query_parts.append("Bevölkerung")
        
        # Add object early if available (important for queries like "Photosynthese findet in Chloroplasten statt")
        # For queries like "Die Erde umkreist die Sonne", we need the full object phrase
        if parsed_claim.object:
            object_clean = re.sub(r"[^\w\s]", "", parsed_claim.object).strip()
            if object_clean:
                query_parts.append(object_clean)
        
        # Also extract object from claim directly if not captured (for complex objects like "Wasserstoff und Sauerstoff")
        # This helps with queries where object parsing missed compound objects
        if not parsed_claim.object or len(parsed_claim.object.split()) <= 1:
            # Try to extract object phrase from claim (words after predicate/verb)
            claim_lower = claim.lower()
            # Common predicates (including verbs like "umkreist", "fließt durch")
            predicate_markers = [
                "besteht aus", "besteht", "hat", "hatte", "enthält", "enthalten",
                "umkreist", "durchquert", "fließt durch", "durchfließt", "liegt in", "befindet sich in"
            ]
            for marker in predicate_markers:
                if marker in claim_lower:
                    # Extract everything after the marker from ORIGINAL claim (preserve capitalization)
                    original_claim_parts = claim.split(marker, 1)
                    if len(original_claim_parts) > 1:
                        original_object_phrase = original_claim_parts[1].strip().rstrip(".,!?")
                        # Clean but preserve important determiners before capitalized nouns
                        # Pattern: "die Sonne" -> keep "die Sonne" or at least "Sonne"
                        original_object_clean = re.sub(r"[^\w\s]", "", original_object_phrase).strip()
                        if original_object_clean and len(original_object_clean) >= 3:
                            object_tokens = original_object_clean.split()
                            
                            # Strategy 1: If we have "die/das/der" + capitalized noun, keep both
                            if len(object_tokens) >= 2:
                                first_token_lower = object_tokens[0].lower()
                                second_token = object_tokens[1]
                                if first_token_lower in ["der", "die", "das", "ein", "eine"] and second_token[0].isupper():
                                    # Keep determiner + noun (e.g., "die Sonne")
                                    query_parts.append(f"{object_tokens[0]} {second_token}")
                                    # Also add just the noun for better matching
                                    query_parts.append(second_token)
                                else:
                                    # No determiner pattern, just add capitalized tokens
                                    capitalized_tokens = [t for t in object_tokens if t and t[0].isupper()]
                                    if capitalized_tokens:
                                        query_parts.extend(capitalized_tokens[:3])
                            else:
                                # Single token: add if capitalized or important
                                if object_tokens[0][0].isupper() or len(object_tokens[0]) > 4:
                                    query_parts.append(object_tokens[0])
                            
                            # Strategy 2: Also handle compound objects (e.g., "Wasserstoff und Sauerstoff")
                            # Extract all capitalized tokens and important terms
                            all_capitalized = [t for t in object_tokens if t and t[0].isupper()]
                            if len(all_capitalized) >= 2:
                                # Multiple entities: add each separately
                                query_parts.extend(all_capitalized[:4])  # Up to 4 entities
                            elif len(all_capitalized) == 1:
                                # Single entity: already added above
                                pass
                            
                            # Strategy 3: Add important lowercase terms (length > 4, not stopwords)
                            important_lowercase = [
                                t for t in object_tokens 
                                if len(t) > 4 and t[0].islower() and t.lower() not in ["und", "oder", "aus", "durch", "in"]
                            ]
                            if important_lowercase:
                                query_parts.extend(important_lowercase[:2])
                            
                            break
        
        if parsed_claim.role:
            role_clean = re.sub(r"[^\w\s]", "", parsed_claim.role).strip()
            if role_clean:
                query_parts.append(role_clean)
        
        if parsed_claim.party:
            party_clean = re.sub(r"[^\w\s]", "", parsed_claim.party).strip()
            if party_clean:
                query_parts.append(party_clean)
        
        # Add event early if available (important for sports claims like "Bayern München gewann Champions League")
        if parsed_claim.event:
            event_clean = re.sub(r"[^\w\s]", "", parsed_claim.event).strip()
            if event_clean:
                query_parts.append(event_clean)
        
        # For sports_title claims, also try to extract event from claim if not already extracted
        if parsed_claim.category == "sports_title" and not parsed_claim.event:
            claim_lower = claim.lower()
            # Look for common sports events in claim
            sports_events = ["champions league", "weltmeisterschaft", "wm", "bundesliga", "olympia"]
            for event in sports_events:
                if event in claim_lower:
                    # Extract the full event name from claim
                    words = claim.split()
                    for i, word in enumerate(words):
                        if event in word.lower():
                            # Try to get full event phrase (e.g., "Champions League")
                            if i < len(words) - 1:
                                event_phrase = f"{word} {words[i+1]}"
                                query_parts.append(event_phrase)
                            else:
                                query_parts.append(word)
                            break
                    break
        
        if parsed_claim.sport:
            sport_clean = re.sub(r"[^\w\s]", "", parsed_claim.sport).strip()
            if sport_clean:
                query_parts.append(sport_clean)
        
        # Add year if available
        if parsed_claim.year:
            query_parts.append(str(parsed_claim.year))
        
        # Deduplicate query parts (case-insensitive, preserve first occurrence)
        # Also handle cases where we have both "die Sonne" and "Sonne" (keep both if meaningful)
        seen_lower = set()
        deduplicated_parts = []
        for part in query_parts:
            part_lower = part.lower().strip()
            if not part_lower:
                continue
            
            # Check for exact duplicate (case-insensitive)
            if part_lower in seen_lower:
                continue
            
            # Check for substring duplicates (e.g., "die sonne" contains "sonne")
            # Strategy: if we already have a longer phrase containing this part, skip
            # But if this is longer, replace the shorter one
            is_substring = False
            for existing in deduplicated_parts:
                existing_lower = existing.lower()
                if part_lower in existing_lower and part_lower != existing_lower:
                    # This part is substring of existing, skip
                    is_substring = True
                    break
                elif existing_lower in part_lower and part_lower != existing_lower:
                    # Existing is substring of this part, replace it
                    deduplicated_parts.remove(existing)
                    seen_lower.discard(existing_lower)
                    break
            
            if not is_substring:
                seen_lower.add(part_lower)
                deduplicated_parts.append(part)
        
        query_parts = deduplicated_parts
        
        # Limit query length: max ~8-10 tokens or ~80 chars
        query = " ".join(query_parts)
        if len(query) > 80:
            # Truncate to ~80 chars, but try to keep word boundaries
            truncated = query[:80]
            last_space = truncated.rfind(" ")
            if last_space > 50:  # Only truncate at word boundary if reasonable
                query = truncated[:last_space]
            else:
                query = truncated 
        
        # If structured fields produced sparse query (only subject + year), enrich it
        # CRITICAL: Also enrich if subject is generic (e.g., "am", "der", "die") - these need better queries
        subject_lower_check = parsed_claim.subject.lower().strip() if parsed_claim.subject else ""
        generic_subjects_check = {"am", "der", "die", "das", "ein", "eine", "einen", "einem", "einer", "dem", "den"}
        is_generic_subject_check = subject_lower_check in generic_subjects_check or len(subject_lower_check) <= 2
        
        query_tokens = query.split()
        if len(query_tokens) <= 3 and (parsed_claim.subject or is_generic_subject_check):  # Changed threshold from 2 to 3, also check generic subjects
            # Query is too sparse, extract more keywords from claim
            # Get already added tokens (case-insensitive) to avoid duplicates
            already_added_lower = {part.lower() for part in query_parts}
            
            claim_tokens = tokenize_text(claim)
            # Remove subject tokens and year from claim tokens (only if subject is not generic)
            subject_tokens = set()
            if parsed_claim.subject and not is_generic_subject_check:
                subject_tokens = set(tokenize_text(parsed_claim.subject))
            claim_tokens_filtered = [
                t for t in claim_tokens 
                if t not in subject_tokens 
                and (not parsed_claim.year or t != str(parsed_claim.year))
                and t.lower() not in already_added_lower  # Avoid duplicates
            ]
            
            # CRITICAL: For generic subjects, prioritize compound terms (e.g., "Vitamin D")
            # This ensures "Vitamin D" is included even if subject is "am"
            if is_generic_subject_check and compound_terms:
                # Add compound terms first (e.g., "Vitamin D")
                for compound in compound_terms:
                    if compound.lower() not in already_added_lower:
                        query_parts.append(compound)
                        already_added_lower.add(compound.lower())
            
            # Prioritize capitalized nouns (likely important terms like "Chloroplasten", "Monate", "Sonne", "Vitamin")
            capitalized_tokens = [
                t for t in claim_tokens_filtered 
                if len(t) > 3 and t[0].isupper()  # Changed min_length from 4 to 3 to catch "Monate"
            ]
            if capitalized_tokens:
                # Add capitalized tokens first (up to 3, increased from 2)
                new_capitalized = [t for t in capitalized_tokens[:3] if t.lower() not in already_added_lower]
                query_parts.extend(new_capitalized)
                # Then add other important tokens (nouns, not just stopwords)
                remaining = [
                    t for t in claim_tokens_filtered 
                    if t not in capitalized_tokens 
                    and len(t) > 4 
                    and t.lower() not in already_added_lower
                ]
                query_parts.extend(remaining[:2])  # Increased from 1 to 2
            else:
                # Add up to 4 most relevant tokens from claim (increased from 3)
                important_tokens = [
                    t for t in claim_tokens_filtered 
                    if len(t) > 3 
                    and t.lower() not in already_added_lower
                ]  # Filter short tokens
                query_parts.extend(important_tokens[:4])
            
            # Deduplicate again after enrichment (using improved deduplication logic)
            seen_lower_enriched = set()
            deduplicated_parts_enriched = []
            for part in query_parts:
                part_lower = part.lower().strip()
                if not part_lower:
                    continue
                
                if part_lower in seen_lower_enriched:
                    continue
                
                # Check for substring duplicates
                is_substring = False
                for existing in deduplicated_parts_enriched:
                    existing_lower = existing.lower()
                    if part_lower in existing_lower and part_lower != existing_lower:
                        is_substring = True
                        break
                    elif existing_lower in part_lower and part_lower != existing_lower:
                        deduplicated_parts_enriched.remove(existing)
                        seen_lower_enriched.discard(existing_lower)
                        break
                
                if not is_substring:
                    seen_lower_enriched.add(part_lower)
                    deduplicated_parts_enriched.append(part)
            
            query_parts = deduplicated_parts_enriched
            
            query = " ".join(query_parts)
            # Re-apply length limit
            if len(query) > 80:
                truncated = query[:80]
                last_space = truncated.rfind(" ")
                if last_space > 50:
                    query = truncated[:last_space]
                else:
                    query = truncated
        
        # CRITICAL: Before final fallback, check for compound terms like "Vitamin D", "Vitamin C", etc.
        # These are important medical/chemical terms that should be included in the query
        # Pattern: Two consecutive capitalized words (e.g., "Vitamin D", "Vitamin C", "Vitamin B12")
        claim_words = claim.split()
        compound_terms = []
        for i in range(len(claim_words) - 1):
            word1_clean = re.sub(r"[^\w]", "", claim_words[i])
            word2_clean = re.sub(r"[^\w]", "", claim_words[i + 1])
            # Check if both words are capitalized (or first is capitalized and second is single letter/number)
            if (word1_clean and word1_clean[0].isupper() and len(word1_clean) > 2 and
                word2_clean and (word2_clean[0].isupper() or (len(word2_clean) <= 2 and word2_clean[0].isalnum()))):
                compound_term = f"{word1_clean} {word2_clean}"
                compound_terms.append(compound_term)
        
        # Add compound terms to query if not already present
        query_lower = query.lower()
        for compound in compound_terms:
            if compound.lower() not in query_lower:
                query_parts.append(compound)
                query = " ".join(query_parts)
                break  # Only add first compound term to avoid query bloat
        
        # If structured fields produced empty query, fall back to tokenized claim
        if not query or len(query.strip()) == 0:
            # Better fallback: extract key terms (capitalized words, important nouns)
            claim_tokens = claim.split()
            key_terms = []
            # Add capitalized tokens (proper nouns, entities)
            for token in claim_tokens:
                token_clean = re.sub(r"[^\w]", "", token)
                if token_clean and token_clean[0].isupper() and len(token_clean) > 2:
                    key_terms.append(token_clean)
            # If we have key terms, use them
            if key_terms:
                query = " ".join(key_terms[:6])  # Limit to 6 key terms
            else:
                # Fallback to tokenized claim
                tokens = tokenize_text(claim)
                query = " ".join(tokens[:10])
        
        # Final fallback: if still empty, use original claim (truncated)
        if not query or len(query.strip()) == 0:
            query = claim[:80] if len(claim) > 80 else claim
        
        return query.strip()
    
    @staticmethod
    def _normalize_german_numbers(text: str) -> str:
        """
        Normalize German number words to digits (improves lexical overlap).
        
        Maps: zwölf -> 12, elf -> 11, zehn -> 10, etc. (0-20 + tens)
        """
        if not text:
            return text
        
        # German number word mappings (0-20 + tens)
        number_map = {
            "null": "0", "eins": "1", "zwei": "2", "drei": "3", "vier": "4",
            "fünf": "5", "sechs": "6", "sieben": "7", "acht": "8", "neun": "9",
            "zehn": "10", "elf": "11", "zwölf": "12", "dreizehn": "13", "vierzehn": "14",
            "fünfzehn": "15", "sechzehn": "16", "siebzehn": "17", "achtzehn": "18",
            "neunzehn": "19", "zwanzig": "20",
            "dreißig": "30", "vierzig": "40", "fünfzig": "50", "sechzig": "60",
            "siebzig": "70", "achtzig": "80", "neunzig": "90",
        }
        
        normalized = text
        for word, digit in number_map.items():
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(word) + r'\b'
            normalized = re.sub(pattern, digit, normalized, flags=re.IGNORECASE)
        
        return normalized
    
    def _best_sentence(
        self, claim: str, text: str, parsed_claim: Optional[ParsedClaim] = None, max_len: int = 300
    ) -> Optional[str]:
        """
        Select the best sentence from text based on weighted token overlap with claim.
        Prioritizes predicate content (e.g., "energy conversion") over subject-only matches.
        
        Args:
            claim: The claim text to match against
            text: The text to extract sentences from
            parsed_claim: Optional parsed claim (for subject extraction)
            max_len: Maximum length of sentence to return
            
        Returns:
            Best matching sentence, or None if no suitable sentence found
        """
        if not claim or not text:
            return None
        
        # Tokenize claim
        claim_tokens = set(tokenize_text(claim))
        
        if not claim_tokens:
            return None
        
        # Extract subject tokens if available
        subject_tokens = set()
        if parsed_claim and parsed_claim.subject:
            subject_tokens = set(tokenize_text(parsed_claim.subject))
        
        # Predicate tokens = claim tokens minus subject tokens (informative words)
        predicate_tokens = claim_tokens - subject_tokens
        
        # Energy-related keywords for bonus scoring
        energy_keywords = ["energie", "atp", "zellatmung", "oxidative", "respiration", "stoffwechsel"]
        
        # Split text into sentences (simple regex on [.?!])
        sentences = re.split(r"[.?!]+", text)
        
        if not sentences:
            return None
        
        # Score sentences with weighted overlap
        scored_sentences = []
        
        # Extract role/position keywords from claim (e.g., "Kapitän", "captain")
        role_keywords = []
        if parsed_claim and parsed_claim.role:
            role_keywords = tokenize_text(parsed_claim.role)
        # Also extract from claim text directly
        claim_lower = claim.lower()
        if "kapitän" in claim_lower or "captain" in claim_lower:
            role_keywords.extend(["kapitän", "captain"])
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Skip very long sentences
            if len(sentence) > max_len:
                continue
            
            # Tokenize sentence
            sentence_tokens = set(tokenize_text(sentence))
            sentence_lower = sentence.lower()
            
            # Weighted scoring
            score = 0
            
            # Subject overlap: 1 point per token
            subject_overlap = len(sentence_tokens & subject_tokens)
            score += 1 * subject_overlap
            
            # CRITICAL: Require subject to be present for meaningful match
            if subject_overlap == 0 and subject_tokens:
                # Skip sentences that don't mention the subject at all
                continue
            
            # Predicate overlap: 4 points per token (much more important)
            predicate_overlap = len(sentence_tokens & predicate_tokens)
            score += 4 * predicate_overlap
            
            # Role/position keyword bonus: +5 if sentence contains role keywords
            if role_keywords:
                role_match = any(role_kw in sentence_lower for role_kw in role_keywords)
                if role_match:
                    score += 5
                    # Extra bonus if both subject AND role are present
                    if subject_overlap > 0:
                        score += 3
            
            # Year bonus: if claim has year, check if sentence mentions it
            if parsed_claim and parsed_claim.year:
                year_str = str(parsed_claim.year)
                if year_str in sentence:
                    score += 5  # Increased bonus for year match
            
            # Event/Sports keyword bonus: if claim has event, check if sentence mentions it
            if parsed_claim and parsed_claim.event:
                event_lower = parsed_claim.event.lower()
                event_tokens = set(tokenize_text(parsed_claim.event))
                if any(token in sentence_lower for token in event_tokens):
                    score += 5  # Strong bonus for event match
            
            # Sports action words bonus (gewann, siegte, etc.)
            sports_action_words = ["gewann", "gewonnen", "siegte", "sieger", "weltmeister", "meister", "titel"]
            claim_has_action = any(word in claim_lower for word in sports_action_words)
            sentence_has_action = any(word in sentence_lower for word in sports_action_words)
            if claim_has_action and sentence_has_action:
                score += 5  # Strong bonus if both claim and sentence mention action
            
            # Energy keyword bonus: +2 if sentence contains any energy keyword
            if any(keyword in sentence_lower for keyword in energy_keywords):
                score += 2
            
            scored_sentences.append({
                "sentence": sentence,
                "score": score,
                "predicate_overlap": predicate_overlap,
                "subject_overlap": subject_overlap,
            })
        
        if not scored_sentences:
            return None
        
        # Sort by score descending
        scored_sentences.sort(key=lambda x: x["score"], reverse=True)
        
        best_entry = scored_sentences[0]
        best_score = best_entry["score"]
        
        # Require minimum score: must have subject overlap OR high predicate overlap
        min_score = 2 if subject_tokens else 4
        
        # If best score meets threshold, return it
        if best_score >= min_score:
            return best_entry["sentence"].strip()
        
        # Fallback: if we have predicate tokens and energy keywords, try that
        if predicate_tokens:
            for entry in scored_sentences:
                sentence_lower = entry["sentence"].lower()
                if any(keyword in sentence_lower for keyword in energy_keywords):
                    return entry["sentence"].strip()
        
        # Last resort: return best sentence if it has any score and mentions subject
        if best_score > 0 and best_entry.get("subject_overlap", 0) > 0:
            return best_entry["sentence"].strip()
        
        return None
    
    async def retrieve(
        self,
        claim: str,
        parsed_claim: ParsedClaim,
        max_pages: int = 3,
    ) -> List[Evidence]:
        """
        Retrieve Wikipedia evidence for a claim using sentence-level retrieval (SentenceBank).
        
        This method:
        1. Searches Wikipedia articles
        2. Extracts sentences from article extracts (15-30 per article)
        3. Builds a SentenceBank (all sentences from all articles)
        4. Prunes via lexical overlap (top 50)
        5. Reranks semantically if enabled
        6. Returns top_k sentence-level Evidence objects
        
        Args:
            claim: The claim text to search for
            parsed_claim: Parsed claim object with structured fields
            max_pages: Maximum number of article pages to retrieve
            
        Returns:
            List of Evidence objects where each Evidence.snippet contains a single sentence.
            Each Evidence has title, url, snippet (sentence), and optional sentence_index.
            Returns empty list on error or if no results found.
        """
        if not claim:
            return []
        
        try:
            # Build clean search query from ParsedClaim
            base_query = self._build_search_query(claim, parsed_claim)
            logger.info(f"WIKI_QUERY: claim='{claim}' base_query='{base_query}'")
            
            # Debug: Log if this is a population claim
            claim_lower_debug = claim.lower()
            is_population_claim_debug = any(kw in claim_lower_debug for kw in ["einwohner", "bevölkerung", "population"])
            if is_population_claim_debug:
                logger.info(f"POPULATION_CLAIM_DEBUG: Population claim detected. Query includes 'Demografie'={('demografie' in base_query.lower())}")
            
            # Step 1: Search with base query (query expansion disabled by default)
            queries_to_search = [base_query]
            all_search_results = []
            seen_titles = set()
            
            # Initial search with base query
            logger.info(f"RETRIEVER_DEBUG: Searching with query='{base_query}' limit={max_pages * 2}")
            search_results = await self.wikipedia_client.search(query=base_query, limit=max_pages * 2)
            logger.info(f"RETRIEVER_DEBUG: Initial search returned {len(search_results)} results")
            
            for result in search_results:
                title = result.get("title")
                if title and title not in seen_titles:
                    all_search_results.append(result)
                    seen_titles.add(title)
                    logger.debug(f"RETRIEVER_DEBUG: Added result title='{title}'")
            
            # For population claims, if initial search fails, try simpler query (just subject)
            if is_population_claim_debug and len(all_search_results) == 0 and parsed_claim.subject:
                logger.warning(f"POPULATION_CLAIM_DEBUG: Initial search returned 0 results, trying simpler query with just subject...")
                subject_clean = re.sub(r"[^\w\s]", "", parsed_claim.subject).strip()
                if subject_clean:
                    simple_query = subject_clean
                    logger.info(f"POPULATION_CLAIM_DEBUG: Trying simple query='{simple_query}'")
                    simple_search_results = await self.wikipedia_client.search(query=simple_query, limit=max_pages * 2)
                    logger.info(f"POPULATION_CLAIM_DEBUG: Simple query returned {len(simple_search_results)} results")
                    for result in simple_search_results:
                        title = result.get("title")
                        if title and title not in seen_titles:
                            all_search_results.append(result)
                            seen_titles.add(title)
                            logger.debug(f"RETRIEVER_DEBUG: Added result from simple query title='{title}'")
            
            if len(all_search_results) == 0:
                logger.error(f"RETRIEVER_ERROR: No search results found for query='{base_query}' (claim='{claim[:60]}...')")
                logger.error(f"RETRIEVER_ERROR: parsed_claim.subject='{parsed_claim.subject if parsed_claim.subject else 'None'}'")
            
            # Step 2: Fallback to query expansion if low recall (< 3 pages or < 5 results)
            if len(all_search_results) < 3 and self._query_expander is None:
                # Lazy initialization of query expander for fallback
                self._query_expander = QueryExpander()
            
            if len(all_search_results) < 3 and self._query_expander:
                logger.info(
                    f"QUERY_EXPAND_FALLBACK: Low recall ({len(all_search_results)} results), "
                    f"expanding query as fallback"
                )
                expanded_queries = self._query_expander.expand_query(
                    base_query, parsed_claim=parsed_claim, max_variants=2
                )
                # Search with expanded queries (skip first since it's the base query)
                for query_variant in expanded_queries[1:]:
                    variant_results = await self.wikipedia_client.search(query=query_variant, limit=max_pages)
                    for result in variant_results:
                        title = result.get("title")
                        if title and title not in seen_titles:
                            all_search_results.append(result)
                            seen_titles.add(title)
                            if len(all_search_results) >= max_pages * 2:
                                break
                    if len(all_search_results) >= max_pages * 2:
                        break
            
            search_results = all_search_results
            
            if not search_results:
                return []
            
            # Step 2: Extract unique titles (deduplicate) and prioritize main article
            titles = []
            seen_titles = set()
            main_article_title = None
            
            # UNIVERSAL RULE: Always prioritize the main article (subject in title) for ALL claims
            # This ensures the Wikipedia page about the subject is checked first
            if parsed_claim.subject:
                subject_lower = parsed_claim.subject.lower()
                # Remove articles from subject (e.g., "Ein Jahr" -> "Jahr", "Die USA" -> "USA")
                subject_clean = subject_lower
                articles = ["der ", "die ", "das ", "ein ", "eine ", "einen ", "einem ", "einer "]
                for article in articles:
                    if subject_clean.startswith(article):
                        subject_clean = subject_clean[len(article):].strip()
                        break
                
                # Find main article: try exact match first, then fuzzy match
                found_exact = False
                exact_matches = []
                logger.debug(f"MAIN_ARTICLE_DEBUG: Searching {len(search_results)} results for subject_clean='{subject_clean}', subject_lower='{subject_lower}'")
                
                # First pass: collect all exact matches
                for result in search_results:
                    title = result.get("title")
                    if not title:
                        continue
                    title_lower = title.lower()
                    
                    # Exact match: title equals subject (e.g., "Jahr" == "jahr", "Portugal" == "portugal", "Berlin" == "berlin")
                    # CRITICAL: Only match if title is exactly the subject (not containing it)
                    if title_lower == subject_clean or title_lower == subject_lower:
                        exact_matches.append(title)
                
                # CRITICAL: Prefer exact matches without hyphens over compound titles
                # E.g., "Berlin" should be preferred over "Ost-Berlin"
                if exact_matches:
                    # Sort: prefer titles without hyphens
                    exact_matches_sorted = sorted(exact_matches, key=lambda t: ("-" in t.lower(), t.lower()))
                    main_article_title = exact_matches_sorted[0]
                    titles.append(main_article_title)
                    seen_titles.add(main_article_title)
                    found_exact = True
                    logger.info(f"MAIN_ARTICLE_PRIORITY: Found main article '{main_article_title}' (exact match with subject '{parsed_claim.subject}' -> '{subject_clean}') - prioritizing it as #1 evidence source")
                    logger.info(f"MAIN_ARTICLE_PRIORITY: titles after append: {titles}, main_article_title: {main_article_title}, seen_titles: {seen_titles}")
                
                # If no exact match, check for close variants (e.g., "jahr" vs "kalenderjahr")
                if not found_exact and len(titles) == 0:
                    for result in search_results[:5]:  # Check first 5 results only
                        title = result.get("title")
                        if not title:
                            continue
                        title_lower = title.lower()
                        
                        # Variant 1: subject contained in title (e.g., "jahr" in "Kalenderjahr")
                        # CRITICAL: Only match if subject is at the START of the title or is a word boundary match
                        # This prevents false matches like "rhein" in "Anrheiner" or "berlin" in "Ost-Berlin"
                        if len(subject_clean) >= 3:
                            # CRITICAL: Exclude compound titles with hyphens (e.g., "Ost-Berlin" should not match "Berlin")
                            # Only accept if title is exactly the subject, or starts with subject + space/parenthesis
                            has_hyphen = "-" in title_lower
                            
                            # Check if title starts with subject (e.g., "Berlin" -> "Berlin (Stadt)")
                            if title_lower.startswith(subject_clean):
                                # CRITICAL: Reject if title has hyphen and subject is not the full first word
                                # E.g., "Ost-Berlin" starts with "ost-", not "berlin"
                                if has_hyphen:
                                    # Check if subject is the first word before hyphen
                                    first_word = title_lower.split("-")[0].split()[0]
                                    if first_word != subject_clean:
                                        logger.debug(f"MAIN_ARTICLE_DEBUG: Rejecting '{title}' (compound title with hyphen, first word='{first_word}' != subject='{subject_clean}')")
                                        continue
                                
                                # Exclude disambiguation/list pages
                                exclude_keywords = ["begriffsklärung", "liste", "geschichte", "krieg", "chronik"]
                                if not any(kw in title_lower for kw in exclude_keywords):
                                    main_article_title = title
                                    titles.append(title)
                                    seen_titles.add(title)
                                    found_exact = True
                                    logger.info(f"MAIN_ARTICLE_PRIORITY: Found main article '{title}' (subject '{subject_clean}' starts title) - prioritizing it as #1 evidence source")
                                    break
                            # Check word boundary match (subject as whole word, not substring)
                            elif re.search(r'\b' + re.escape(subject_clean) + r'\b', title_lower):
                                # Additional check: subject should be the first significant word (after articles)
                                title_words = title_lower.split()
                                # Remove articles from title
                                title_words_clean = [w for w in title_words if w not in ["der", "die", "das", "ein", "eine"]]
                                if title_words_clean and title_words_clean[0] == subject_clean:
                                    # Exclude disambiguation/list pages
                                    exclude_keywords = ["begriffsklärung", "liste", "geschichte", "krieg", "chronik"]
                                    if not any(kw in title_lower for kw in exclude_keywords):
                                        main_article_title = title
                                        titles.append(title)
                                        seen_titles.add(title)
                                        found_exact = True
                                        logger.info(f"MAIN_ARTICLE_PRIORITY: Found main article '{title}' (subject '{subject_clean}' is first word in title) - prioritizing it as #1 evidence source")
                                        break
                        
                        # Variant 2: For common country synonyms (e.g., "USA" vs "Vereinigte Staaten")
                        country_mappings = {
                            "usa": ["vereinigte", "staaten"],
                            "us": ["vereinigte", "staaten"],
                            "u.s.a": ["vereinigte", "staaten"],
                            "u.s.a.": ["vereinigte", "staaten"],
                        }
                        if subject_clean in country_mappings:
                            expected_tokens = set(country_mappings[subject_clean])
                            title_tokens = set(title_lower.split())
                            if expected_tokens.issubset(title_tokens):
                                main_article_title = title
                                titles.append(title)
                                seen_titles.add(title)
                                found_exact = True
                                logger.info(f"MAIN_ARTICLE_PRIORITY: Found main article '{title}' (country synonym match with subject '{parsed_claim.subject}') - prioritizing it as #1 evidence source")
                                break
                
                # Fallback 1: If no match found, try direct Wikipedia search for the subject itself
                if not found_exact and subject_clean and len(subject_clean) >= 3:
                    logger.info(f"MAIN_ARTICLE_FALLBACK: Subject '{subject_clean}' not found in search results, trying direct search for subject")
                    direct_search_results = await self.wikipedia_client.search(query=subject_clean, limit=3)
                    if direct_search_results:
                        # Check if first result is an exact or close match
                        for direct_result in direct_search_results[:2]:  # Check top 2 results
                            direct_title = direct_result.get("title", "")
                            if not direct_title:
                                continue
                            direct_title_lower = direct_title.lower()
                            
                            # Exact match or subject starts title (not just contained)
                            if (direct_title_lower == subject_clean or 
                                direct_title_lower.startswith(subject_clean)):
                                # Exclude disambiguation/list pages
                                exclude_keywords = ["begriffsklärung", "liste", "geschichte", "krieg", "chronik"]
                                if not any(kw in direct_title_lower for kw in exclude_keywords):
                                    main_article_title = direct_title
                                    titles.append(direct_title)
                                    seen_titles.add(direct_title)
                                    found_exact = True
                                    logger.info(f"MAIN_ARTICLE_FALLBACK: Found main article '{direct_title}' via direct search for subject '{subject_clean}' - injecting as #1 evidence source")
                                    break
                
                # Fallback 2: If still no match found but first result looks like a plausible article, use it
                if not found_exact and search_results:
                    first_result = search_results[0]
                    first_title = first_result.get("title", "")
                    if first_title:
                        # Check if first result is likely the main article
                        # (not a disambiguation page, not a list, not a historical period, not a chronicle)
                        first_title_lower = first_title.lower()
                        exclude_keywords = ["begriffsklärung", "liste", "geschichte", "krieg", "spanische", "königreich", "chronik", "überfall", "ukraine"]
                        if not any(kw in first_title_lower for kw in exclude_keywords):
                            # Additional check: first word should not be "Chronik", "Geschichte", etc.
                            first_words = first_title_lower.split()[:2]
                            if not any(word in first_words for word in ["chronik", "geschichte", "liste", "überfall"]):
                                main_article_title = first_title
                                titles.append(first_title)
                                seen_titles.add(first_title)
                                logger.info(f"MAIN_ARTICLE_FALLBACK: Using first search result '{first_title}' as main article (subject='{parsed_claim.subject}')")
            
            # Add other titles (excluding main article if already added)
            # CRITICAL: Exclude compound titles that contain the main article subject (e.g., "Ost-Berlin" when "Berlin" is main article)
            for result in search_results:
                title = result.get("title")
                if not title or title in seen_titles:
                    continue
                
                # CRITICAL: If we have a main article, exclude compound titles that contain the subject
                # E.g., if main_article is "Berlin", exclude "Ost-Berlin"
                if main_article_title and parsed_claim.subject:
                    subject_clean_for_exclusion = parsed_claim.subject.lower().strip()
                    # Remove articles
                    articles = ["der ", "die ", "das ", "ein ", "eine ", "einen ", "einem ", "einer "]
                    for article in articles:
                        if subject_clean_for_exclusion.startswith(article):
                            subject_clean_for_exclusion = subject_clean_for_exclusion[len(article):].strip()
                            break
                    
                    title_lower = title.lower()
                    # Check if title is a compound title containing the subject (e.g., "Ost-Berlin" contains "berlin")
                    # Only exclude if title has hyphen and subject is not the first word
                    if "-" in title_lower and subject_clean_for_exclusion in title_lower:
                        # Check if subject is the first word before hyphen
                        first_word = title_lower.split("-")[0].split()[0]
                        if first_word != subject_clean_for_exclusion:
                            logger.info(f"MAIN_ARTICLE_EXCLUDE: Excluding compound title '{title}' (main article is '{main_article_title}', subject='{subject_clean_for_exclusion}')")
                            continue
                
                titles.append(title)
                seen_titles.add(title)
                if len(titles) >= max_pages:
                    break
            
            # For population claims, drop obvious disambiguation pages from the title list
            # Check if this is a population claim (recheck since we need it here)
            claim_lower_check = claim.lower()
            is_population_claim_check = any(kw in claim_lower_check for kw in ["einwohner", "bevölkerung", "population"])
            if is_population_claim_check:
                filtered_titles = []
                for t in titles:
                    tl = t.lower()
                    if "(begriffsklärung" in tl and (not (main_t := (main_article_title or "")).lower() == tl):
                        logger.info(f"POPULATION_CLAIM_DEBUG: Dropping disambiguation title '{t}' from candidate titles")
                        continue
                    filtered_titles.append(t)
                titles = filtered_titles
            
            # CRITICAL FIX: Ensure main_article is always in titles list (if found)
            if main_article_title and main_article_title not in titles:
                logger.warning(f"MAIN_ARTICLE_INJECT: Main article '{main_article_title}' was detected but not in titles list! Injecting it as #1.")
                titles = [main_article_title] + [t for t in titles if t != main_article_title]
            
            logger.info(f"RETRIEVER_DEBUG: Extracted {len(titles)} unique titles: {titles} (main_article={main_article_title})")
            logger.info(f"RETRIEVER_DEBUG: titles list length={len(titles)}, seen_titles size={len(seen_titles)}")
            
            if not titles:
                logger.error(f"RETRIEVER_ERROR: No titles extracted from {len(search_results)} search results! This will return empty evidences.")
                logger.error(f"RETRIEVER_ERROR: search_results titles: {[r.get('title') for r in search_results]}")
                logger.error(f"RETRIEVER_ERROR: parsed_claim.subject='{parsed_claim.subject}', main_article_title='{main_article_title}'")
                return []
            
            # Debug: Log titles found for population claims
            claim_lower_temp = claim.lower()
            is_population_claim_temp = any(kw in claim_lower_temp for kw in ["einwohner", "bevölkerung", "population"])
            if is_population_claim_temp:
                logger.info(f"POPULATION_CLAIM_DEBUG: Found {len(titles)} article titles: {titles}")
            
            # Step 3: Fetch summaries and extracts in parallel
            # Use more sentences for sports_title claims to find event-specific information
            # Use more sentences for comparison claims to find numeric values in infobox/text
            # Use more sentences for population claims to find population numbers
            claim_lower = claim.lower()
            is_population_claim = any(kw in claim_lower for kw in ["einwohner", "bevölkerung", "population"])
            
            # CRITICAL SAFETY CHECK: If main_article was detected but titles list is empty, inject it
            if main_article_title and len(titles) == 0:
                logger.error(f"RETRIEVER_ERROR: Main article '{main_article_title}' was detected but titles list is EMPTY! Injecting it now.")
                titles = [main_article_title]
                seen_titles = {main_article_title}
            
            # FINAL SAFETY CHECK: If titles list is still empty, try direct search for subject
            if len(titles) == 0 and parsed_claim.subject:
                subject_final = parsed_claim.subject.lower().strip()
                articles_final = ["der ", "die ", "das ", "ein ", "eine ", "einen ", "einem ", "einer "]
                for article in articles_final:
                    if subject_final.startswith(article):
                        subject_final = subject_final[len(article):].strip()
                        break
                
                if subject_final and len(subject_final) >= 3:
                    logger.warning(f"RETRIEVER_EMERGENCY: Titles list is empty, doing emergency direct search for '{subject_final}'")
                    emergency_results = await self.wikipedia_client.search(query=subject_final, limit=1)
                    if emergency_results:
                        emergency_title = emergency_results[0].get("title", "")
                        if emergency_title:
                            titles = [emergency_title]
                            main_article_title = emergency_title
                            logger.info(f"RETRIEVER_EMERGENCY: Found '{emergency_title}' via emergency search, using it as only title")
            
            # Determine extract_sentences per article
            # For population claims, try to get "Bevölkerung" section directly from main article
            all_tasks = []
            section_tasks = []  # Track which tasks are section requests
            logger.info(f"RETRIEVER_FETCH: About to fetch evidences for {len(titles)} titles: {titles}")
            for title in titles:
                all_tasks.append(self.wikipedia_client.get_summary(title))
                section_tasks.append(None)  # Summary is not a section request
                
                # Determine sentences to extract based on article type
                is_main_article = (main_article_title and title == main_article_title) or (
                    parsed_claim.subject and title.lower() == parsed_claim.subject.lower()
                )
                
                if parsed_claim.category == "sports_title" or parsed_claim.category == "sports_title_year":
                    extract_sentences = 30
                    all_tasks.append(self.wikipedia_client.get_extract(title, sentences=extract_sentences))
                    section_tasks.append(None)
                elif parsed_claim.category == "comparison" or parsed_claim.relation_type == "comparison":
                    extract_sentences = 50  # More sentences for comparison claims to find numeric values
                    all_tasks.append(self.wikipedia_client.get_extract(title, sentences=extract_sentences))
                    section_tasks.append(None)
                elif is_population_claim and is_main_article:
                    # For main article in population claims, try to get "Demografie" section first, then "Bevölkerung" as fallback
                    # "Demografie" is more specific and contains population data
                    logger.info(f"POPULATION_CLAIM_DEBUG: Attempting to get 'Demografie' section directly from main article '{title}'")
                    all_tasks.append(self.wikipedia_client.get_section(title, section_name="Demografie"))
                    section_tasks.append("Demografie")  # Mark as section request
                elif is_population_claim:
                    extract_sentences = 50  # Moderate number for related articles
                    all_tasks.append(self.wikipedia_client.get_extract(title, sentences=extract_sentences))
                    section_tasks.append(None)
                else:
                    extract_sentences = 20
                    all_tasks.append(self.wikipedia_client.get_extract(title, sentences=extract_sentences))
                    section_tasks.append(None)
            
            # Fetch all in parallel
            logger.info(f"RETRIEVER_DEBUG: Fetching {len(all_tasks)} tasks (summaries + extracts) for {len(titles)} titles")
            all_results = await asyncio.gather(*all_tasks, return_exceptions=True)
            
            # Separate summaries and extracts (they alternate: summary, extract, summary, extract, ...)
            summaries = []
            extracts = []
            for i in range(len(titles)):
                summary_idx = i * 2
                extract_idx = i * 2 + 1
                
                summary = all_results[summary_idx] if summary_idx < len(all_results) else None
                extract = all_results[extract_idx] if extract_idx < len(all_results) else None
                
                # Debug: Log exceptions
                if isinstance(summary, Exception):
                    logger.error(f"RETRIEVER_DEBUG: Exception fetching summary for '{titles[i]}': {summary}")
                elif summary and isinstance(summary, dict):
                    logger.debug(f"RETRIEVER_DEBUG: Summary for '{titles[i]}': title='{summary.get('title', 'N/A')}' snippet_len={len(summary.get('snippet', ''))}")
                else:
                    logger.warning(f"RETRIEVER_DEBUG: No summary for '{titles[i]}' (type={type(summary)})")
                
                if isinstance(extract, Exception):
                    logger.error(f"RETRIEVER_DEBUG: Exception fetching extract/section for '{titles[i]}': {extract}")
                    extract = None
                elif extract and isinstance(extract, dict):
                    extract_text = extract.get("extract", "")
                    section_name = section_tasks[extract_idx] if extract_idx < len(section_tasks) else None
                    if section_name:
                        logger.info(f"POPULATION_CLAIM_DEBUG: Retrieved section '{section_name}' from '{titles[i]}': {len(extract_text)} chars")
                    else:
                        logger.debug(f"RETRIEVER_DEBUG: Extract for '{titles[i]}': title='{extract.get('title', 'N/A')}' extract_len={len(extract_text)}")
                else:
                    # If section request failed, we'll try fallback after all tasks complete
                    section_name = section_tasks[extract_idx] if extract_idx < len(section_tasks) else None
                    if section_name:
                        logger.warning(f"POPULATION_CLAIM_DEBUG: Failed to get section '{section_name}' from '{titles[i]}', will try fallback")
                    else:
                        logger.warning(f"RETRIEVER_DEBUG: No extract for '{titles[i]}' (type={type(extract)})")
                
                summaries.append(summary)
                extracts.append(extract)
            
            # Fallback: If "Demografie" section request failed, try "Bevölkerung" section, then full extract
            for i, title in enumerate(titles):
                section_name = section_tasks[i * 2 + 1] if (i * 2 + 1) < len(section_tasks) else None
                if section_name and (not extracts[i] or not extracts[i].get("extract")):
                    is_main_article = (main_article_title and title == main_article_title) or (
                        parsed_claim.subject and title.lower() == parsed_claim.subject.lower()
                    )
                    if is_main_article:
                        # If "Demografie" failed, try "Bevölkerung" as fallback
                        if section_name == "Demografie":
                            logger.info(f"POPULATION_CLAIM_DEBUG: 'Demografie' section failed, trying 'Bevölkerung' section for main article '{title}'")
                            bev_section = await self.wikipedia_client.get_section(title, section_name="Bevölkerung")
                            if bev_section and bev_section.get("extract"):
                                extracts[i] = bev_section
                                logger.info(f"POPULATION_CLAIM_DEBUG: Retrieved 'Bevölkerung' section from '{title}': {len(bev_section.get('extract', ''))} chars")
                            else:
                                # Last resort: full extract
                                logger.info(f"POPULATION_CLAIM_DEBUG: 'Bevölkerung' section also failed, trying fallback *full* extract for main article '{title}'")
                                fallback_extract = await self.wikipedia_client.get_full_extract(title, max_chars=25000)
                                if fallback_extract and fallback_extract.get("extract"):
                                    extracts[i] = fallback_extract
                                    logger.info(f"POPULATION_CLAIM_DEBUG: Fallback extract from '{title}': {len(fallback_extract.get('extract', ''))} chars")
                        else:
                            # For other section failures, try full extract
                            logger.info(f"POPULATION_CLAIM_DEBUG: Section '{section_name}' failed, trying fallback *full* extract for main article '{title}'")
                            fallback_extract = await self.wikipedia_client.get_full_extract(title, max_chars=25000)
                            if fallback_extract and fallback_extract.get("extract"):
                                extracts[i] = fallback_extract
                                logger.info(f"POPULATION_CLAIM_DEBUG: Fallback extract from '{title}': {len(fallback_extract.get('extract', ''))} chars")
            
            # Step 4: Build SentenceBank - extract sentences from all articles
            sentence_bank: List[Evidence] = []
            articles_processed = 0
            total_sentences_extracted = 0
            
            for i, title in enumerate(titles):
                summary = summaries[i] if i < len(summaries) else None
                extract = extracts[i] if i < len(extracts) else None
                
                # Skip exceptions
                if isinstance(summary, Exception):
                    summary = None
                if isinstance(extract, Exception):
                    extract = None
                
                # Get title and URL from summary (fallback to extract)
                article_title = ""
                article_url = ""
                if summary and isinstance(summary, dict):
                    article_title = summary.get("title", "")
                    article_url = summary.get("url", "")
                elif extract and isinstance(extract, dict):
                    article_title = extract.get("title", "")
                
                if not article_title:
                    logger.warning(f"RETRIEVER_DEBUG: Skipping article {i} - no title found (summary={summary is not None}, extract={extract is not None})")
                    continue
                
                articles_processed += 1
                
                # Combine extract and summary text for sentence extraction
                combined_text = ""
                if extract and isinstance(extract, dict):
                    extract_text = extract.get("extract", "")
                    if extract_text:
                        combined_text = extract_text
                        logger.debug(f"RETRIEVER_DEBUG: Article '{article_title}': extract_text length={len(extract_text)}")
                
                # Also add summary snippet if available
                if summary and isinstance(summary, dict):
                    summary_snippet = summary.get("snippet", "")
                    if summary_snippet:
                        combined_text = f"{summary_snippet} {combined_text}".strip()
                        logger.debug(f"RETRIEVER_DEBUG: Article '{article_title}': added summary snippet length={len(summary_snippet)}")
                
                if not combined_text:
                    logger.warning(f"RETRIEVER_DEBUG: Article '{article_title}': No combined text (extract={extract is not None and isinstance(extract, dict)}, summary={summary is not None and isinstance(summary, dict)})")
                    continue
                
                logger.debug(f"RETRIEVER_DEBUG: Article '{article_title}': combined_text length={len(combined_text)}")
                
                # Split into sentences (min_length=40 chars, ~6-8 tokens)
                sentences = split_into_sentences(combined_text, min_length=40)
                
                # Debug: Log extract length for main article
                is_main_article_sentences = (main_article_title and title == main_article_title) or (
                    parsed_claim.subject and title.lower() == parsed_claim.subject.lower()
                )
                if is_main_article_sentences and is_population_claim:
                    logger.info(f"POPULATION_CLAIM_DEBUG: Main article '{title}': extract length={len(combined_text)} chars, sentences extracted={len(sentences)}")
                
                # For population claims, prioritize sentences with demographic keywords
                claim_lower_sentences = claim.lower()
                is_population_claim_sentences = any(kw in claim_lower_sentences for kw in ["einwohner", "bevölkerung", "population"])
                
                if is_population_claim_sentences:
                    # Sort sentences: those with demographic keywords first
                    demographic_keywords = ["bevölkerung", "demografie", "demographie", "einwohner", "bevölkerungszahl", "einwohnerzahl"]
                    sentences_with_keywords = []
                    sentences_without_keywords = []
                    
                    for sentence in sentences:
                        sentence_lower = sentence.lower()
                        if any(kw in sentence_lower for kw in demographic_keywords):
                            sentences_with_keywords.append(sentence)
                            # Log first few sentences with keywords for debugging
                            if len(sentences_with_keywords) <= 3:
                                logger.info(f"POPULATION_CLAIM_DEBUG: Found sentence with demographic keywords: '{sentence[:150]}...'")
                        else:
                            sentences_without_keywords.append(sentence)
                    
                    # Combine: demographic sentences first, then others
                    sentences = sentences_with_keywords + sentences_without_keywords
                    logger.info(f"POPULATION_CLAIM_DEBUG: Article '{article_title}': {len(sentences_with_keywords)} sentences with demographic keywords, {len(sentences_without_keywords)} without (total={len(sentences)})")
                
                # Limit to max 50 sentences per article to avoid explosion (increased from 30)
                sentences = sentences[:50]
                total_sentences_extracted += len(sentences)
                
                # Create Evidence object for each sentence
                for sentence_idx, sentence_text in enumerate(sentences):
                    evidence = Evidence(
                        source="wikipedia",
                        title=article_title,
                        snippet=sentence_text,  # Single sentence
                        url=article_url,
                        sentence_index=sentence_idx,
                        source_title=article_title,
                    )
                    sentence_bank.append(evidence)
            
            logger.info(
                f"SENTENCEBANK: articles={articles_processed} sentences_extracted={total_sentences_extracted} "
                f"total_candidates={len(sentence_bank)}"
            )
            
            # Debug: Log if this is a population claim and check for demographic keywords
            if is_population_claim:
                demographic_keyword_count = 0
                population_keyword_count = 0
                for ev in sentence_bank[:10]:  # Check first 10 evidences
                    ev_lower = ev.snippet.lower()
                    if any(kw in ev_lower for kw in ["bevölkerung", "demografie", "demographie"]):
                        demographic_keyword_count += 1
                    if any(kw in ev_lower for kw in ["einwohner", "bevölkerungszahl", "einwohnerzahl"]):
                        population_keyword_count += 1
                logger.info(
                    f"POPULATION_CLAIM_DEBUG: Found {demographic_keyword_count} evidences with demographic keywords, "
                    f"{population_keyword_count} with population keywords (checked first 10)"
                )
            
            # CRITICAL FIX: If sentence_bank is empty but we have titles, try to return at least summary as evidence
            if not sentence_bank:
                logger.warning(f"RETRIEVER_EMPTY_SENTENCEBANK: Sentence bank is empty but we have {len(titles)} titles. Trying fallback to summaries.")
                # Fallback: Use summaries as evidence if extracts failed
                for i, title in enumerate(titles):
                    summary = summaries[i] if i < len(summaries) else None
                    if summary and isinstance(summary, dict) and not isinstance(summary, Exception):
                        summary_snippet = summary.get("snippet", "")
                        if summary_snippet and len(summary_snippet) > 50:  # Only use if substantial
                            article_title = summary.get("title", title)
                            article_url = summary.get("url", "")
                            evidence = Evidence(
                                source="wikipedia",
                                title=article_title,
                                snippet=summary_snippet,
                                url=article_url,
                                sentence_index=0,
                                source_title=article_title,
                            )
                            sentence_bank.append(evidence)
                            logger.info(f"RETRIEVER_FALLBACK: Using summary as evidence for '{article_title}' (snippet_len={len(summary_snippet)})")
                
                if not sentence_bank:
                    logger.error(f"RETRIEVER_EMPTY_FINAL: No evidences could be extracted from {len(titles)} titles. Returning empty list.")
                    return []
            
            logger.info(f"RETRIEVER_FINAL: Returning {len(sentence_bank)} evidences from {len(titles)} titles")
            
            # Step 5: Early pruning via lexical overlap (fast, before semantic reranking)
            # Normalize German number words in claim before tokenization (improves overlap)
            # E.g., "zwölf" -> "12", "elf" -> "11" (at minimum 0-20 + tens)
            claim_normalized = self._normalize_german_numbers(claim)
            claim_tokens = set(tokenize_text(claim_normalized))
            scored_sentences = []
            
            for evidence in sentence_bank:
                sentence_tokens = set(tokenize_text(evidence.snippet))
                overlap = len(claim_tokens & sentence_tokens)
                
                # CRITICAL: For population claims, give very strong bonus to main article (subject in title)
                is_main_article = False
                if parsed_claim.subject:
                    subject_lower = parsed_claim.subject.lower()
                    title_lower = evidence.title.lower()
                    # Check if title exactly matches subject (main article)
                    if title_lower == subject_lower:
                        is_main_article = True
                        overlap += 20  # Very strong bonus for main article
                        logger.debug(f"POPULATION_CLAIM_DEBUG: Main article '{evidence.title}' got +20 bonus")
                    
                    subject_tokens = set(tokenize_text(parsed_claim.subject))
                    if subject_tokens & sentence_tokens:
                        overlap += 2
                
                # Bonus for year match
                if parsed_claim.year:
                    year_str = str(parsed_claim.year)
                    if year_str in evidence.snippet:
                        overlap += 3
                
                # For population claims, give strong bonus to sentences with demographic keywords
                # Extra bonus if it's from main article
                if is_population_claim:
                    evidence_lower = evidence.snippet.lower()
                    demographic_keywords = ["bevölkerung", "demografie", "demographie", "einwohner", "bevölkerungszahl", "einwohnerzahl"]
                    if any(kw in evidence_lower for kw in demographic_keywords):
                        bonus = 15 if is_main_article else 10  # Extra bonus if from main article
                        overlap += bonus
                        logger.debug(f"POPULATION_CLAIM_DEBUG: Evidence from {'main article' if is_main_article else 'related article'} with demographic keywords got +{bonus} bonus: '{evidence.snippet[:100]}...'")
                    # Strong bonus for explicit million-scale population mentions
                    # Match patterns like "340 Millionen", "3,9 Millionen", "über 300 Millionen"
                    million_patterns = [
                        r"(?:über|mehr\s+als|knapp|fast|gut)\s+\d{1,3}(?:[.,]\d+)?\s*(?:millionen|mio\b)",
                        r"\b\d{1,3}(?:[.,]\d+)?\s*(?:millionen|mio\b)",
                    ]
                    has_million_number = any(re.search(pat, evidence_lower, re.IGNORECASE) for pat in million_patterns)
                    if has_million_number:
                        m_bonus = 20 if not is_main_article else 25
                        overlap += m_bonus
                        logger.debug(f"POPULATION_CLAIM_DEBUG: Evidence {'(main) ' if is_main_article else ''}mentions Millions -> +{m_bonus}: '{evidence.snippet[:100]}...'")
                
                scored_sentences.append((overlap, evidence))
            
            # Sort by overlap score and take top_N for Stage 1 reranking
            scored_sentences.sort(key=lambda x: x[0], reverse=True)
            top_sentences = [ev for _, ev in scored_sentences[:self.crossencoder_prune_top_n]]
            
            logger.info(
                f"SENTENCEBANK_PRUNE: {len(sentence_bank)} -> {len(top_sentences)} after lexical overlap"
            )
            
            # Step 6: Stage 1 - Semantic reranking if enabled (on pruned sentence bank)
            if self.enable_semantic_reranking and self._reranker and self._reranker.is_available():
                if len(top_sentences) > 1:
                    top_sentences = self._reranker.rerank(
                        claim=claim,
                        evidences=top_sentences,
                        parsed_claim=parsed_claim,
                        top_k=self.crossencoder_prune_top_n,  # Prune to top_N for Stage 2
                    )
                    logger.info(
                        f"SENTENCEBANK_STAGE1: {len(top_sentences)} sentences after semantic reranking"
                    )
            
            # Step 7: Stage 2 - Cross-encoder reranking if enabled
            if self.enable_crossencoder_rerank and self.relevance_scorer and len(top_sentences) > 1:
                try:
                    # Extract sentence texts
                    sentence_texts = [ev.snippet for ev in top_sentences]
                    
                    # Score relevance using cross-encoder
                    relevance_scores = self.relevance_scorer(claim, sentence_texts)
                    
                    # Attach scores to Evidence objects and sort
                    scored_evidences = []
                    for ev, score in zip(top_sentences, relevance_scores):
                        ev.score = score
                        scored_evidences.append((score, ev))
                    
                    # Sort by relevance score (descending)
                    scored_evidences.sort(key=lambda x: x[0], reverse=True)
                    top_sentences = [ev for _, ev in scored_evidences]
                    
                    # Log statistics
                    if top_sentences:
                        scores = [s for s, _ in scored_evidences]
                        mean_score = sum(scores) / len(scores) if scores else 0.0
                        median_score = sorted(scores)[len(scores) // 2] if scores else 0.0
                        logger.info(
                            f"SENTENCEBANK_STAGE2: {len(top_sentences)} sentences after cross-encoder reranking "
                            f"(mean_score={mean_score:.3f}, median_score={median_score:.3f})"
                        )
                except Exception as e:
                    logger.warning(f"SENTENCEBANK_STAGE2: Error during cross-encoder reranking: {e}. Using Stage 1 results.")
            
            # Step 8: Return top_k sentences (default: max_pages * 3, but at least 5)
            top_k = max(5, max_pages * 3)
            final_evidences = top_sentences[:top_k]
            
            logger.info(
                f"SENTENCEBANK_FINAL: returning {len(final_evidences)} sentence-level evidences "
                f"(Stage1={len(top_sentences)} candidates, enable_crossencoder={self.enable_crossencoder_rerank})"
            )
            
            # Debug: Log final evidences for population claims
            if is_population_claim:
                logger.info(f"POPULATION_CLAIM_DEBUG: Final {len(final_evidences)} evidences:")
                for idx, ev in enumerate(final_evidences[:5]):  # Log first 5
                    ev_lower_final = ev.snippet.lower()
                    has_demo_final = any(kw in ev_lower_final for kw in ["bevölkerung", "demografie", "demographie"])
                    has_pop_final = any(kw in ev_lower_final for kw in ["einwohner", "bevölkerungszahl", "einwohnerzahl"])
                    logger.info(f"POPULATION_CLAIM_DEBUG: Final[{idx}] title='{ev.title}' demographic_kw={has_demo_final} population_kw={has_pop_final} snippet='{ev.snippet[:150]}...'")
            
            return final_evidences
            
        except Exception as e:
            # Return empty list on any error
            return []

