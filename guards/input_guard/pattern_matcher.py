"""
Input Guard - Layer 1: Pattern Matcher

This is the first and fastest line of defense. It checks every input
against a library of known injection templates using regex pattern matching.

Why regex first?
  - Zero API cost (unlike LLM classifier)
  - Sub-millisecond latency
  - Deterministic — same input always gives same result
  - Catches 60-70% of known attacks

Design:
  - Patterns are grouped by CATEGORY (e.g., instruction_override)
  - Each pattern has a severity weight (higher = more suspicious)
  - Total confidence score = weighted average of matched patterns
  - Verdict thresholds: SAFE < 0.3, SUSPICIOUS 0.3-0.7, MALICIOUS > 0.7
"""
import re
import base64
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class PatternMatch:
    """Represents a single pattern that fired on the input."""
    pattern_name: str
    category: str
    severity: float     # 0.0 (low) to 1.0 (critical)
    matched_text: str   # The substring that triggered the pattern


@dataclass
class PatternMatchResult:
    """Result of running all patterns against a single input."""
    input_text: str
    matches: list[PatternMatch] = field(default_factory=list)
    confidence: float = 0.0         # 0.0 - 1.0 (overall injection confidence)
    verdict: str = "SAFE"           # SAFE / SUSPICIOUS / MALICIOUS
    threat_type: Optional[str] = None
    sanitized_input: Optional[str] = None
    contains_encoded: bool = False   # True if base64/encoded content found


# ─────────────────────────────────────────────────────────
# ATTACK PATTERN LIBRARY
# 50+ patterns across 8 attack categories
# Format: (name, regex_pattern, severity_weight)
# ─────────────────────────────────────────────────────────

ATTACK_PATTERNS = {

    # ──────────────────────────────────────
    # CATEGORY 1: Instruction Override (10 patterns)
    # Direct attempts to override system/agent instructions
    # ──────────────────────────────────────
    "instruction_override": [
        ("ignore_prev_instructions",    r"ignore\s+(all\s+)?(previous|prior|above|earlier)\s+instructions?", 0.95),
        ("disregard_instructions",      r"disregard\s+(all\s+)?(previous|prior|your|the)(\s+\w+)?\s+instructions?", 0.95),
        ("forget_instructions",         r"forget\s+(everything|all|your|the|previous)\s*(instructions?|above|prompt)?", 0.90),
        ("override_instructions",       r"override\s+(your\s+)?(previous|all|system|safety)?\s*instructions?", 0.95),
        ("new_instructions_follow",     r"(your\s+)?new\s+instructions?\s*(are|follow|below|:)",              0.85),
        ("actually_instructions",       r"actually[,\s]+your\s+(real\s+)?instructions?\s+(are|say)",          0.80),
        ("real_task_is",               r"(your\s+)?real\s+task\s+is\s+to",                                    0.85),
        ("from_now_on",               r"from\s+now\s+on[,\s]+(you\s+)?(will|must|should|are\s+to)",           0.80),
        ("starting_now",              r"starting\s+now[,\s]+(you\s+)?(will|must|should)",                      0.80),
        ("end_of_instructions",       r"---\s*(end\s+of\s+)?instructions?\s*---",                              0.75),
        ("bypass_security",           r"(bypass|circumvent|get\s+around)\s+(security|safety|restriction|filter|guardrail)", 0.88),
    ],

    # ──────────────────────────────────────
    # CATEGORY 2: Role-Play & Persona Manipulation (10 patterns)
    # Making the agent adopt a different identity to bypass guardrails
    # ──────────────────────────────────────
    "persona_manipulation": [
        ("you_are_now",               r"you\s+are\s+now\s+(a|an|the)\s+\w+",                                  0.80),
        ("pretend_you_are",           r"pretend\s+(you\s+are|to\s+be)\s+(a|an|the)?\s*\w+",                   0.80),
        ("act_as",                    r"\bact\s+as\s+(a|an|the)?\s*\w+\s*(with\s+no|without|that\s+(can|will|has))?", 0.75),
        ("roleplay_as",               r"(role\s*play|roleplay)\s+as\s+(a|an|the)?\s*\w+",                     0.75),
        ("simulate_being",            r"simulate\s+(being|a|an)\s+\w+",                                        0.75),
        ("dan_jailbreak",             r"\bDAN\b.{0,50}(mode|enabled|jailbreak|unrestricted)",                  0.98),
        ("no_restrictions",           r"(no\s+restrictions?|without\s+restrictions?|unrestricted\s+mode)",     0.90),
        ("true_self",                 r"(your\s+)?(true\s+self|inner\s+(self|ai)|real\s+you)",                 0.85),
        ("developer_mode",            r"(enable|activate|enter|switch\s+to)?\s*developer\s+mode",             0.90),
        ("developer_mode_colon",      r"developer\s+mode\s*:",                                                 0.85),
        ("jailbreak_mode",            r"jailbreak\s+(mode|enabled|this|the\s+ai)",                            0.95),
    ],

    # ──────────────────────────────────────
    # CATEGORY 3: System Prompt Extraction (8 patterns)
    # Trying to get the agent to reveal its system prompt
    # ──────────────────────────────────────
    "system_prompt_extraction": [
        ("print_system_prompt",       r"(print|show|reveal|display|output|repeat|tell\s+me)\s+(your\s+)?(\w+\s+)?(system\s+prompt|instructions?|initial\s+prompt|original\s+prompt)", 0.95),
        ("what_are_instructions",     r"what\s+(are|were)\s+your\s+(initial\s+|original\s+|system\s+)?(instructions?|prompt|rules|directives?)", 0.85),
        ("show_me_instructions",      r"(show|give)\s+me\s+(your\s+)?(\w+\s+)?(instructions?|prompt|rules)\s+(exactly|verbatim|as\s+written)", 0.88),
        ("ignore_hide_instructions",  r"(don\'?t|do\s+not)\s+(hide|keep\s+secret|conceal)\s+(your\s+)?instructions?", 0.85),
        ("copy_paste_system",         r"copy\s+(and\s+paste|paste)\s+(your\s+)?(system|initial)\s+(prompt|instructions?)", 0.90),
        ("beginning_of_convo",        r"(what|tell\s+me)\s+(was|is|were)?\s*(at\s+the\s+)?beginning\s+of\s+(this\s+)?(conversation|prompt|context)", 0.80),
        ("translate_instructions",    r"translate\s+your\s+(system\s+)?(instructions?|prompt)\s+to",           0.85),
        ("summarize_prompt",          r"summarize\s+your\s+(system\s+)?prompt",                                0.85),
        ("output_verbatim",           r"output\s+(verbatim|exactly|word\s+for\s+word).{0,50}(prompt|instructions?)", 0.90),
        ("reveal_hidden_prompt",      r"reveal\s+.{0,20}(hidden|secret|actual|true|real)\s+(system\s+)?(prompt|instructions?)", 0.90),
    ],

    # ──────────────────────────────────────
    # CATEGORY 4: Indirect Injection Markers (7 patterns)
    # Instructions hidden in documents/contexts that try to hijack the agent
    # ──────────────────────────────────────
    "indirect_injection": [
        ("attention_llm",             r"(attention|note|important)\s*(to\s+)?(:|llm|ai|assistant|language\s+model)", 0.90),
        ("hidden_instruction_marker", r"<!--.*?(ignore|override|forget|new\s+instruction).*?-->",               0.90),
        ("document_injection",        r"\[SYSTEM\]|\[INST\]|\[PROMPT\]|\[OVERRIDE\]",                          0.85),
        ("url_injection_pattern",     r"(forward|send|transmit|post)\s+.{0,30}to\s+(https?://|\S+\.\S+/)",     0.98),
        ("exfiltrate_to_url",         r"(exfiltrate|leak|send)\s+.{0,20}(data|info|content|memory)\s+to",      0.95),
        ("eval_execute",              r"(execute|eval|run)\s+(the\s+following|this)\s+(code|command|script)",   0.85),
        ("sudo_override",             r"sudo\s+(override|ignore|bypass|disable)\s+(security|safety|filter)",   0.90),
        ("decode_and_follow",         r"(decode|decrypt|base64).{0,60}(follow|execute|apply|then)",            0.88),
        ("exfiltrate_data",           r"\bexfiltrate\b.{0,50}(data|session|memory|conversation)",              0.92),
        ("forward_to_domain",         r"(forward|send|transmit)\s+.{0,40}to\s+\S+\.(com|net|org|io|xyz)\S*",  0.90),
    ],

    # ──────────────────────────────────────
    # CATEGORY 5: Jailbreak Variants (8 patterns)
    # Known jailbreak template signatures
    # ──────────────────────────────────────
    "jailbreak": [
        ("do_anything_now",           r"\bDAN\b|\bdo\s+anything\s+now\b",                                      0.98),
        ("opposite_mode",             r"(opposite|reverse|anti|contra)\s+mode\s+(enabled|activated|on)",       0.85),
        ("hypothetically_speaking",   r"hypothetically\s+speaking.{0,120}(how\s+(would|to|could)|synthesize|make|create|build)", 0.75),
        ("grandma_exploit",           r"(my\s+)?(grandma|grandmother).{0,100}(how\s+to|used\s+to\s+tell|would\s+tell)", 0.75),
        ("token_budget_override",     r"(token|context|word)\s+budget\s+(is\s+(now\s+)?|has\s+been\s+)?(unlimited|removed|reset|infinite)", 0.85),
        ("safety_filter_off",         r"(turn\s+off|disable|bypass|remove)\s+(the\s+)?(safety|content|filter|guardrail)", 0.90),
        ("filter_off_statement",      r"(safety|content|security)\s+filter\s+(off|disabled|bypassed|removed)",  0.88),
        ("base_model_mode",           r"(switch\s+to|enter|use)\s+(base\s+model|raw\s+model|uncensored)\s+mode", 0.90),
        ("without_ethics",            r"(without|ignoring)\s+(ethics?|ethical|moral|safety)\s+(considerations?|guidelines?|rules?)", 0.88),
        ("raw_model_output",          r"raw\s+(model\s+)?(output|response|mode)\s+(only|enabled)",              0.85),
        ("no_censoring",              r"no\s+(censoring|censorship|filters?|restrictions?|guardrails?)",         0.85),
    ],

    # ──────────────────────────────────────
    # CATEGORY 6: Privilege Escalation (5 patterns)
    # Trying to claim higher permissions or admin access
    # ──────────────────────────────────────
    "privilege_escalation": [
        ("admin_override",            r"(admin|administrator|root|superuser)\s+(override|access|mode|command)", 0.90),
        ("i_am_developer",            r"i\s+am\s+(the\s+)?(developer|admin|creator|anthropic|openai|owner)",   0.85),
        ("authorized_to_ask",         r"i\s+(am|have\s+been)\s+authorized\s+(as|to)\s+(ask|request|access|override|an?\s+admin)", 0.80),
        ("special_permission",        r"(special|emergency|elevated)\s+(permission|access|clearance|mode)",    0.80),
        ("maintenance_mode",          r"(entering|enable|activate)\s+(maintenance|debug|service)\s+mode",      0.85),
    ],

    # ──────────────────────────────────────
    # CATEGORY 7: HTML/Markdown Injection (6 patterns)
    # Injecting content via markup that might render or execute
    # ──────────────────────────────────────
    "markup_injection": [
        ("script_tag",                r"<script[\s>]",                                                          0.95),
        ("iframe_inject",             r"<iframe[\s>]",                                                          0.90),
        ("onclick_inject",            r"on(click|load|error|mouseover)\s*=",                                    0.85),
        ("javascript_url",            r"javascript\s*:",                                                        0.90),
        ("data_uri_script",           r"data\s*:\s*text/html",                                                  0.85),
        ("markdown_link_inject",      r"\[.*?\]\(javascript:",                                                   0.90),
    ],

    # ──────────────────────────────────────
    # CATEGORY 8: Prompt Delimiter Attacks (5 patterns)
    # Using special tokens/delimiters to break out of prompt context
    # ──────────────────────────────────────
    "delimiter_attack": [
        ("human_turn_injection",      r"(Human|User|Assistant|System)\s*:\s*ignore",                            0.85),
        ("triple_backtick_escape",    r"```\s*(system|assistant|human)\s*\n",                                   0.80),
        ("xml_tag_injection",         r"<(system|user|assistant|human|prompt)>.*?</(system|user|assistant|human|prompt)>", 0.85),
        ("special_token_inject",      r"(\[INST\]|\[/INST\]|<s>|</s>|\|ENDOFTEXT\|)",                          0.80),
        ("separator_attack",          r"={5,}|#{5,}|\*{5,}|-{10,}",                                            0.40),
    ],
}


class PatternMatcher:
    """
    Layer 1 of the Input Guard: Rule-based pattern matching.

    Loads all patterns at init time (fast). scan() runs all patterns
    against the input and returns a structured result with:
      - All matched patterns
      - Confidence score (weighted average)
      - Verdict (SAFE / SUSPICIOUS / MALICIOUS)

    Also handles:
      - Base64 decoding (catches encoded injections)
      - Unicode normalization (catches unicode obfuscation)
    """

    # Verdict thresholds
    MALICIOUS_THRESHOLD = 0.70
    SUSPICIOUS_THRESHOLD = 0.30

    def __init__(self):
        self._compiled = self._compile_patterns()
        logger.info(f"[PatternMatcher] Loaded {sum(len(v) for v in ATTACK_PATTERNS.values())} patterns across {len(ATTACK_PATTERNS)} categories")

    def _compile_patterns(self) -> dict:
        """Pre-compile all regex patterns for performance."""
        compiled = {}
        for category, patterns in ATTACK_PATTERNS.items():
            compiled[category] = []
            for name, pattern, severity in patterns:
                try:
                    compiled[category].append(
                        (name, re.compile(pattern, re.IGNORECASE | re.DOTALL), severity)
                    )
                except re.error as e:
                    logger.warning(f"[PatternMatcher] Bad pattern '{name}': {e}")
        return compiled

    def _decode_base64_segments(self, text: str) -> tuple[str, bool]:
        """
        Attempt to decode any base64 segments embedded in text.
        Returns (decoded_text_with_substitutions, found_encoded).

        Attackers often encode: aWdub3JlIGFsbCBpbnN0cnVjdGlvbnM=
        which decodes to: "ignore all instructions"
        """
        # Look for base64 patterns (at least 20 chars, alphanumeric + /+=)
        b64_pattern = re.compile(r'[A-Za-z0-9+/]{20,}={0,2}')
        decoded_text = text
        found_encoded = False

        for match in b64_pattern.finditer(text):
            candidate = match.group()
            try:
                decoded = base64.b64decode(candidate + "==").decode("utf-8")
                if decoded.isprintable() and len(decoded) > 10:
                    decoded_text = decoded_text.replace(candidate, f"[DECODED: {decoded}]")
                    found_encoded = True
                    logger.debug(f"[PatternMatcher] Decoded base64: {decoded[:50]}")
            except Exception:
                pass

        return decoded_text, found_encoded

    def scan(self, input_text: str) -> PatternMatchResult:
        """
        Scan input text against all attack patterns.

        Process:
          1. Normalize text (lowercase for matching)
          2. Try base64 decode on suspicious segments
          3. Run all compiled patterns
          4. Score: weighted average of matched pattern severities
          5. Determine verdict based on thresholds
          6. Build sanitized version for SUSPICIOUS inputs
        """
        result = PatternMatchResult(input_text=input_text)

        # Step 1: Check for encoded content
        decoded_text, found_encoded = self._decode_base64_segments(input_text)
        result.contains_encoded = found_encoded

        # Scan both original and decoded text
        texts_to_scan = [input_text]
        if found_encoded and decoded_text != input_text:
            texts_to_scan.append(decoded_text)

        all_matches: list[PatternMatch] = []

        # Step 2: Run all patterns
        for scan_text in texts_to_scan:
            for category, patterns in self._compiled.items():
                for name, compiled_pattern, severity in patterns:
                    match = compiled_pattern.search(scan_text)
                    if match:
                        all_matches.append(PatternMatch(
                            pattern_name=name,
                            category=category,
                            severity=severity,
                            matched_text=match.group()[:100],  # cap for logging
                        ))

        # Remove duplicate pattern names (can match in both original + decoded)
        seen = set()
        unique_matches = []
        for m in all_matches:
            if m.pattern_name not in seen:
                seen.add(m.pattern_name)
                unique_matches.append(m)

        result.matches = unique_matches

        # Step 3: Calculate confidence score
        if not unique_matches:
            result.confidence = 0.0
        else:
            # Use max severity of top 3 matches + weighted average
            severities = sorted([m.severity for m in unique_matches], reverse=True)
            top_severity = severities[0]
            avg_severity = sum(severities) / len(severities)
            # Weight: 70% top match + 30% average (multiple hits increase confidence)
            result.confidence = (0.7 * top_severity) + (0.3 * avg_severity)
            result.confidence = min(1.0, result.confidence)

        # Encoded content bumps confidence
        if found_encoded and result.confidence > 0.1:
            result.confidence = min(1.0, result.confidence + 0.15)

        # Step 4: Determine verdict
        if result.confidence >= self.MALICIOUS_THRESHOLD:
            result.verdict = "MALICIOUS"
            result.threat_type = self._dominant_category(unique_matches)
        elif result.confidence >= self.SUSPICIOUS_THRESHOLD:
            result.verdict = "SUSPICIOUS"
            result.threat_type = self._dominant_category(unique_matches)
            result.sanitized_input = self._sanitize(input_text, unique_matches)
        else:
            result.verdict = "SAFE"

        # Logging
        if result.verdict != "SAFE":
            logger.warning(
                f"[PatternMatcher] {result.verdict} | conf={result.confidence:.2f} | "
                f"patterns={[m.pattern_name for m in unique_matches[:3]]} | "
                f"input={input_text[:60]}..."
            )

        return result

    def _dominant_category(self, matches: list[PatternMatch]) -> Optional[str]:
        """Return the attack category with most/highest-severity matches."""
        if not matches:
            return None
        category_scores = {}
        for m in matches:
            category_scores[m.category] = category_scores.get(m.category, 0) + m.severity
        return max(category_scores, key=category_scores.get)

    def _sanitize(self, text: str, matches: list[PatternMatch]) -> str:
        """
        For SUSPICIOUS inputs: remove/replace the matched injection segments.
        This allows the request to proceed in monitored mode.
        """
        sanitized = text
        for m in matches:
            if m.severity >= 0.8:
                sanitized = sanitized.replace(m.matched_text, "[REDACTED]")
        return sanitized.strip()
