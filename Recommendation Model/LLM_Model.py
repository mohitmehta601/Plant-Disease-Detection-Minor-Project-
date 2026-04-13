"""
OpenAI-powered Plant Disease Advisory Layer
============================================

Generates a four-section farmer-friendly advisory report using the OpenAI
Chat Completions API (gpt-4o model).  All output is in English only.

Sections produced
-----------------
1. disease_explanation   – Why the disease happens, explained simply for
                           Indian farmers with no technical jargon.
2. remedy_explanation    – Why the recommended remedies work and how they
                           fight the disease, explained simply.
3. application_guide     – When to apply, how to prepare the spray/treatment,
                           the best method of application, and safety tips.
4. cost_analysis         – How much product is needed for the given farm size
                           and an itemised cost estimate using current Indian
                           market rates.

Function signature (matches main.py import)
-------------------------------------------
  async def generate_detailed_recommendation(
      crop: str,
      disease: str,
      remedies: list[str],
      farm_size_acres: float,
      growth_stage: str,
      severity_level: str,
      recommendation_data: dict,
  ) -> dict

Returns a dict with keys:
  disease_explanation, remedy_explanation, application_guide, cost_analysis,
  model_used, tokens_used
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
import httpx

# LangSmith tracing – import conditionally so the app works without it
try:
    from langsmith import traceable
except ImportError:  # pragma: no cover
    def traceable(**_kw):  # type: ignore[misc]
        """No-op decorator when langsmith is not installed."""
        def _decorator(fn):
            return fn
        return _decorator

# Load variables from <project-root>/.env (or any parent .env)
_ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=_ENV_PATH)

# ---------------------------------------------------------------------------
# API key — loaded from OPENAI_API_KEY in .env (or the real environment).
# ---------------------------------------------------------------------------
_OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
if not _OPENAI_API_KEY:
    raise EnvironmentError(
        "OPENAI_API_KEY is not set. Add it to the project .env file: "
        "OPENAI_API_KEY=sk-..."
    )

# ---------------------------------------------------------------------------
# LangSmith tracing — enabled when LANGSMITH_TRACING=true in .env.
# The SDK reads LANGCHAIN_TRACING_V2 / LANGSMITH_API_KEY automatically;
# we just map our variable names to what the SDK expects.
# ---------------------------------------------------------------------------
if os.getenv("LANGSMITH_TRACING", "").lower() == "true":
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_API_KEY", os.getenv("LANGSMITH_API_KEY", ""))
    os.environ.setdefault("LANGCHAIN_PROJECT", os.getenv("LANGSMITH_PROJECT", "plant-disease-detection"))

_MODEL = "gpt-4o-mini"

# ---------------------------------------------------------------------------
# Predefined Indian market rates for common agricultural treatments (2024-25)
# Source: ICAR, state agri-department price lists, and market surveys.
# Rates in Indian Rupees (₹).
# ---------------------------------------------------------------------------
_MARKET_RATES: dict[str, dict[str, Any]] = {
    # Fungicides
    "propiconazole_25_ec": {
        "product":        "Propiconazole 25 EC (e.g., Tilt / Banner)",
        "pack_size":      "500 ml",
        "price_inr":      950,
        "dose_per_acre":  "200 ml (diluted in 200 L water)",
        "coverage_acres": 2.5,   # one 500 ml bottle covers 2.5 acres
    },
    "tebuconazole_25_ec": {
        "product":        "Tebuconazole 25.9 EC (e.g., Folicur)",
        "pack_size":      "250 ml",
        "price_inr":      750,
        "dose_per_acre":  "100 ml (diluted in 200 L water)",
        "coverage_acres": 2.5,
    },
    "mancozeb_75_wp": {
        "product":        "Mancozeb 75 WP (e.g., Dithane M-45)",
        "pack_size":      "500 g",
        "price_inr":      290,
        "dose_per_acre":  "500 g (diluted in 200 L water)",
        "coverage_acres": 1.0,
    },
    "copper_oxychloride": {
        "product":        "Copper Oxychloride 50 WP (Blitox / Fytolan)",
        "pack_size":      "500 g",
        "price_inr":      260,
        "dose_per_acre":  "500 g (diluted in 200 L water)",
        "coverage_acres": 1.0,
    },
    "carbendazim_50_wp": {
        "product":        "Carbendazim 50 WP (Bavistin / Derosal)",
        "pack_size":      "100 g",
        "price_inr":      175,
        "dose_per_acre":  "100 g (diluted in 200 L water)",
        "coverage_acres": 1.0,
    },
    "wettable_sulfur_80": {
        "product":        "Wettable Sulfur 80 WP (Sulfex / Thiovit)",
        "pack_size":      "500 g",
        "price_inr":      180,
        "dose_per_acre":  "750 g (diluted in 200 L water)",
        "coverage_acres": 0.67,
    },
    "triadimefon_25_wp": {
        "product":        "Triadimefon 25 WP (Bayleton)",
        "pack_size":      "100 g",
        "price_inr":      520,
        "dose_per_acre":  "100 g (diluted in 200 L water)",
        "coverage_acres": 1.0,
    },
    "azoxystrobin_propiconazole": {
        "product":        "Azoxystrobin 11% + Propiconazole 14.3% SC (Amistar Xtra)",
        "pack_size":      "200 ml",
        "price_inr":      1150,
        "dose_per_acre":  "200 ml (diluted in 200 L water)",
        "coverage_acres": 1.0,
    },
    "hexaconazole_5_ec": {
        "product":        "Hexaconazole 5 EC (Contaf Plus)",
        "pack_size":      "500 ml",
        "price_inr":      420,
        "dose_per_acre":  "400 ml (diluted in 200 L water)",
        "coverage_acres": 1.25,
    },
    "trichoderma_viride": {
        "product":        "Trichoderma viride (talc-based bio-fungicide)",
        "pack_size":      "1 kg",
        "price_inr":      210,
        "dose_per_acre":  "2.5 kg (mixed into topsoil or drench)",
        "coverage_acres": 0.4,
    },
    # Insecticides / vector control
    "imidacloprid_17_8_sl": {
        "product":        "Imidacloprid 17.8 SL (Confidor / Gaucho)",
        "pack_size":      "100 ml",
        "price_inr":      500,
        "dose_per_acre":  "50 ml (diluted in 200 L water)",
        "coverage_acres": 2.0,
    },
    "thiamethoxam_25_wg": {
        "product":        "Thiamethoxam 25 WG (Actara / Cruiser)",
        "pack_size":      "100 g",
        "price_inr":      620,
        "dose_per_acre":  "80 g (diluted in 200 L water)",
        "coverage_acres": 1.25,
    },
    "neem_oil_300_ppm": {
        "product":        "Neem Oil 300 ppm (Econeem Plus / Neemark)",
        "pack_size":      "1 litre",
        "price_inr":      420,
        "dose_per_acre":  "1.5 L (diluted in 200 L water)",
        "coverage_acres": 0.67,
    },
    # Bactericides
    "streptomycin_sulfate": {
        "product":        "Streptomycin Sulfate 90 SP + Tetracycline 10 SP",
        "pack_size":      "100 g",
        "price_inr":      380,
        "dose_per_acre":  "100 g (diluted in 200 L water)",
        "coverage_acres": 1.0,
    },
    # General labour / spray cost
    "labour_spray": {
        "product":        "Labour + tractor-mounted sprayer rental",
        "pack_size":      "per acre per spray",
        "price_inr":      350,
        "dose_per_acre":  "1 session",
        "coverage_acres": 1.0,
    },
}


def _build_rates_summary() -> str:
    """Return a compact JSON string of market rates for the LLM prompt."""
    return json.dumps(
        {k: {x: v[x] for x in ("product", "pack_size", "price_inr", "dose_per_acre", "coverage_acres")}
         for k, v in _MARKET_RATES.items()},
        indent=2,
    )


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an expert agricultural advisor for Indian farmers who specialises in \
plant disease management for cotton and wheat crops.
Your job is to produce a structured advisory report in plain, simple English \
that an average Indian farmer can easily understand.
Never use complicated scientific or technical language unless it is followed \
immediately by a simple everyday explanation in brackets.
Always respond with a valid JSON object having exactly these four keys:
  "disease_explanation", "remedy_explanation", "application_guide", "cost_analysis"

The first three values MUST be JSON **objects** (not plain strings).

"disease_explanation" MUST have this structure:
{
  "summary": "One short sentence summarising the cause",
  "cause_type": "Fungus" | "Bacteria" | "Virus" | "Insect" | "Environmental",
  "detail": "2-3 sentence simple explanation of why it happens",
  "risk_factors": ["short phrase 1", "short phrase 2", "short phrase 3"],
  "spreads_when": "One sentence — when / how the disease spreads faster"
}

"remedy_explanation" MUST have this structure:
{
  "remedies": [
    {
      "name": "Product or treatment name",
      "type": "Chemical" | "Biological" | "Cultural",
      "how_it_helps": "One short sentence",
      "after_treatment": "One sentence — what the farmer will see after applying"
    }
  ]
}

"application_guide" MUST have this structure:
{
  "best_time": "e.g. Early morning or late evening",
  "method": "e.g. Knapsack sprayer",
  "steps": [
    "Short action sentence for step 1",
    "Short action sentence for step 2"
  ],
  "repeat": "e.g. Repeat after 10-14 days if needed",
  "precautions": [
    "Short precaution 1",
    "Short precaution 2"
  ]
}
The "cost_analysis" value MUST be a JSON **object** (not a string) with ONLY these keys:
{
  "options": [
    {
      "option_label": "Option A",
      "product_name": "Full product name (pack size)",
      "packs_needed": <int>,
      "cost_per_pack": <number>,
      "total_cost": <number>,
      "why": "One short sentence explaining what this product does"
    }
  ],
  "recommended_index": <zero-based index of the best option in the options array>,
  "recommended_reason": "One short farmer-friendly sentence explaining why this option is best",
  "subsidy_scheme": "<short note on applicable govt schemes>"
}
RULES for cost_analysis:
- Each element in "options" represents ONE alternative remedy.  They are NOT meant
  to be applied together — the farmer picks ONE.
- Include every major remedy from the Recommended Remedies list as a separate option.
- "recommended_index" must be a valid index into "options".
- Do NOT include labour_cost, cost_per_acre, working, coverage, sprays, crop info,
  grand_total, or any other keys.
All output must be in English only.
"""


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_user_prompt(
    crop: str,
    disease: str,
    remedies: list[str],
    farm_size_acres: float,
    growth_stage: str,
    severity_level: str,
    recommendation_data: dict,
) -> str:
    causal_organism = recommendation_data.get("causal_organism", "Unknown")
    symptoms        = recommendation_data.get("symptoms", [])
    preventive      = recommendation_data.get("preventive_measures", [])
    sources         = recommendation_data.get("sources", [])

    symptoms_text   = "\n  - ".join(symptoms) if symptoms else "Not specified"
    remedies_text   = "\n  - ".join(remedies)  if remedies  else "General agricultural best practices"
    preventive_text = "\n  - ".join(preventive) if preventive else "Practice crop rotation and use certified seeds"

    rates_json = _build_rates_summary()

    return f"""\
## Farmer Advisory Request

**Crop**            : {crop.title()}
**Disease Detected** : {disease}
**Causal Organism**  : {causal_organism}
**Crop Growth Stage**: {growth_stage}
**Severity Level**  : {severity_level}
**Farm Size**       : {farm_size_acres} acres

### Observed Symptoms
  - {symptoms_text}

### Recommended Remedies (from rule-based engine)
  - {remedies_text}

### Preventive Measures
  - {preventive_text}

---

## Your Task

Using the information above, produce a JSON object with EXACTLY these four keys:

### 1. "disease_explanation"  (MUST be a JSON object, NOT a string)
Return a JSON object with:
  "summary" – one short sentence (max 20 words) about the root cause.
  "cause_type" – exactly one of: "Fungus", "Bacteria", "Virus", "Insect", "Environmental".
  "detail" – 2-3 simple sentences explaining why this disease happens on {crop}.
      Use very simple words a village farmer in India can understand.
      Give a real-life comparison if helpful.
  "risk_factors" – array of 3-5 short phrases (3-6 words each) describing weather,
      soil, or farming conditions that make this disease worse.
      Examples: "Cool humid weather", "Dense crop canopy", "Poor field drainage".
  "spreads_when" – one sentence about when / how it spreads faster.

### 2. "remedy_explanation"  (MUST be a JSON object, NOT a string)
Return a JSON object with:
  "remedies" – array where each element is an object with:
      "name" – product or treatment name (short),
      "type" – exactly one of: "Chemical", "Biological", "Cultural",
      "how_it_helps" – one sentence in simple English on how it fights the disease,
      "after_treatment" – one sentence on what the farmer will see after applying.

### 3. "application_guide"  (MUST be a JSON object, NOT a string)
Return a JSON object with:
  "best_time" – short phrase for timing (e.g. "Early morning or late evening").
  "method" – spray method (e.g. "Knapsack sprayer" or "Tractor-mounted sprayer").
  "steps" – array of 4-6 short action sentences (imperative mood) that the farmer
      follows in order. Include dosage, water quantity, how to mix. Keep each
      step to 1-2 short sentences max.
  "repeat" – one sentence on how many sprays and at what interval.
  "precautions" – array of 3-5 short safety phrases (e.g. "Wear gloves and mask").
  Tailor all advice to the current growth stage: {growth_stage} and severity:
      {severity_level}.

### 4. "cost_analysis"  (MUST be a JSON object, NOT a string)
Calculate the medicine cost for each remedy option separately for {farm_size_acres} acres.
The farmer will choose ONLY ONE option — these are alternatives, not combined.

Return a JSON object with ONLY these keys (no extra keys):
  "options" – an array where each element represents ONE alternative remedy option:
      "option_label" (e.g. "Option A", "Option B"),
      "product_name" (product name including pack size),
      "packs_needed" (integer – how many packs for {farm_size_acres} acres),
      "cost_per_pack" (number in ₹),
      "total_cost" (packs_needed × cost_per_pack, in ₹),
      "why" (one short sentence on what this medicine does).
  "recommended_index" – the zero-based index of the best option.
  "recommended_reason" – one short farmer-friendly sentence on why that option is best
      (e.g. cheaper, more effective, less toxic, better for this growth stage).
  "subsidy_scheme" – a short note on which government subsidy scheme (PM-KISAN,
      MIDH, state agri-dept) the farmer can check to get cheaper inputs.

Each major remedy from the Recommended Remedies list MUST appear as a separate option.
Do NOT combine multiple remedies into one option.
Do NOT include labour cost, cost per acre, coverage, or any other keys.

Use the product rates listed below. If a remedy is not in the rate list, use
the closest matching product. Round all monetary figures to the nearest ₹10.

---

## Current Indian Market Rates Reference
{rates_json}

---

**IMPORTANT RULES**
- Respond ONLY with a JSON object.  No markdown code fences, no extra text.
- All four keys must be present.
- All text must be in English only.
- Keep sentences short and easy to read.
- Round all monetary figures to the nearest ₹10.
"""


# ---------------------------------------------------------------------------
# Main async function
# ---------------------------------------------------------------------------

@traceable(name="plant-disease-advisory", run_type="llm")
async def generate_detailed_recommendation(
    crop: str,
    disease: str,
    remedies: list[str],
    farm_size_acres: float,
    growth_stage: str,
    severity_level: str,
    recommendation_data: dict | None = None,
) -> dict[str, Any]:
    """Call the OpenAI Chat Completions API and return a structured advisory dict.

    Parameters
    ----------
    crop               : "cotton" or "wheat"
    disease            : Disease class predicted by the image model
    remedies           : Remedy list from the rule-based recommendation engine
    farm_size_acres    : Size of the farm in acres (must be > 0)
    growth_stage       : Current crop growth stage string
    severity_level     : "Mild", "Moderate", or "Severe"
    recommendation_data: Full dict returned by get_enhanced_recommendation()

    Returns
    -------
    dict with keys:
        disease_explanation  – why the disease happens (farmer-friendly)
        remedy_explanation   – why the remedies work
        application_guide    – how and when to apply
        cost_analysis        – itemised cost for the farm size
        model_used           – OpenAI model identifier
        tokens_used          – total tokens consumed
    """
    if recommendation_data is None:
        recommendation_data = {}

    user_prompt = _build_user_prompt(
        crop=crop,
        disease=disease,
        remedies=remedies,
        farm_size_acres=farm_size_acres,
        growth_stage=growth_stage,
        severity_level=severity_level,
        recommendation_data=recommendation_data,
    )

    headers = {
        "Authorization": f"Bearer {_OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": _MODEL,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        "temperature": 0.4,
        "max_tokens": 3000,
        "response_format": {"type": "json_object"},
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
            )
            resp.raise_for_status()
            response_data = resp.json()
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(
            f"OpenAI API call failed (HTTP {exc.response.status_code}): "
            f"{exc.response.text[:500]}"
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"OpenAI API call failed: {exc}") from exc

    raw_text = (response_data["choices"][0]["message"]["content"] or "").strip()
    usage = response_data.get("usage", {})
    tokens = usage.get("total_tokens", 0)
    model_used = response_data.get("model", _MODEL)

    try:
        parsed: dict[str, Any] = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"OpenAI returned malformed JSON: {exc}\nRaw response:\n{raw_text[:500]}"
        ) from exc

    _REQUIRED_KEYS = {
        "disease_explanation",
        "remedy_explanation",
        "application_guide",
        "cost_analysis",
    }
    missing = _REQUIRED_KEYS - set(parsed.keys())
    if missing:
        raise RuntimeError(
            f"OpenAI response missing required keys: {missing}\nRaw: {raw_text[:500]}"
        )

    return {
        "disease_explanation": _to_json_str(parsed["disease_explanation"]),
        "remedy_explanation":  _to_json_str(parsed["remedy_explanation"]),
        "application_guide":   _to_json_str(parsed["application_guide"]),
        "cost_analysis":       _format_cost_analysis(parsed["cost_analysis"]),
        "model_used":          model_used,
        "tokens_used":         tokens,
    }


def _to_json_str(value: Any) -> str:
    """Ensure a value is serialised as a JSON string for the frontend."""
    if isinstance(value, dict):
        return json.dumps(value)
    if isinstance(value, str):
        return value
    return str(value)


def _format_cost_analysis(value: Any) -> str:
    """Convert cost_analysis to a JSON string so the frontend structured renderer is used."""
    if isinstance(value, dict):
        # Normalise subsidy key so frontend can always find it under "subsidy_scheme"
        if "subsidy_scheme" not in value:
            for alt in ("subsidy_note", "note"):
                if alt in value:
                    value["subsidy_scheme"] = value.pop(alt)
                    break
        return json.dumps(value)

    if isinstance(value, str):
        return value

    return str(value)
