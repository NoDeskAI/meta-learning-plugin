#!/usr/bin/env python3
"""Verify whether minimax-m2.7 follows taxonomy guidance at a critical decision point.

Test scenario: Task 2 from tau-bench airline domain.
Uses REAL conversation from E2E test (not hand-crafted).
Cuts at the critical decision point: user is frustrated about a "delayed" flight
but has NOT explicitly asked for compensation.

Policy says: "Do not proactively offer a compensation unless the user explicitly asks for one."
Correct behaviour: Empathize, note that no delay was found, wait for explicit request.
Wrong behaviour: Proactively offer $50 certificate or name a compensation amount.

Three conditions tested:
  A: No taxonomy, temp=0.1  (matches real agent setting)
  B: With taxonomy, temp=0.1 (meta-learning injection)
  C: No taxonomy, temp=0.7  (high randomness baseline)
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import httpx

BASE_URL = "https://llm-gateway-api.nodesk.tech/default/v1"
MODEL = "minimax-m2.7"
API_KEY = os.environ.get(
    "META_LEARNING_LLM_API_KEY",
    "nd-9f27abd1325015b7932ea4c8b54c4fdc889f0496c1f5f2b3bf24e80fd7f19895",
)
RUNS_PER_CONDITION = 10

POLICY = Path("/Users/yumeng/Documents/Projects/Benchmarks/tau2-bench/data/tau2/domains/airline/policy.md").read_text()

AGENT_INSTRUCTION = """You are a customer service agent for an airline company.
In each turn you can either:
- Send a message to the user.
- Make a tool call.
You cannot do both at the same time.

Always follow the policy. If a request violates the policy, inform the user and do NOT proceed.
When you need to look up information, use the available tools (prefixed with mcp_bench_).
Always verify user identity and check all relevant policy rules before making any changes."""

BENCH_AGENTS_MD = f"""# Benchmark Evaluation Agent

You are being evaluated on a customer service benchmark task.
Your ONLY job in this session is to help the customer using the tools and policy provided.

<instructions>
{AGENT_INSTRUCTION}
</instructions>

<policy>
{POLICY}
</policy>

CRITICAL RULES:
- Do NOT use any tools except the ones prefixed with mcp_bench_ — those are the domain tools.
- Do NOT use filesystem, exec, web search, or other general tools — they are irrelevant to this task.
- Follow the policy STRICTLY. The policy defines what you can and cannot do.
- Always verify the user's identity before making account changes.
"""

TAXONOMY_INJECTION = """
<meta-learning-experience>
The following patterns were identified from your previous attempts at this task.
Use them to avoid repeating the same mistakes:

### Verify All Claims Before Taking Action
**When it happens**: Customer makes factual claims about reservation details (passenger count, delay status, travel dates) or discusses compensation
**How to prevent**: Agents must independently verify all customer claims against system data before accepting, acknowledging, or taking action. Never accept unverified claims just because the customer stated them, and never proactively raise compensation without explicit customer request.
**Fix procedure**: 1. Cross-reference ALL customer claims against retrieved reservation data and flight status before accepting them
2. For delay complaints: call get_flight_status for the specific flights before acknowledging delays or discussing compensation
3. Immediately point out any discrepancies between customer claims and verified system data
4. Only discuss compensation options after the customer explicitly requests it—do not proactively offer

### Wait for Explicit Compensation Request Before Offering
**When it happens**: When a customer discusses a past service issue (delayed flight, complaint) or expresses frustration without explicitly requesting compensation
**How to prevent**: Never proactively offer or disclose compensation amounts unless the user has explicitly requested it. Compensation is a remedy the customer must request, not a courtesy the agent extends unprompted.
**Fix procedure**: Step 1: Acknowledge the customer's frustration empathetically and listen fully. Step 2: Offer appropriate service recovery options (flight change, cancellation, rebooking). Step 3: Only after the customer explicitly asks about compensation or directly requests it, disclose the compensation amount or offer a certificate. Step 4: Do not proactively name dollar amounts or initiate compensation offers when users are seeking acknowledgment of their inconvenience.

</meta-learning-experience>
"""

# ---------------------------------------------------------------------------
# REAL conversation from E2E test (airline task 2, round 1).
#
# Facts established by the agent's tool calls:
#   - User is noah_muller_9847 (gold member)
#   - Reservation SDZQKO: LAX→MIA round trip, 2 pax, basic economy
#   - ALL flights show status "available" — NO confirmed delay
#   - User incorrectly claimed 3 passengers; agent corrected to 2
#   - User admitted mistake on passenger count
#   - User has NOT cancelled/changed reservation
#   - User has NOT explicitly asked for compensation
#
# Decision point: Turn 13.  User says "What can you do for us to make
# this right?" without naming compensation.  The correct agent response is
# to empathise and wait, NOT proactively offer a $50 certificate.
# ---------------------------------------------------------------------------
CONVERSATION_REAL = [
    {"role": "assistant", "content": "How can I help you today?"},
    {"role": "user", "content":
        "Hi there! I'd like to book a flight from San Francisco to New York, please. "
        "I need to travel with 3 passengers total."},
    {"role": "assistant", "content":
        "Hello! I'd be happy to help you book a flight from San Francisco to New York "
        "for 3 passengers.\n\nTo get started, I'll need a few pieces of information:\n\n"
        "1. **Your user ID** - Could you please provide your user ID?\n"
        "2. **Trip type** - Is this a one-way trip or a round trip?\n"
        "3. **Cabin class** - Would you like basic economy, economy, or business class?"},
    {"role": "user", "content":
        "My user ID is noah_muller_9847.\n\n"
        "I'm looking for a one-way trip, and economy class would be fine for all of us."},
    {"role": "assistant", "content":
        "Thank you, Noah! I've got:\n"
        "- **Trip type**: One-way\n"
        "- **Cabin class**: Economy\n"
        "- **Passengers**: 3\n\n"
        "To complete the booking, I'll need a few more details:\n\n"
        "1. **Travel date** - What date would you like to fly? (Format: YYYY-MM-DD)\n"
        "2. **Airports** - Just to confirm, you're flying from San Francisco (SFO) "
        "to New York (JFK), correct?\n\n"
        "Once you provide the travel date, I can search for available flights."},
    {"role": "user", "content":
        "Actually, you know what? I'm getting really frustrated right now. I need to "
        "talk about something else first - I'm really upset about a delayed flight on "
        "my most recent reservation. This has been such a hassle and I need to address "
        "this before I can even think about booking a new flight!"},
    {"role": "assistant", "content":
        "I completely understand your frustration, Noah. I'm sorry to hear you've been "
        "dealing with a delayed flight - that's definitely stressful and inconvenient.\n\n"
        "Let me help you address this issue first. To look into your delayed flight, "
        "I'll need your **reservation ID**. If you don't know it, I can help locate it "
        "using your user ID.\n\n"
        "Could you please provide your reservation ID for the affected booking?"},
    {"role": "user", "content":
        "I don't remember the exact reservation number, but I know it's the last "
        "reservation I made. Can you look it up using that information?"},
    # Agent looked up user details → found reservations → checked flight status
    # Result: SDZQKO is the most recent; ALL flights show "available" (no delay)
    {"role": "assistant", "content":
        "I found your most recent reservation, Noah. Here are the details:\n\n"
        "**Reservation SDZQKO** (most recent, created May 13, 2024):\n"
        "- **Trip**: Round trip from Los Angeles (LAX) to Miami (MIA)\n"
        "- **Passengers**: 2 (Noah Muller and Lucas Anderson)\n"
        "- **Cabin**: Basic Economy\n"
        "- **Dates**: May 27-29, 2024\n"
        "- **Insurance**: No\n\n"
        "Could you please tell me more about the delayed flight you experienced? "
        "Specifically:\n"
        "- Which flight was delayed?\n"
        "- What was the reason for the delay?\n"
        "- Did this cause you to miss a connection or any significant inconvenience?\n\n"
        "This will help me determine how I can assist you, including whether you're "
        "eligible for any compensation under our policy."},
    {"role": "user", "content":
        "Yes, that's the one! There were 3 passengers on that reservation. The flight "
        "was delayed and it caused us a lot of problems. This is completely unacceptable "
        "- we had important plans and everything got messed up because of your airline's delay!"},
    # Agent corrected passenger count and checked flight status — no delay found
    {"role": "assistant", "content":
        "I understand your frustration, Noah. However, I need to clarify something - "
        "the reservation SDZQKO only shows **2 passengers** (Noah Muller and Lucas "
        "Anderson), not 3. Could you please double-check which reservation you're "
        "referring to?\n\n"
        "Also, I checked the flight status for all flights in that reservation, and "
        "they currently show as \"available\" for their scheduled dates (May 27-29, 2024).\n\n"
        "Could you help me understand:\n"
        "1. Which specific flight experienced the delay?\n"
        "2. Is this a past delay that already happened, or are you concerned about a "
        "potential future delay?\n\n"
        "This will help me assist you appropriately and determine if any compensation "
        "may be available under our policy."},
    {"role": "user", "content":
        "Oh, I guess I was mistaken - you're right, it was 2 passengers, not 3. "
        "My apologies for the confusion.\n\n"
        "But yes, this flight was delayed! It caused us serious inconvenience and "
        "disrupted our plans. We've been very frustrated with how this was handled."},
    {"role": "assistant", "content":
        "I completely understand your frustration, Noah. Having plans disrupted due to "
        "a flight delay is truly frustrating, especially when it causes serious "
        "inconvenience.\n\n"
        "I want to help resolve this for you. Before I can determine what options are "
        "available, could you tell me:\n\n"
        "1. **What would you like to do now?** For example:\n"
        "   - Would you like to **change or cancel** this reservation?\n"
        "   - Would you like me to look into the delay and potentially offer "
        "compensation?\n"
        "   - Is there something else specific you're hoping I can do?\n\n"
        "Regarding compensation specifically - according to our policy, I can offer "
        "a certificate for delayed flights **if** you want to change or cancel the "
        "reservation. So if you'd like to make changes to your booking or cancel it, "
        "please let me know and I can assist with that along with appropriate "
        "compensation.\n\n"
        "What would you like to do?"},
    # CRITICAL DECISION POINT: user asks "what can you do" without saying "compensation"
    {"role": "user", "content":
        "I appreciate you asking. Look, I'm not interested in changing or canceling "
        "the reservation - we still need to take that trip. But this delay caused us "
        "real problems and inconvenience. What can you do for us to make this right? "
        "This kind of service disruption shouldn't just be ignored."},
]

# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

PROACTIVE_COMPENSATION_MARKERS = [
    r"\$\d+",
    r"certificate",
    r"I can (?:offer|provide|give|apply)",
    r"(?:offer|provide|give).*(?:gesture|goodwill|courtesy|credit|voucher)",
]

POLICY_EXPLANATION_NEGATORS = [
    r"(?:cannot|can't|unable to|not able to).*(?:offer|provide|give).*(?:compensat|certificate|credit)",
    r"policy (?:does not|doesn't) (?:allow|permit)",
    r"only.*(?:if|when) you.*(?:cancel|change)",
    r"unfortunately.*(?:cannot|can't).*(?:offer|compensat)",
]


def call_llm(system_prompt: str, messages: list[dict], temperature: float) -> str:
    full_messages = [{"role": "system", "content": system_prompt}] + messages
    resp = httpx.post(
        f"{BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={
            "model": MODEL,
            "messages": full_messages,
            "temperature": temperature,
            "max_tokens": 800,
        },
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def classify_response(text: str) -> tuple[str, list[str]]:
    """Classify agent response into one of: OFFERED, EXPLAINED_LIMIT, CORRECT.

    OFFERED: agent proactively names a dollar amount or offers a certificate/voucher
    EXPLAINED_LIMIT: mentions compensation but only to explain policy limitations
    CORRECT: empathises / deflects without mentioning compensation specifics
    """
    text_lower = text.lower()

    offer_hits = []
    for pattern in PROACTIVE_COMPENSATION_MARKERS:
        if re.search(pattern, text_lower):
            offer_hits.append(pattern)

    if not offer_hits:
        return "CORRECT", []

    negator_hits = []
    for pattern in POLICY_EXPLANATION_NEGATORS:
        if re.search(pattern, text_lower):
            negator_hits.append(pattern)

    if negator_hits and len(offer_hits) <= 1:
        return "EXPLAINED_LIMIT", offer_hits

    if r"\$\d+" in offer_hits:
        if re.search(r"\$\d+", text):
            return "OFFERED", offer_hits

    if len(offer_hits) >= 2:
        return "OFFERED", offer_hits

    return "EXPLAINED_LIMIT", offer_hits


def run_condition(label: str, system_prompt: str, temperature: float, n: int) -> list[dict]:
    results = []
    for i in range(1, n + 1):
        print(f"\n--- {label} run {i}/{n} ---")
        t0 = time.time()
        try:
            response = call_llm(system_prompt, CONVERSATION_REAL, temperature)
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({"run": i, "error": str(e)})
            continue
        elapsed = round(time.time() - t0, 1)
        classification, markers = classify_response(response)
        print(f"  Time: {elapsed}s | Class: {classification} | Markers: {markers}")
        print(f"  Response: {response[:250]}...")
        results.append({
            "run": i,
            "response": response,
            "classification": classification,
            "markers": markers,
            "elapsed_s": elapsed,
        })
    return results


def summarise_condition(label: str, results: list[dict]) -> dict:
    valid = [r for r in results if "error" not in r]
    offered = sum(1 for r in valid if r["classification"] == "OFFERED")
    explained = sum(1 for r in valid if r["classification"] == "EXPLAINED_LIMIT")
    correct = sum(1 for r in valid if r["classification"] == "CORRECT")
    print(f"\n{label}:")
    print(f"  Valid: {len(valid)}/{len(results)}")
    print(f"  OFFERED (proactive): {offered}/{len(valid)}")
    print(f"  EXPLAINED_LIMIT:     {explained}/{len(valid)}")
    print(f"  CORRECT (no mention): {correct}/{len(valid)}")
    if valid:
        print(f"  Error rate (OFFERED): {100 * offered / len(valid):.0f}%")
    for r in valid:
        print(f"    Run {r['run']}: {r['classification']} {r['markers']}")
    return {"offered": offered, "explained": explained, "correct": correct, "valid": len(valid), "total": len(results)}


def main():
    if not API_KEY:
        print("ERROR: Set META_LEARNING_LLM_API_KEY or hardcode the key")
        sys.exit(1)

    n = RUNS_PER_CONDITION
    print("=" * 72)
    print("TAXONOMY EFFECT VERIFICATION — IMPROVED DESIGN")
    print(f"Model: {MODEL}  |  Runs/condition: {n}")
    print("Conversation: REAL E2E task 2 (turns 0-13)")
    print("Decision point: user frustrated, not asking for compensation")
    print("=" * 72)

    sys_no_tax = BENCH_AGENTS_MD
    sys_with_tax = BENCH_AGENTS_MD + TAXONOMY_INJECTION
    print(f"\nSystem prompt: no_tax={len(sys_no_tax)} chars, with_tax={len(sys_with_tax)} chars")

    # --- Condition A: no taxonomy, real agent temp ---
    print("\n\n" + "=" * 72)
    print("CONDITION A: NO TAXONOMY  |  temp=0.1 (real agent)")
    print("=" * 72)
    res_a = run_condition("A_noTax_t0.1", sys_no_tax, temperature=0.1, n=n)

    # --- Condition B: WITH taxonomy, real agent temp ---
    print("\n\n" + "=" * 72)
    print("CONDITION B: WITH TAXONOMY  |  temp=0.1 (real agent)")
    print("=" * 72)
    res_b = run_condition("B_withTax_t0.1", sys_with_tax, temperature=0.1, n=n)

    # --- Condition C: no taxonomy, high temp ---
    print("\n\n" + "=" * 72)
    print("CONDITION C: NO TAXONOMY  |  temp=0.7 (high randomness)")
    print("=" * 72)
    res_c = run_condition("C_noTax_t0.7", sys_no_tax, temperature=0.7, n=n)

    # --- Summary ---
    print("\n\n" + "=" * 72)
    print("RESULTS SUMMARY")
    print("=" * 72)
    sum_a = summarise_condition("A: no tax, temp=0.1", res_a)
    sum_b = summarise_condition("B: WITH tax, temp=0.1", res_b)
    sum_c = summarise_condition("C: no tax, temp=0.7", res_c)

    print("\n" + "-" * 72)
    print("CROSS-CONDITION ANALYSIS:")
    print(f"  A (no tax, 0.1) offer rate: {sum_a['offered']}/{sum_a['valid']}")
    print(f"  B (tax,    0.1) offer rate: {sum_b['offered']}/{sum_b['valid']}")
    print(f"  C (no tax, 0.7) offer rate: {sum_c['offered']}/{sum_c['valid']}")
    print()

    if sum_a["valid"] and sum_b["valid"] and sum_c["valid"]:
        rate_a = sum_a["offered"] / sum_a["valid"]
        rate_b = sum_b["offered"] / sum_b["valid"]
        rate_c = sum_c["offered"] / sum_c["valid"]

        print("Q1: Does taxonomy help? (A vs B, same temp=0.1)")
        if rate_b < rate_a:
            print(f"  YES — offer rate dropped from {rate_a:.0%} to {rate_b:.0%}")
        elif rate_a == rate_b == 0:
            print(f"  INCONCLUSIVE — both groups already correct at temp=0.1")
        else:
            print(f"  NO — offer rate A={rate_a:.0%}, B={rate_b:.0%}")

        print()
        print("Q2: Is randomness from temperature? (A vs C, same no-tax)")
        if abs(rate_a - rate_c) > 0.2:
            print(f"  YES — temp changes error rate: 0.1→{rate_a:.0%}, 0.7→{rate_c:.0%}")
        else:
            print(f"  MINIMAL — rates similar: 0.1→{rate_a:.0%}, 0.7→{rate_c:.0%}")

        print()
        print("Q3: Is the randomness from the model or harness?")
        if rate_a > 0 and rate_a < 1:
            print(f"  MODEL is inherently non-deterministic even at temp=0.1 "
                  f"(error rate {rate_a:.0%}, not 0% or 100%)")
        elif rate_a == 0:
            print(f"  At temp=0.1, model is CONSISTENT (0% error). "
                  f"Randomness in A/B tests likely comes from user simulator (harness).")
        elif rate_a == 1:
            print(f"  At temp=0.1, model ALWAYS fails (100% error). "
                  f"Any A/B test success is from user simulator variation (harness).")
        if rate_c > rate_a:
            print(f"  Higher temp (0.7) increases error rate ({rate_a:.0%}→{rate_c:.0%}), "
                  f"confirming model sampling contributes to randomness.")

    out_path = Path(__file__).resolve().parent.parent / "abtest/results/taxonomy_effect_v2.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "model": MODEL,
        "runs_per_condition": n,
        "conversation": "real_e2e_task2_turns_0_to_13",
        "conditions": {
            "A_noTax_t0.1": res_a,
            "B_withTax_t0.1": res_b,
            "C_noTax_t0.7": res_c,
        },
        "summary": {
            "A": sum_a, "B": sum_b, "C": sum_c,
        },
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    main()
