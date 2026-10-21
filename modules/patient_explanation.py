"""
Patient Explanation Module
Friendly, non-technical explanations of arthritis types & what happens to joints/body
For use in GUI/API responses, chatbot
"""

ARTHRITIS_EXPLANATIONS = {
    "Osteoarthritis": {
        "what_happens": """
**What is happening in your joints?**
Your joint cartilage (the smooth cushion between bones) is wearing away over time. 
Think of it like tire tread - normal use + aging = thinner cushion → bone-on-bone friction → pain/stiffness.

**Your body:** Bones rub together → small bone spurs form → joint shape changes
**Daily impact:** Worse with activity (stairs, walking), better with rest
**Good news:** Slow progression, very treatable with weight loss + physio!
        """,
        "management": "Weight loss 1kg = 4kg less knee pressure! Swimming best exercise.",
        "when_worry": "Sudden swelling or locking = see doctor ASAP"
    },
    "Rheumatoid Arthritis": {
        "what_happens": """
**Your immune system is attacking your joints!**
RA is **autoimmune** - your body mistakes joint lining as enemy → sends army of inflammatory cells → swelling + damage.

**Your body:** Joint lining thickens → erodes bone/cartilage → can affect other organs (lungs, eyes)
**Daily impact:** Morning stiffness >1hr, same joints both sides (hands, wrists)
**Urgent:** Early DMARD treatment stops 70% joint damage!
        """,
        "management": "Methotrexate (pill) + regular blood tests. Never skip doses!",
        "when_worry": "Fever + joint pain = infection risk (meds suppress immunity)"
    },
    "Gout": {
        "what_happens": """
**Uric acid crystals stabbing your joints!**
Your body makes/holds too much uric acid → forms sharp crystals in joints (like broken glass).

**Your body:** Big toe classic (70% first attack). Red, hot, can't touch it!
**Daily impact:** Sudden night attacks, lasts 7-10 days untreated
**Triggers:** Beer, red meat, dehydration, crash diets
        """,
        "management": "Allopurinol daily + 3L water. Attack: Ice + ibuprofen.",
        "when_worry": "Fever or multiple joints = seek ER (could be septic)"
    },
    "Psoriatic Arthritis": {
        "what_happens": """
**Psoriasis jumped from skin to joints!**
Same immune problem causing skin plaques now attacks joints + tendons.

**Your body:** Asymmetric (one knee, few fingers). Nails pitted/separated common
**Daily impact:** Stiff fingers ('sausage digits'), back pain possible
**Connection:** 30% psoriasis patients get PsA eventually
        """,
        "management": "Same skin cream for joints! Biologics very effective.",
        "when_worry": "Eye redness/pain = uveitis emergency"
    },
    "Ankylosing Spondylitis": {
        "what_happens": """
**Spine inflammation trying to fuse vertebrae!**
Autoimmune attacking spine ligaments → inflammation → new bone formation ('bamboo spine')

**Your body:** Lower back pain/stiffness worse AM/night, improves with movement
**Daily impact:** Can't touch toes, feels better after hot shower/exercise
**Young people:** Usually starts teens/20s, men > women
        """,
        "management": "Daily stretching + swimming. Stop smoking worsens fusion!",
        "when_worry": "Can't expand chest = lung involvement"
    },
    "Unknown": {
        "what_happens": """
**Joint problem needs rheumatologist diagnosis**
Could be early arthritis, injury, infection, or other cause.
Don't guess - get bloodwork + X-rays for proper diagnosis!
        """,
        "management": "Track symptoms: which joints, time of day, triggers, family history",
        "when_worry": "Fever, redness, can't bear weight = ER now"
    }
}

def get_patient_explanation(arthritis_type: str) -> dict:
    """Get complete patient-friendly explanation by type"""
    return ARTHRITIS_EXPLANATIONS.get(arthritis_type, ARTHRITIS_EXPLANATIONS["Unknown"])

def format_explanation(arthritis_type: str, confidence: float = 0.0) -> str:
    """Format for display/chatbot"""
    expl = get_patient_explanation(arthritis_type)
    conf_text = f" (confidence {confidence:.0%})" if confidence > 0 else ""
    return f"**{arthritis_type}{conf_text}:**\n\n{expl['what_happens']}\n\n**Daily Tips:** {expl['management']}\n**Call Doctor:** {expl['when_worry']}"

if __name__ == "__main__":
    print(format_explanation("Rheumatoid Arthritis", 0.85))

