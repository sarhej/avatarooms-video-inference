"""40-prompt evaluation set for the LTX-2 POC.

Six categories, six languages, all designed to stress different aspects of
text-to-video generation:

- C1 — single avatar dialogue (10 prompts)
- C2 — two-character conversation (6 prompts)
- C3 — cinematic narrative, no dialogue (8 prompts)
- C4 — stylised / non-photoreal (6 prompts)
- C5 — explicit camera movement (5 prompts)
- C6 — long-form (5 prompts, capped to 10s)

All clips are 9:16, 6-10 seconds, audio on where applicable.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Prompt:
    id: str
    category: str
    lang: str
    duration: int
    text: str


PROMPTS: list[Prompt] = [
    # ------------------------------------------------------------------
    # Category 1 — single avatar, dialogue
    # ------------------------------------------------------------------
    Prompt(
        id="EN-1", category="C1", lang="en", duration=10,
        text=(
            "A bearded fantasy knight in silver armour stands in a stone-walled great hall. "
            "Looking directly at camera, he speaks in a low, calm voice: \"We march at dawn. "
            "Tell the others to be ready.\" Warm torchlight flickers across his face. "
            "Close-up portrait, 9:16, 10 seconds."
        ),
    ),
    Prompt(
        id="EN-2", category="C1", lang="en", duration=10,
        text=(
            "A young woman with red hair and a leather jacket leans against a brick wall in a "
            "rain-soaked alley at night. Looking off-camera, she says with quiet determination: "
            "\"I'm not running anymore. Not from this.\" Neon signs reflect in puddles behind "
            "her. Close-up, 9:16, 10 seconds."
        ),
    ),
    Prompt(
        id="EN-3", category="C1", lang="en", duration=10,
        text=(
            "An elderly fisherman with deep wrinkles and a wool cap sits on a wooden dock at "
            "dawn. Looking out to sea, he speaks gently: \"My father said the same thing, and "
            "his father before him. The tide always turns.\" Soft pink sunrise light. "
            "Close-up portrait, 9:16, 10 seconds."
        ),
    ),
    Prompt(
        id="FR-1", category="C1", lang="fr", duration=10,
        text=(
            "Une jeune femme aux cheveux bruns dans un petit café parisien sourit à la caméra "
            "et dit doucement: « Je crois que je suis prête à recommencer. » Lumière chaude du "
            "matin par la fenêtre. Plan rapproché, 9:16, 10 secondes."
        ),
    ),
    Prompt(
        id="FR-2", category="C1", lang="fr", duration=10,
        text=(
            "Un homme barbu en imperméable beige marche sur un quai au bord de la Seine. Il "
            "regarde la caméra et dit avec assurance: « Demain, tout sera différent. » Lumière "
            "grise d'après-midi. Plan rapproché en mouvement, 9:16, 10 secondes."
        ),
    ),
    Prompt(
        id="DE-1", category="C1", lang="de", duration=10,
        text=(
            "Eine junge Frau mit blonden Haaren sitzt in einem Berliner Café und blickt zur "
            "Kamera. Sie sagt leise auf Deutsch: \"Manchmal muss man einfach loslassen, um "
            "weiterzukommen.\" Warmes Nachmittagslicht. Nahaufnahme, 9:16, 10 Sekunden."
        ),
    ),
    Prompt(
        id="DE-2", category="C1", lang="de", duration=10,
        text=(
            "Ein älterer Mann mit Brille und grauem Anzug steht in einer Bibliothek vor hohen "
            "Bücherregalen. Er schaut zur Kamera und sagt nachdenklich: \"Geschichten sind das, "
            "was uns verbindet.\" Gedämpftes Goldlicht. Porträt-Nahaufnahme, 9:16, 10 Sekunden."
        ),
    ),
    Prompt(
        id="ES-1", category="C1", lang="es", duration=10,
        text=(
            "Una mujer joven con pelo oscuro está en una azotea de Madrid al atardecer. Mira a "
            "la cámara y dice con calma: \"Ya no tengo miedo de empezar de nuevo.\" Luz dorada "
            "del crepúsculo detrás de ella. Primer plano, 9:16, 10 segundos."
        ),
    ),
    Prompt(
        id="IT-1", category="C1", lang="it", duration=10,
        text=(
            "Un giovane uomo con barba corta seduto in un bar di Roma. Guarda la telecamera e "
            "dice con un sorriso: \"La vita è troppo breve per non provare.\" Luce calda del "
            "pomeriggio. Primo piano, 9:16, 10 secondi."
        ),
    ),
    Prompt(
        id="PL-1", category="C1", lang="pl", duration=10,
        text=(
            "Młoda kobieta z długimi włosami stoi w warszawskim parku jesienią. Patrzy w "
            "kamerę i mówi spokojnie: \"Czasem trzeba po prostu zacząć od nowa.\" Złote światło "
            "popołudnia. Zbliżenie, 9:16, 10 sekund."
        ),
    ),

    # ------------------------------------------------------------------
    # Category 2 — two-character conversation
    # ------------------------------------------------------------------
    Prompt(
        id="CONV-1", category="C2", lang="en", duration=10,
        text=(
            "Two friends sit across from each other at a small wooden café table. A woman with "
            "curly brown hair leans forward and says, \"Are you sure about this?\" A man with "
            "glasses opposite her smiles and replies, \"I've never been more sure.\" Warm "
            "interior lighting, slight depth-of-field. Wide two-shot, 9:16, 10 seconds."
        ),
    ),
    Prompt(
        id="CONV-2", category="C2", lang="en", duration=10,
        text=(
            "Two business people stand in a glass-walled office at dusk. A tall man in a navy "
            "suit says firmly, \"We need an answer by Friday.\" A woman in a grey blazer "
            "crosses her arms and answers, \"You'll have it by Thursday.\" Office skyline "
            "behind them. Medium two-shot, 9:16, 10 seconds."
        ),
    ),
    Prompt(
        id="CONV-3", category="C2", lang="en", duration=10,
        text=(
            "A couple walks along a beach at sunset. The woman stops and says with frustration, "
            "\"You promised this would be different.\" The man turns to face her and replies "
            "softly, \"And I meant it. I still do.\" Golden hour light. Wide two-shot, 9:16, "
            "10 seconds."
        ),
    ),
    Prompt(
        id="CONV-4", category="C2", lang="en", duration=10,
        text=(
            "Two old friends sit on a park bench in autumn. The first, an older man in a tweed "
            "coat, says, \"It's been twenty years, can you believe it?\" The second, his old "
            "friend with a grey beard, laughs and replies, \"Feels like yesterday.\" Soft "
            "golden light through orange leaves. Medium two-shot, 9:16, 10 seconds."
        ),
    ),
    Prompt(
        id="CONV-5", category="C2", lang="fr", duration=10,
        text=(
            "Deux amies dans un restaurant parisien. La première, brune, dit en souriant: « Tu "
            "vas vraiment déménager à Lyon ? » La seconde, blonde, répond avec hésitation: « Je "
            "pense que oui. » Lumière douce de bougies. Plan moyen, 9:16, 10 secondes."
        ),
    ),
    Prompt(
        id="CONV-6", category="C2", lang="de", duration=10,
        text=(
            "Zwei Freunde gehen durch einen verschneiten Park. Der Mann links sagt: \"Hast du "
            "dich entschieden?\" Die Frau rechts antwortet ruhig: \"Ja. Ich gehe nach Berlin.\" "
            "Kaltes Winterlicht, sanfter Schneefall. Medium-Shot, 9:16, 10 Sekunden."
        ),
    ),

    # ------------------------------------------------------------------
    # Category 3 — cinematic narrative, no dialogue
    # ------------------------------------------------------------------
    Prompt(
        id="CINE-1", category="C3", lang="en", duration=10,
        text=(
            "A lone hiker stands on a snowy mountain ridge at dawn, silhouetted against a sky "
            "turning from deep blue to soft orange. Mist drifts through the valley below. The "
            "camera slowly pulls back to reveal more peaks. 9:16, 10 seconds."
        ),
    ),
    Prompt(
        id="CINE-2", category="C3", lang="en", duration=10,
        text=(
            "A vintage red telephone box on an empty rain-soaked London street at night. "
            "Streetlights reflect on wet cobblestones. A black cab passes slowly in the "
            "distance. Atmospheric ambient sound — rain, distant traffic. 9:16, 10 seconds."
        ),
    ),
    Prompt(
        id="CINE-3", category="C3", lang="en", duration=10,
        text=(
            "An aerial shot descending toward a small oasis in a vast golden desert. Date "
            "palms cluster around a still pond. The shadow of the drone moves across the sand "
            "as the camera approaches. Bright midday light. 9:16, 10 seconds."
        ),
    ),
    Prompt(
        id="CINE-4", category="C3", lang="en", duration=10,
        text=(
            "A diver swims through a glowing underwater cave, beams of sunlight piercing the "
            "blue-green water from cracks above. Schools of small silver fish dart away. Slow "
            "tracking shot following the diver. 9:16, 10 seconds."
        ),
    ),
    Prompt(
        id="CINE-5", category="C3", lang="en", duration=10,
        text=(
            "A red fox steps cautiously through a misty pine forest at sunrise. Shafts of "
            "golden light cut through the trees. The fox pauses, ears twitching, then "
            "continues. Naturalistic ambient sound — birds, distant water. 9:16, 10 seconds."
        ),
    ),
    Prompt(
        id="CINE-6", category="C3", lang="en", duration=10,
        text=(
            "A young woman in a long coat walks across a wet plaza in Tokyo at night. Neon "
            "signs in Japanese reflect in puddles. She doesn't look at the camera. Crowd noise "
            "softens into rain. Medium tracking shot, 9:16, 10 seconds."
        ),
    ),
    Prompt(
        id="CINE-7", category="C3", lang="en", duration=10,
        text=(
            "Slow tracking shot through a Diwali festival in an Indian street. Hundreds of "
            "small oil lamps line the doorways. Children run past with sparklers. Warm golden "
            "lighting, gentle festive music. 9:16, 10 seconds."
        ),
    ),
    Prompt(
        id="CINE-8", category="C3", lang="en", duration=10,
        text=(
            "A small Alpine village blanketed in fresh snow at twilight. Smoke rises from "
            "chimneys. The camera slowly pushes in toward a single lit window. Distant church "
            "bells ring softly. 9:16, 10 seconds."
        ),
    ),

    # ------------------------------------------------------------------
    # Category 4 — stylised / non-photoreal
    # ------------------------------------------------------------------
    Prompt(
        id="STYLE-1", category="C4", lang="en", duration=8,
        text=(
            "Anime-style portrait of a teenage swordswoman with long dark hair, standing on a "
            "cliff at sunset, wind blowing her hair. Stylised cel-shaded look, vibrant colours. "
            "The camera slowly orbits her. 9:16, 8 seconds."
        ),
    ),
    Prompt(
        id="STYLE-2", category="C4", lang="en", duration=8,
        text=(
            "A watercolour-style animation of a small wooden boat drifting down a river "
            "through cherry blossom petals. Soft pastel colours, visible brush textures, "
            "gentle motion. 9:16, 8 seconds."
        ),
    ),
    Prompt(
        id="STYLE-3", category="C4", lang="en", duration=10,
        text=(
            "A cute round robot with big expressive eyes rolls through a futuristic garden "
            "full of bioluminescent plants. Pixar-like 3D animation style, soft global "
            "illumination. 9:16, 10 seconds."
        ),
    ),
    Prompt(
        id="STYLE-4", category="C4", lang="en", duration=8,
        text=(
            "Comic-book style panel of a costumed superhero leaping between rooftops at night. "
            "Bold inked outlines, halftone shading, motion lines. Dynamic low-angle. 9:16, "
            "8 seconds."
        ),
    ),
    Prompt(
        id="STYLE-5", category="C4", lang="en", duration=8,
        text=(
            "Stop-motion clay animation of a small clay rabbit hopping through a clay forest. "
            "Visible fingerprints in the clay, charming imperfections. Warm key light. 9:16, "
            "8 seconds."
        ),
    ),
    Prompt(
        id="STYLE-6", category="C4", lang="en", duration=10,
        text=(
            "Vaporwave aesthetic: a pink convertible drives slowly down a neon-lit palm-tree "
            "boulevard at sunset. Pink and teal gradients, retro grid floor, soft glitchy "
            "effects. Synthwave music. 9:16, 10 seconds."
        ),
    ),

    # ------------------------------------------------------------------
    # Category 5 — camera movement explicit
    # ------------------------------------------------------------------
    Prompt(
        id="CAM-1", category="C5", lang="en", duration=8,
        text=(
            "A weathered wooden door at the end of a long stone corridor. The camera slowly "
            "dollies in toward the door at a steady speed. Flickering torchlight on the walls. "
            "9:16, 8 seconds."
        ),
    ),
    Prompt(
        id="CAM-2", category="C5", lang="en", duration=10,
        text=(
            "A young woman stands alone in a vast empty wheat field at golden hour. The camera "
            "starts at eye level and cranes straight up, revealing the field stretches to the "
            "horizon. 9:16, 10 seconds."
        ),
    ),
    Prompt(
        id="CAM-3", category="C5", lang="en", duration=10,
        text=(
            "A vintage motorcycle parked in a garage. The camera orbits 360 degrees around it "
            "at a slow constant speed. Warm tungsten lighting from above. 9:16, 10 seconds."
        ),
    ),
    Prompt(
        id="CAM-4", category="C5", lang="en", duration=6,
        text=(
            "The camera whip-pans rapidly from a chef chopping vegetables in a restaurant "
            "kitchen to a customer at the counter taking their first bite. Fast cut feeling "
            "within a single shot. 9:16, 6 seconds."
        ),
    ),
    Prompt(
        id="CAM-5", category="C5", lang="en", duration=10,
        text=(
            "The camera pushes in toward a closed book on an old desk, then pulls back rapidly "
            "to reveal a library full of similar books. 9:16, 10 seconds."
        ),
    ),

    # ------------------------------------------------------------------
    # Category 6 — long-form (capped to 10s by the LTX-2 text-to-video path)
    # ------------------------------------------------------------------
    Prompt(
        id="LONG-1", category="C6", lang="en", duration=10,
        text=(
            "A young woman in a vintage 1940s dress sits at a small writing desk in a sunlit "
            "room. She picks up a fountain pen, looks at the camera, smiles slightly, and "
            "begins writing a letter. Camera slowly pushes in over 10 seconds. Soft music. "
            "9:16, 10 seconds."
        ),
    ),
    Prompt(
        id="LONG-2", category="C6", lang="en", duration=10,
        text=(
            "A drone shot starting low over a calm lake at sunrise, then slowly rising and "
            "pushing forward over forest, then over a small town, finally cresting a hill to "
            "reveal a large city skyline in golden morning light. 9:16, 10 seconds."
        ),
    ),
    Prompt(
        id="LONG-3", category="C6", lang="en", duration=10,
        text=(
            "A barista in a busy café finishes pouring a latte, hands it across the counter, "
            "smiles, and says, \"Here you go — flat white, extra hot. Anything else?\" The "
            "customer (visible only as their hand) takes the cup. 9:16, 10 seconds."
        ),
    ),
    Prompt(
        id="LONG-4", category="C6", lang="en", duration=10,
        text=(
            "A parkour runner navigates a series of obstacles across rooftops in a "
            "Mediterranean coastal town at noon. Vaults a railing, rolls onto a lower roof, "
            "sprints across, and leaps to the next building. Continuous motion. 9:16, "
            "10 seconds."
        ),
    ),
    Prompt(
        id="LONG-5", category="C6", lang="en", duration=10,
        text=(
            "A weathered fisherman repairs a net on a wooden boat in a small Greek harbour. He "
            "works carefully, hands moving methodically. He pauses, looks up at the seagulls, "
            "sighs softly, then continues working. Warm Mediterranean afternoon light. 9:16, "
            "10 seconds."
        ),
    ),
]
