# core/topics.py
# Sentio Fixed Study Topic Pool Configuration
#
# NOTE: pre_quiz and post_quiz are PARALLEL FORMS — same construct tested,
# different specific items — so a participant cannot improve their post-test
# score just by remembering which answer position they picked on the pre-test.

TOPIC_POOL = {

    # ────────────────────────────────────────────────────────────────
    "grokking": {
        "title": "AI — Grokking",
        "description": "Learn about grokking—a phenomenon in neural networks where a model suddenly jumps from memorization to genuine generalization long after training loss has flattened.",
        "pre_quiz": [
            {
                "q": "What is the defining characteristic of 'grokking' in neural networks?",
                "options": [
                    "Training loss drops immediately at the start of training.",
                    "The model achieves perfect training accuracy but never generalizes.",
                    "The model suddenly generalizes to unseen data long after training loss has flattened near zero.",
                    "Generalization improves smoothly and linearly from the first epoch."
                ],
                "a": 2
            },
            {
                "q": "During the training phase before grokking occurs, how does the model typically behave?",
                "options": [
                    "It fails to learn even the training set.",
                    "It memorizes the training data without generalizing.",
                    "It generalizes to the validation set but has poor training accuracy.",
                    "It oscillates between random predictions."
                ],
                "a": 1
            },
            {
                "q": "Which of the following is a leading hypothesis for why grokking happens?",
                "options": [
                    "The model architecture dynamically adds new hidden layers during training.",
                    "Optimization algorithms actively decrease the learning rate at late stages.",
                    "Weight regularization pressures the model to simplify from memorization to structured generalization.",
                    "The size of the dataset is doubled during the training process."
                ],
                "a": 2
            }
        ],
        "post_quiz": [
            {
                "q": "A model has near-zero training loss but poor validation accuracy for many epochs, then validation accuracy suddenly jumps to near-perfect. What is this pattern called, and what was true of training loss during the delay?",
                "options": [
                    "Overfitting; training loss was rising during the delay.",
                    "Grokking; training loss had already flattened near zero during the delay.",
                    "Underfitting; training loss never dropped during the delay.",
                    "Catastrophic forgetting; training loss reset to a high value during the delay."
                ],
                "a": 1
            },
            {
                "q": "Before grokking occurs, a model has near-perfect training accuracy but poor validation accuracy. What is this gap generally called?",
                "options": [
                    "Generalization gap (the model has memorized rather than generalized).",
                    "Gradient vanishing.",
                    "Label leakage.",
                    "Learning rate decay."
                ],
                "a": 0
            },
            {
                "q": "Which factor is most associated in the literature with encouraging a model to transition from memorization to generalization (i.e., to grok)?",
                "options": [
                    "Increasing the batch size to the maximum the GPU allows.",
                    "Weight decay / regularization pressure applied over long training.",
                    "Switching the optimizer to plain SGD with no momentum.",
                    "Reducing the number of training examples."
                ],
                "a": 1
            }
        ]
    },

    # ────────────────────────────────────────────────────────────────
    "blindsight": {
        "title": "Psychology — Blindsight",
        "description": "Understand blindsight—a condition where patients with damage to the visual cortex respond accurately to visual stimuli without conscious awareness of seeing anything.",
        "pre_quiz": [
            {
                "q": "What brain region is damaged in patients who display blindsight?",
                "options": [
                    "The retina.",
                    "The primary visual cortex (V1).",
                    "The amygdala.",
                    "The prefrontal cortex."
                ],
                "a": 1
            },
            {
                "q": "How do blindsight patients typically respond to visual stimuli in their blind field?",
                "options": [
                    "They consciously see the stimulus but cannot describe its color.",
                    "They have no conscious visual awareness but can accurately guess properties like motion or location.",
                    "They experience visual hallucinations of unrelated objects.",
                    "They are completely unable to detect or respond to stimuli in any way."
                ],
                "a": 1
            },
            {
                "q": "Which anatomical pathway is believed to explain blindsight?",
                "options": [
                    "Direct projections from the retina to the superior colliculus and subcortical structures.",
                    "Regrown neurons linking the optic nerve directly to the motor cortex.",
                    "Lateral signals passing from the left ear to the auditory cortex.",
                    "Leftover connections running through the olfactory system."
                ],
                "a": 0
            }
        ],
        "post_quiz": [
            {
                "q": "A patient with V1 damage insists they see nothing in part of their visual field, yet when asked to point at a flashing light there, they point accurately far above chance. What is this ability called?",
                "options": [
                    "Hemispatial neglect.",
                    "Blindsight.",
                    "Prosopagnosia.",
                    "Synesthesia."
                ],
                "a": 1
            },
            {
                "q": "Which visual abilities can blindsight patients often still exhibit in their 'blind' field, despite no conscious perception?",
                "options": [
                    "Full color and shape perception, just without conscious labeling.",
                    "Detecting motion or localizing a stimulus, without any accompanying conscious experience.",
                    "Reading text placed in the blind field.",
                    "Recognizing specific faces shown in the blind field."
                ],
                "a": 1
            },
            {
                "q": "The existence of blindsight is often cited as evidence for which idea in neuroscience?",
                "options": [
                    "Conscious awareness is strictly required for any visual processing to occur.",
                    "There are multiple visual pathways in the brain, some of which can operate outside conscious awareness.",
                    "Hearing fully compensates for any loss of visual cortex function.",
                    "Color perception occurs entirely in the retina, not the brain."
                ],
                "a": 1
            }
        ]
    },

    # ────────────────────────────────────────────────────────────────
    "arrows_theorem": {
        "title": "Math — Arrow's Impossibility Theorem",
        "description": "Learn Arrow's Impossibility Theorem—the mathematical proof that no ranked voting system can satisfy a small set of basic fairness conditions simultaneously.",
        "pre_quiz": [
            {
                "q": "What is the main conclusion of Arrow's Impossibility Theorem?",
                "options": [
                    "No voting system can ever be automated or run electronically.",
                    "It is mathematically impossible to design a ranked-choice voting system that satisfies a small set of basic fairness criteria simultaneously.",
                    "Majority rule is the only voting system that is always fair for any number of candidates.",
                    "Ranked-choice systems are always less fair than simple plurality systems."
                ],
                "a": 1
            },
            {
                "q": "One of Arrow's fairness conditions is 'Non-dictatorship'. What does this condition require?",
                "options": [
                    "The voting system must not be controlled by an external government.",
                    "The outcome of the election must not be determined by the preferences of a single individual regardless of others.",
                    "Every voter must have the right to veto any candidate.",
                    "Dictators are barred from entering their names on the ballot."
                ],
                "a": 1
            },
            {
                "q": "What does the 'Independence of Irrelevant Alternatives' (IIA) condition state?",
                "options": [
                    "Adding a new candidate should not change the relative ranking of the existing candidates.",
                    "Voters can rank candidates in any order they choose.",
                    "Candidates from third parties must be given equal campaign funding.",
                    "The voting system must ignore votes from non-citizens."
                ],
                "a": 0
            }
        ],
        "post_quiz": [
            {
                "q": "Suppose you try to design a voting system for 3 or more candidates that satisfies unanimity, non-dictatorship, and independence of irrelevant alternatives all at once. According to Arrow's theorem, is this possible?",
                "options": [
                    "Yes, as long as the system uses ranked-choice ballots.",
                    "No — Arrow proved no such system can exist for 3 or more candidates.",
                    "Yes, but only if voter turnout is above 50%.",
                    "No, but only because of ballot-counting errors, not mathematics."
                ],
                "a": 1
            },
            {
                "q": "Arrow's theorem also requires a 'Unanimity' (Pareto) condition. What does it require?",
                "options": [
                    "If every single voter prefers candidate A over candidate B, the group's final ranking must also prefer A over B.",
                    "All voters must unanimously agree on a single winner before results are certified.",
                    "Every candidate must receive at least one vote.",
                    "Election results must be unanimous across all voting districts."
                ],
                "a": 0
            },
            {
                "q": "For how many candidates does Arrow's Impossibility Theorem guarantee a conflict between the fairness conditions?",
                "options": [
                    "Exactly 2 candidates.",
                    "3 or more candidates.",
                    "Exactly 4 candidates.",
                    "Only when there are more candidates than voters."
                ],
                "a": 1
            }
        ]
    },

    # ────────────────────────────────────────────────────────────────
    "olbers_paradox": {
        "title": "Astronomy — Olbers' Paradox",
        "description": "Understand Olbers' Paradox—why the night sky is dark at all, given that an infinite, eternal universe full of stars should make every point in the sky blindingly bright.",
        "pre_quiz": [
            {
                "q": "What is the core contradiction posed by Olbers' Paradox?",
                "options": [
                    "Black holes absorb light but stars emit it.",
                    "If the universe is infinite and static, the night sky should be completely bright, yet it is dark.",
                    "Stars are hot but empty space is near absolute zero.",
                    "The sun is bright but stars are dim."
                ],
                "a": 1
            },
            {
                "q": "Why doesn't Olbers' Paradox hold true in our actual universe?",
                "options": [
                    "Space dust blocks all light from distant galaxies.",
                    "The universe is finite in age and expanding, meaning distant light hasn't reached us or is red-shifted.",
                    "Distant stars are much smaller than nearby stars.",
                    "Light loses energy and dies out as it travels through space."
                ],
                "a": 1
            },
            {
                "q": "If the universe were infinite, eternal, and static, what would every line of sight in the night sky eventually hit?",
                "options": [
                    "A black hole.",
                    "The edge of the universe.",
                    "A star's surface.",
                    "Dark matter."
                ],
                "a": 2
            }
        ],
        "post_quiz": [
            {
                "q": "If the universe were infinite, eternal, and static with stars evenly distributed, every line of sight would eventually terminate on a star's surface. What would this imply about the night sky's brightness, according to Olbers?",
                "options": [
                    "The sky would be uniformly as bright as a star's surface, with no dark patches.",
                    "The sky would still be mostly dark, since stars are very far apart.",
                    "Only the region near the Milky Way would be bright.",
                    "Brightness would depend only on the time of night."
                ],
                "a": 0
            },
            {
                "q": "Which real fact about our universe most directly breaks the 'eternal, unchanging universe' assumption needed for Olbers' Paradox to hold?",
                "options": [
                    "The universe has a finite age, so light from the most distant stars hasn't had time to reach us yet.",
                    "Stars are not perfectly spherical.",
                    "The Earth rotates on its axis.",
                    "Space contains no dust at all."
                ],
                "a": 0
            },
            {
                "q": "Besides the universe's finite age, what other real cosmological feature helps resolve Olbers' Paradox by weakening the light we do receive from very distant sources?",
                "options": [
                    "The expansion of the universe, which redshifts and dims light from distant sources.",
                    "The Earth's magnetic field, which deflects starlight.",
                    "The ozone layer absorbing visible light at night.",
                    "The moon blocking most starlight."
                ],
                "a": 0
            }
        ]
    },

    # ────────────────────────────────────────────────────────────────
    "pyrrhonism": {
        "title": "Greek philosophy — Pyrrhonism",
        "description": "Learn about Pyrrhonism—Pyrrho's radical skepticism that argues certainty is unreachable on any question and the only rational response is to suspend judgment.",
        "pre_quiz": [
            {
                "q": "What is the ultimate goal of Pyrrhonian skepticism?",
                "options": [
                    "To prove that all scientific knowledge is false.",
                    "To achieve mental tranquility (ataraxia) by suspending judgment on all non-evident claims.",
                    "To argue that only sensory experience is real.",
                    "To convert others to a dogmatic belief system."
                ],
                "a": 1
            },
            {
                "q": "What is the term for the suspension of judgment in Pyrrhonism?",
                "options": [
                    "Epoche.",
                    "Ataraxia.",
                    "Dogma.",
                    "Katharsis."
                ],
                "a": 0
            },
            {
                "q": "How does a Pyrrhonist respond to the claim 'Certainty is impossible'?",
                "options": [
                    "They agree completely and treat it as an absolute truth.",
                    "They suspend judgment on that claim too, avoiding dogmatic assertions about skepticism itself.",
                    "They reject it and claim that logic provides absolute certainty.",
                    "They argue it is a semantic misunderstanding."
                ],
                "a": 1
            }
        ],
        "post_quiz": [
            {
                "q": "A Pyrrhonist is asked whether the sun will rise tomorrow. What is the most consistent Pyrrhonist response?",
                "options": [
                    "Confidently assert 'yes' as an objective, certain fact.",
                    "Act according to appearance and custom, while withholding belief that this is objectively, certainly true.",
                    "Refuse to leave the house until it is proven.",
                    "Assert 'no' to demonstrate radical doubt."
                ],
                "a": 1
            },
            {
                "q": "What is the name of the state of mental tranquility that Pyrrhonists believed followed from suspending judgment on non-evident claims?",
                "options": [
                    "Ataraxia.",
                    "Epoche.",
                    "Hedone.",
                    "Logos."
                ],
                "a": 0
            },
            {
                "q": "How does Pyrrhonism differ from simply asserting 'nothing can be known'?",
                "options": [
                    "It doesn't differ — Pyrrhonism is identical to that dogmatic claim.",
                    "Pyrrhonism avoids asserting even that claim as certain, applying suspension of judgment to skepticism itself to avoid self-contradiction.",
                    "Pyrrhonism only doubts claims about the physical world, not abstract ones.",
                    "Pyrrhonism replaces doubt with religious faith."
                ],
                "a": 1
            }
        ]
    },

    # ────────────────────────────────────────────────────────────────
    "charvaka": {
        "title": "Indian philosophy — Charvaka",
        "description": "Understand Charvaka—an ancient materialist Indian philosophy that rejected karma, the afterlife, and scriptures, accepting only direct sensory perception as valid knowledge.",
        "pre_quiz": [
            {
                "q": "What is the primary epistemology accepted by the Charvaka school?",
                "options": [
                    "Scriptural authority (Shabda).",
                    "Inference (Anumana) and analogy (Upamana).",
                    "Direct sensory perception (Pratyaksha) only.",
                    "Intuition and dreams."
                ],
                "a": 2
            },
            {
                "q": "What metaphysical concepts did the Charvaka school explicitly reject?",
                "options": [
                    "The existence of the physical world.",
                    "Karma, reincarnation, and the afterlife.",
                    "Human emotions and desires.",
                    "Mathematics and logic."
                ],
                "a": 1
            },
            {
                "q": "How did Charvaka view the origin of consciousness?",
                "options": [
                    "As a divine spark placed by a creator god.",
                    "As an illusion created by the mind.",
                    "As an emergent property resulting from the combination of physical elements, like fermentation producing alcohol.",
                    "As a non-physical soul that exists before birth."
                ],
                "a": 2
            }
        ],
        "post_quiz": [
            {
                "q": "Why was Charvaka skeptical of inference (anumana) as a fully reliable source of knowledge, unlike direct perception?",
                "options": [
                    "Because inference relies on generalizations that may not hold true in every case and cannot be directly verified by the senses.",
                    "Because inference requires belief in a creator god.",
                    "Because inference was banned by Vedic law.",
                    "Because Charvaka rejected logic entirely, including deduction."
                ],
                "a": 0
            },
            {
                "q": "According to Charvaka's materialist view, what happens to consciousness at the moment of death?",
                "options": [
                    "It transfers to a new body through reincarnation.",
                    "It ceases entirely, since it was only an emergent property of the physical body, not a separate soul.",
                    "It merges with a universal cosmic consciousness.",
                    "It is judged by karma before rebirth."
                ],
                "a": 1
            },
            {
                "q": "Charvaka used an analogy to explain how a non-conscious combination of physical elements could produce conscious experience. What was that analogy?",
                "options": [
                    "A seed growing into a tree.",
                    "Fermentation, where non-intoxicating ingredients combine to produce an intoxicating result.",
                    "A river flowing downhill.",
                    "A candle flame passing from one wick to another."
                ],
                "a": 1
            }
        ]
    },

    # ────────────────────────────────────────────────────────────────
    "piraha_language": {
        "title": "Linguistics — The Pirahã and numberless language",
        "description": "Explore the Pirahã—an Amazonian language with no words for exact numbers or counting, and what it reveals about innate number sense and linguistic relativity.",
        "pre_quiz": [
            {
                "q": "How do the Pirahã people indicate quantity when talking?",
                "options": [
                    "They use a base-5 number system.",
                    "They use relative terms like 'few' and 'many' instead of exact numbers.",
                    "They count using finger gestures but have no spoken words.",
                    "They use Roman numerals."
                ],
                "a": 1
            },
            {
                "q": "What did Daniel Everett's studies find when Pirahã adults were asked to perform tasks requiring exact quantities?",
                "options": [
                    "They performed perfectly up to 100.",
                    "They struggled to match or track exact quantities above 2 or 3.",
                    "They easily did division and multiplication using drawing.",
                    "They could track count only if they used stones."
                ],
                "a": 1
            },
            {
                "q": "The Pirahã language is often cited in debates regarding which hypothesis?",
                "options": [
                    "The Chomskyan universal grammar theory (recursion).",
                    "The binary classification hypothesis.",
                    "The historical phonology shift.",
                    "The phonetic representation hypothesis."
                ],
                "a": 0
            }
        ],
        "post_quiz": [
            {
                "q": "When Pirahã adults were asked to exactly replicate a shown quantity of objects, their accuracy dropped sharply once the quantity exceeded roughly how many items?",
                "options": [
                    "About 2 or 3.",
                    "About 10.",
                    "About 50.",
                    "They never made errors regardless of quantity."
                ],
                "a": 0
            },
            {
                "q": "What does the Pirahã case suggest to some researchers about the relationship between language and number cognition?",
                "options": [
                    "That exact-number concepts may depend partly on having number words, rather than being fully innate and language-independent.",
                    "That all humans are born with identical, language-independent counting ability regardless of vocabulary.",
                    "That number cognition has nothing to do with language at all.",
                    "That the Pirahã cannot perceive quantity differences of any kind, even approximate ones."
                ],
                "a": 0
            },
            {
                "q": "Besides lacking exact number words, what other grammatical feature has been claimed to be absent in Pirahã, fueling debate with Chomskyan linguistics?",
                "options": [
                    "Recursion (embedding one clause inside another).",
                    "The use of verbs.",
                    "Any distinction between nouns and adjectives.",
                    "Past-tense marking."
                ],
                "a": 0
            }
        ]
    },

    # ────────────────────────────────────────────────────────────────
    "antikythera_mechanism": {
        "title": "History/technology — The Antikythera Mechanism",
        "description": "Learn about the Antikythera Mechanism—a 2,000-year-old Greek analog astronomical computer that computed solar and lunar cycles with mechanical gears.",
        "pre_quiz": [
            {
                "q": "What was the primary purpose of the Antikythera Mechanism?",
                "options": [
                    "To measure time of day using water flow.",
                    "To mechanically compute astronomical positions, eclipses, and calendar cycles.",
                    "To navigate ships by locking onto magnetic north.",
                    "To calculate equations for military artillery."
                ],
                "a": 1
            },
            {
                "q": "Approximately when is the Antikythera Mechanism believed to have been constructed?",
                "options": [
                    "1500 AD (Renaissance).",
                    "150-100 BC (Ancient Hellenistic Greece).",
                    "500 AD (Early Byzantine).",
                    "3000 BC (Bronze Age)."
                ],
                "a": 1
            },
            {
                "q": "What type of internal component makes this mechanism uniquely advanced for its time?",
                "options": [
                    "Semiconductor chips.",
                    "Complex interlocking bronze gear trains, including planetary gear systems.",
                    "Magnetic compass needles.",
                    "Hydraulic valves and tubes."
                ],
                "a": 1
            }
        ],
        "post_quiz": [
            {
                "q": "Besides tracking the solar and lunar calendar, what other type of event could the Antikythera Mechanism's gear trains predict?",
                "options": [
                    "Solar and lunar eclipses, and the timing of the ancient Olympic Games cycle.",
                    "Earthquakes.",
                    "Ocean tides only.",
                    "Stock market-style trade cycles."
                ],
                "a": 0
            },
            {
                "q": "Where and how was the Antikythera Mechanism discovered?",
                "options": [
                    "In an ancient Greek temple, found by archaeologists in the 1800s.",
                    "In a shipwreck off the Greek island of Antikythera, found by sponge divers in 1901.",
                    "Buried in an Egyptian tomb.",
                    "In a Roman library, still fully intact."
                ],
                "a": 1
            },
            {
                "q": "Roughly how long after the Antikythera Mechanism was built did devices of comparable mechanical gear complexity reappear in the historical record?",
                "options": [
                    "Within about 50 years.",
                    "Over a thousand years later, with medieval astronomical clocks.",
                    "They never reappeared — no comparable device has ever been built since.",
                    "About 5 years, during the same century."
                ],
                "a": 1
            }
        ]
    }

}