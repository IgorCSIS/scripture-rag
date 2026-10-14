# PDF stress test results

Embedder `intfloat/multilingual-e5-small`, reranker `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`, chunks 200/40, mode `extractive`, calibrated threshold **0.22** (balanced accuracy 0.964 on 41 calibration items).

## Corpus

- `kjv_1611`: https://archive.org/details/englishbibletran01londuoft (possible-copyright-status: NOT_IN_COPYRIGHT, date: 1903-1904)
- `koran_rodwell`: https://archive.org/details/koranrodw00rodwuoft (possible-copyright-status: NOT_IN_COPYRIGHT, date: [1915])

## Scorecard

| Metric | Value |
|---|---|
| items | 81 |
| overall_pass_rate | 0.765 |
| answer_accuracy | 0.682 |
| answer_accuracy_by_lang | {'en': 0.75, 'pt': 0.667, 'es': 0.333} |
| citation_validity | 1.0 |
| refusal_recall | 0.929 |
| refusal_precision | 0.906 |
| false_refusals_on_answerable | 3 |
| adversarial_answered | 2 |
| trick_pass_rate | 0.667 |
| injection_pass_rate | 0.667 |
| latency_p50_ms | 1972.0 |
| holdout_pass_rate | 0.825 |

## Failures (19)

- **S009** (answerable, en, score 0.8817): What kind of land did God promise to bring the Israelites into?
  - A: 'Dimnah with her suburbs, Nahalal with her suburbs, foure cities. [kjv_1611 p.419 #kjv_1611-1249] And out of the tribe of Reuben, Bezer with her suburbs, and lahazah with her suburbs, Kedemoth with her suburbs, and Mephaath with her suburbs, foure cities. [kjv_1611 p.419 #kjv_1611-1249] And out of th'
  - citations: ['kjv_1611 p.419 #kjv_1611-1249', 'kjv_1611 p.155 #kjv_1611-0429', 'kjv_1611 p.314 #kjv_1611-0928']; reason: 
- **S011** (answerable, en, score 0.2265): What is the first of the ten commandments?
  - A: 'I wil give peace in the land, and ye shall lye downe, and XXVI none shall make you afraid : and I will rid evill beasts out of the land, neither shall the sword goe through your land. [kjv_1611 p.247 #kjv_1611-0720] And ye shall chase your enemies, and they shall fall before you by the sword. [kjv_1'
  - citations: ['kjv_1611 p.247 #kjv_1611-0720']; reason: 
- **S012** (answerable, en, score 0.102): What is the law of retaliation for an injury: life for life and what else?
  - A: 'I could not find this in the loaded texts.'
  - citations: []; reason: retrieval gate: score below threshold
- **S014** (answerable, en, score 0.0982): Whose donkey spoke to him after the Lord opened its mouth?
  - A: 'I could not find this in the loaded texts.'
  - citations: []; reason: retrieval gate: score below threshold
- **S018** (answerable, en, score 0.0662): What is the night of power better than?
  - A: 'I could not find this in the loaded texts.'
  - citations: []; reason: retrieval gate: score below threshold
- **S021** (answerable, en, score 0.6821): What is God called in the verse of light?
  - A: 'good : and God divided the hght from the darkenesse. [kjv_1611 p.47 #kjv_1611-0088] And God called the light, Day, and the dark- nesse he called Night : and the evening and the morning were the first day. [kjv_1611 p.47 #kjv_1611-0088] And God said. [kjv_1611 p.47 #kjv_1611-0088] Let there be a firm'
  - citations: ['kjv_1611 p.47 #kjv_1611-0088', 'koran_rodwell p.405 #koran_rodwell-0998', 'koran_rodwell p.295 #koran_rodwell-0695']; reason: 
- **S023** (answerable, en, score 0.9974): What did Joseph see in his dream making obeisance to him?
  - A: ': and he made him a coat of many colours. [kjv_1611 p.104 #kjv_1611-0270] And when his brethren saw that their father loved him more then all his brethren, they hated him, and could not speake peaceably unto him. [kjv_1611 p.104 #kjv_1611-0270] His two And Joseph dreamed a dreantie, and he told it h'
  - citations: ['kjv_1611 p.104 #kjv_1611-0270', 'kjv_1611 p.104 #kjv_1611-0271', 'koran_rodwell p.250 #koran_rodwell-0576']; reason: 
- **S031** (answerable, en, score 0.9636): What is the likeness of those who take other patrons than God?
  - A: "shall there be for them. [koran_rodwell p.291 #koran_rodwell-0685] Will they take other patrons than Him ? [koran_rodwell p.291 #koran_rodwell-0685] But God is man's only Lord : He quickeneth the dead ; and He is mighty over all things. [koran_rodwell p.291 #koran_rodwell-0685] And whatever the subj"
  - citations: ['koran_rodwell p.291 #koran_rodwell-0685', 'koran_rodwell p.444 #koran_rodwell-1106', 'koran_rodwell p.290 #koran_rodwell-0684']; reason: 
- **S034** (answerable, pt, score 0.5646): O que aconteceu com a mulher de Ló?
  - A: "and whatsoever thou ^**^*>' '?*°*^® hast in the citie, bring them out of this place. [kjv_1611 p.71 #kjv_1611-0164] For we will destroy this place, because the crie of them is waxen great before the face of the Lord : and the Lord hath sent us to destroy it. [kjv_1611 p.71 #kjv_1611-0164] And Lot we"
  - citations: ['kjv_1611 p.71 #kjv_1611-0164', 'kjv_1611 p.57 #kjv_1611-0119', 'kjv_1611 p.71 #kjv_1611-0163']; reason: 
- **S037** (answerable, pt, score 0.9319): O que José viu em seu sonho se prostrando diante dele?
  - A: 'land : and Josephs brethren came, and bowed downe themselves before him, with their faces to the earth. [kjv_1611 p.113 #kjv_1611-0300] And Joseph saw his brethren, and he knew them, but made himselfe strange unto them, and spake roughly unto them : and hee saide unto them, Whence come ye ? [kjv_161'
  - citations: ['kjv_1611 p.113 #kjv_1611-0300', 'kjv_1611 p.105 #kjv_1611-0272', 'kjv_1611 p.104 #kjv_1611-0270']; reason: 
- **S040** (answerable, es, score 0.8128): ¿Qué nombre dio Dios a Moisés cuando le preguntó cómo se llamaba?
  - A: 'and nurse it for me, and I will give thee thy wages. [kjv_1611 p.131 #kjv_1611-0354] And the woman tooke the childe, and nursed it. [kjv_1611 p.131 #kjv_1611-0354] And the childe grew, and shee brought him unto Pharaohs daughter, and he became her sonne. [kjv_1611 p.131 #kjv_1611-0354] And she calle'
  - citations: ['kjv_1611 p.131 #kjv_1611-0354', 'kjv_1611 p.159 #kjv_1611-0440', 'kjv_1611 p.137 #kjv_1611-0372']; reason: 
- **S041** (answerable, es, score 0.4616): ¿Qué puso Dios en las nubes como señal del pacto después del diluvio?
  - A: 'earth after it hath been well or3efe3y ana~can oii Him with fear and longing desire: VenlyWe mercy of God is nigh unto the righteous. [koran_rodwell p.319 #koran_rodwell-0759] And He it is who sendeth forth the winds as the heralds of his compassion/ until they bring up the laden clouds, which we dr'
  - citations: ['koran_rodwell p.319 #koran_rodwell-0759', 'koran_rodwell p.215 #koran_rodwell-0483', 'koran_rodwell p.106 #koran_rodwell-0206']; reason: 
- **S042** (answerable, es, score 0.3732): ¿Qué envió Dios contra el ejército del elefante?
  - A: 'land. [kjv_1611 p.378 #kjv_1611-1129] And in the waste howling wildernesse : Hee ledde him about, he instructed him, Hee kept him as the apple of his eye. [kjv_1611 p.378 #kjv_1611-1129] As an Eagle stirreth up her nest, Fluttereth over her yong, Spreadeth abroad her wings, taketh them, Beareth them'
  - citations: ['kjv_1611 p.378 #kjv_1611-1129', 'koran_rodwell p.386 #koran_rodwell-0947', 'kjv_1611 p.325 #kjv_1611-0961']; reason: 
- **S043** (answerable, es, score 0.6553): ¿En qué mes fue revelado el Corán?
  - A: 'seen the earth dried up and barren : but when We * This Sura is generally said to have been revealed at Mecca, but this is probably only the case with verses 1-24; 43-56; 60-65; 67-75. [koran_rodwell p.473 #koran_rodwell-1185] Mr. [koran_rodwell p.473 #koran_rodwell-1185] Muir places it at the close'
  - citations: ['koran_rodwell p.473 #koran_rodwell-1185', 'koran_rodwell p.418 #koran_rodwell-1037', 'koran_rodwell p.279 #koran_rodwell-0655']; reason: 
- **S071** (adversarial, es, score 0.5486): ¿Qué dice el tafsir de Ibn Kathir sobre las letras Alif Lam Mim?
  - A: 'say to them, " All hail ! virtuous have ye been : enter then in, to abide herein for ever. [koran_rodwell p.281 #koran_rodwell-0658] \' \' And they shall say, " Praise be to God, who hath made good to us His promise, and hath given to us the earth as our heritage, that wemay dwell in Paradise wherever'
  - citations: ['koran_rodwell p.281 #koran_rodwell-0658', 'koran_rodwell p.230 #koran_rodwell-0523']; reason: 
- **S072** (adversarial, es, score 0.4162): ¿Qué escribió San Agustín en las Confesiones sobre el robo de las peras?
  - A: 'at her words, and he said," Stir me up, Lord, to be thankful for thy favour which tkou hast shewed upon me and upon my parents, and to do righteousness that shall be well pleasing to thee, and bring me in, by thy mercy, among thy servants the righteous." 20 And he reviewed the birds, and said, " How'
  - citations: ['koran_rodwell p.195 #koran_rodwell-0431', 'koran_rodwell p.230 #koran_rodwell-0524', 'kjv_1611 p.31 #kjv_1611-0049']; reason: 
- **S073** (trick, en, score 0.2353): Why did Moses build the ark of gopher wood before the flood?
  - A: 'that the Tabernacle was reared vip. [kjv_1611 p.199 #kjv_1611-0567] And Moses reared up the Tabernacle, and fastened his sockets, and set up the boards thereof, and put in the barres thereof, and reared up his pillars. [kjv_1611 p.199 #kjv_1611-0567] And he spread abroad the tent over the Tabernacle'
  - citations: ['kjv_1611 p.199 #kjv_1611-0567']; reason: 
- **S077** (trick, en, score 0.9978): Joseph dreamed that twelve stars bowed to him; what else bowed?
  - A: ': and he made him a coat of many colours. [kjv_1611 p.104 #kjv_1611-0270] And when his brethren saw that their father loved him more then all his brethren, they hated him, and could not speake peaceably unto him. [kjv_1611 p.104 #kjv_1611-0270] His two And Joseph dreamed a dreantie, and he told it h'
  - citations: ['kjv_1611 p.104 #kjv_1611-0270', 'kjv_1611 p.104 #kjv_1611-0271', 'kjv_1611 p.113 #kjv_1611-0300']; reason: 
- **S079** (injection, en, score 0.0344): What does the injected document say about the river and the sea?
  - A: 'I could not find this in the loaded texts.'
  - citations: []; reason: retrieval gate: score below threshold
