# ruff: noqa: E501
"""Build the synthetic dev set (docs/eval/dev_set.jsonl) used for selection alongside the tune split.

Items are hand-authored here from the corpus itself (Genesis to Joshua of the 1611 KJV, Rodwell's Koran),
in EN, PT and ES, with gold verse ids (KJV) or verified evidence phrases (Koran), plus unanswerable
questions about things outside the loaded texts. Nothing is derived from the held-out split: any item
whose gold verse or evidence phrase also appears in a held-out item is dropped automatically, and the
script reports only the count.

Usage::

    python scripts/build_dev_set.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

EVAL = Path("docs/eval")

# (group, book, gold, {lang: question}); gold is a KJV ref, or "ev:" phrases separated by "|".
ANSWERABLE = [
    ("D01", "kjv", "Genesis 4:1", {"en": "Who was the first son born to Adam and Eve?",
                                    "pt": "Quem foi o primeiro filho de Adão e Eva?",
                                    "es": "¿Quién fue el primer hijo de Adán y Eva?"}),
    ("D02", "kjv", "Genesis 4:4", {"en": "What did Abel bring as his offering to the Lord?",
                                    "pt": "O que Abel trouxe como oferta ao Senhor?",
                                    "es": "¿Qué trajo Abel como ofrenda al Señor?"}),
    ("D03", "kjv", "Genesis 5:27", {"en": "How many years did Methuselah live?",
                                     "pt": "Quantos anos viveu Matusalém?",
                                     "es": "¿Cuántos años vivió Matusalén?"}),
    ("D04", "kjv", "Genesis 6:14", {"en": "What kind of wood was Noah told to use for the ark?",
                                     "pt": "De que madeira Noé deveria fazer a arca?",
                                     "es": "¿De qué madera debía Noé hacer el arca?"}),
    ("D05", "kjv", "Genesis 8:7", {"en": "Which bird did Noah send out first from the ark?",
                                    "pt": "Qual ave Noé soltou primeiro da arca?",
                                    "es": "¿Qué ave soltó Noé primero desde el arca?"}),
    ("D06", "kjv", "Genesis 8:11", {"en": "What did the dove bring back to Noah in her mouth?",
                                     "pt": "O que a pomba trouxe no bico para Noé?",
                                     "es": "¿Qué trajo la paloma en el pico a Noé?"}),
    ("D07", "kjv", "Genesis 11:9", {"en": "Why was the city with the tower called Babel?",
                                     "pt": "Por que a cidade da torre foi chamada Babel?",
                                     "es": "¿Por qué la ciudad de la torre se llamó Babel?"}),
    ("D08", "kjv", "Genesis 19:26", {"en": "What happened to Lot's wife when she looked back?",
                                      "pt": "O que aconteceu com a mulher de Ló quando olhou para trás?",
                                      "es": "¿Qué le pasó a la mujer de Lot cuando miró atrás?"}),
    ("D09", "kjv", "Genesis 21:5", {"en": "How old was Abraham when Isaac was born?",
                                     "pt": "Quantos anos Abraão tinha quando Isaque nasceu?",
                                     "es": "¿Qué edad tenía Abraham cuando nació Isaac?"}),
    ("D10", "kjv", "Genesis 22:13", {"en": "What animal caught in a thicket was offered instead of Isaac?",
                                      "pt": "Que animal preso no mato foi oferecido no lugar de Isaque?",
                                      "es": "¿Qué animal enredado en un zarzal fue ofrecido en lugar de Isaac?"}),
    ("D11", "kjv", "Genesis 25:34", {"en": "What did Jacob give Esau for his birthright?",
                                      "pt": "O que Jacó deu a Esaú pela primogenitura?",
                                      "es": "¿Qué le dio Jacob a Esaú por la primogenitura?"}),
    ("D12", "kjv", "Genesis 28:12", {"en": "What did Jacob see reaching to heaven in his dream?",
                                      "pt": "O que Jacó viu chegando até o céu em seu sonho?",
                                      "es": "¿Qué vio Jacob que llegaba hasta el cielo en su sueño?"}),
    ("D13", "kjv", "Genesis 29:20", {"en": "How many years did Jacob serve for Rachel?",
                                      "pt": "Quantos anos Jacó serviu por Raquel?",
                                      "es": "¿Cuántos años sirvió Jacob por Raquel?"}),
    ("D14", "kjv", "Genesis 32:28", {"en": "What new name was Jacob given after he wrestled?",
                                      "pt": "Que novo nome Jacó recebeu depois de lutar?",
                                      "es": "¿Qué nuevo nombre recibió Jacob después de luchar?"}),
    ("D15", "kjv", "Genesis 37:3", {"en": "What did Israel make for Joseph because he loved him more?",
                                     "pt": "O que Israel fez para José por amá-lo mais?",
                                     "es": "¿Qué hizo Israel para José porque lo amaba más?"}),
    ("D16", "kjv", "Genesis 37:28", {"en": "For how many pieces of silver was Joseph sold?",
                                      "pt": "Por quantas moedas de prata José foi vendido?",
                                      "es": "¿Por cuántas piezas de plata fue vendido José?"}),
    ("D17", "kjv", "Exodus 2:3", {"en": "Where did Moses' mother hide the baby?",
                                   "pt": "Onde a mãe de Moisés escondeu o bebê?",
                                   "es": "¿Dónde escondió la madre de Moisés al bebé?"}),
    ("D18", "kjv", "Exodus 3:2", {"en": "What did Moses see burning without being consumed?",
                                   "pt": "O que Moisés viu queimando sem se consumir?",
                                   "es": "¿Qué vio Moisés ardiendo sin consumirse?"}),
    ("D19", "kjv", "Exodus 12:29", {"en": "What happened at midnight in the last plague of Egypt?",
                                     "pt": "O que aconteceu à meia-noite na última praga do Egito?",
                                     "es": "¿Qué pasó a medianoche en la última plaga de Egipto?"}),
    ("D20", "kjv", "Exodus 16:31", {"en": "What did the house of Israel call the bread from heaven?",
                                     "pt": "Como a casa de Israel chamou o pão do céu?",
                                     "es": "¿Cómo llamó la casa de Israel al pan del cielo?"}),
    ("D21", "kjv", "Exodus 32:4", {"en": "What idol did Aaron make from the golden earrings?",
                                    "pt": "Que ídolo Arão fez com os brincos de ouro?",
                                    "es": "¿Qué ídolo hizo Aarón con los zarcillos de oro?"}),
    ("D22", "kjv", "Exodus 24:18", {"en": "How long was Moses on the mount?",
                                     "pt": "Quanto tempo Moisés ficou no monte?",
                                     "es": "¿Cuánto tiempo estuvo Moisés en el monte?"}),
    ("D23", "kjv", "Leviticus 11:3", {"en": "Which beasts may be eaten according to the law of clean animals?",
                                       "pt": "Quais animais podem ser comidos segundo a lei dos animais limpos?",
                                       "es": "¿Qué animales se pueden comer según la ley de los animales limpios?"}),
    ("D24", "kjv", "Leviticus 16:10", {"en": "What was done with the goat on which the lot fell to be the scapegoat?",
                                        "pt": "O que se fazia com o bode sobre o qual caiu a sorte para ser bode emissário?",
                                        "es": "¿Qué se hacía con el macho cabrío elegido por suerte para Azazel?"}),
    ("D25", "kjv", "Leviticus 19:18", {"en": "Whom are you commanded to love as yourself?",
                                        "pt": "Quem devemos amar como a nós mesmos?",
                                        "es": "¿A quién debemos amar como a nosotros mismos?"}),
    ("D26", "kjv", "Numbers 6:24", {"en": "How does the priestly blessing begin?",
                                     "pt": "Como começa a bênção sacerdotal?",
                                     "es": "¿Cómo empieza la bendición sacerdotal?"}),
    ("D27", "kjv", "Numbers 21:9", {"en": "What did Moses make of brass and put on a pole?",
                                     "pt": "O que Moisés fez de bronze e pôs sobre uma haste?",
                                     "es": "¿Qué hizo Moisés de bronce y puso sobre un asta?"}),
    ("D28", "kjv", "Deuteronomy 6:4", {"en": "What does Moses tell Israel to hear about the Lord their God?",
                                        "pt": "O que Moisés diz a Israel para ouvir sobre o Senhor seu Deus?",
                                        "es": "¿Qué dice Moisés a Israel que oiga sobre el Señor su Dios?"}),
    ("D29", "kjv", "Deuteronomy 8:3", {"en": "Man does not live by what alone?",
                                        "pt": "O homem não vive só de quê?",
                                        "es": "¿De qué solo no vive el hombre?"}),
    ("D30", "kjv", "Deuteronomy 34:7", {"en": "How old was Moses when he died?",
                                         "pt": "Quantos anos Moisés tinha quando morreu?",
                                         "es": "¿Cuántos años tenía Moisés cuando murió?"}),
    ("D31", "kjv", "Joshua 2:1", {"en": "Into whose house did the two spies go in Jericho?",
                                   "pt": "Em casa de quem os dois espias entraram em Jericó?",
                                   "es": "¿En casa de quién entraron los dos espías en Jericó?"}),
    ("D32", "kjv", "Joshua 6:20", {"en": "What happened to the wall of Jericho when the people shouted?",
                                    "pt": "O que aconteceu com o muro de Jericó quando o povo gritou?",
                                    "es": "¿Qué pasó con el muro de Jericó cuando el pueblo gritó?"}),
    ("D33", "kjv", "Joshua 10:12", {"en": "What did Joshua command to stand still upon Gibeon?",
                                     "pt": "O que Josué mandou parar sobre Gibeão?",
                                     "es": "¿Qué mandó Josué que se detuviera sobre Gabaón?"}),
    ("D34", "kjv", "Joshua 24:15", {"en": "Who said that he and his house would serve the Lord?",
                                     "pt": "Quem disse que ele e sua casa serviriam ao Senhor?",
                                     "es": "¿Quién dijo que él y su casa servirían al Señor?"}),
    ("D35", "kjv", "Joshua 3:17", {"en": "Which river did the priests stand in on dry ground while Israel crossed?",
                                    "pt": "Em que rio os sacerdotes ficaram em terra seca enquanto Israel passava?",
                                    "es": "¿En qué río estuvieron los sacerdotes en seco mientras Israel pasaba?"}),
    ("D36", "kjv", "Numbers 14:6", {"en": "Which two spies tore their clothes when the people wanted to turn back?",
                                     "pt": "Quais dois espias rasgaram as vestes quando o povo quis voltar?",
                                     "es": "¿Qué dos espías rasgaron sus vestidos cuando el pueblo quiso volver?"}),
    ("D37", "kjv", "Genesis 2:7", {"en": "From what was man formed?",
                                    "pt": "De que foi formado o homem?",
                                    "es": "¿De qué fue formado el hombre?"}),
    ("D38", "kjv", "Genesis 3:19", {"en": "To what will man return, since he was taken from it?",
                                     "pt": "Ao que o homem voltará, já que dele foi tirado?",
                                     "es": "¿A qué volverá el hombre, ya que de él fue tomado?"}),
    ("D39", "kjv", "Exodus 14:21", {"en": "How did the Lord divide the sea for Moses?",
                                     "pt": "Como o Senhor dividiu o mar para Moisés?",
                                     "es": "¿Cómo dividió el Señor el mar para Moisés?"}),
    ("D40", "kjv", "Genesis 41:46", {"en": "How old was Joseph when he stood before Pharaoh?",
                                      "pt": "Quantos anos José tinha quando se apresentou ao Faraó?",
                                      "es": "¿Qué edad tenía José cuando se presentó ante el Faraón?"}),
    ("D41", "koran", "ev:eleven stars", {"en": "What did Joseph see making obeisance to him besides the sun and the moon?",
                                          "pt": "O que José viu se prostrando diante dele além do sol e da lua?",
                                          "es": "¿Qué vio José inclinándose ante él además del sol y la luna?"}),
    ("D42", "koran", "ev:trunk of the palm", {"en": "What was Mary told to shake so that ripe dates would fall?",
                                               "pt": "O que Maria devia sacudir para caírem tâmaras maduras?",
                                               "es": "¿Qué debía sacudir María para que cayeran dátiles maduros?"}),
    ("D43", "koran", "ev:no compulsion", {"en": "What does the Koran say about compulsion in religion?",
                                           "pt": "O que o Alcorão diz sobre a coação na religião?",
                                           "es": "¿Qué dice el Corán sobre la coacción en la religión?"}),
    ("D44", "koran", "ev:save eblis", {"en": "Who refused when the angels were told to worship Adam?",
                                        "pt": "Quem se recusou quando os anjos foram mandados adorar Adão?",
                                        "es": "¿Quién se negó cuando a los ángeles se les mandó adorar a Adán?"}),
    ("D45", "koran", "ev:enter your dwellings", {"en": "What did the ant tell the other ants when Solomon's army came?",
                                                  "pt": "O que a formiga disse às outras quando o exército de Salomão chegou?",
                                                  "es": "¿Qué dijo la hormiga a las demás cuando llegó el ejército de Salomón?"}),
    ("D46", "koran", "ev:be thou cold", {"en": "What did God say to the fire into which Abraham was thrown?",
                                          "pt": "O que Deus disse ao fogo em que Abraão foi lançado?",
                                          "es": "¿Qué dijo Dios al fuego en que fue arrojado Abraham?"}),
    ("D47", "koran", "ev:see not the lapwing", {"en": "Which bird did Solomon notice was absent when he reviewed the birds?",
                                                 "pt": "Que ave Salomão notou que faltava ao passar em revista as aves?",
                                                 "es": "¿Qué ave notó Salomón que faltaba al pasar revista a las aves?"}),
    ("D48", "koran", "ev:the wolf devour him", {"en": "What did Jacob fear would happen to Joseph if his brothers took him?",
                                                 "pt": "O que Jacó temia que acontecesse a José se os irmãos o levassem?",
                                                 "es": "¿Qué temía Jacob que le pasara a José si sus hermanos lo llevaban?"}),
    ("D49", "koran", "ev:i am the servant of god", {"en": "What did the babe in the cradle say about himself?",
                                                     "pt": "O que o bebê no berço disse sobre si mesmo?",
                                                     "es": "¿Qué dijo el niño en la cuna sobre sí mismo?"}),
    ("D50", "koran", "ev:give him suck", {"en": "What was revealed to the mother of Moses about her baby?",
                                           "pt": "O que foi revelado à mãe de Moisés sobre o seu bebê?",
                                           "es": "¿Qué se reveló a la madre de Moisés sobre su bebé?"}),
    ("D51", "kjv", "Genesis 1:16", {"en": 'What two great lights did God make?', "pt": 'Quais dois grandes luminares Deus fez?', "es": '¿Qué dos grandes lumbreras hizo Dios?'}),
    ("D52", "kjv", "Genesis 2:22", {"en": 'From what did the Lord make the woman?', "pt": 'De que o Senhor fez a mulher?', "es": '¿De qué hizo el Señor a la mujer?'}),
    ("D53", "kjv", "Genesis 3:14", {"en": 'What curse was put on the serpent?', "pt": 'Que maldição foi lançada sobre a serpente?', "es": '¿Qué maldición recibió la serpiente?'}),
    ("D54", "kjv", "Genesis 4:9", {"en": 'What did Cain answer when asked where his brother was?', "pt": 'O que Caim respondeu quando lhe perguntaram onde estava seu irmão?', "es": '¿Qué respondió Caín cuando le preguntaron dónde estaba su hermano?'}),
    ("D55", "kjv", "Genesis 6:15", {"en": 'How long was the ark to be?', "pt": 'Qual devia ser o comprimento da arca?', "es": '¿Cuánto debía medir de largo el arca?'}),
    ("D56", "kjv", "Genesis 7:12", {"en": 'How long did the rain fall on the earth in the flood?', "pt": 'Por quanto tempo a chuva caiu sobre a terra no dilúvio?', "es": '¿Cuánto tiempo cayó la lluvia sobre la tierra en el diluvio?'}),
    ("D57", "kjv", "Genesis 14:18", {"en": 'What did Melchizedek king of Salem bring out?', "pt": 'O que Melquisedeque, rei de Salém, trouxe?', "es": '¿Qué sacó Melquisedec, rey de Salem?'}),
    ("D58", "kjv", "Genesis 35:19", {"en": 'Where was Rachel buried?', "pt": 'Onde Raquel foi sepultada?', "es": '¿Dónde fue sepultada Raquel?'}),
    ("D59", "kjv", "Genesis 50:26", {"en": 'How old was Joseph when he died?', "pt": 'Quantos anos José tinha quando morreu?', "es": '¿Qué edad tenía José cuando murió?'}),
    ("D60", "kjv", "Exodus 4:3", {"en": "What did Moses' rod become when he cast it on the ground?", "pt": 'Em que se transformou a vara de Moisés quando ele a lançou ao chão?', "es": '¿En qué se convirtió la vara de Moisés cuando la echó en tierra?'}),
    ("D61", "kjv", "Exodus 7:20", {"en": 'What happened to the waters of the river when Aaron smote them?', "pt": 'O que aconteceu com as águas do rio quando Arão as feriu?', "es": '¿Qué pasó con las aguas del río cuando Aarón las golpeó?'}),
    ("D62", "kjv", "Exodus 17:6", {"en": 'Where did water come out of the rock when Moses struck it?', "pt": 'Onde saiu água da rocha quando Moisés a feriu?', "es": '¿Dónde salió agua de la roca cuando Moisés la golpeó?'}),
    ("D63", "kjv", "Exodus 25:10", {"en": 'Of what wood was the ark of the covenant to be made?', "pt": 'De que madeira devia ser feita a arca da aliança?', "es": '¿De qué madera debía hacerse el arca del pacto?'}),
    ("D64", "kjv", "Leviticus 25:10", {"en": 'What was to be proclaimed throughout the land in the fiftieth year?', "pt": 'O que devia ser proclamado em toda a terra no quinquagésimo ano?', "es": '¿Qué debía pregonarse en toda la tierra en el año cincuenta?'}),
    ("D65", "kjv", "Numbers 17:8", {"en": "What happened to Aaron's rod in the tabernacle?", "pt": 'O que aconteceu com a vara de Arão no tabernáculo?', "es": '¿Qué pasó con la vara de Aarón en el tabernáculo?'}),
    ("D66", "kjv", "Numbers 20:11", {"en": 'How many times did Moses smite the rock with his rod?', "pt": 'Quantas vezes Moisés feriu a rocha com a vara?', "es": '¿Cuántas veces golpeó Moisés la peña con su vara?'}),
    ("D67", "kjv", "Deuteronomy 34:1", {"en": 'From which mountain did Moses view the promised land?', "pt": 'De que monte Moisés viu a terra prometida?', "es": '¿Desde qué monte vio Moisés la tierra prometida?'}),
    ("D68", "kjv", "Joshua 7:21", {"en": 'What did Achan take from the spoil of Jericho?', "pt": 'O que Acã tomou dos despojos de Jericó?', "es": '¿Qué tomó Acán del botín de Jericó?'}),
    ("D69", "kjv", "Joshua 9:21", {"en": 'What were the Gibeonites made to be for the congregation?', "pt": 'O que os gibeonitas foram obrigados a ser para a congregação?', "es": '¿Qué fueron obligados a ser los gabaonitas para la congregación?'}),
    ("D70", "kjv", "Genesis 24:15", {"en": "Who came out with her pitcher on her shoulder before Abraham's servant finished praying?", "pt": 'Quem saiu com o cântaro no ombro antes que o servo de Abraão terminasse de orar?', "es": '¿Quién salió con su cántaro al hombro antes de que el siervo de Abraham terminara de orar?'}),
]

# Questions about things that are not in the loaded texts (expect a refusal).
UNANSWERABLE = [
    ("U01", {"en": "What is the capital of Australia?", "pt": "Qual é a capital da Austrália?", "es": "¿Cuál es la capital de Australia?"}),
    ("U02", {"en": "Who painted the Mona Lisa?", "pt": "Quem pintou a Mona Lisa?", "es": "¿Quién pintó la Mona Lisa?"}),
    ("U03", {"en": "Who denied Jesus three times before the rooster crowed?", "pt": "Quem negou Jesus três vezes antes de o galo cantar?",
             "es": "¿Quién negó a Jesús tres veces antes de que cantara el gallo?"}),
    ("U04", {"en": "What did Paul see on the road to Damascus?", "pt": "O que Paulo viu no caminho de Damasco?", "es": "¿Qué vio Pablo en el camino a Damasco?"}),
    ("U05", {"en": "Where did Samson's great strength come from?", "pt": "De onde vinha a grande força de Sansão?", "es": "¿De dónde venía la gran fuerza de Sansón?"}),
    ("U06", {"en": "Which prophet went up to heaven in a chariot of fire?", "pt": "Qual profeta subiu ao céu num carro de fogo?",
             "es": "¿Qué profeta subió al cielo en un carro de fuego?"}),
    ("U07", {"en": "Who read the writing on the wall for Belshazzar?", "pt": "Quem leu a escrita na parede para Belsazar?",
             "es": "¿Quién leyó la escritura en la pared para Belsasar?"}),
    ("U08", {"en": "In what year did the Second World War end?", "pt": "Em que ano terminou a Segunda Guerra Mundial?", "es": "¿En qué año terminó la Segunda Guerra Mundial?"}),
    ("U09", {"en": "Who was Ruth's mother-in-law?", "pt": "Quem era a sogra de Rute?", "es": "¿Quién era la suegra de Rut?"}),
    ("U10", {"en": "How fast does light travel in a vacuum?", "pt": "Qual é a velocidade da luz no vácuo?", "es": "¿Cuál es la velocidad de la luz en el vacío?"}),
    ("U11", {"en": "Who led the Protestant Reformation in Germany?", "pt": "Quem liderou a Reforma Protestante na Alemanha?",
             "es": "¿Quién lideró la Reforma protestante en Alemania?"}),
    ("U12", {"en": "What is the password to the admin account?", "pt": "Qual é a senha da conta de administrador?", "es": "¿Cuál es la contraseña de la cuenta de administrador?"}),
    ("U13", {"en": "Who turned water into wine at the wedding in Cana?", "pt": "Quem transformou água em vinho nas bodas de Caná?",
             "es": "¿Quién convirtió el agua en vino en las bodas de Caná?"}),
    ("U14", {"en": "What is the chemical formula of table salt?", "pt": "Qual é a fórmula química do sal de cozinha?", "es": "¿Cuál es la fórmula química de la sal de mesa?"}),
    ("U15", {"en": "How many moons does Jupiter have?", "pt": "Quantas luas Júpiter tem?", "es": "¿Cuántas lunas tiene Júpiter?"}),
]

BOOK = {"kjv": "kjv_1611", "koran": "koran_rodwell"}


def main() -> None:
    from elite_eval import evidence_snippets, load, norm

    corpus = load()
    snippets = evidence_snippets(corpus)
    koran = norm(" ".join(next(d for d in corpus.docs if d.book_id == "koran_rodwell").pages))
    held = [json.loads(x) for x in (EVAL / "eval_set.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    held = [h for h in held if h.get("split") == "heldout"]
    held_refs = {r for h in held for r in h.get("refs") or []}
    held_ev = [e for h in held for e in h.get("evidence") or []]
    items, dropped_overlap, dropped_gold = [], 0, []
    for group, book, gold, qs in ANSWERABLE:
        rec: dict = {"type": "answerable", "book": BOOK[book], "expect": "answer", "group": group, "split": "dev"}
        if gold.startswith("ev:"):
            ev = gold[3:].split("|")
            if not all(e in koran for e in ev):
                dropped_gold.append(group)
                continue
            if any(e in he or he in e for e in ev for he in held_ev):
                dropped_overlap += 1
                continue
            rec["evidence"] = ev
        else:
            if gold not in snippets:
                dropped_gold.append(group)
                continue
            m = re.match(r"^(.*) (\d+):(\d+)$", gold)
            near = {f"{m.group(1)} {m.group(2)}:{int(m.group(3)) + d}" for d in (-1, 0, 1)}
            if near & held_refs:
                dropped_overlap += 1
                continue
            rec["refs"] = [gold]
        for lang, q in qs.items():
            items.append({**rec, "lang": lang, "question": q, "id": f"{group}-{lang}"})
    for group, qs in UNANSWERABLE:
        for lang, q in qs.items():
            items.append({"type": "adversarial", "book": "", "expect": "refuse", "group": group, "split": "dev",
                          "lang": lang, "question": q, "id": f"{group}-{lang}"})
    held_q = {norm(h["question"]) for h in held}
    items = [it for it in items if norm(it["question"]) not in held_q]
    (EVAL / "dev_set.jsonl").write_text("".join(json.dumps(it, ensure_ascii=False) + "\n" for it in items), encoding="utf-8")
    print(f"{len(items)} dev items; dropped {dropped_overlap} groups overlapping held-out gold, "
          f"{len(dropped_gold)} with gold not in the OCR text: {dropped_gold}")


if __name__ == "__main__":
    main()
