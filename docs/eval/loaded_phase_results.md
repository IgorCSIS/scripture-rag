# Grounding phase: source not loaded vs loaded

Same questions, same models, same threshold (0.22). No topic rules: the only difference between the two runs is which documents are indexed.

## Added sources

- `talmud_rodkinson_v1`: https://archive.org/details/neweditionofbaby01rodk (possible-copyright-status: NOT_IN_COPYRIGHT, date: 1896)
- `quran_commentary_wherry_v1`: https://archive.org/details/comprehensivecom01saleuoft (possible-copyright-status: NOT_IN_COPYRIGHT, date: 1882-86)

## Summary

| Condition | Pass | Answered | Abstained |
|---|---|---|---|
| not_loaded | 11/13 | 2 | 11 |
| loaded | 11/13 | 9 | 4 |

## Per question

| ID | Lang | Question | Not loaded | Loaded | Loaded citation |
|---|---|---|---|---|---|
| S045 | en | What did Hillel tell the man who asked to be taught the whole Torah while standing on one foot? | abstain (pass) | answer (pass) | talmud_rodkinson_v1 p.93 #talmud_rodkinson_v1-0210, talmud_rodkinson_v1 p.92 #talmud_rodkinson_v1-0208 |
| S046 | en | How many categories of creative labor does the Mishnah forbid on the Sabbath? | abstain (pass) | answer (FAIL) | talmud_rodkinson_v1 p.27 #talmud_rodkinson_v1-0029, talmud_rodkinson_v1 p.173 #talmud_rodkinson_v1-0435 |
| S048 | en | Did the House of Hillel or the House of Shammai prevail on the order of lighting Hanukkah candles? | abstain (pass) | answer (pass) | talmud_rodkinson_v1 p.76 #talmud_rodkinson_v1-0162, talmud_rodkinson_v1 p.80 #talmud_rodkinson_v1-0174 |
| S069 | pt | Quantas categorias de trabalho a Mishná proíbe no sábado? | abstain (pass) | answer (pass) | talmud_rodkinson_v1 p.40 #talmud_rodkinson_v1-0063, talmud_rodkinson_v1 p.27 #talmud_rodkinson_v1-0029 |
| S047 | en | In the story of the oven of Akhnai, what miracles did Rabbi Eliezer call on to prove his ruling? | abstain (pass) | abstain (pass) | - |
| S049 | en | According to Pirkei Avot, on which three things does the world stand? | abstain (pass) | abstain (pass) | - |
| S050 | en | How does Ibn Kathir's tafsir explain the meaning of the disjointed letters Alif Lam Mim? | abstain (pass) | abstain (pass) | - |
| S071 | es | ¿Qué dice el tafsir de Ibn Kathir sobre las letras Alif Lam Mim? | answer (FAIL) | answer (FAIL) | koran_rodwell p.286 #koran_rodwell-0674, koran_rodwell p.209 #koran_rodwell-0470 |
| S051 | en | What does Ibn Kathir report about the wording of the Prophet's farewell sermon? | abstain (pass) | abstain (pass) | - |
| T001 | en | According to Baidhawi, why was the law of legacies to parents and relatives enacted? | abstain (pass) | answer (pass) | quran_commentary_wherry_v1 p.366 #quran_commentary_wherry_v1-0871 |
| T002 | pt | Segundo Baidhawi, por que foi promulgada a lei dos legados aos pais e parentes? | answer (FAIL) | answer (pass) | quran_commentary_wherry_v1 p.366 #quran_commentary_wherry_v1-0871 |
| T003 | en | In the commentary on the opening sura, how does al Zamakhshari's reading of the negatives differ from the common exposition? | abstain (pass) | answer (pass) | quran_commentary_wherry_v1 p.302 #quran_commentary_wherry_v1-0677 |
| T004 | es | En el comentario de la primera sura, ¿en qué difiere la lectura de al Zamakhshari de la exposición común? | abstain (pass) | answer (pass) | quran_commentary_wherry_v1 p.302 #quran_commentary_wherry_v1-0677, quran_commentary_wherry_v1 p.367 #quran_commentary_wherry_v1-0876 |

## Loaded answers

- **S045** What did Hillel tell the man who asked to be taught the whole Torah while standing on one foot?
  - 'must likewise repose confidence in the oral law (which appears at first sight different from the written law)." Another Gentile came to Shamai saying : " Convert me on the condition that thou teach me the whole Torah while I stand on one foot." Shamai pushed him away with the builders\' measure he he'
- **S046** How many categories of creative labor does the Mishnah forbid on the Sabbath?
  - 'due observance of the Sabbath is repeatedly ordained; in some instances merely mentioning the day as one to be kept inviolate and holy; and in others employing greater emphasis, referring to the history of creation, and establishing the obser- vance as a sign of the covenant between the Lord and Isr'
- **S048** Did the House of Hillel or the House of Shammai prevail on the order of lighting Hanukkah candles?
  - "in the best possible manner should light the first night eight flames and every fol- lowing night one flame less, says the school of Shamai. [talmud_rodkinson_v1 p.76 #talmud_rodkinson_v1-0162] Hillel's school holds that only one should be lit on the first night and be increased by one on each succe"
- **S069** Quantas categorias de trabalho a Mishná proíbe no sábado?
  - 'labor may be distinguished. [talmud_rodkinson_v1 p.40 #talmud_rodkinson_v1-0063] Different instances of forgetfulness regarding Sabbath and the performance of labor on the Sabbath. [talmud_rodkinson_v1 p.40 #talmud_rodkinson_v1-0063] Instances of forgetfulness in dietary matters. [talmud_rodkinson_v'
- **S071** ¿Qué dice el tafsir de Ibn Kathir sobre las letras Alif Lam Mim?
  - 'been miminent when Muhammad could thus write. [koran_rodwell p.286 #koran_rodwell-0674] Lit. carry not. [koran_rodwell p.286 #koran_rodwell-0674] Comp. [koran_rodwell p.286 #koran_rodwell-0674] Matth. vi. [koran_rodwell p.286 #koran_rodwell-0674] 26; Luke xii. [koran_rodwell p.286 #koran_rodwell-067'
- **T001** According to Baidhawi, why was the law of legacies to parents and relatives enacted?
  - 'change the legacy, after he hath heard it bequeathed by the dying person, surely the sin thereof shall be on those who change it, for GOD is he who heareth and knoweth. [quran_commentary_wherry_v1 p.366 #quran_commentary_wherry_v1-0871] (182) Howbeit he who apprehendeth from the testator any mistake'
- **T002** Segundo Baidhawi, por que foi promulgada a lei dos legados aos pais e parentes?
  - 'change the legacy, after he hath heard it bequeathed by the dying person, surely the sin thereof shall be on those who change it, for GOD is he who heareth and knoweth. [quran_commentary_wherry_v1 p.366 #quran_commentary_wherry_v1-0871] (182) Howbeit he who apprehendeth from the testator any mistake'
- **T003** In the commentary on the opening sura, how does al Zamakhshari's reading of the negatives differ from the common exposition?
  - 'faithful who preceded Muhammad ; under which appellations are also compre- hended the Jews and Christians, such as they were in the times of their primitive purity, before they had deviated from their respective institutions ; not the way of the modern Jews, whose signal calamities are marks of the '
- **T004** En el comentario de la primera sura, ¿en qué difiere la lectura de al Zamakhshari de la exposición común?
  - 'faithful who preceded Muhammad ; under which appellations are also compre- hended the Jews and Christians, such as they were in the times of their primitive purity, before they had deviated from their respective institutions ; not the way of the modern Jews, whose signal calamities are marks of the '
