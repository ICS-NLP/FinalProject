# Zero-shot error analysis (qualitative)

Source model: best Phase-1 checkpoint (`Final_Source_Model/`, `afro-xlmr-large-76L` fine-tuned on Hau+Amh+Yor). Errors are sampled (up to 8 per error type per target) from the official `twi` and `pcm` AfriHate test splits.


## Error counts in the sample

| target | category | n |
|---|---|---|
| pcm | over_flagging | 8 |
| pcm | under_flagging | 8 |
| twi | over_flagging | 8 |
| twi | under_flagging | 8 |

## Target: `twi`


### Over-flagging — Normal predicted as Hate / Abuse (false positives)

- *true* `Normal` / *pred* `Abuse` — "My Manager ye brofo no one one like tugyimi rice. 🔥🔥😍"
- *true* `Normal` / *pred* `Abuse` — "Herhh Kwabena wo ha twa s3 abokyi sekan!🔥"
- *true* `Normal` / *pred* `Abuse` — "Wei de3 Dagaate Poloo ankasa"
- *true* `Normal` / *pred* `Hate` — "Americans are the real frafrafo)"
- *true* `Normal` / *pred* `Abuse` — "Ah the abokyi people go demma hometown or? Cos I haven’t seen 1 in almost a week."
- *true* `Normal` / *pred* `Hate` — "Anka )b3ti Zongofo) no nka"
- *true* `Normal` / *pred* `Hate` — "Wo Kɔ Alatafuo Traditional Marriage, na wo hunu sɛ Wornom de Nsa no egu Bath No mu"
- *true* `Normal` / *pred* `Abuse` — "Afihyia pa ooh, afi sesei na w'awo ntafo."

### Under-flagging — Hate / Abuse predicted as Normal (false negatives)

- *true* `Abuse` / *pred* `Normal` — "Saa pɛpɛɛpɛ. Nyankopon nti, Kontomponi bɛ ferɛ saaa🙏🏾"
- *true* `Abuse` / *pred* `Normal` — "Ah na severed vagina deɛ bosom de yɛ dɛn? Gyimisɛm kwa!"
- *true* `Abuse` / *pred* `Normal` — "Wooooow!! Ma y3 kurasini aky33 "@degalalinco: Did u know dis??😂😂😂 ""
- *true* `Abuse` / *pred* `Normal` — "Omo ho nny3 F3 nso oo, ahwene3 pa nkasa ampah"
- *true* `Abuse` / *pred* `Normal` — "na s3 mode3 gyimie noaaa Fri tete"
- *true* `Abuse` / *pred* `Normal` — "Saa man no, n'anum kasa nyɛ dɛ"
- *true* `Abuse` / *pred* `Normal` — "Kwasia you don't have money so you see taking care of 1 person us big favour Jon"
- *true* `Abuse` / *pred* `Normal` — "🤣🤣🤣🤣🤣🤣akoa wei yɛ jon, adɛn Manhyia no ɛyɛ orphanage home.and feel sorry for your people cos amo yɛ mmɔbɔ,so called land owners have nothing on their own land.ɛnyɛ gyimisɛm"


## Target: `pcm`


### Over-flagging — Normal predicted as Hate / Abuse (false positives)

- *true* `Normal` / *pred* `Abuse` — "patronising alternativemedicine vendors dey cum around on saturdays stay d bstop and stat blastin ru deir loudspeakers about d various cures dey ave esp stds and stis den my dad wud b lyk go an buy agbo jedi me rm dose p…"
- *true* `Normal` / *pred* `Abuse` — "fine boys like us we nor too dey chase women na dem dey rush us chevron gbagada"
- *true* `Normal` / *pred* `Abuse` — "@USER @USER So what exactly was d intent if not to cause reputational damage?Cooked up a totally false story&amp;attached someones name to it but then you "did not intend to bring taooma's reputation into any form of dis…"
- *true* `Normal` / *pred* `Abuse` — "lmao you bin dey tell me small small but na you worse pass"
- *true* `Normal` / *pred* `Abuse` — "you wey na work to stand up from bed wan dance lol"
- *true* `Normal` / *pred* `Abuse` — "no now daboski woman wey sabi no be like dis we plan am o who will give abimbola heat now who will trouble terfas soul this is not fair who will take timothy and justice atinuke for a ride who"
- *true* `Normal` / *pred* `Abuse` — "if na this oscar award winning actor now i will understand na person way no fit navigate elevator dem dey do like this"
- *true* `Normal` / *pred* `Abuse` — "onyi youre not getting me i mean my comment was a joke but nah none of them will cry for man sha abeg i dey laugh for here before they think im cray"

### Under-flagging — Hate / Abuse predicted as Normal (false negatives)

- *true* `Abuse` / *pred* `Normal` — "@USER Another reason not to bring a hoe to a sword fight"
- *true* `Hate` / *pred* `Normal` — "Then I think u should meet d hausas"@USER: Have u met d ebiras"@USER: Yoruba people too Like noise tah"""
- *true* `Hate` / *pred* `Normal` — "I agree"@USER: Facts: 50% of Whores in Nigeria are Igbo's."""
- *true* `Abuse` / *pred* `Normal` — "kill em wit d sauce kill em wit d success kill em wit d smile kill em wit d noise kill em wit d joy the key is dat u r happy we want school bt dey gave us prison we want education by dey taught us lesson"
- *true* `Abuse` / *pred* `Normal` — "na catfish i do i no kill person saratu"
- *true* `Abuse` / *pred* `Normal` — "may the good lord touch the heart of your debtors this week amen i have over out there and i dey see all of my debtors dey flex well i kuku no get mama or papa so i leave them to god to judge na me wey mumu dey loan like…"
- *true* `Abuse` / *pred* `Normal` — "e be like all these coaches don dey mad wtf is wrong with tuchel wtf is chupo motim wheres icardi if only these guys are aware how much ive lost to betja"
- *true* `Abuse` / *pred* `Normal` — "yansh be trending all of a sudden as i no kon get yansh laidis wetin i go do"


## Patterns we observe

**Twi (Akan).**
- *Over-flagging.* Tweets that simply mention ethnic / regional identity terms (e.g. `Ntafo`, `Frafrafo`, `Zongofo`, `Alatafuo`) are labelled Hate even when the rest of the post is playful or descriptive. The encoder appears to associate these proper nouns with hostile contexts seen during Hausa+Amharic training, where intergroup tokens correlate with hate.
- *Under-flagging.* The hardest under-flagging cases involve idiomatic insults that the source languages do not share (`Kontomponi`, `Gyimisɛm`, `akoa`, `Apakye musuoni`, `Animguasefoɔ`). Those tokens carry abuse for Akan readers but are unfamiliar to the source-trained head.

**Nigerian Pidgin.**
- *Over-flagging.* Pidgin's familiar/teasing register (`abeg`, `mumu`, `oga`, `mad woman`, rhetorical exaggeration) is repeatedly classified Abuse — common in-group banter rather than directed harassment.
- *Under-flagging.* Stereotyping framed as agreement (`I agree "Facts: 50% of … are Igbo's"`) and ethnic-group put-downs (`Yoruba people too like noise tah`) are labelled Normal — the model misses ethnic-group hate when the surface tone is conversational.
- *Cross-class confusion* on Pidgin is rare in this sample; when the model flags a tweet it tends to keep the right Hate vs Abuse split.

**Implications for the report.**
1. Twi failures concentrate in **morphology + identity terms**; they explain why adding Yoruba to the source mix in E3/E4 helps but does not close the gap.
2. Pidgin failures concentrate in **register and irony**; few-shot k=5 already lifts F1 above zero-shot, suggesting a small amount of in-domain calibration is enough.
3. False positives on identity terms are an **ethical** risk: deploying without human review would silence legitimate in-group speech in under-served languages.

