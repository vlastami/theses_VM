**Název**: Vícesnímkové zpracování účtenek v mobilní aplikaci CheckChecker s využitím OpenCV a LLM

**Název v angličtině**: Multi-Frame Receipt Processing in the CheckChecker Mobile App Using OpenCV and LLMs

**Zásady pro vypracování**

Diplomová práce navazuje na bakalářskou práci, v níž byla představena mobilní aplikace CheckChecker využívající OCR a LLM ke skenování účetních dokladů a automatické kategorizaci výdajů. Z analýzy existujících dat je evidentní, že současná implementace je limitována kvalitou pořízeného snímku; zejména u dlouhých účtenek se projevuje nízká kvalita výstupu.
Cílem diplomové práce je návrh a implementace rozšíření aplikace umožňující zpracování účtenek z videostreamu. Uživatel natočí účtenku kamerou a systém automaticky detekuje kvalitní snímky vhodné pro zpracování. Díky vyššímu počtu detailních snímků dojde ke zlepšení výstupu celého procesu.

Teoretická část se zaměří na problematiku vícesnímkového zpracování obrazu, hodnocení kvality snímku, OCR a metody využití velkých jazykových modelů (LLM) pro validaci a doplnění údajů.

Praktická část se zaměří na implementaci funkčního řešení do produkční aplikace CheckChecker s využitím knihovny OpenCV, služby Azure Document Intelligence a nástrojů LLM.
Součástí řešení bude implementace následujících funkcí:
1.	Detekce optimálních snímků z videostreamu na základě např. ostrosti, kontrastu a stability.
2.	Automatické pořízení snímku při dosažení dostatečné kvality („auto-capture“).
3.	Spojení více snímků do jednoho dokumentu (OCR stitching) pro dlouhé účtenky.
4.	OCR zpracování s validací a doplněním klíčových polí pomocí LLM (např. obchodník, datum, částky, DPH).

Výstupem je kromě zdokumentované aplikace i statistické vyhodnocení přesnosti, rychlosti a uživatelské úspěšnosti oproti současnému řešení.


**Osnova**

Teoretická část
1.	OpenCV a metody vícesnímkového zpracování obrazu
2.	OCR a Azure Document Intelligence
3.	využití LLM pro strukturovanou extrakci a validaci dat
      
Praktická část
4.	návrh architektury vícesnímkové pipeline.
5.	návrh metrik pro hodnocení kvality snímku a automatické spouště
6.	integrace OpenCV, OCR a LLM modulů do aplikace CheckChecker
7.	implementace uživatelského rozhraní s okamžitou zpětnou vazbou
8.	testování (experimentální porovnání s jednosnímkovým přístupem)
9.	zhodnocení (včetně statistického vyhodnocení přesnosti a efektivity)

**Literatura**
1.	Bradski, G., Kaehler, A. Learning OpenCV 3. O’Reilly Media, 2017.
2.	Kim, G. et al. Donut: Document Understanding Transformer without OCR. ECCV Workshops 2022.
3.	Microsoft. Azure Document Intelligence – Developer Guide. 2025.
4.	OpenCV.org. OpenCV Documentation: Feature Detection, Image Stitching and Camera Calibration [online]. Verze 4.x, 2024. Dostupné z: https://docs.opencv.org/4.x/ [cit. 2025-10-19].
5.	Huang, Y.; Lv, T.; Cui, L.; Lu, Y.; Wei, F. LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking. In: Proceedings of the 30th ACM International Conference on Multimedia (MM ’22). New York: ACM, 2022. DOI: 10.1145/3503161.3548112. Dostupné z: https://arxiv.org/abs/2204.08387 [cit. 2025-10-19].

