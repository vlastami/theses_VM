**Název**: Vícesnímkové zpracování účtenek v mobilní aplikaci CheckChecker s využitím OpenCV a LLM
**Název v angličtině**: Multi-Frame Receipt Processing in the CheckChecker Mobile App Using OpenCV and LLMs

**Zásady pro vypracování**
Cílem diplomové práce je návrh a implementace rozšíření mobilní aplikace CheckChecker umožňující zpracování účtenek z videostreamu. Uživatel natočí účtenku kamerou, systém automaticky detekuje kvalitní snímky, provádí OCR a živě zobrazuje rozpoznané informace. Tato funkcionalita řeší problém s kvalitou vstupu pro OCR u dlouhých účtenek.

Teoretická část se zaměří na problematiku vícesnímkového zpracování obrazu, hodnocení kvality snímku, OCR a metody využití velkých jazykových modelů (LLM) pro validaci a doplnění údajů.

Praktická část se zaměří na implementaci funkčního řešení do produkční aplikace CheckChecker s využitím knihovny OpenCV, služby Azure Document Intelligence a nástrojů LLM.
Součástí řešení bude implementace následujících funkcí:
1.	Detekce optimálních snímků z videostreamu na základě např. ostrosti, kontrastu a stability.
2.	Automatické pořízení snímku při dosažení dostatečné kvality („auto-capture“).
3.	Spojení více snímků do jednoho dokumentu (OCR stitching) pro dlouhé účtenky.
4.	OCR zpracování s validací a doplněním klíčových polí pomocí LLM (např. obchodník, datum, částky, DPH).
5.	Statistické vyhodnocení přesnosti, rychlosti a uživatelské úspěšnosti oproti současnému řešení.


**Osnova**
1.	Úvod
o	Motivace, cíl práce a vymezení problému.
2.	Teoretický základ
o	OpenCV a metody vícesnímkového zpracování obrazu.
o	OCR a Azure Document Intelligence.
o	Využití LLM pro strukturovanou extrakci a validaci dat.
3.	Návrh řešení
o	Architektura vícesnímkové pipeline.
o	Návrh metrik pro hodnocení kvality snímku a automatické spouště.
4.	Implementace
o	Integrace OpenCV, OCR a LLM modulů do aplikace CheckChecker.
o	Návrh a implementace uživatelského rozhraní s okamžitou zpětnou vazbou.
5.	Testování a vyhodnocení
o	Experimentální porovnání s jednosnímkovým přístupem.
o	Statistické vyhodnocení přesnosti a efektivity.
6.	Závěr
o	Shrnutí výsledků a návrhy na další rozvoj.

**Literatura**
1.	Bradski, G., Kaehler, A. Learning OpenCV 3. O’Reilly Media, 2017.
2.	Kim, G. et al. Donut: Document Understanding Transformer without OCR. ECCV Workshops 2022.
3.	Microsoft. Azure Document Intelligence – Developer Guide. 2025.
4.	OpenCV Documentation. Feature Detection, Image Stitching and Camera Calibration. OpenCV.org, 2024.
5.	Xu, Y. et al. LayoutLMv3. EMNLP 2022.
