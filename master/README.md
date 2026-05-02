# Vícesnímkové zpracování účtenek v mobilní aplikaci CheckChecker s využitím OpenCV a LLM

## Přidělené téma

Toto téma je řešeno jako diplomová práce na Katedře informatiky. Téma se zabývá rozšířením mobilní aplikace CheckChecker o vícesnímkové zpracování obrazu s využitím knihovny OpenCV, cloudové služby Azure Document Intelligence a velkých jazykových modelů (LLM).

### Zadání

**Vedoucí práce:** Mgr. Jiří Fišer, Ph.D.  
**Typ práce:** Diplomová (Mgr.)  
**Akademický rok zadání a obhajoby:** 2025/2026  
**Pracoviště:** KI – Katedra informatiky

### Cíl práce

Diplomová práce navazuje na bakalářskou práci, v níž byla představena mobilní aplikace CheckChecker využívající OCR a LLM ke skenování účetních dokladů a automatické kategorizaci výdajů. Z analýzy existujících dat je evidentní, že současná implementace je limitována kvalitou pořízeného snímku; zejména u dlouhých účtenek se projevuje nízká kvalita výstupu.

Cílem práce je návrh a implementace rozšíření aplikace umožňujícího zpracování účtenek z videostreamu. Uživatel natočí účtenku kamerou a systém automaticky detekuje kvalitní snímky vhodné pro zpracování. Díky vyššímu počtu detailních snímků dojde ke zlepšení výstupu celého procesu.

### Osnova práce

#### Teoretická část

1. OpenCV a metody vícesnímkového zpracování obrazu
2. OCR a Azure Document Intelligence
3. Využití LLM pro strukturovanou extrakci a validaci dat

#### Praktická část

4. Návrh architektury vícesnímkové pipeline
5. Návrh metrik pro hodnocení kvality snímku a automatické spouště
6. Integrace OpenCV, OCR a LLM modulů do aplikace CheckChecker
7. Implementace uživatelského rozhraní s okamžitou zpětnou vazbou
8. Testování (experimentální porovnání s jednosnímkovým přístupem)
9. Zhodnocení (včetně statistického vyhodnocení přesnosti a efektivity)

### Implementované funkce

- Detekce optimálních snímků z videostreamu na základě ostrosti, kontrastu a stability obrazu
- Automatické pořízení snímku při dosažení dostatečné kvality (auto-capture)
- Spojení více snímků do jednoho dokumentu (OCR stitching) pro dlouhé účtenky
- OCR zpracování s validací a doplněním klíčových polí pomocí LLM (obchodník, datum, částky, DPH)

### Použité technologie

- React Native (TypeScript)
- Kotlin / Android (OpenCV4Android SDK)
- Azure Document Intelligence (prebuilt-receipt)
- Azure OpenAI Service (GPT-4o-mini)
- OpenCV 4.x
- Java Spring framework
- PostgreSQL

### Literatura

1. BRADSKI, G.; KAEHLER, A. *Learning OpenCV 3*. O'Reilly Media, 2017.
2. KIM, G. et al. Donut: Document Understanding Transformer without OCR. *ECCV Workshops 2022*.
3. MICROSOFT. *Azure Document Intelligence – Developer Guide* [online]. 2025. Dostupné z: https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/
4. OPENCV.ORG. *OpenCV Documentation: Feature Detection, Image Stitching and Camera Calibration* [online]. Verze 4.x, 2024. Dostupné z: https://docs.opencv.org/4.x/ [cit. 2025-10-19].
5. HUANG, Y.; LV, T.; CUI, L.; LU, Y.; WEI, F. LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking. In: *Proceedings of the 30th ACM International Conference on Multimedia (MM '22)*. New York: ACM, 2022. DOI: 10.1145/3503161.3548112. Dostupné z: https://arxiv.org/abs/2204.08387 [cit. 2025-10-19].
