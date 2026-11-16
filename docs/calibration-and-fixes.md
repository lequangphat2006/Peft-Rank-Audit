\# Calibration audit + kế hoạch hoàn thiện phản biện



Tài liệu nội bộ. Tổng hợp kết quả chạy lại `code/ext\_plot\_reliability\_all.py`

(16 seed × 5 rank × 2 model, T fit trên calibration split) và kế hoạch xử lý

8 điểm phản biện còn lại. Ngày cập nhật: 2026-11-16.



\---



\## PHẦN A — KẾT QUẢ CALIBRATION (ĐÃ XONG)



\### A.1. Kiểm chứng tính nhất quán



NLL raw từ script khớp tuyệt đối với Table 3 của bài:



| Rank | NLL raw (script) | NLL raw (Table 3) | Khớp |

|---|---|---|---|

| XLM-R r=1  | 0.2089 | 0.2089 | ✅ |

| XLM-R r=4  | 0.2102 | 0.2102 | ✅ |

| XLM-R r=8  | 0.2087 | 0.2087 | ✅ |

| XLM-R r=16 | 0.2148 | 0.2148 | ✅ |

| XLM-R r=32 | 0.2127 | 0.2127 | ✅ |



`n=16` ở mọi ô. T nằm trong 1.50–1.70. Số liệu dùng được.



\### A.2. Bốn phát hiện mới



\*\*Phát hiện 1 — NLL trend sau rescaling khác nhau giữa hai encoder.\*\*



| | NLL raw r1→r32 | NLL TS r1→r32 | Diễn giải |

|---|---|---|---|

| XLM-R  | +0.0038 | \*\*+0.0080\*\* | Trend tăng gấp đôi → không phải overconfidence đồng đều |

| PhoBERT | +0.0163 | \*\*+0.0037\*\* | Trend giảm 4 lần → phần lớn là overconfidence rescale được |



Củng cố trực tiếp lập luận mục 5.3: "NLL increase is not just a simple

overconfidence that rescaling can remove" — nhưng chỉ đúng với XLM-R,

không đúng với PhoBERT. Đây là bằng chứng mới cho luận điểm

"model dependence" ở Đóng góp 3.



\*\*Phát hiện 2 — ECE không theo kịp NLL.\*\*



| | ECE raw r1→r32 | NLL raw r1→r32 | Nhất quán? |

|---|---|---|---|

| XLM-R  | 0.0360 → 0.0355 (\*\*−0.0005\*\*) | +0.0038 | ❌ Ngược chiều |

| PhoBERT | 0.0372 → 0.0423 (\*\*+0.0051\*\*) | +0.0163 | ✅ Cùng chiều |



→ Không thể thay NLL bằng ECE trong audit. Một audit chỉ dùng ECE sẽ

kết luận ngược với XLM-R.



\*\*Phát hiện 3 — ECE non-monotone, chênh lệch nằm trong \~2 SD seed.\*\*



| | ECE raw range | SD trong rank | Chênh cực đại / SD |

|---|---|---|---|

| XLM-R  | 0.0336–0.0372 | 0.002–0.003 | \~1.7 |

| PhoBERT | 0.0372–0.0423 | 0.003 | \~1.7 |



→ Không được viết "ECE thay đổi theo rank" mà không test. Viết:

"ECE shows no monotone rank trend; the range across ranks is within

about two seed standard deviations for both encoders."



\*\*Cảnh báo:\*\* XLM-R r=8 có ECE raw thấp nhất (0.0336) và NLL raw thấp

nhất (0.2087). KHÔNG được viết "rank 8 calibrates best" — đây là rank

có LR thấp nhất trong grid mở rộng (1e-4), trùng với phản biện #3 về

nhiễu chọn LR. Chỉ nói "non-monotone".



\*\*Phát hiện 4 — PhoBERT khó hiệu chỉnh hơn XLM-R.\*\*



| | ECE raw → TS | Mức giảm |

|---|---|---|

| XLM-R  | 0.0336–0.0372 → 0.0125–0.0154 | 58–63% |

| PhoBERT | 0.0372–0.0423 → 0.0173–0.0212 | 47–54% |



→ Có thể do khác biệt tokenizer/segmenter ảnh hưởng phân bố confidence.

Một câu thảo luận ngắn ở mục 5.3 là đủ.



\### A.3. Ba chỗ trong bài PHẢI sửa



| Vị trí | Câu hiện tại | Sửa thành |

|---|---|---|

| Mục 5.3, câu 2 | "T between 1.50 and 1.62" | "T = 1.50–1.62 for XLM-R and 1.53–1.70 for PhoBERT" |

| Mục 5.3, câu 3 | "ECE to 0.0125–0.0154" | "ECE to 0.0125–0.0154 (XLM-R) and 0.0173–0.0212 (PhoBERT)" |

| Mục 5.3, câu cuối | "(+0.0071, 95% CI \[+0.0038, +0.0103])" | Kiểm tra lại cách tính; nếu là endpoint diff → "+0.0080" |



\### A.4. Table 4 — dán trực tiếp



| Model | Rank | LR | ECE raw | T | ECE TS | NLL raw | NLL TS |

|---|---|---|---|---|---|---|---|

| XLM-R | 1 | 1e-3 | 0.0360 ± 0.0028 | 1.58 | 0.0151 ± 0.0031 | 0.2089 | 0.1732 |

| XLM-R | 4 | 5e-4 | 0.0353 ± 0.0023 | 1.62 | 0.0142 ± 0.0036 | 0.2102 | 0.1709 |

| XLM-R | 8 | 1e-4 | 0.0336 ± 0.0021 | 1.50 | 0.0125 ± 0.0029 | 0.2087 | 0.1795 |

| XLM-R | 16 | 2e-4 | 0.0372 ± 0.0019 | 1.62 | 0.0154 ± 0.0026 | 0.2148 | 0.1733 |

| XLM-R | 32 | 5e-5 | 0.0355 ± 0.0021 | 1.50 | 0.0139 ± 0.0024 | 0.2127 | 0.1812 |

| PhoBERT | 1 | 5e-4 | 0.0372 ± 0.0032 | 1.59 | 0.0174 ± 0.0026 | 0.2198 | 0.1871 |

| PhoBERT | 4 | 5e-4 | 0.0405 ± 0.0025 | 1.70 | 0.0195 ± 0.0031 | 0.2337 | 0.1865 |

| PhoBERT | 8 | 1e-4 | 0.0378 ± 0.0025 | 1.53 | 0.0173 ± 0.0036 | 0.2260 | 0.1954 |

| PhoBERT | 16 | 1e-4 | 0.0396 ± 0.0028 | 1.58 | 0.0209 ± 0.0040 | 0.2261 | 0.1903 |

| PhoBERT | 32 | 1e-4 | 0.0423 ± 0.0027 | 1.64 | 0.0212 ± 0.0033 | 0.2361 | 0.1908 |



\*\*Caption:\*\* Table 4: Calibration audit on the test split, mean ± SD over

16 seeds. T is fitted on the calibration split per seed. NLL raw matches

Table 3 exactly. Raw ECE is non-monotone in rank; the rank range is

within \~2 SD of seed variability for both encoders. Temperature scaling

reduces ECE by 58–63% (XLM-R) and 47–54% (PhoBERT). The NLL trend

survives rescaling for XLM-R and steepens (raw +0.0038, scaled +0.0080),

but shrinks for PhoBERT (raw +0.0163, scaled +0.0037).



\### A.5. Caption cho hai hình



\*\*Fig. 3\*\* (`fig\_reliability\_all\_ranks`):

> Reliability diagrams (15 equal-width bins) for all five global LoRA

> ranks and both encoders, before (grey) and after (dark) temperature

> scaling. T is fitted on the calibration split; the panel shows the

> seed with median raw ECE. Title reports mean ECE before → after across

> 16 seeds. Dashed line = perfect calibration.



\*\*Fig. 4\*\* (`fig\_ece\_vs\_rank`):

> Expected calibration error versus global LoRA rank on a log2 axis,

> before (dashed) and after (solid) temperature scaling, mean over 16

> seeds. Raw ECE is non-monotone in rank for XLM-R and rises only

> slightly for PhoBERT; the range across ranks is within about two seed

> standard deviations. Compare with the NLL trend in Table 3, which is

> in the opposite direction for XLM-R.



\### A.6. Đoạn văn mục 5.3 — viết lại (dùng số liệu thật)



> At 6 epochs NLL rises with rank by +0.0049 over the full range for

> XLM-R (95% CI \[+0.0010, +0.0087]; p^Holm = 0.014), about 2% of NLL and

> below the practical margin of 0.01. Raw ECE is 0.0336–0.0372 for

> XLM-R and 0.0372–0.0423 for PhoBERT (Table 4, Fig. 3), with no

> monotone trend; the range across ranks is within about two seed

> standard deviations (SD ≤ 0.003) for both encoders. Temperature

> scaling fitted on the calibration split (T = 1.50–1.62 for XLM-R,

> 1.53–1.70 for PhoBERT) reduces ECE by 58–63% and 47–54%

> respectively, to 0.0125–0.0154 and 0.0173–0.0212, and lowers NLL to

> 0.171–0.181 and 0.187–0.195. Three observations bear on the rank

> question. First, the NLL trend survives rescaling for XLM-R and in

> fact steepens (raw +0.0038, scaled +0.0080), so it is not a uniform

> overconfidence removable by a single temperature; for PhoBERT the

> opposite holds (raw +0.0163, scaled +0.0037). Second, ECE and NLL

> disagree in direction for XLM-R (ECE −0.0005, NLL +0.0038 from rank 1

> to 32), so an ECE-only audit would have suggested the opposite

> conclusion; the two metrics capture different aspects of probabilistic

> quality. Third, PhoBERT is harder to recalibrate than XLM-R (residual

> ECE 0.017–0.021 versus 0.013–0.015), possibly reflecting tokeniser and

> segmentation differences that affect the confidence distribution. We

> return to the role of α = 2r in Section 5.8.



\---



\## PHẦN B — TÁM ĐIỂM PHẢN BIỆN CÒN LẠI



\### B.1. Session effect (#1 — NGHIÊM TRỌNG NHẤT)



\*\*Vấn đề reviewer nêu:\*\*

\- XLM-R giống hệt 6 chữ số giữa R1 và T2 → pipeline tất định.

\- PhoBERT lệch \~0.02 F1 (0.04–0.05 calibration) với cùng code/seed/LR.

\- Input tách từ không được lưu → không kiểm chứng được.

\- E3 đổi từ p=0.97 sang p=0.0007 giữa hai lần chạy.

\- Cần làm rõ T2 XLM-R chạy lại hay tái sử dụng.



\*\*Việc phải làm:\*\*



1\. Chạy `code/ext\_session\_effect\_diagnostic.py` (xem Phần C) để kiểm tra

&#x20;  bit-identical giữa R1/T2 (XLM-R) và B1X/T2 (PhoBERT). Kết quả mong đợi:

&#x20;  - XLM-R: max |Δlogit| = 0 → xác nhận tất định.

&#x20;  - PhoBERT: max |Δlogit| > 0, đo được % mẫu khác.

&#x20;  - Gold labels: phải giống hệt (cùng split).

2\. Kiểm tra log git/run: T2 XLM-R có thực sự chạy lại không. Nếu tái sử

&#x20;  dụng → sửa mục 3.6(e) thành "re-ran PhoBERT; XLM-R reused R1 logits

&#x20;  after verifying bit-identical output".

3\. Thêm Limitations paragraph (xem text dưới).

4\. Hạ cấp kết luận E3: nếu p đổi từ 0.97 → 0.0007, không thể coi E3 là

&#x20;  confirmatory nữa. Đổi framing thành "exploratory, sensitive to session".



\*\*Text cho Limitations (mục 6.x):\*\*



> A diagnostic comparison of R1 and T2 logits (Appendix D) shows that

> XLM-R output is bit-identical between the two sessions for all 16

> seeds, confirming that the training pipeline is deterministic given

> identical inputs. For PhoBERT, however, logits differ on X% of test

> samples (mean |Δlogit| = Y) despite identical code, seed and LR. The

> most likely sources are segmentation or tokeniser version drift, but

> we cannot confirm this retroactively because tokenised inputs were not

> saved. We have added SHA256 hashing of the tokenised text, per seed,

> to the pipeline and will save tokenised inputs for all future runs.

> The consequence is that the 16-seed confidence intervals in this paper

> measure training variance only; they underestimate the true

> between-session uncertainty, and the PhoBERT extension (E1–E3) has not

> been independently replicated. We have therefore downgraded the E3

> result from confirmatory to exploratory.



\### B.2. LR selection noise (#3)



\*\*Vấn đề reviewer nêu:\*\*

\- LR chọn trên 1.598 câu calib (\~67 neutral), 3 seed, lưới thô, tie 0.005.

\- XLM-R LR map không đơn điệu (1e-3, 5e-4, 1e-4, 2e-4, 5e-5).

\- Rank 8 bị chọn LR thấp rồi thành điểm thấp nhất ở cả 3 và 6 epoch.

\- Rank 32 chọn 5e-5 ở biên lưới mở rộng.



\*\*Việc phải làm:\*\*



1\. Nếu có file kết quả tuning per-run → chạy

&#x20;  `code/ext\_lr\_tuning\_stability.py` (xem Phần C) để báo gap best vs

&#x20;  second-best LR cho từng rank.

2\. Nếu không có → thêm Limitations paragraph và ít nhất báo LR map kèm

&#x20;  SD calibration F1 trong Table 2 (hiện chỉ báo mean).



\*\*Text cho Limitations (mục 6.x):\*\*



> Learning rates were selected on a 1,598-sentence calibration split

> (≈67 neutral examples) with three tuning seeds and a coarse grid. For

> XLM-R, rank 8 received the lowest LR in the grid (1e-4) and became the

> worst rank at both 3 and 6 epochs; rank 32 required the extended grid

> boundary (5e-5). We cannot fully separate the rank-8 result from

> noise in LR selection: the gap between the best and second-best LR at

> rank 8 was within the 0.005 tie threshold, so the selected LR is

> near-arbitrary. More tuning seeds or a finer grid would be needed to

> settle this. The rank-4 and rank-16 results, which received interior

> LR values, are more robust.



\### B.3. Abstract và Conclusion (#2 + #4)



\*\*#2 — Điều kiện α/r = 2:\*\*



Sửa Abstract, Contribution 1, và Conclusion. Cụ thể Abstract đổi câu:



> ...increasing global LoRA rank provides no consistent improvement in

> Macro-F1 under rank-specific LR tuning \*\*and constant α/r = 2\*\*.



\*\*#4 — Equivalence gần như theo định nghĩa:\*\*



Sửa Abstract để đưa two-level interval lên làm kết quả chính:



> Equivalence testing (TOST, δ = 0.01) showed that Macro-F1 for XLM-R

> did not change from rank 1 to 32 across seeds (-0.003, 90% CI

> \[-0.008, +0.002]); \*\*when test-set sampling noise is added, the

> rank-32 minus rank-1 difference lies in \[-0.013, +0.006] F1 points,

> and equivalence is not established\*\*.



Thêm câu biện minh δ trong mục 3.4:



> δ = 0.01 corresponds to roughly one to two neutral test sentences

> (\~67 neutral in total); it is about 1.2% of the rank-8 mean F1.



Thêm TOST trực tiếp cho rank 32 − rank 1 trong mục 5.2 (song song với

slope estimand), và báo kiểm tra độ lệch tuyến tính của estimand.



\### B.4. Positive control (#5)



\*\*Text cho Limitations:\*\*



> LoRA rank 1 already reaches 0.838 Macro-F1 versus 0.849 for full

> fine-tuning on UIT-VSFC. With only \~4% neutral examples and label

> noise in the corpus, the task may not be sensitive enough to

> distinguish ranks: a null result on this task does not imply a null

> result where rank is known to matter. A positive-control task with a

> larger label space or a harder domain is left for future work.



\### B.5. H3 không nhất quán (#6)



\*\*Vấn đề:\*\* Tóm tắt và Đóng góp 2 coi H3 là confirmatory, nhưng mục 5.3

gọi ngân sách 3 epoch là exploratory và LR cho 3 epoch không được tune.



\*\*Việc phải làm:\*\* Chọn một trong hai:



\- \*\*(A) Đổi framing (nhanh):\*\* Trong Abstract và Contribution 2, gọi H3

&#x20; là "a preregistered hypothesis whose confirmatory status was

&#x20; undermined by a design flaw (LR inherited from 6-epoch tuning). The

&#x20; fixed-LR branch in Section 5.4 shows the apparent interaction is not

&#x20; robust."

\- \*\*(B) Tune lại LR cho 3 epoch (tốn compute):\*\* Chạy lưới LR cho 3

&#x20; epoch, 3 tuning seed, chọn LR riêng, rồi báo H3 như confirmatory thực

&#x20; sự.



Khuyến nghị (A) vì (B) tốn \~30 run GPU mà lợi ích khoa học thấp.



\### B.6. Meta-analysis (#7)



\*\*Vấn đề:\*\* k=4, T3 chỉ 4 seed, B1X và T2 dùng chung XLM-R, một iteration

có thể là artefact của session effect.



\*\*Sửa text:\*\*



> Đổi "the interaction is zero" → "there is \*\*no evidence\*\* for a

> model × rank interaction".

> Thêm câu: "With k = 4 iterations and T3 contributing only 4 seeds,

> τ is imprecisely estimated; one iteration (T2) may be affected by the

> session effect discussed in Section 5.6. The pooled estimate should

> therefore be read as a summary of the available iterations, not as a

> precise null."



\### B.7. AdaLoRA không tune (#8)



\*\*Việc phải làm:\*\* Chọn một trong hai:



\- Chuyển xuống \*\*Appendix B\*\* ("untuned reference, not used to support

&#x20; any claim about rank").

\- Hoặc tune LR cho AdaLoRA như các phương pháp khác, rồi báo.



Khuyến nghị chuyển xuống Appendix — nhanh và trung thực.



\### B.8. Khuyến nghị rank 4–8 (#9)



\*\*Sửa mục 6.3:\*\*



> Đổi "we recommend rank 4–8" → "small ranks (1–8) are sufficient in

> this setting; the data do not support a narrower recommendation.

> Rank 1 is statistically indistinguishable from the best rank, and

> rank 8 is the lowest of the five, so the observed ordering does not

> justify a specific range."



\### B.9. Tài liệu tham khảo 2026 (\[7], \[8], \[12])



\*\*Việc phải làm trước khi nộp:\*\*

\- Mở arXiv, nhập mã arXiv ID của \[7], \[8], \[12], xác nhận tên tác giả

&#x20; và năm.

\- Kiểm tra DOI của OSF preregistration (10.17605/OSF.IO/A2MN5) còn

&#x20; truy cập được.

\- Kiểm tra phiên bản `peft 0.21.2`, `transformers 5.16.1`, `torch 2.13`

&#x20; — các phiên bản này có vẻ không tồn tại (transformers 5.x chưa ra

&#x20; vào 2026?). Nếu sai, sửa lại theo `pip freeze` thực tế.



\---



\## PHẦN C — SCRIPTS



\### C.1. `code/ext\_session\_effect\_diagnostic.py`



```python

"""So sanh logits giua hai phien (R1 vs T2 cho XLM-R; B1X vs T2 cho PhoBERT).



Muc dich: xac nhan XLM-R tai lap tuyet doi, do luong khac biet PhoBERT,

va kiem tra gold label co trung khong.

"""

import os, glob, hashlib

import numpy as np

from scipy.special import softmax



SEEDS\_XLM = list(range(200, 216))

SEEDS\_PHO = list(range(300, 316))



def load(path):

&#x20;   d = np.load(path)

&#x20;   return dict(

&#x20;       logits=d\["eval\_logits"],

&#x20;       gold=d\["eval\_gold"],

&#x20;       cal\_logits=d\["cal\_logits"] if "cal\_logits" in d.files else None,

&#x20;       cal\_gold=d\["cal\_gold"] if "cal\_gold" in d.files else None,

&#x20;   )



def sha(arr):

&#x20;   return hashlib.sha256(arr.tobytes()).hexdigest()\[:16]



def compare(p1, p2, label):

&#x20;   if not os.path.exists(p1) or not os.path.exists(p2):

&#x20;       print(f"  MISSING: {label}  ({os.path.basename(p1) if os.path.exists(p1) else '-'} | {os.path.basename(p2) if os.path.exists(p2) else '-'})")

&#x20;       return None

&#x20;   a, b = load(p1), load(p2)

&#x20;   gold\_match = np.array\_equal(a\["gold"], b\["gold"])

&#x20;   max\_diff = float(np.max(np.abs(a\["logits"].astype(np.float64) - b\["logits"].astype(np.float64))))

&#x20;   n\_diff = int(np.sum(np.any(a\["logits"] != b\["logits"], axis=1)))

&#x20;   p\_a = softmax(a\["logits"].astype(np.float64), axis=1).max(1)

&#x20;   p\_b = softmax(b\["logits"].astype(np.float64), axis=1).max(1)

&#x20;   corr = float(np.corrcoef(p\_a, p\_b)\[0, 1])

&#x20;   print(f"  {label}: gold\_match={gold\_match}  max|Δlogit|={max\_diff:.8f}  "

&#x20;         f"n\_diff={n\_diff}/{len(a\['logits'])}  corr(max\_prob)={corr:.8f}  "

&#x20;         f"sha(gold)={sha(a\['gold'])}/{sha(b\['gold'])}")

&#x20;   return dict(max\_diff=max\_diff, n\_diff=n\_diff, gold\_match=gold\_match)



def main():

&#x20;   print("="\*70)

&#x20;   print("XLM-R: R1 vs T2 (phai bit-identical neu pipeline tat dinh)")

&#x20;   print("="\*70)

&#x20;   for s in SEEDS\_XLM:

&#x20;       p1 = glob.glob(f"results/ext/xlmr/logits/R1\_\_s{s}\_\_global\_\_L-1\_\_r1\_\_e6\_\_\*.npz")

&#x20;       p2 = glob.glob(f"results/ext/xlmr/logits/T2\_XLM\_\_s{s}\_\_global\_\_L-1\_\_r1\_\_e6\_\_\*.npz")

&#x20;       if p1 and p2:

&#x20;           compare(p1\[0], p2\[0], f"XLM-R s{s} r1")



&#x20;   print()

&#x20;   print("="\*70)

&#x20;   print("PhoBERT: B1X vs T2 (nghi ngo session effect)")

&#x20;   print("="\*70)

&#x20;   for s in SEEDS\_PHO:

&#x20;       # B1X co the co nhieu ten khac nhau; glob rong

&#x20;       cands = (glob.glob(f"results/ext/phobert/logits/B1X\*s{s}\*r1\*.npz") or

&#x20;                glob.glob(f"results/ext/phobert/logits/\*s{s}\*r1\*e6\*.npz"))

&#x20;       # loai T2 ra khoi cands

&#x20;       cands = \[c for c in cands if "T2\_" not in c]

&#x20;       p2 = glob.glob(f"results/ext/phobert/logits/T2\_PHO\_\_s{s}\_\_global\_\_L-1\_\_r1\_\_e6\_\_\*.npz")

&#x20;       if cands and p2:

&#x20;           compare(cands\[0], p2\[0], f"PhoBERT s{s} r1")



if \_\_name\_\_ == "\_\_main\_\_":

&#x20;   main()

