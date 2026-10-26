"""analyze_tost_b1.py - TOST + ước lượng hiệu ứng rank cho PhoBERT (B1).

README đã nhắc tới script này nhưng bản nộp trước không có file; đây là bản mỏng gọi `ext_analyze_phobert.py`.
Mặc định phân tích B1 cũ (4 rank, 8 seed). Sau EXT-1, truyền CSV mới:
  python code/analyze_tost_b1.py                                     # B1 cũ (kết quả đã công bố trong README)
  python code/analyze_tost_b1.py results/ext/phobert/B1X_merged.csv results/ext/phobert/phobert_lr_map_v2.json
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ext_analyze_phobert as M  # noqa: E402

if __name__ == "__main__":
    csv = sys.argv[1] if len(sys.argv) > 1 else "results/v4_phobert/B1_final_merged.csv"
    lr = sys.argv[2] if len(sys.argv) > 2 else "results/v4_phobert/phobert_lr_map.json"
    legacy = csv.endswith("B1_final_merged.csv")
    sys.argv = [sys.argv[0], csv, "--lr_map", lr] + (["--ranks", "1", "4", "16", "32", "--first_seeds", "4"] if legacy else [])
    M.main()
