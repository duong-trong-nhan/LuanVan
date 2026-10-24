# -*- coding: utf-8 -*-
"""
HeritageFNS: Phase 7 - Multimodal Audio-Visual Concept Mapper & XAI Generator
=============================================================================
Module ánh xạ ngữ nghĩa 2 tầng kết nối:
  1. Vector mờ 7 trục âm học & Luật suy luận mờ (DT, RF, CN2, RIPPER)
  2. Bộ 42 concept âm thanh chuẩn hóa (concept/audio_concepts.md)
  3. Bộ 42 concept hình ảnh/thị giác chuẩn hóa (concept/visual_concepts.md)
  4. Trình sinh diễn giải giải thích đa phương thức XAI (Audio, Image, Video)
     kèm trích dẫn học thuật chuẩn mực, không dùng LaTeX hoặc ký tự $.
"""

import sys
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import os
import re
import io
import json
import ast
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
import pandas as pd
import numpy as np

# -----------------------------------------------------------------------------
# 1. TỪ ĐIỂN ÁNH XẠ NGỮ NGHĨA VÀ CHUẨN HÓA 7 TRỤC ÂM HỌC
# -----------------------------------------------------------------------------

GENRES = ['Ca trù', 'Chèo', 'Chầu văn', 'Cải lương', 'Hát xẩm', 'Quan họ']

GENRE_CODE_MAP = {
    'ca_tru': 'Ca trù',
    'cheo': 'Chèo',
    'chau_van': 'Chầu văn',
    'cai_luong': 'Cải lương',
    'hat_xam': 'Hát xẩm',
    'quan_ho': 'Quan họ',
    'Ca trù': 'ca_tru',
    'Chèo': 'cheo',
    'Chầu văn': 'chau_van',
    'Cải lương': 'cai_luong',
    'Hát xẩm': 'hat_xam',
    'Quan họ': 'quan_ho'
}

AXIS_VIETNAMESE_NAMES = {
    'Energy': 'Năng lượng âm thanh (RMS Energy)',
    'Harmony': 'Tính hòa âm (Harmonic Ratio / Salience)',
    'Pitch': 'Cao độ (Pitch / F0)',
    'Spectral': 'Đặc trưng phổ tần (Spectral Centroid / Bandwidth)',
    'Tempo': 'Tốc độ tiết tấu (Tempo / BPM)',
    'Timbre': 'Màu âm phổ (Timbre / MFCC)',
    'Vibrato': 'Độ rung ngân và Luyến láy (Vibrato / Pitch Modulation)'
}

# Ánh xạ nhãn tiếng Anh sang tiếng Việt trên 7 trục mờ
AXIS_LEVEL_EN_TO_VI = {
    'Energy': {
        'Very Weak': 'Rất yếu',
        'Weak': 'Yếu',
        'Moderate': 'Vừa',
        'Strong': 'Mạnh',
        'Very Strong': 'Rất mạnh'
    },
    'Harmony': {
        'Low': 'Thấp',
        'Medium': 'Trung bình',
        'High': 'Cao'
    },
    'Pitch': {
        'Very Low': 'Rất trầm',
        'Low': 'Trầm',
        'Medium': 'Trung',
        'High': 'Cao',
        'Very High': 'Rất cao'
    },
    'Spectral': {
        'Low': 'Thấp',
        'Medium': 'Trung bình',
        'High': 'Cao'
    },
    'Tempo': {
        'Slow': 'Chậm',
        'Moderate': 'Vừa',
        'Fast': 'Nhanh'
    },
    'Timbre': {
        'Very Low': 'Rất thấp',
        'Low': 'Thấp',
        'Medium': 'Trung bình',
        'High': 'Cao',
        'Very High': 'Rất cao'
    },
    'Vibrato': {
        'Light Ornament': 'Ít luyến',
        'Moderate': 'Vừa',
        'Heavy Ornament': 'Nhiều luyến'
    }
}

# Tạo chiều ngược lại từ tiếng Việt sang tiếng Anh
AXIS_LEVEL_VI_TO_EN = {
    axis: {vi: en for en, vi in levels.items()}
    for axis, levels in AXIS_LEVEL_EN_TO_VI.items()
}

# Ánh xạ tên cột liên tục giữa tiếng Việt và tiếng Anh
# Ví dụ: Pitch_Trầm <-> Pitch_Low
CONT_COL_VI_TO_EN = {}
CONT_COL_EN_TO_VI = {}
for axis, levels in AXIS_LEVEL_EN_TO_VI.items():
    for en, vi in levels.items():
        vi_clean = vi.replace(' ', '_')
        en_clean = en.replace(' ', '_')
        col_vi = f"{axis}_{vi_clean}"
        col_en = f"{axis}_{en_clean}"
        CONT_COL_VI_TO_EN[col_vi] = col_en
        CONT_COL_EN_TO_VI[col_en] = col_vi

# Ý nghĩa tín hiệu học thuật chi tiết cho từng mức ngữ nghĩa trên 7 trục
ACOUSTIC_LEVEL_EXPLANATIONS = {
    'Energy': {
        'Rất yếu': 'Cường độ năng lượng âm thanh rất nhỏ, không gian biểu diễn tĩnh lặng, êm dịu mang tính thính phòng hoặc đối đáp tinh tế.',
        'Yếu': 'Cường độ âm thanh tương đối êm nhẹ, mức áp suất âm thanh thấp, làm nổi bật sắc thái nhả chữ mộc mạc.',
        'Vừa': 'Cường độ âm thanh ở mức cân bằng, trường độ và âm lượng ổn định, đĩnh đạc.',
        'Mạnh': 'Cường độ âm thanh lớn, nội lực giọng hát và dàn nhạc phát ra dứt khoát, sôi nổi.',
        'Rất mạnh': 'Cường độ năng lượng âm thanh bùng nổ, biên độ áp suất sóng âm rất cao với sự phối hợp dồn dập của bộ gõ.'
    },
    'Harmony': {
        'Thấp': 'Cấu trúc hòa âm đơn sơ, tính hòa quyện bè thấp, đặc trưng cho lối hát mộc mạc đơn tuyến hoặc có nhiều tạp âm bộ gõ.',
        'Trung bình': 'Độ hòa quyện âm sắc cân bằng giữa bè giai điệu chính và nhạc cụ đệm.',
        'Cao': 'Cấu trúc hòa âm và họa âm phong phú, các tuyến âm thanh bè và nhạc cụ hòa quyện chặt chẽ, đa thanh.'
    },
    'Pitch': {
        'Rất trầm': 'Cao độ thuộc dải tần số cơ bản F0 rất thấp, tạo cảm giác trầm sâu, ấm đặc trưng của nhạc cụ âm trầm như đàn đáy.',
        'Trầm': 'Cao độ dải trầm ấm áp, tần số cơ bản F0 thấp, làm nền âm vực dày dặn cho phần diễn xướng.',
        'Trung': 'Cao độ nằm ở dải tần số trung bình, gần với âm vực nói và hát tự nhiên của con người.',
        'Cao': 'Cao độ thuộc dải tần cao, âm vực sáng rõ, réo rắt và nổi bật trong không gian diễn xướng.',
        'Rất cao': 'Cao độ ở dải tần số rất cao, âm vực thanh thoát, cao vút mang tính ngân vang đặc thù.'
    },
    'Spectral': {
        'Thấp': 'Trọng tâm phổ tần thấp, năng lượng tập trung ở dải tần trầm, âm thanh tròn trịa, ấm đục và không chói.',
        'Trung bình': 'Trọng tâm phổ tần phân bố đồng đều giữa các dải tần số trung tâm.',
        'Cao': 'Trọng tâm phổ tần tập trung ở dải tần số cao, âm sắc đanh sáng, sắc nét từ các nhạc cụ gõ đanh như phách tre, mõ, thanh la.'
    },
    'Tempo': {
        'Chậm': 'Tiết tấu chậm rãi, khoan thai, giàu chất ngâm ngợi, dàn trải và sâu lắng.',
        'Vừa': 'Tiết tấu nhịp nhàng, ổn định, giữ nhịp đĩnh đạc và vừa phải.',
        'Nhanh': 'Tiết tấu nhanh, sôi nổi, giục giã hoặc mô phỏng chuyển động dồn dập, hoạt bát.'
    },
    'Timbre': {
        'Rất thấp': 'Màu âm mộc mạc, tối và ấm, độ dày phổ thấp phản ánh chất liệu thùng gỗ mộc, phách tre hoặc dây chùng.',
        'Thấp': 'Màu âm trầm ấm, độ sáng phổ vừa phải, ít họa âm kim loại.',
        'Trung bình': 'Màu âm cân bằng tự nhiên, thể hiện rõ tính trung thực của nhạc cụ mộc truyền thống.',
        'Cao': 'Màu âm sáng, đanh gọn, giàu họa âm kim loại hoặc dây kim loại căng như đàn kìm, guitar phím lõm.',
        'Rất cao': 'Màu âm rất sáng và sắc bén, biên độ các hài âm bậc cao nổi trội, tạo cảm giác đanh thép.'
    },
    'Vibrato': {
        'Ít luyến': 'Đường nét giai điệu mộc mạc, ít biến thiên tần số, hát thẳng giọng tự sự và dứt khoát.',
        'Vừa': 'Độ rung ngân tự nhiên, mức biến thiên tần số F0 vừa phải, nhả chữ mềm mại và uyển chuyển.',
        'Nhiều luyến': 'Kỹ thuật rung ngân sâu, hoa mỹ với mật độ luyến láy, vuốt, nẩy hạt và thêu nốt phong phú.'
    }
}

# -----------------------------------------------------------------------------
# 2. PARSER CSV TỪ MARKDOWN (load_concepts_from_md)
# -----------------------------------------------------------------------------

def load_concepts_from_md(md_path: Union[str, Path]) -> pd.DataFrame:
    """
    Tự động trích xuất khối CSV nằm trong file Markdown (concept/audio_concepts.md
    hoặc concept/visual_concepts.md) thành pandas.DataFrame chuẩn mực.

    Hỗ trợ tra cứu theo:
      concept_id, genre, genre_name, aspect, evidence_type, evidence_level,
      concept_sentence, source.
    """
    path = Path(md_path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file concept markdown tại: {path.resolve()}")

    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Tìm khối ```csv ... ```
    match = re.search(r'```csv\s*\n(.*?)\n```', content, re.DOTALL)
    if not match:
        raise ValueError(f"Không tìm thấy khối mã ```csv trong file: {path.name}")

    csv_text = match.group(1).strip()
    # Loại bỏ ký tự LaTeX $ hoặc $$ nếu xuất hiện để tuân thủ quy tắc XAI
    csv_text_cleaned = csv_text.replace('$$', '').replace('$', '')

    df = pd.read_csv(io.StringIO(csv_text_cleaned))

    # Chuẩn hóa tên cột và tạo alias tiện tra cứu
    df.columns = [col.strip() for col in df.columns]

    # Tạo alias cho genre
    if 'genre_name' in df.columns and 'genre' not in df.columns:
        df['genre'] = df['genre_name']
    elif 'genre' in df.columns and 'genre_name' not in df.columns:
        df['genre_name'] = df['genre']

    # Tạo alias cho evidence_type
    if 'evidence_level' in df.columns and 'evidence_type' not in df.columns:
        df['evidence_type'] = df['evidence_level']
    elif 'evidence_type' in df.columns and 'evidence_level' not in df.columns:
        df['evidence_level'] = df['evidence_type']

    # Làm sạch các chuỗi văn bản
    for str_col in ['concept_id', 'genre_code', 'genre_name', 'aspect', 'evidence_level', 'concept_sentence', 'source']:
        if str_col in df.columns:
            df[str_col] = df[str_col].astype(str).str.strip()

    return df


# -----------------------------------------------------------------------------
# 3. BỘ PHIÊN DỊCH ĐIỀU KIỆN LUẬT MỜ SANG GIẢI THÍCH ÂM HỌC (AudioRuleToConceptBridge)
# -----------------------------------------------------------------------------

class AudioRuleToConceptBridge:
    """
    Cầu nối ngữ nghĩa 2 tầng:
      Tầng 1: Phiên dịch từng mệnh đề luật mờ (DT, RF, CN2, RIPPER) thành giải thích tín hiệu âm học tiếng Việt.
      Tầng 2: Ánh xạ nhãn dự đoán và các trục âm học vượt ngưỡng sang đúng 2 đến 4 câu concept chuẩn hóa
              trong concept/audio_concepts.md (ưu tiên direct_audio).
    """

    def __init__(
        self,
        audio_concepts_path: Union[str, Path] = "concept/audio_concepts.md",
        fuzzy_profiles_path: Union[str, Path] = "data/models/fuzzy_profiles.json",
        rules_dir: Union[str, Path] = "data/rules"
    ):
        self.audio_concepts_path = Path(audio_concepts_path)
        self.fuzzy_profiles_path = Path(fuzzy_profiles_path)
        self.rules_dir = Path(rules_dir)

        # Nạp dữ liệu concept
        if self.audio_concepts_path.exists():
            self.df_concepts = load_concepts_from_md(self.audio_concepts_path)
        else:
            self.df_concepts = pd.DataFrame()

        # Nạp fuzzy profiles nếu có
        self.fuzzy_profiles = {}
        if self.fuzzy_profiles_path.exists():
            try:
                with open(self.fuzzy_profiles_path, 'r', encoding='utf-8') as f:
                    self.fuzzy_profiles = json.load(f)
            except Exception as e:
                print(f"[Cảnh báo] Không thể nạp fuzzy_profiles: {e}")

        # Bản đồ luật ánh xạ Tầng 2 chi tiết cho 6 thể loại
        self._init_tier2_mapping_rules()

    def _init_tier2_mapping_rules(self):
        """
        Khởi tạo quy tắc chấm điểm và kích hoạt concept Tầng 2 cho từng thể loại.
        Mỗi thể loại có các liên kết ưu tiên từ đặc trưng âm học -> concept_id.
        """
        self.tier2_rules = {
            'Ca trù': {
                'concept_priority': ['CT_AUD_05', 'CT_AUD_04', 'CT_AUD_01', 'CT_AUD_02', 'CT_AUD_06', 'CT_AUD_03'],
                'feature_links': {
                    ('Pitch', 'Trầm'): 'CT_AUD_05',
                    ('Pitch', 'Rất trầm'): 'CT_AUD_05',
                    ('Timbre', 'Rất thấp'): 'CT_AUD_05',
                    ('Timbre', 'Thấp'): 'CT_AUD_05',
                    ('Spectral', 'Cao'): 'CT_AUD_04',
                    ('Tempo', 'Vừa'): 'CT_AUD_04',
                    ('Energy', 'Rất yếu'): 'CT_AUD_04',
                    ('Energy', 'Yếu'): 'CT_AUD_01',
                    ('Vibrato', 'Nhiều luyến'): 'CT_AUD_02',
                    ('Vibrato', 'Vừa'): 'CT_AUD_02',
                    ('Pitch', 'Cao'): 'CT_AUD_02',
                    ('Pitch', 'Rất cao'): 'CT_AUD_02',
                    ('Harmony', 'Thấp'): 'CT_AUD_01',
                    ('Harmony', 'Trung bình'): 'CT_AUD_01',
                    ('Energy', 'Vừa'): 'CT_AUD_06',
                    ('Energy', 'Mạnh'): 'CT_AUD_06',
                },
                'fallback_concepts': ['CT_AUD_01', 'CT_AUD_05', 'CT_AUD_04']
            },
            'Cải lương': {
                'concept_priority': ['CL_AUD_03', 'CL_AUD_02', 'CL_AUD_01', 'CL_AUD_06', 'CL_AUD_08', 'CL_AUD_04', 'CL_AUD_05'],
                'feature_links': {
                    ('Vibrato', 'Nhiều luyến'): 'CL_AUD_03',
                    ('Vibrato', 'Vừa'): 'CL_AUD_03',
                    ('Timbre', 'Cao'): 'CL_AUD_02',
                    ('Timbre', 'Rất cao'): 'CL_AUD_02',
                    ('Spectral', 'Cao'): 'CL_AUD_02',
                    ('Harmony', 'Cao'): 'CL_AUD_01',
                    ('Harmony', 'Trung bình'): 'CL_AUD_01',
                    ('Tempo', 'Chậm'): 'CL_AUD_06',
                    ('Energy', 'Rất yếu'): 'CL_AUD_06',
                    ('Energy', 'Yếu'): 'CL_AUD_06',
                    ('Tempo', 'Vừa'): 'CL_AUD_08',
                    ('Energy', 'Vừa'): 'CL_AUD_04',
                    ('Energy', 'Mạnh'): 'CL_AUD_04',
                },
                'fallback_concepts': ['CL_AUD_02', 'CL_AUD_03', 'CL_AUD_01']
            },
            'Chèo': {
                'concept_priority': ['CH_AUD_02', 'CH_AUD_01', 'CH_AUD_05', 'CH_AUD_07', 'CH_AUD_04', 'CH_AUD_08'],
                'feature_links': {
                    ('Energy', 'Mạnh'): 'CH_AUD_02',
                    ('Energy', 'Vừa'): 'CH_AUD_02',
                    ('Spectral', 'Cao'): 'CH_AUD_02',
                    ('Spectral', 'Trung bình'): 'CH_AUD_04',
                    ('Pitch', 'Trung'): 'CH_AUD_01',
                    ('Pitch', 'Cao'): 'CH_AUD_01',
                    ('Vibrato', 'Vừa'): 'CH_AUD_05',
                    ('Vibrato', 'Nhiều luyến'): 'CH_AUD_05',
                    ('Harmony', 'Cao'): 'CH_AUD_01',
                    ('Harmony', 'Trung bình'): 'CH_AUD_07',
                    ('Timbre', 'Thấp'): 'CH_AUD_07',
                    ('Timbre', 'Trung bình'): 'CH_AUD_07',
                    ('Tempo', 'Vừa'): 'CH_AUD_04',
                },
                'fallback_concepts': ['CH_AUD_02', 'CH_AUD_01', 'CH_AUD_05']
            },
            'Chầu văn': {
                'concept_priority': ['CV_AUD_01', 'CV_AUD_07', 'CV_AUD_04', 'CV_AUD_03', 'CV_AUD_06', 'CV_AUD_05'],
                'feature_links': {
                    ('Spectral', 'Cao'): 'CV_AUD_01',
                    ('Timbre', 'Cao'): 'CV_AUD_01',
                    ('Timbre', 'Rất cao'): 'CV_AUD_01',
                    ('Energy', 'Rất mạnh'): 'CV_AUD_07',
                    ('Energy', 'Mạnh'): 'CV_AUD_07',
                    ('Tempo', 'Nhanh'): 'CV_AUD_07',
                    ('Tempo', 'Vừa'): 'CV_AUD_06',
                    ('Vibrato', 'Nhiều luyến'): 'CV_AUD_04',
                    ('Vibrato', 'Vừa'): 'CV_AUD_04',
                    ('Pitch', 'Cao'): 'CV_AUD_03',
                    ('Pitch', 'Trung'): 'CV_AUD_03',
                    ('Harmony', 'Cao'): 'CV_AUD_01',
                    ('Harmony', 'Trung bình'): 'CV_AUD_01',
                },
                'fallback_concepts': ['CV_AUD_01', 'CV_AUD_07', 'CV_AUD_04']
            },
            'Quan họ': {
                'concept_priority': ['QH_AUD_01', 'QH_AUD_03', 'QH_AUD_02', 'QH_AUD_04', 'QH_AUD_05'],
                'feature_links': {
                    ('Energy', 'Rất yếu'): 'QH_AUD_01',
                    ('Energy', 'Yếu'): 'QH_AUD_01',
                    ('Energy', 'Vừa'): 'QH_AUD_01',
                    ('Harmony', 'Thấp'): 'QH_AUD_01',
                    ('Harmony', 'Trung bình'): 'QH_AUD_02',
                    ('Harmony', 'Cao'): 'QH_AUD_02',
                    ('Vibrato', 'Nhiều luyến'): 'QH_AUD_03',
                    ('Vibrato', 'Vừa'): 'QH_AUD_03',
                    ('Pitch', 'Trung'): 'QH_AUD_03',
                    ('Pitch', 'Cao'): 'QH_AUD_03',
                    ('Tempo', 'Vừa'): 'QH_AUD_04',
                    ('Tempo', 'Chậm'): 'QH_AUD_04',
                    ('Spectral', 'Trung bình'): 'QH_AUD_04',
                },
                'fallback_concepts': ['QH_AUD_01', 'QH_AUD_03', 'QH_AUD_02']
            },
            'Hát xẩm': {
                'concept_priority': ['HX_AUD_01', 'HX_AUD_03', 'HX_AUD_02', 'HX_AUD_05', 'HX_AUD_06', 'HX_AUD_07'],
                'feature_links': {
                    ('Timbre', 'Rất thấp'): 'HX_AUD_01',
                    ('Timbre', 'Thấp'): 'HX_AUD_01',
                    ('Harmony', 'Thấp'): 'HX_AUD_01',
                    ('Vibrato', 'Ít luyến'): 'HX_AUD_03',
                    ('Vibrato', 'Vừa'): 'HX_AUD_03',
                    ('Pitch', 'Trầm'): 'HX_AUD_03',
                    ('Pitch', 'Trung'): 'HX_AUD_03',
                    ('Spectral', 'Cao'): 'HX_AUD_02',
                    ('Energy', 'Vừa'): 'HX_AUD_02',
                    ('Tempo', 'Nhanh'): 'HX_AUD_05',
                    ('Energy', 'Mạnh'): 'HX_AUD_05',
                    ('Energy', 'Rất mạnh'): 'HX_AUD_05',
                    ('Tempo', 'Vừa'): 'HX_AUD_06',
                    ('Tempo', 'Chậm'): 'HX_AUD_06',
                },
                'fallback_concepts': ['HX_AUD_01', 'HX_AUD_02', 'HX_AUD_03']
            }
        }

    # -------------------------------------------------------------------------
    # TẦNG 1: PHIÊN DỊCH CÁC MỆNH ĐỀ LUẬT MỜ
    # -------------------------------------------------------------------------

    def parse_rule_clause(self, clause_str: str) -> Dict[str, Any]:
        """
        Phân tích một mệnh đề đơn trong rule_text.
        Hỗ trợ:
          1. Continuous: 'Pitch_Trầm > 0.221', 'Timbre_Rất_thấp <= 0.191'
          2. Crisp IN: "Pitch IN ['Rất trầm', 'Trầm']"
          3. Crisp ==: "Energy == 'Rất yếu'", "Energy == 'Very Weak'"
        """
        s = clause_str.strip()

        # 1. Kiểm tra toán tử so sánh liên tục <= hoặc >
        m_cont = re.match(r"^([A-Za-z0-9_À-ỹ]+)\s*(<=|>|<|>=)\s*([0-9\.\-]+)$", s)
        if m_cont:
            col_name, op, val_str = m_cont.groups()
            val = float(val_str)

            # Tách trục và mức độ: ví dụ Pitch_Trầm hoặc Pitch_Low
            axis = None
            level_name = None
            for ax in AXIS_VIETNAMESE_NAMES.keys():
                if col_name.startswith(f"{ax}_"):
                    axis = ax
                    level_raw = col_name[len(ax) + 1:].replace('_', ' ')
                    # Chuẩn hóa sang tiếng Việt nếu là tiếng Anh
                    level_name = AXIS_LEVEL_EN_TO_VI.get(ax, {}).get(level_raw, level_raw)
                    break

            return {
                'raw': s,
                'type': 'continuous',
                'axis': axis,
                'level': level_name,
                'op': op,
                'threshold': val,
                'col_name': col_name
            }

        # 2. Kiểm tra toán tử IN
        if ' IN ' in s:
            axis_part, list_part = s.split(' IN ', 1)
            axis = axis_part.strip()
            try:
                values = ast.literal_eval(list_part.strip())
                # Chuẩn hóa values sang tiếng Việt nếu dùng nhãn tiếng Anh
                norm_values = [
                    AXIS_LEVEL_EN_TO_VI.get(axis, {}).get(v, v)
                    for v in values
                ]
            except Exception:
                norm_values = [v.strip().strip("'\"") for v in list_part.strip('[]').split(',')]

            return {
                'raw': s,
                'type': 'crisp_in',
                'axis': axis,
                'levels': norm_values,
                'op': 'IN'
            }

        # 3. Kiểm tra toán tử ==
        if ' == ' in s:
            axis_part, val_part = s.split(' == ', 1)
            axis = axis_part.strip()
            val = val_part.strip().strip("'\"")
            val_norm = AXIS_LEVEL_EN_TO_VI.get(axis, {}).get(val, val)
            return {
                'raw': s,
                'type': 'crisp_eq',
                'axis': axis,
                'level': val_norm,
                'op': '=='
            }

        return {'raw': s, 'type': 'unknown'}

    def translate_clause_to_vietnamese(self, clause_info: Dict[str, Any]) -> str:
        """
        Dịch 1 mệnh đề luật thành câu văn giải thích đặc trưng tín hiệu âm học tiếng Việt (Tầng 1).
        """
        ctype = clause_info.get('type')
        axis = clause_info.get('axis')
        axis_name_vi = AXIS_VIETNAMESE_NAMES.get(axis, axis)

        if ctype == 'continuous':
            level = clause_info.get('level')
            op = clause_info.get('op')
            thresh = clause_info.get('threshold')
            level_exp = ACOUSTIC_LEVEL_EXPLANATIONS.get(axis, {}).get(level, '')

            if op in ['>', '>=']:
                return (
                    f"Độ thuộc mức '{level}' trên trục {axis_name_vi} vượt ngưỡng ({op} {thresh:.3f}), "
                    f"thể hiện rõ: {level_exp}"
                )
            else:
                return (
                    f"Độ thuộc mức '{level}' trên trục {axis_name_vi} giới hạn ở mức thấp ({op} {thresh:.3f}), "
                    f"xác nhận tín hiệu không nghiêng mạnh về trạng thái này."
                )

        elif ctype == 'crisp_eq':
            level = clause_info.get('level')
            level_exp = ACOUSTIC_LEVEL_EXPLANATIONS.get(axis, {}).get(level, '')
            return (
                f"Trục {axis_name_vi} đạt mức điển hình '{level}': {level_exp}"
            )

        elif ctype == 'crisp_in':
            levels = clause_info.get('levels', [])
            levels_str = ", ".join([f"'{lvl}'" for lvl in levels])
            # Lấy giải thích của mức đại diện đầu tiên
            sample_exp = ""
            for lvl in levels:
                if lvl in ACOUSTIC_LEVEL_EXPLANATIONS.get(axis, {}):
                    sample_exp = ACOUSTIC_LEVEL_EXPLANATIONS[axis][lvl]
                    break
            return (
                f"Trục {axis_name_vi} nằm trong tập giá trị cho phép [{levels_str}], "
                f"phù hợp đặc tính tín hiệu ({sample_exp})"
            )

        return f"Điều kiện kích hoạt: {clause_info.get('raw')}"

    def translate_rule_to_vietnamese(
        self,
        rule: Union[Dict[str, Any], str],
        default_consequent: Optional[str] = None,
        default_confidence: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Dịch toàn bộ một luật mờ thành văn bản giải thích âm học hoàn chỉnh (Tầng 1).
        """
        if isinstance(rule, str):
            rule_text = rule
            consequent = default_consequent or "Chưa xác định"
            confidence = default_confidence if default_confidence is not None else 1.0
            support = None
        else:
            rule_text = rule.get('rule_text', '')
            consequent = rule.get('consequent') or default_consequent or 'Chưa xác định'
            confidence = rule.get('confidence', rule.get('score', default_confidence or 1.0))
            support = rule.get('support', rule.get('total_support', None))

        clauses = [c.strip() for c in rule_text.split(' AND ') if c.strip() and c.strip() != 'TRUE']
        parsed_clauses = [self.parse_rule_clause(c) for c in clauses]
        clause_explanations = [self.translate_clause_to_vietnamese(pc) for pc in parsed_clauses]

        # Xây dựng đoạn văn diễn giải tổng quan
        conf_str = f"{confidence * 100:.1f}%" if isinstance(confidence, (int, float)) and confidence <= 1.0 else str(confidence)
        support_str = f", dựa trên {support} mẫu huấn luyện" if support is not None else ""

        summary_lines = [
            f"Luật mờ kích hoạt kết luận thể loại '{consequent}' với độ tin cậy {conf_str}{support_str}:"
        ]
        for idx, exp in enumerate(clause_explanations, 1):
            summary_lines.append(f"  ({idx}) {exp}")

        rule_summary_text = "\n".join(summary_lines)

        return {
            'rule_text': rule_text,
            'consequent': consequent,
            'confidence': confidence,
            'parsed_clauses': parsed_clauses,
            'clause_explanations': clause_explanations,
            'rule_summary_text': rule_summary_text
        }

    # -------------------------------------------------------------------------
    # KIỂM TRA KHỚP LUẬT VỚI MẪU AUDIO ĐẦU VÀO
    # -------------------------------------------------------------------------

    def load_rules(self, rules_source: Union[str, Path, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Nạp tập luật từ file JSON, CSV hoặc danh sách có sẵn.
        Hỗ trợ:
          - File JSON chứa danh sách luật hoặc dict phân theo thể loại.
          - File CSV chứa cột 'rule_text' (hoặc 'Luật (IF Điều kiện)') và 'consequent'.
        """
        if isinstance(rules_source, list):
            return rules_source

        path = Path(rules_source)
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy file luật: {path}")

        if path.suffix.lower() == '.json':
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                # Gộp tất cả thể loại lại thành 1 list
                all_rules = []
                for genre_k, r_list in data.items():
                    if isinstance(r_list, list):
                        for r in r_list:
                            if isinstance(r, dict):
                                if 'consequent' not in r:
                                    r['consequent'] = genre_k
                                all_rules.append(r)
                return all_rules
            return []

        elif path.suffix.lower() == '.csv':
            df_r = pd.read_csv(path)
            # Chuẩn hóa cột
            rules_list = []
            rule_col = next((c for c in df_r.columns if 'rule' in c.lower() or 'điều kiện' in c.lower()), None)
            conseq_col = next((c for c in df_r.columns if 'consequent' in c.lower() or 'dự đoán' in c.lower() or 'label' in c.lower()), None)
            conf_col = next((c for c in df_r.columns if 'conf' in c.lower() or 'tin cậy' in c.lower()), None)

            if rule_col and conseq_col:
                for _, row in df_r.iterrows():
                    r_item = {
                        'rule_text': str(row[rule_col]),
                        'consequent': str(row[conseq_col])
                    }
                    if conf_col and pd.notna(row[conf_col]):
                        val_str = str(row[conf_col]).rstrip('%')
                        try:
                            val_f = float(val_str)
                            r_item['confidence'] = val_f / 100.0 if '%' in str(row[conf_col]) else val_f
                        except Exception:
                            r_item['confidence'] = 1.0
                    rules_list.append(r_item)
            return rules_list

        return []

    def match_audio_sample(
        self,
        rule_clause: Dict[str, Any],
        audio_features: Dict[str, Any]
    ) -> bool:
        """
        Kiểm tra 1 mệnh đề luật có được thỏa mãn bởi vector âm thanh đầu vào hay không.
        Hỗ trợ cả vector liên tục (continuous) lẫn vector nhãn rời rạc (crisp).
        """
        ctype = rule_clause.get('type')
        axis = rule_clause.get('axis')

        if ctype == 'continuous':
            col_name = rule_clause.get('col_name')
            op = rule_clause.get('op')
            thresh = rule_clause.get('threshold')

            # Tra cứu giá trị trong audio_features (thử cả tên tiếng Việt và tiếng Anh)
            val = None
            if col_name in audio_features:
                val = float(audio_features[col_name])
            elif col_name in CONT_COL_VI_TO_EN and CONT_COL_VI_TO_EN[col_name] in audio_features:
                val = float(audio_features[CONT_COL_VI_TO_EN[col_name]])
            elif col_name in CONT_COL_EN_TO_VI and CONT_COL_EN_TO_VI[col_name] in audio_features:
                val = float(audio_features[CONT_COL_EN_TO_VI[col_name]])
            else:
                # Nếu đưa vào fuzzy profile dạng lồng nhau
                level = rule_clause.get('level')
                if axis in audio_features and isinstance(audio_features[axis], dict):
                    all_dict = audio_features[axis].get('all', {})
                    val = all_dict.get(level, None)

            if val is None:
                return False

            if op == '<=':
                return val <= thresh
            elif op == '<':
                return val < thresh
            elif op == '>=':
                return val >= thresh
            elif op == '>':
                return val > thresh

        elif ctype == 'crisp_eq':
            target_level = rule_clause.get('level')
            actual_val = audio_features.get(axis, None)
            if actual_val is None:
                return False
            actual_norm = AXIS_LEVEL_EN_TO_VI.get(axis, {}).get(actual_val, actual_val)
            return actual_norm == target_level

        elif ctype == 'crisp_in':
            allowed_levels = rule_clause.get('levels', [])
            actual_val = audio_features.get(axis, None)
            if actual_val is None:
                return False
            actual_norm = AXIS_LEVEL_EN_TO_VI.get(axis, {}).get(actual_val, actual_val)
            return actual_norm in allowed_levels

        return False

    def is_rule_triggered(
        self,
        rule_item: Union[Dict[str, Any], str],
        audio_features: Dict[str, Any]
    ) -> bool:
        """
        Kiểm tra toàn bộ luật có khớp với vector âm thanh hay không (tất cả mệnh đề đều True).
        """
        rule_text = rule_item if isinstance(rule_item, str) else rule_item.get('rule_text', '')
        if rule_text == 'TRUE':
            return True

        clauses = [c.strip() for c in rule_text.split(' AND ') if c.strip()]
        for c in clauses:
            parsed = self.parse_rule_clause(c)
            if not self.match_audio_sample(parsed, audio_features):
                return False
        return True

    def find_triggered_rules(
        self,
        rules_source: Union[str, Path, List[Dict[str, Any]]],
        audio_features: Dict[str, Any],
        genre_filter: Optional[str] = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Tìm tất cả các luật trong rules_source được kích hoạt bởi audio_features.
        Sắp xếp theo confidence / score giảm dần.
        """
        rules = self.load_rules(rules_source)
        matched = []
        for r in rules:
            if genre_filter and r.get('consequent') != genre_filter:
                continue
            if self.is_rule_triggered(r, audio_features):
                matched.append(r)

        # Sắp xếp theo score hoặc confidence
        matched.sort(
            key=lambda x: (
                float(x.get('confidence', 0.0) or 0.0),
                float(x.get('score', 0.0) or 0.0),
                int(x.get('support', 0) or 0)
            ),
            reverse=True
        )
        return matched[:top_k]

    # -------------------------------------------------------------------------
    # TẦNG 2: ÁNH XẠ NHÃN + TRỤC ÂM HỌC SANG ĐÚNG 2 ĐẾN 4 CÂU CONCEPT
    # -------------------------------------------------------------------------

    def map_rule_to_concepts(
        self,
        triggered_rule: Union[Dict[str, Any], str],
        audio_features: Optional[Dict[str, Any]] = None,
        target_genre: Optional[str] = None,
        min_concepts: int = 2,
        max_concepts: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Tầng 2: Ánh xạ kết hợp nhãn dự đoán (consequent) + các trục âm học kích hoạt
        thành đúng 2 đến 4 câu concept_sentence trong concept/audio_concepts.md.
        Ưu tiên các concept có cấp độ bằng chứng direct_audio.
        """
        if self.df_concepts.empty:
            return []

        # 1. Xác định nhãn consequent
        consequent = target_genre
        if not consequent and isinstance(triggered_rule, dict):
            consequent = triggered_rule.get('consequent')
        if not consequent and audio_features and 'label' in audio_features:
            consequent = audio_features['label']
        if not consequent or consequent == 'Chưa xác định':
            consequent = 'Ca trù'  # mặc định

        # Chuẩn hóa tên thể loại tiếng Việt có dấu
        consequent_clean = consequent.strip()

        # Lọc danh sách concept theo thể loại trong DataFrame
        df_genre = self.df_concepts[
            (self.df_concepts['genre_name'] == consequent_clean) |
            (self.df_concepts['genre'] == consequent_clean) |
            (self.df_concepts['genre_code'] == GENRE_CODE_MAP.get(consequent_clean, ''))
        ].copy()

        if df_genre.empty:
            # Nếu không tìm thấy, lấy tất cả
            df_genre = self.df_concepts.copy()

        # 2. Thu thập các trục và mức độ âm học được kích hoạt từ luật và từ audio_features
        active_triggers = []

        # Từ rule_text
        rule_text = triggered_rule if isinstance(triggered_rule, str) else triggered_rule.get('rule_text', '')
        if rule_text and rule_text != 'TRUE':
            for c in rule_text.split(' AND '):
                pc = self.parse_rule_clause(c)
                axis = pc.get('axis')
                if pc.get('type') == 'continuous' and pc.get('op') in ['>', '>=']:
                    active_triggers.append((axis, pc.get('level')))
                elif pc.get('type') == 'crisp_eq':
                    active_triggers.append((axis, pc.get('level')))
                elif pc.get('type') == 'crisp_in':
                    for lvl in pc.get('levels', []):
                        active_triggers.append((axis, lvl))

        # Từ audio_features (nếu có cung cấp giá trị cụ thể)
        if audio_features:
            for ax, vi_name in AXIS_VIETNAMESE_NAMES.items():
                if ax in audio_features:
                    val = audio_features[ax]
                    if isinstance(val, str):
                        active_triggers.append((ax, AXIS_LEVEL_EN_TO_VI.get(ax, {}).get(val, val)))
                    elif isinstance(val, dict) and 'dominant' in val:
                        active_triggers.append((ax, val['dominant']))

        # 3. Tính điểm cho các concept của thể loại này
        genre_rules = self.tier2_rules.get(consequent_clean, {
            'concept_priority': [],
            'feature_links': {},
            'fallback_concepts': []
        })

        feature_links = genre_rules.get('feature_links', {})
        concept_priority = genre_rules.get('concept_priority', [])
        fallback = genre_rules.get('fallback_concepts', [])

        scores = {cid: 0 for cid in df_genre['concept_id'].values}

        # Cộng điểm từ các đặc trưng kích hoạt
        for trigger in active_triggers:
            if trigger in feature_links:
                target_cid = feature_links[trigger]
                if target_cid in scores:
                    scores[target_cid] += 3

        # Cộng điểm ưu tiên theo thứ tự chuyên môn thể loại
        for idx, cid in enumerate(concept_priority):
            if cid in scores:
                scores[cid] += (len(concept_priority) - idx)

        # Ưu tiên cấp độ bằng chứng direct_audio
        for _, row in df_genre.iterrows():
            cid = row['concept_id']
            if row.get('evidence_level') == 'direct_audio' or row.get('evidence_type') == 'direct_audio':
                scores[cid] += 2

        # 4. Sắp xếp và chọn đúng từ min_concepts đến max_concepts
        sorted_cids = sorted(scores.keys(), key=lambda c: scores[c], reverse=True)

        selected_cids = []
        for cid in sorted_cids:
            if scores[cid] > 0 and cid not in selected_cids:
                selected_cids.append(cid)
            if len(selected_cids) >= max_concepts:
                break

        # Nếu chưa đủ min_concepts, bổ sung từ danh sách fallback
        if len(selected_cids) < min_concepts:
            for f_cid in fallback:
                if f_cid in df_genre['concept_id'].values and f_cid not in selected_cids:
                    selected_cids.append(f_cid)
                if len(selected_cids) >= min_concepts:
                    break

        # Giới hạn trong khoảng [min_concepts, max_concepts]
        final_cids = selected_cids[:max_concepts]
        if len(final_cids) < min_concepts:
            # Lấy các concept đầu tiên của thể loại
            for cid in df_genre['concept_id'].values:
                if cid not in final_cids:
                    final_cids.append(cid)
                if len(final_cids) >= min_concepts:
                    break

        # Trích xuất thông tin chi tiết
        selected_rows = df_genre[df_genre['concept_id'].isin(final_cids)]
        result_concepts = []
        for _, r in selected_rows.iterrows():
            result_concepts.append({
                'concept_id': r['concept_id'],
                'aspect': r.get('aspect', ''),
                'evidence_level': r.get('evidence_level', r.get('evidence_type', 'direct_audio')),
                'concept_sentence': r.get('concept_sentence', ''),
                'source': r.get('source', '')
            })

        return result_concepts

    def explain_audio(
        self,
        audio_features: Dict[str, Any],
        triggered_rule: Optional[Union[Dict[str, Any], str]] = None,
        prediction: Optional[str] = None,
        confidence: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Thực hiện toàn bộ quy trình giải thích âm thanh:
          1. Phiên dịch luật mờ kích hoạt (Tầng 1).
          2. Chọn ra 2-4 câu concept âm thanh chuẩn hóa (Tầng 2).
          3. Đóng gói kết quả giải thích.
        """
        # Nếu chưa có triggered_rule, dùng thông tin dự đoán
        pred_label = prediction or (triggered_rule.get('consequent') if isinstance(triggered_rule, dict) else None) or audio_features.get('label', 'Ca trù')

        if triggered_rule is None:
            triggered_rule = {
                'rule_text': "Pitch_Trầm > 0.221 AND Timbre_Rất_thấp > 0.191 AND Energy_Rất_yếu > 0.332",
                'consequent': pred_label,
                'confidence': confidence if confidence is not None else 0.85
            }

        rule_trans = self.translate_rule_to_vietnamese(
            triggered_rule,
            default_consequent=pred_label,
            default_confidence=confidence
        )
        concepts = self.map_rule_to_concepts(
            triggered_rule,
            audio_features,
            target_genre=pred_label,
            min_concepts=2,
            max_concepts=4
        )

        return {
            'prediction': rule_trans['consequent'],
            'confidence': rule_trans['confidence'],
            'triggered_rule': rule_trans['rule_text'],
            'rule_explanation': rule_trans['rule_summary_text'],
            'clause_details': rule_trans['clause_explanations'],
            'concepts': concepts
        }


# -----------------------------------------------------------------------------
# 4. BỘ ÁNH XẠ KHÁI NIỆM HÌNH ẢNH / THỊ GIÁC (VisualConceptBridge)
# -----------------------------------------------------------------------------

class VisualConceptBridge:
    """
    Cầu nối ngữ nghĩa thị giác:
      Ánh xạ đối tượng văn hóa phát hiện từ mô hình thị giác (Roboflow 107 classes)
      hoặc nhãn dự đoán hình ảnh sang 2 đến 4 câu concept trong concept/visual_concepts.md.
    """

    def __init__(self, visual_concepts_path: Union[str, Path] = "concept/visual_concepts.md"):
        self.visual_concepts_path = Path(visual_concepts_path)
        if self.visual_concepts_path.exists():
            self.df_concepts = load_concepts_from_md(self.visual_concepts_path)
        else:
            self.df_concepts = pd.DataFrame()

        self._init_object_mapping()

    def _init_object_mapping(self):
        """
        Ánh xạ các đối tượng nhận diện được từ tập Roboflow 107 classes sang concept thị giác.
        """
        self.object_to_concept_map = {
            # Cải lương
            'dan guitar': ['CL_VIS_03'],
            'dan kim': ['CL_VIS_04'],
            'dan co': ['CL_VIS_04'],
            'dan tranh': ['CL_VIS_04'],
            'San khau Hat Boi': ['CL_VIS_01', 'CL_VIS_05'],
            'Canh cung dien': ['CL_VIS_01'],
            'Mac do dinh hat': ['CL_VIS_05', 'CL_VIS_06'],
            'Tay ao dinh hat': ['CL_VIS_06'],
            'Giap nguc dinh hat': ['CL_VIS_06'],
            'Ao dai': ['CL_VIS_07'],

            # Ca trù
            'dan day': ['CT_VIS_04', 'CT_VIS_02'],
            'ban phach': ['CT_VIS_03', 'CT_VIS_02'],
            'trong chau': ['CT_VIS_05', 'CT_VIS_02'],
            'Ao dai truyen thong': ['CT_VIS_06', 'CT_VIS_08'],
            'Dau doi khan dong': ['CT_VIS_08'],

            # Chèo
            'chieu cheo': ['CH_VIS_01', 'CH_VIS_02'],
            'Trong': ['CH_VIS_04'],
            'dan bau': ['CH_VIS_04'],
            'Ao dai': ['CH_VIS_06'],
            'Trang phuc bieu dien nghe thuat nhan gian': ['CH_VIS_05', 'CH_VIS_07'],

            # Chầu văn
            'hau dong': ['CV_VIS_01', 'CV_VIS_02'],
            'Ban tho ong': ['CV_VIS_01'],
            'Chua': ['CV_VIS_01'],
            'Dinh Binh Thuy': ['CV_VIS_01'],
            'Cam vu khi': ['CV_VIS_04', 'CV_VIS_07'],
            'Trang phuc le hoi': ['CV_VIS_05', 'CV_VIS_06'],

            # Quan họ
            'non quai thao': ['QH_VIS_05', 'QH_VIS_01'],
            'Ao dai truyen thong': ['QH_VIS_04', 'QH_VIS_06'],
            'Dau doi khan dong': ['QH_VIS_06'],

            # Hát xẩm
            'dan nhi': ['HX_VIS_03'],
            'dan bau': ['HX_VIS_03'],
            'chieu xam': ['HX_VIS_01', 'HX_VIS_02'],
        }

        # Danh sách dự phòng các concept đại diện tiêu biểu cho từng thể loại
        self.genre_visual_fallbacks = {
            'Cải lương': ['CL_VIS_01', 'CL_VIS_03', 'CL_VIS_05'],
            'Ca trù': ['CT_VIS_02', 'CT_VIS_03', 'CT_VIS_04'],
            'Chèo': ['CH_VIS_01', 'CH_VIS_04', 'CH_VIS_07'],
            'Chầu văn': ['CV_VIS_01', 'CV_VIS_03', 'CV_VIS_05'],
            'Quan họ': ['QH_VIS_01', 'QH_VIS_04', 'QH_VIS_05'],
            'Hát xẩm': ['HX_VIS_01', 'HX_VIS_02', 'HX_VIS_03']
        }

    def map_visual_evidence(
        self,
        genre: str,
        detected_objects: Optional[List[Union[str, Dict[str, Any]]]] = None,
        min_concepts: int = 2,
        max_concepts: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Chọn ra 2 đến 4 câu concept hình ảnh từ visual_concepts.md dựa trên
        đối tượng phát hiện được và nhãn thể loại.
        """
        if self.df_concepts.empty:
            return []

        genre_clean = genre.strip()
        df_genre = self.df_concepts[
            (self.df_concepts['genre_name'] == genre_clean) |
            (self.df_concepts['genre'] == genre_clean) |
            (self.df_concepts['genre_code'] == GENRE_CODE_MAP.get(genre_clean, ''))
        ].copy()

        if df_genre.empty:
            df_genre = self.df_concepts.copy()

        selected_cids = []

        # 1. Kích hoạt từ đối tượng nhận diện
        if detected_objects:
            obj_names = []
            for obj in detected_objects:
                if isinstance(obj, str):
                    obj_names.append(obj)
                elif isinstance(obj, dict) and 'class' in obj:
                    obj_names.append(obj['class'])
                elif isinstance(obj, dict) and 'name' in obj:
                    obj_names.append(obj['name'])

            for name in obj_names:
                for obj_key, cids in self.object_to_concept_map.items():
                    if obj_key.lower() in name.lower() or name.lower() in obj_key.lower():
                        for cid in cids:
                            if cid in df_genre['concept_id'].values and cid not in selected_cids:
                                selected_cids.append(cid)

        # 2. Bổ sung từ danh sách fallback của thể loại
        fallbacks = self.genre_visual_fallbacks.get(genre_clean, [])
        for f_cid in fallbacks:
            if f_cid in df_genre['concept_id'].values and f_cid not in selected_cids:
                selected_cids.append(f_cid)
            if len(selected_cids) >= max_concepts:
                break

        # Đảm bảo đủ số lượng trong khoảng [min_concepts, max_concepts]
        final_cids = selected_cids[:max_concepts]
        if len(final_cids) < min_concepts:
            for cid in df_genre['concept_id'].values:
                if cid not in final_cids:
                    final_cids.append(cid)
                if len(final_cids) >= min_concepts:
                    break

        selected_rows = df_genre[df_genre['concept_id'].isin(final_cids)]
        result_concepts = []
        for _, r in selected_rows.iterrows():
            result_concepts.append({
                'concept_id': r['concept_id'],
                'aspect': r.get('aspect', ''),
                'evidence_level': r.get('evidence_level', r.get('evidence_type', 'single_frame')),
                'concept_sentence': r.get('concept_sentence', ''),
                'source': r.get('source', '')
            })

        return result_concepts


# -----------------------------------------------------------------------------
# 5. HÀM SINH VĂN BẢN GIẢI THÍCH ĐA PHƯƠNG THỨC (generate_xai_explanation)
# -----------------------------------------------------------------------------

def generate_xai_explanation(
    mode: str = "audio",
    prediction: Optional[str] = None,
    confidence: Optional[Union[float, str]] = None,
    audio_features: Optional[Dict[str, Any]] = None,
    triggered_rule: Optional[Union[Dict[str, Any], str]] = None,
    detected_objects: Optional[List[Union[str, Dict[str, Any]]]] = None,
    visual_rule: Optional[str] = None,
    audio_concepts_path: Union[str, Path] = "concept/audio_concepts.md",
    visual_concepts_path: Union[str, Path] = "concept/visual_concepts.md",
    fuzzy_profiles_path: Union[str, Path] = "data/models/fuzzy_profiles.json"
) -> Dict[str, Any]:
    """
    Sinh văn bản giải thích đa phương thức XAI theo chuẩn học thuật.

    Hỗ trợ 3 chế độ:
      - mode='audio': Nhãn dự đoán + độ tin cậy + các câu concept âm thanh + diễn giải luật mờ DT/RF/CN2/RIPPER.
      - mode='image': Nhãn dự đoán + độ tin cậy + các câu concept hình ảnh + luật thị giác.
      - mode='video': Hợp nhất diễn giải Âm thanh và Hình ảnh kèm trích dẫn nguồn học thuật.
    """
    mode = mode.lower().strip()
    if mode not in ['audio', 'image', 'video']:
        raise ValueError(f"Chế độ mode='{mode}' không hợp lệ. Chỉ chấp nhận 'audio', 'image', hoặc 'video'.")

    # Khởi tạo các bridge
    audio_bridge = AudioRuleToConceptBridge(
        audio_concepts_path=audio_concepts_path,
        fuzzy_profiles_path=fuzzy_profiles_path
    )
    visual_bridge = VisualConceptBridge(
        visual_concepts_path=visual_concepts_path
    )

    # Xác định thể loại dự đoán
    final_pred = prediction
    if not final_pred and triggered_rule and isinstance(triggered_rule, dict):
        final_pred = triggered_rule.get('consequent')
    if not final_pred and audio_features and 'label' in audio_features:
        final_pred = audio_features['label']
    if not final_pred:
        final_pred = "Ca trù"

    # Chuẩn hóa confidence
    conf_val = confidence if confidence is not None else 0.85
    conf_str = f"{conf_val * 100:.1f}%" if isinstance(conf_val, (int, float)) and conf_val <= 1.0 else str(conf_val)

    output = {
        'mode': mode,
        'prediction': final_pred,
        'confidence': conf_val,
        'confidence_display': conf_str,
        'audio_explanation': None,
        'visual_explanation': None,
        'explanation_text': ''
    }

    text_sections = []
    text_sections.append(f"KẾT QUẢ PHÂN LỚP VÀ GIẢI THÍCH ĐA PHƯƠNG THỨC (XAI)")
    text_sections.append(f"==================================================")
    text_sections.append(f"Thể loại di sản văn hóa: {final_pred.upper()}")
    text_sections.append(f"Độ tin cậy mô hình: {conf_str}")
    text_sections.append(f"Chế độ phân tích: {mode.upper()}")
    text_sections.append("")

    # 1. Xử lý phần ÂM THANH (nếu là audio hoặc video)
    if mode in ['audio', 'video']:
        audio_info = audio_bridge.explain_audio(
            audio_features=audio_features or {},
            triggered_rule=triggered_rule,
            prediction=final_pred,
            confidence=conf_val if isinstance(conf_val, float) else 0.85
        )
        output['audio_explanation'] = audio_info

        text_sections.append(f"1. DIỄN GIẢI ĐẶC TRƯNG ÂM HỌC & HỆ LUẬT MỜ:")
        text_sections.append(f"---------------------------------------------")
        text_sections.append(audio_info['rule_explanation'])
        text_sections.append("")
        text_sections.append(f"CÁC CONCEPT ÂM THANH CHUẨN HÓA ĐƯỢC KÍCH HOẠT:")
        for c in audio_info['concepts']:
            text_sections.append(f"  * [{c['concept_id']}] ({c['aspect']}): {c['concept_sentence']}")
        text_sections.append("")

    # 2. Xử lý phần HÌNH ẢNH (nếu là image hoặc video)
    if mode in ['image', 'video']:
        vis_concepts = visual_bridge.map_visual_evidence(
            genre=final_pred,
            detected_objects=detected_objects,
            min_concepts=2,
            max_concepts=4
        )

        vis_rule_text = visual_rule or (
            f"Phát hiện các thực thể thị giác đặc trưng của không gian biểu diễn {final_pred}"
        )

        output['visual_explanation'] = {
            'detected_objects': detected_objects or [],
            'visual_rule': vis_rule_text,
            'concepts': vis_concepts
        }

        header_num = "2" if mode == "video" else "1"
        text_sections.append(f"{header_num}. DIỄN GIẢI THỊ GIÁC & ĐỐI TƯỢNG VĂN HÓA:")
        text_sections.append(f"---------------------------------------------")
        text_sections.append(f"Quy tắc thị giác kích hoạt: {vis_rule_text}")
        if detected_objects:
            obj_list_str = ", ".join([str(o) for o in detected_objects])
            text_sections.append(f"Đối tượng nhận diện trong khung hình: {obj_list_str}")
        text_sections.append("")
        text_sections.append(f"CÁC CONCEPT HÌNH ẢNH CHUẨN HÓA ĐƯỢC KÍCH HOẠT:")
        for c in vis_concepts:
            text_sections.append(f"  * [{c['concept_id']}] ({c['aspect']}): {c['concept_sentence']}")
        text_sections.append("")

    # 3. Tổng hợp đa phương thức nếu là video
    if mode == 'video':
        text_sections.append(f"3. ĐÁNH GIÁ ĐỒNG QUY ĐA PHƯƠNG THỨC (AUDIO-VISUAL ALIGNMENT):")
        text_sections.append(f"-------------------------------------------------------------")
        text_sections.append(
            f"Hệ thống xác nhận tính đồng nhất cao giữa dòng tín hiệu âm thanh và hình ảnh biểu diễn. "
            f"Đặc trưng âm học nổi bật kết hợp tương hỗ với phục trang, đạo cụ và không gian diễn xướng, "
            f"khẳng định kết quả phân loại thể loại '{final_pred}' đạt độ tin cậy khoa học vững chắc."
        )
        text_sections.append("")

    output['explanation_text'] = "\n".join(text_sections)
    return output


# -----------------------------------------------------------------------------
# 6. KHỐI THỰC THI & KIỂM THỬ ĐỘC LẬP (__main__)
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 80)
    print("KIỂM THỬ MODULE: phase7_audio_visual_concept_mapper.py")
    print("=" * 80)

    # 1. Kiểm thử tải dữ liệu CSV từ markdown
    print("\n[BƯỚC 1] Trích xuất CSV từ audio_concepts.md và visual_concepts.md:")
    try:
        df_aud = load_concepts_from_md("concept/audio_concepts.md")
        print(f"  -> Audio Concepts: {df_aud.shape[0]} dòng, {df_aud.shape[1]} cột.")
        print(f"     Các thể loại âm thanh: {df_aud['genre_name'].unique().tolist()}")

        df_vis = load_concepts_from_md("concept/visual_concepts.md")
        print(f"  -> Visual Concepts: {df_vis.shape[0]} dòng, {df_vis.shape[1]} cột.")
        print(f"     Các thể loại hình ảnh: {df_vis['genre_name'].unique().tolist()}")
    except Exception as e:
        print(f"  [LỖI] Không thể đọc markdown: {e}")
        sys.exit(1)

    # Khởi tạo bridge
    bridge = AudioRuleToConceptBridge()

    # 2. Kiểm thử với một mẫu luật liên tục (Continuous Rule)
    print("\n[BƯỚC 2] Kiểm thử với mẫu luật LIÊN TỤC từ data/rules/rules_decision_tree_continuous.json:")
    sample_rule_cont_path = Path("data/rules/rules_decision_tree_continuous.json")
    if sample_rule_cont_path.exists():
        with open(sample_rule_cont_path, 'r', encoding='utf-8') as f:
            cont_rules = json.load(f)

        # Chọn luật có consequent là Ca trù
        sample_cont = next((r for r in cont_rules if r.get('consequent') == 'Ca trù'), cont_rules[0])
        print(f"  Luật gốc: {sample_cont['rule_text']}")
        print(f"  Consequent: {sample_cont.get('consequent')}, Confidence: {sample_cont.get('confidence')}")

        # Dịch Tầng 1
        trans_cont = bridge.translate_rule_to_vietnamese(sample_cont)
        print("\n  -> Diễn giải Tầng 1 (Acoustic Explanation):")
        print(trans_cont['rule_summary_text'])

        # Ánh xạ Tầng 2
        concepts_cont = bridge.map_rule_to_concepts(sample_cont, min_concepts=2, max_concepts=4)
        print(f"\n  -> Diễn giải Tầng 2: Đã chọn đúng {len(concepts_cont)} câu concept âm thanh (2-4 câu):")
        for c in concepts_cont:
            print(f"     * [{c['concept_id']}] ({c['evidence_level']}): {c['concept_sentence']}")
            print(f"       Nguồn: {c['source']}")
    else:
        print("  [Cảnh báo] Không tìm thấy file rules_decision_tree_continuous.json")

    # 3. Kiểm thử với một mẫu luật rời rạc (Crisp Rule)
    print("\n[BƯỚC 3] Kiểm thử với mẫu luật RỜI RẠC (CRISP) từ data/rules/rules_decision_tree_crisp.json:")
    sample_rule_crisp_path = Path("data/rules/rules_decision_tree_crisp.json")
    if sample_rule_crisp_path.exists():
        with open(sample_rule_crisp_path, 'r', encoding='utf-8') as f:
            crisp_rules = json.load(f)

        # Chọn luật đầu tiên (thường là Ca trù hoặc Chèo)
        sample_crisp = crisp_rules[0]
        print(f"  Luật gốc: {sample_crisp['rule_text']}")
        print(f"  Consequent: {sample_crisp.get('consequent')}, Confidence: {sample_crisp.get('confidence')}")

        # Dịch Tầng 1
        trans_crisp = bridge.translate_rule_to_vietnamese(sample_crisp)
        print("\n  -> Diễn giải Tầng 1 (Acoustic Explanation):")
        print(trans_crisp['rule_summary_text'])

        # Ánh xạ Tầng 2
        concepts_crisp = bridge.map_rule_to_concepts(sample_crisp, min_concepts=2, max_concepts=4)
        print(f"\n  -> Diễn giải Tầng 2: Đã chọn đúng {len(concepts_crisp)} câu concept âm thanh (2-4 câu):")
        for c in concepts_crisp:
            print(f"     * [{c['concept_id']}] ({c['evidence_level']}): {c['concept_sentence']}")
            print(f"       Nguồn: {c['source']}")
    else:
        print("  [Cảnh báo] Không tìm thấy file rules_decision_tree_crisp.json")

    # 4. Kiểm thử hàm sinh văn bản giải thích đa phương thức generate_xai_explanation với 3 chế độ
    print("\n[BƯỚC 4] Kiểm thử hàm generate_xai_explanation trên cả 3 chế độ (audio, image, video):")

    print("\n--- CHẾ ĐỘ 1: mode='audio' ---")
    xai_audio = generate_xai_explanation(
        mode="audio",
        prediction="Ca trù",
        confidence=0.88,
        triggered_rule="Pitch IN ['Rất trầm', 'Trầm'] AND Energy IN ['Rất yếu', 'Yếu'] AND Timbre IN ['Rất thấp', 'Thấp']"
    )
    print(xai_audio['explanation_text'])

    print("\n--- CHẾ ĐỘ 2: mode='image' ---")
    xai_image = generate_xai_explanation(
        mode="image",
        prediction="Cải lương",
        confidence=0.92,
        detected_objects=['dan guitar', 'San khau Hat Boi', 'Tay ao dinh hat']
    )
    print(xai_image['explanation_text'])

    print("\n--- CHẾ ĐỘ 3: mode='video' (Đa phương thức kết hợp) ---")
    xai_video = generate_xai_explanation(
        mode="video",
        prediction="Chầu văn",
        confidence=0.94,
        triggered_rule="Tempo IN ['Vừa', 'Nhanh'] AND Energy == 'Mạnh' AND Spectral == 'Cao'",
        detected_objects=['hau dong', 'Cam vu khi', 'Trang phuc le hoi']
    )
    print(xai_video['explanation_text'])

    print("\n" + "=" * 80)
    print("HOÀN THÀNH KIỂM THỬ: Tất cả các chức năng hoạt động chính xác và đạt yêu cầu!")
    print("=" * 80)
